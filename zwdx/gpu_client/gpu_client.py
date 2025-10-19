import torch
import requests
import os
import time
import threading
import socketio
import socket
import base64
import datetime
import dill
import traceback
import argparse
import io

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.utils.data import DataLoader

from zwdx import Reporter
from zwdx.utils import getenv

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("client")

class GPUClient:
    def __init__(self, server_url, room_token=None):
        self.server_url = server_url
        self.room_token = room_token
        self.sio = socketio.Client(reconnection_attempts=5, reconnection_delay=2, request_timeout=30)
        self.rank = None
        self.world_size = None
        self.master_addr = None
        self.gpus = []
        self.ip = None
        self.client_id = None

        self._register_socket_handlers()
    
    def _register_socket_handlers(self):
        @self.sio.event
        def connect():
            self.register()

        @self.sio.event
        def connect_error(data):
            logger.error(f"Connection failed: {data}")

        @self.sio.on("assign_rank")
        def on_assign_rank(data):
            self.rank = data["rank"]
            self.world_size = data["world_size"]
            self.master_addr = data["master_addr"]
            logger.info(f"[{self.client_id}] Assigned rank={self.rank}, world_size={self.world_size}, master_addr={self.master_addr}")

        @self.sio.on("registration_ack")
        def registration_ack(data):
            self.sid = data.get("sid")
            self.client_id = data.get("client_id")
            status = data.get("status")
            room = data.get("room")

            if status == "ok":
                logger.info(f"Registered successfully — client_id={self.client_id}, sid={self.sid}, room={room}")
            else:
                logger.error(f"Registration failed: {data}.")

        @self.sio.on("start_training")
        def on_start_training(data):
            logger.info(f"[{self.client_id}]  start_training received")
            threading.Thread(target=self._run_training_worker, args=(data,), daemon=True).start()
            
    def start(self):
        threading.Thread(target=self._send_heartbeat, daemon=True).start()
        logger.info(f"Connecting to {self.server_url}")
        self.sio.connect(self.server_url, transports=["websocket"], wait_timeout=30)
        self.sio.wait()

    def register(self):
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            self.gpus.append({
                "name": torch.cuda.get_device_name(i),
                "capability": torch.cuda.get_device_capability(i),
                "memory": props.total_memory
            })
        local_ip = self._get_local_ip()
        payload = {
            "gpus": self.gpus,
            "client_ip": self.ip,
            "room_token": self.room_token,
        }
        logger.info(f"Registering client with room token={self.room_token}")
        self.sio.emit("register_client", payload)

    def _get_local_ip(self):
        try:
            self.ip = socket.gethostbyname(socket.gethostname())
            return self.ip if not self.ip.startswith("127.") else "127.0.0.1"
        except:
            return "127.0.0.1"

    def _send_heartbeat(self):
        session = requests.Session()
        while True:
            try:
                session.post(f"{self.server_url}/heartbeat", json={"client_id": self.client_id}, timeout=10)
            except Exception as e:
                logger.error(f"[{self.client_id}] Heartbeat failed: {e}")
            time.sleep(5)

    def _run_training_worker(self, data):
        """Enhanced training worker with error handling."""
        job_id = data.get("job_id")
        try:
            from zwdx.gpu_client.training import run_training_worker
            run_training_worker(data, self.rank, self.world_size, self.master_addr, self.sio)
        except Exception as e:
            logger.error(f"[{self.client_id}] Training worker crashed: {e}")
            logger.error(traceback.format_exc())
            
            # Emit failure to server
            if self.sio and self.sio.connected:
                try:
                    self.sio.emit("training_failed", {
                        "job_id": job_id,
                        "reason": f"Training exception: {str(e)}",
                        "client_id": self.client_id,
                        "rank": self.rank
                    })
                    logger.info(f"[{self.client_id}] Emitted training_failed for job {job_id}")
                except Exception as emit_error:
                    logger.error(f"[{self.client_id}] Failed to emit training_failed: {emit_error}")


def parse_args():
    parser = argparse.ArgumentParser(description="GPU client for distributed training")
    parser.add_argument(
        "-rt", "--room_token",
        type=str,
        default=None,
        required=True,
        help="Room token to join a private training room."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    server_url = getenv("SERVER_URL")

    client = GPUClient(server_url=server_url, room_token=args.room_token)
    client.start()