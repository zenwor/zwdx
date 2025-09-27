import torch
import requests
import os
import time
import threading
import socket
import socketio
import cloudpickle
import base64
import logging
import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

import torch.optim as optim
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from zwdx.utils import getenv
from zwdx import Reporter

SERVER_URL = getenv("SERVER_URL")
HOSTNAME = f"{os.uname().nodename}_{os.getpid()}"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("client")

sio = socketio.Client(reconnection_attempts=5, reconnection_delay=2, request_timeout=30)

rank, world_size, master_addr = None, None, None

@sio.event
def connect():
    logger.info("Connected to server")
    register_client()

@sio.event
def connect_error(data):
    logger.error(f"Connection failed: {data}")

@sio.on("assign_rank")
def on_assign_rank(data):
    global rank, world_size, master_addr
    rank = data["rank"]
    world_size = data["world_size"]
    master_addr = data["master_addr"]
    logger.info(f"Assigned rank={rank}, world_size={world_size}, master_addr={master_addr}")

def rebuild_optimizer(model, optimizer_config):
    cls = getattr(optim, optimizer_config["class"])
    return cls(model.parameters(), **optimizer_config["kwargs"])

@sio.on("start_training")
def on_start_training(data):
    global job_id
    logger.info("Starting training...")
    
    # Deserialize
    parallelism = data["parallelism"]
    master_port = data["master_port"]
    job_id = data["job_id"]
    
    if rank is None:
        logger.error("Rank not assigned yet; cannot start training.")
        return
    
    try:
        model = cloudpickle.loads(base64.b64decode(data["model_bytes"]))
        data_loader_func = cloudpickle.loads(base64.b64decode(data["data_loader_bytes"]))
        train_func = cloudpickle.loads(base64.b64decode(data.get("train_func_bytes"))) if data.get("train_func_bytes") else None
        eval_func = cloudpickle.loads(base64.b64decode(data.get("eval_func_bytes"))) if data.get("eval_func_bytes") else None
    except Exception as e:
        logger.error(f"Deserialization failed: {e}")
        return

    # Distributed setup
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    try:
        torch.distributed.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=60)
        )
    except Exception as e:
        logger.error(f"DDP init failed: {e}")
        return

    # Device placement + parallelism
    if torch.cuda.is_available():
        model = model.cuda()

    if parallelism == "DDP":
        model = DDP(model, device_ids=[0] if torch.cuda.is_available() else None)
    elif parallelism == "FSDP":
        model = FSDP(model)

    torch.manual_seed(42 + rank)

    # Data loader
    try:
        data_loader = data_loader_func(rank, world_size)
        logger.info(f"Rank {rank}: dataset size={len(data_loader.dataset)}, loader len={len(data_loader)}")
    except Exception as e:
        logger.error(f"Data loader failed: {e}")
        torch.distributed.destroy_process_group()
        return

    # Training
    try:
        if train_func is not None:
            # Optimizer
            if data.get("optimizer_bytes"):
                optimizer_config = cloudpickle.loads(base64.b64decode(data["optimizer_bytes"]))
                optimizer = rebuild_optimizer(model, optimizer_config)
            else:
                optimizer = None
                logger.error(f"Optimizer not passed.")
            
            # Reporter
            reporter = Reporter(sio, job_id, rank)
            
            # Run training
            final_loss = train_func(
                model=model,
                data_loader=data_loader,
                rank=rank,
                world_size=world_size,
                eval_func=eval_func,
                optimizer=optimizer,
                reporter=reporter,
            )

            # Emit results back to server
            model_state_dict_bytes = None
            if rank == 0:
                try:
                    if isinstance(model, (DDP, FSDP)):
                        underlying_model = model.module
                    else:
                        underlying_model = model

                    if isinstance(model, FSDP):
                        # Use context manager for full, unflattened state dict
                        with FSDP.state_dict_type(
                            model,
                            StateDictType.FULL_STATE_DICT,
                            FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                        ):
                            model_state_dict = model.state_dict()
                    else:
                        model_state_dict = underlying_model.state_dict()

                    model_state_dict_bytes = base64.b64encode(
                        cloudpickle.dumps(model_state_dict)
                    ).decode("utf-8")

                except Exception as e:
                    logger.error(f"Failed to serialize model state dict: {e}")

                
                if model_state_dict_bytes is not None:
                    try:
                        sio.emit(
                            "training_done",
                            {
                                "final_loss": final_loss,
                                "job_id": job_id,
                                "model_state_dict_bytes": model_state_dict_bytes
                            }
                        )
                        logger.info(f"Rank {rank}: training_done emitted")
                    except Exception as e:
                        logger.error(f"Failed to emit training_done: {e}")

        else:
            logger.warning("No train_func provided; skipping training.")

    except Exception as e:
        logger.error(f"Training failed: {e}")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()

def register_client():
    gpu_count = torch.cuda.device_count()
    gpus = []
    for i in range(gpu_count):
        gpus.append(
            {
                "name": torch.cuda.get_device_name(i) if gpu_count > 0 else "CPU",
                "capability": torch.cuda.get_device_capability(i) if gpu_count > 0 else [0, 0],
                "memory": torch.cuda.get_device_properties(i).total_memory if gpu_count > 0 else 0,
            }
        )

    try:
        local_ip = socket.gethostbyname(socket.gethostname())
        if local_ip.startswith("127."):
            local_ip = "127.0.0.1"
    except Exception:
        local_ip = "127.0.0.1"

    data = {"hostname": HOSTNAME, "gpus": gpus, "client_ip": local_ip}
    logger.info(f"Registering client: {HOSTNAME}, IP={local_ip}")
    sio.emit("register", data)

def send_heartbeat():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    while True:
        try:
            session.post(f"{SERVER_URL}/heartbeat", json={"hostname": HOSTNAME}, timeout=10)
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
        time.sleep(5)

if __name__ == "__main__":
    threading.Thread(target=send_heartbeat, daemon=True).start()
    try:
        logger.info(f"Connecting to {SERVER_URL}")
        sio.connect(SERVER_URL, transports=["websocket"], wait_timeout=30)
        sio.wait()
    except Exception as e:
        logger.error(f"Failed to connect to {SERVER_URL}: {e}")