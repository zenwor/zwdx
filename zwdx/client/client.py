import torch
import requests
import os
import time
import threading
import socketio
import socket
import base64
import logging
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

from zwdx.client.utils import (
    safe_loads, 
    safe_dumps, 
    deserialize_model_payload, 
    deserialize_function_payload
)

import faulthandler
faulthandler.enable()

ROOM_TOKEN = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("client")

SERVER_URL = getenv("SERVER_URL")
HOSTNAME = f"{os.uname().nodename}_{os.getpid()}"

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

@sio.on("registration_ack")
def registration_ack(data):
    status = data.get("status")
    room = data.get("room")
    if status in ("joined_room", "joined_public"):
        logger.info(f"Successfully registered in {room}")
    else:
        logger.error(f"Registration failed: {data.get('message')}")

def rebuild_optimizer(model, optimizer_config):
    cls = getattr(torch.optim, optimizer_config["class"])
    return cls(model.parameters(), **optimizer_config["kwargs"])

def run_training_worker(serialized_data):
    import os
    import datetime

    global rank, world_size, master_addr
    faulthandler.enable()

    model = deserialize_model_payload(serialized_data.get("model_payload"))
    if model is None:
        logger.error("[Worker] Failed to deserialize model")
        return
    logger.info("[Worker] Model deserialized successfully")

    data_loader_func = deserialize_function_payload(serialized_data.get("data_loader_payload"))
    if data_loader_func is None:
        logger.error("[Worker] Failed to deserialize data_loader_func, using fallback")
        # Fallback to a dummy loader
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        def data_loader_func(rank, world_size, path=None):
            dataset = datasets.FakeData(transform=transforms.ToTensor())
            return DataLoader(dataset, batch_size=4)

    train_func = deserialize_function_payload(serialized_data.get("train_payload"))
    eval_func = deserialize_function_payload(serialized_data.get("eval_payload"))
    data_fetch_func = deserialize_function_payload(serialized_data.get("data_fetch_payload"))

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(serialized_data.get("master_port", os.environ.get("MASTER_PORT", "")))

    if world_size > 1:
        try:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=60)
            )
            logger.info("[Worker] Distributed process group initialized.")
        except Exception as e:
            logger.error(f"[Worker] DDP init failed: {e}")
            logger.error(traceback.format_exc())
            return
    else:
        logger.info("[Worker] world_size == 1, running single-process training.")

    device = torch.device("cpu")
    try:
        if torch.cuda.is_available():
            chosen_idx = rank % max(1, torch.cuda.device_count())
            device = torch.device(f"cuda:{chosen_idx}")
            logger.info(f"[Worker] Using GPU device cuda:{chosen_idx}")
        else:
            logger.info(f"[Worker] No GPU available, using CPU")
        
        # Move model to device BEFORE wrapping with DDP
        model = model.to(device)
        logger.info(f"[Worker] Model moved to {device}")
    except Exception as e:
        logger.error(f"[Worker] Device placement failed: {e}")
        logger.error(traceback.format_exc())
        if world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return

    try:
        parallelism = serialized_data.get("parallelism", "DDP")
        if parallelism == "DDP" and world_size > 1 and torch.distributed.is_initialized():
            if device.type == "cuda":
                model = DDP(model, device_ids=[chosen_idx], output_device=chosen_idx)
            else:
                model = DDP(model)
            logger.info(f"[Worker] Model wrapped with DDP (device: {device})")
        elif parallelism == "FSDP":
            model = FSDP(model)
            logger.info("[Worker] Model wrapped with FSDP")
    except Exception as e:
        logger.warning(f"[Worker] Parallel wrapper failed: {e}")
        logger.warning(traceback.format_exc())

    torch.manual_seed(42 + (rank or 0))

    data_path_or_dataset = None
    if data_fetch_func:
        try:
            data_path_or_dataset = data_fetch_func(rank, world_size)
            logger.info(f"[Worker] data_fetch_func returned: {data_path_or_dataset}")
        except Exception as e:
            logger.warning(f"[Worker] data_fetch_func failed: {e}")

    try:
        try:
            data_loader = data_loader_func(rank, world_size, data_path_or_dataset)
        except TypeError:
            data_loader = data_loader_func(rank, world_size)
        logger.info(f"[Worker] DataLoader created - dataset_len={len(data_loader.dataset)}, batches={len(data_loader)}")
    except Exception as e:
        logger.error(f"[Worker] DataLoader creation failed: {e}")
        logger.error(traceback.format_exc())
        if world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return

    try:
        it = iter(data_loader)
        batch = next(it)
        sample = batch[0] if isinstance(batch, (list, tuple)) else batch

        device = next(model.parameters()).device
        sample = sample.to(device) if isinstance(sample, torch.Tensor) else None

        with torch.no_grad():
            out = model(sample) if sample is not None else None
        logger.info(f"[Worker] Smoke forward pass OK; out_shape={getattr(out, 'shape', None)}")
    except Exception as e:
        logger.error(f"[Worker] Smoke forward/test pass failed: {e}")
        logger.error(traceback.format_exc())
        if world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return

    try:
        if serialized_data.get("optimizer_bytes"):
            optimizer_config = safe_loads(base64.b64decode(serialized_data["optimizer_bytes"]))
            cls = getattr(torch.optim, optimizer_config["class"])
            optimizer = cls(model.parameters(), **optimizer_config["kwargs"])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    except Exception as e:
        logger.warning(f"[Worker] Optimizer setup failed: {e}")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    final_loss = None
    try:
        reporter = Reporter(sio, serialized_data.get("job_id"), rank)
        if train_func:
            final_loss = train_func(
                model=model,
                data_loader=data_loader,
                rank=rank,
                world_size=world_size,
                eval_func=eval_func,
                optimizer=optimizer,
                reporter=reporter,
            )
            logger.info(f"[Worker] Training finished, final_loss={final_loss}")
        else:
            logger.warning("[Worker] No train_func provided; skipping training.")
    except Exception as e:
        logger.error(f"[Worker] Training raised: {e}")
        logger.error(traceback.format_exc())

    try:
        if rank == 0:
            underlying_model = getattr(model, "module", model)
            model_state = underlying_model.state_dict()
            sio.emit("training_done", {
                "job_id": serialized_data.get("job_id"),
                "final_loss": final_loss,
                "model_state_dict_bytes": base64.b64encode(dill.dumps(model_state)).decode("utf-8")
            })
            logger.info("[Worker] training_done emitted")
    except Exception as e:
        logger.error(f"[Worker] Emitting training_done failed: {e}")
        logger.error(traceback.format_exc())

    try:
        if world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
    except Exception:
        pass

@sio.on("start_training")
def on_start_training(data):
    logger.info("start_training received; spawning worker thread")
    if rank is None:
        logger.error("Rank not assigned yet; cannot start training")
        return
    threading.Thread(target=run_training_worker, args=(data,), daemon=True).start()
    logger.info("Worker thread started; returning from Socket.IO handler")

def register_client():
    gpus = []
    gpu_count = torch.cuda.device_count()
    for i in range(gpu_count):
        gpus.append({
            "name": torch.cuda.get_device_name(i),
            "capability": torch.cuda.get_device_capability(i),
            "memory": torch.cuda.get_device_properties(i).total_memory
        })

    try:
        local_ip = socket.gethostbyname(socket.gethostname())
        if local_ip.startswith("127."):
            local_ip = "127.0.0.1"
    except Exception:
        local_ip = "127.0.0.1"

    client_id = f"{HOSTNAME}_{ROOM_TOKEN or 'public'}"

    data = {
        "hostname": HOSTNAME,
        "gpus": gpus,
        "client_ip": local_ip,
        "client_id": client_id,
        "room_token": ROOM_TOKEN,
    }
    logger.info(f"Registering client: {HOSTNAME}, IP={local_ip}, room_token={ROOM_TOKEN}")
    sio.emit("register_client", data)

def send_heartbeat():
    session = requests.Session()
    while True:
        try:
            session.post(f"{SERVER_URL}/heartbeat", json={"hostname": HOSTNAME}, timeout=10)
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
        time.sleep(5)

def parse_args():
    parser = argparse.ArgumentParser(description="Client for distributed training")
    parser.add_argument(
        "--room_token",
        type=str,
        default=None,
        help="Room token to join a private training room. If not provided, joins public pool."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    ROOM_TOKEN = args.room_token
    
    threading.Thread(target=send_heartbeat, daemon=True).start()
    try:
        logger.info(f"Connecting to {SERVER_URL}")
        sio.connect(SERVER_URL, transports=["websocket"], wait_timeout=30)
        sio.wait()
    except Exception as e:
        logger.error(f"Failed to connect to {SERVER_URL}: {e}")