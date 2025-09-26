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

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from zwdx.utils import getenv

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


@sio.on("start_training")
def on_start_training(data):
    logger.info("Starting training...")
    parallelism = data["parallelism"]
    model_bytes_enc = data["model_bytes"]
    data_loader_bytes_enc = data["data_loader_bytes"]
    master_port = data["master_port"]
    job_id = data["job_id"]

    try:
        model = cloudpickle.loads(base64.b64decode(model_bytes_enc))
        data_loader_func = cloudpickle.loads(base64.b64decode(data_loader_bytes_enc))
    except Exception as e:
        logger.error(f"Deserialization failed: {e}")
        return

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    try:
        torch.distributed.init_process_group(
            backend="gloo", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=60)
        )
    except Exception as e:
        logger.error(f"DDP init failed: {e}")
        return

    if torch.cuda.is_available():
        model = model.cuda()
    if parallelism == "DDP":
        model = DDP(model, device_ids=[0] if torch.cuda.is_available() else None)
    elif parallelism == "FSDP":
        model = FSDP(model)
        
    torch.manual_seed(42 + rank)

    try:
        data_loader = data_loader_func(rank, world_size)
        logger.info(f"Rank {rank}: dataset size={len(data_loader.dataset)}, loader len={len(data_loader)}")

        try:
            first_batch = next(iter(data_loader))
            images, labels = first_batch
            logger.info(f"Rank {rank}: first 5 sample indices (approx, via sampler) = {list(data_loader.sampler)[:5]}")
            logger.info(f"Rank {rank}: first 5 labels = {labels[:5].tolist()}")
        except Exception as e:
            logger.warning(f"Rank {rank}: failed to inspect first batch: {e}")

    except Exception as e:
        logger.error(f"Data loader failed: {e}")
        torch.distributed.destroy_process_group()
        return

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    final_loss = None

    try:
        for epoch in range(5):
            if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
                data_loader.sampler.set_epoch(epoch)

            epoch_loss, batch_count = 0.0, 0
            for data, target in data_loader:
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1

            local_avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
            logger.info(f"Rank {rank}: Epoch {epoch} local avg loss={local_avg_loss:.4f}")

            loss_tensor = torch.tensor(local_avg_loss, device="cuda" if torch.cuda.is_available() else "cpu")
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)
            global_avg_loss = loss_tensor.item()

            if rank == 0:
                sio.emit("training_progress", {"job_id": job_id, "epoch": epoch, "loss": global_avg_loss})
            final_loss = global_avg_loss
    except Exception as e:
        logger.error(f"Training failed: {e}")

    torch.distributed.barrier()
    if rank == 0:
        sio.emit("training_done", {"final_loss": final_loss, "job_id": job_id})
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