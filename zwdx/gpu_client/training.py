# ==================== 1. UPDATE training.py ====================
# Add server health monitoring during training

import os
import sys
import base64
import logging
import traceback
import datetime
import torch
import dill
import threading
import time

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader

from zwdx import Reporter
from zwdx.gpu_client.utils import (
    safe_loads, 
    deserialize_model_payload, 
    deserialize_function_payload,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("client")

# Global flag to signal training termination
_should_terminate = threading.Event()

def _server_health_monitor(sio, job_id, check_interval=5, max_failures=3):
    """
    Monitor server health in background thread.
    Terminates training if server becomes unreachable.
    """
    consecutive_failures = 0
    
    while not _should_terminate.is_set():
        try:
            if sio and sio.connected:
                # Server is responsive
                consecutive_failures = 0
            else:
                # Server is not connected
                consecutive_failures += 1
                logger.warning(f"[HealthMonitor] Server unreachable ({consecutive_failures}/{max_failures})")
                
                if consecutive_failures >= max_failures:
                    logger.error(f"[HealthMonitor] Server unreachable for {max_failures * check_interval}s - TERMINATING TRAINING")
                    _should_terminate.set()
                    
                    # Force exit after a brief delay
                    time.sleep(2)
                    logger.error("[HealthMonitor] Forcing process exit")
                    os._exit(1)  # Hard exit to kill all threads
                    
        except Exception as e:
            logger.error(f"[HealthMonitor] Error checking server: {e}")
            consecutive_failures += 1
        
        time.sleep(check_interval)


class TrainingTerminated(Exception):
    """Exception raised when training is terminated due to server loss."""
    pass


def _check_should_continue():
    """Check if training should continue. Raises exception if terminated."""
    if _should_terminate.is_set():
        raise TrainingTerminated("Server connection lost - training terminated")


def run_training_worker(data, rank, world_size, master_addr, sio):
    """
    Main training worker function with server health monitoring.
    """
    job_id = data.get("job_id")
    
    # Reset termination flag
    _should_terminate.clear()
    
    # Start server health monitor
    health_monitor = threading.Thread(
        target=_server_health_monitor,
        args=(sio, job_id),
        kwargs={"check_interval": 5, "max_failures": 3},
        daemon=True
    )
    health_monitor.start()
    logger.info("[Worker] Server health monitor started")
    
    def emit_failure(reason):
        """Helper to emit failure to server."""
        if sio and sio.connected:
            try:
                sio.emit("training_failed", {
                    "job_id": job_id,
                    "reason": reason,
                    "rank": rank
                })
                logger.error(f"[Worker] Emitted training failure: {reason}")
            except Exception as e:
                logger.error(f"[Worker] Failed to emit training_failed: {e}")
    
    try:
        # Check server before starting
        _check_should_continue()
        
        # Deserialize components
        model = deserialize_model_payload(data.get("model_payload"))
        if model is None:
            emit_failure("Failed to deserialize model")
            return
        logger.info("[Worker] Model deserialized successfully")
        
        _check_should_continue()

        data_loader_func = deserialize_function_payload(data.get("data_loader_payload"))
        if data_loader_func is None:
            emit_failure("Failed to deserialize data_loader_func")
            return

        train_func = deserialize_function_payload(data.get("train_payload"))
        eval_func = deserialize_function_payload(data.get("eval_payload"))
        data_fetch_func = deserialize_function_payload(data.get("data_fetch_payload"))

        # Setup distributed environment
        try:
            _check_should_continue()
            _setup_distributed(rank, world_size, master_addr, data.get("master_port", os.environ.get("MASTER_PORT", "")))
        except TrainingTerminated:
            raise
        except Exception as e:
            emit_failure(f"Distributed setup failed: {str(e)}")
            return

        # Setup device and move model
        try:
            _check_should_continue()
            device, device_idx = _setup_device(rank)
            model = model.to(device)
            logger.info(f"[Worker] Model moved to {device}")
        except TrainingTerminated:
            raise
        except Exception as e:
            emit_failure(f"Device placement failed: {str(e)}")
            _cleanup_distributed(world_size)
            return

        # Wrap model with parallelization
        try:
            _check_should_continue()
            model = _wrap_model_parallel(model, data.get("parallelism", "DDP"), world_size, device, device_idx)
        except TrainingTerminated:
            raise
        except Exception as e:
            logger.warning(f"[Worker] Parallel wrapper failed: {e}")

        # Set random seed
        torch.manual_seed(42 + (rank or 0))

        # Create data loader
        try:
            _check_should_continue()
            data_loader = _create_data_loader(data_loader_func, data_fetch_func, rank, world_size)
        except TrainingTerminated:
            raise
        except Exception as e:
            emit_failure(f"DataLoader creation failed: {str(e)}")
            _cleanup_distributed(world_size)
            return

        # Smoke test
        try:
            _check_should_continue()
            _smoke_test_model(model, data_loader)
        except TrainingTerminated:
            raise
        except Exception as e:
            emit_failure(f"Model smoke test failed: {str(e)}")
            _cleanup_distributed(world_size)
            return

        # Setup optimizer
        optimizer = _setup_optimizer(data, model)

        # Run training with periodic server checks
        final_loss = None
        try:
            _check_should_continue()
            reporter = ServerAwareReporter(sio, job_id, rank)
            
            if train_func:
                final_loss = train_func(
                    model=model,
                    loader=data_loader,
                    rank=rank,
                    world_size=world_size,
                    eval_func=eval_func,
                    optim=optimizer,
                    reporter=reporter,
                )
                logger.info(f"[Worker] Training finished, final_loss={final_loss}")
            else:
                emit_failure("No train_func provided")
                _cleanup_distributed(world_size)
                return
        except TrainingTerminated as e:
            logger.error(f"[Worker] Training terminated: {e}")
            raise
        except Exception as e:
            emit_failure(f"Training loop exception: {str(e)}")
            _cleanup_distributed(world_size)
            return

        # Emit results (only on success)
        _check_should_continue()
        _emit_training_results(sio, job_id, rank, final_loss, model)

    except TrainingTerminated as e:
        logger.error(f"[Worker] Training terminated due to server loss: {e}")
        # Don't try to emit failure - server is gone
        _cleanup_distributed(world_size)
        # Exit the process
        time.sleep(1)
        os._exit(1)
        
    except Exception as e:
        # Catch-all for any unexpected errors
        emit_failure(f"Unexpected error: {str(e)}")
        logger.error(f"[Worker] Unexpected error: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Signal health monitor to stop
        _should_terminate.set()
        # Always cleanup
        _cleanup_distributed(world_size)


class ServerAwareReporter(Reporter):
    """Reporter that checks server health before logging."""
    
    def log_metrics(self, **kwargs):
        """Log metrics and check if server is still reachable."""
        # Check server health before logging
        _check_should_continue()
        
        # Call parent method
        return super().log_metrics(**kwargs)
    
    def log(self, *args):
        """Log message and check if server is still reachable."""
        # Check server health before logging
        _check_should_continue()
        
        # Call parent method
        return super().log(*args)


def _setup_distributed(rank, world_size, master_addr, master_port):
    """Initialize distributed training environment."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

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
            raise
    else:
        logger.info("[Worker] world_size == 1, running single-process training.")


def _setup_device(rank):
    """Setup and return compute device (GPU or CPU)."""
    if torch.cuda.is_available():
        chosen_idx = rank % max(1, torch.cuda.device_count())
        device = torch.device(f"cuda:{chosen_idx}")
        logger.info(f"[Worker] Using GPU device cuda:{chosen_idx}")
        return device, chosen_idx
    else:
        logger.info(f"[Worker] No GPU available, using CPU")
        return torch.device("cpu"), None


def _wrap_model_parallel(model, parallelism, world_size, device, device_idx):
    """Wrap model with appropriate parallelization strategy."""
    if parallelism == "DDP" and world_size > 1 and torch.distributed.is_initialized():
        if device.type == "cuda":
            model = DDP(model, device_ids=[device_idx], output_device=device_idx)
        else:
            model = DDP(model)
        logger.info(f"[Worker] Model wrapped with DDP (device: {device})")
    elif parallelism == "FSDP":
        model = FSDP(model)
        logger.info("[Worker] Model wrapped with FSDP")
    
    return model


def _create_data_loader(data_loader_func, data_fetch_func, rank, world_size):
    """Create data loader, optionally fetching data first."""
    data_path_or_dataset = None
    if data_fetch_func:
        try:
            data_path_or_dataset = data_fetch_func(rank, world_size)
            logger.info(f"[Worker] data_fetch_func returned: {data_path_or_dataset}")
        except Exception as e:
            logger.warning(f"[Worker] data_fetch_func failed: {e}")

    try:
        data_loader = data_loader_func(rank, world_size, data_path_or_dataset)
    except TypeError:
        data_loader = data_loader_func(rank, world_size)
    
    logger.info(f"[Worker] DataLoader created - dataset_len={len(data_loader.dataset)}, batches={len(data_loader)}")
    return data_loader


def _smoke_test_model(model, data_loader):
    """Run a quick forward pass to verify model works."""
    it = iter(data_loader)
    batch = next(it)
    sample = batch[0] if isinstance(batch, (list, tuple)) else batch

    device = next(model.parameters()).device
    sample = sample.to(device) if isinstance(sample, torch.Tensor) else None

    with torch.no_grad():
        out = model(sample) if sample is not None else None
    
    logger.info(f"[Worker] Smoke forward pass OK; out_shape={getattr(out, 'shape', None)}")


def _setup_optimizer(data, model):
    """Setup optimizer from serialized config or use default."""
    if data.get("optimizer_bytes"):
        try:
            optimizer_config = safe_loads(base64.b64decode(data["optimizer_bytes"]))
            cls = getattr(torch.optim, optimizer_config["class"])
            return cls(model.parameters(), **optimizer_config["kwargs"])
        except Exception as e:
            logger.warning(f"[Worker] Optimizer setup failed: {e}")
    
    return torch.optim.Adam(model.parameters(), lr=1e-3)


def _emit_training_results(sio, job_id, rank, final_loss, model):
    """Emit training completion results (rank 0 only)."""
    if rank != 0:
        return

    try:
        underlying_model = getattr(model, "module", model)
        model_state = underlying_model.state_dict()
        sio.emit("training_done", {
            "job_id": job_id,
            "final_loss": final_loss,
            "model_state_dict_bytes": base64.b64encode(dill.dumps(model_state)).decode("utf-8")
        })
        logger.info("[Worker] training_done emitted")
    except Exception as e:
        logger.error(f"[Worker] Emitting training_done failed: {e}")
        logger.error(traceback.format_exc())


def _cleanup_distributed(world_size):
    """Cleanup distributed training resources."""
    if world_size > 1 and torch.distributed.is_initialized():
        try:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        except Exception:
            pass