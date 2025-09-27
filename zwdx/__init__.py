import requests
import cloudpickle
import base64
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
import logging
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading

logger = logging.getLogger("zwdx")
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

class ZWDX:
    def __init__(self, server_url):
        self.server_url = server_url

    def submit_job(
        self,
        model,
        data_loader_func,
        train_func=None,
        eval_func=None,
        optimizer=None,
        parallelism="DDP",
        memory_required=0
    ):
        try:
            model_bytes = cloudpickle.dumps(model)
            data_loader_bytes = cloudpickle.dumps(data_loader_func)
            train_bytes = base64.b64encode(cloudpickle.dumps(train_func)).decode("utf-8") if train_func else None
            eval_bytes = base64.b64encode(cloudpickle.dumps(eval_func)).decode("utf-8") if eval_func else None
            if optimizer is not None:
                optimizer_config = self.serialize_optimizer(optimizer)
                optimizer_bytes = base64.b64encode(cloudpickle.dumps(optimizer_config)).decode("utf-8")
            else:
                optimizer_bytes = None

            job = {
                "model_bytes": base64.b64encode(model_bytes).decode("utf-8"),
                "data_loader_bytes": base64.b64encode(data_loader_bytes).decode("utf-8"),
                "train_func_bytes": train_bytes,
                "eval_func_bytes": eval_bytes,
                "optimizer_bytes": optimizer_bytes,
                "parallelism": parallelism,
                "memory_required": memory_required,
            }

            logger.info(f"Submitting job to {self.server_url}")
            session = requests.Session()
            retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
            session.mount("https://", HTTPAdapter(max_retries=retries))
            response = session.post(f"{self.server_url}/submit_job", json=job, timeout=10)
            result = response.json()

            if result["status"] != "job started":
                return result

            job_id = result["job_id"]
            logger.info(f"Polling for job {job_id} results")

            last_progress_count = 0
            while True:
                response = session.get(f"{self.server_url}/get_results/{job_id}", timeout=10)
                result = response.json()
                if result["status"] == "pending":
                    new_progress = result["progress"][last_progress_count:]
                    for update in new_progress:
                        log_message = f"Job {job_id}, rank {update.get('rank', 'unknown')}: {update}"
                        logger.info(log_message)
                    last_progress_count = len(result["progress"])
                    time.sleep(2)
                elif result["status"] == "complete":
                    logger.info(f"Job {job_id} completed: final_loss={result['results']['final_loss']}")
                    # Include job_id in returned dict
                    result["job_id"] = job_id
                    return result
                else:
                    logger.error(f"Error retrieving results: {result['message']}")
                    return result
        except Exception as e:
            logger.error(f"Job submission failed: {e}")
            return {"status": "error", "message": str(e)}

    def get_trained_model(self, job_id, model_template):
        """
        Retrieve the trained model's state dictionary for the given job_id and load it into model_template.
        Args:
            job_id (str): The ID of the completed job.
            model_template (nn.Module): A PyTorch model instance to load the state dictionary into.
        Returns:
            nn.Module: The model with loaded trained parameters, or None if retrieval fails.
        """
        try:
            session = requests.Session()
            retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
            session.mount("https://", HTTPAdapter(max_retries=retries))
            response = session.get(f"{self.server_url}/get_results/{job_id}", timeout=10)
            result = response.json()

            if result["status"] != "complete":
                logger.error(f"Cannot retrieve model: Job {job_id} is not complete (status: {result['status']})")
                return None

            model_state_dict_bytes = result.get("model_state_dict_bytes")
            if model_state_dict_bytes is None:
                logger.error(f"No model state dictionary found for job {job_id}")
                return None

            try:
                model_state_dict = cloudpickle.loads(base64.b64decode(model_state_dict_bytes))
                model_template.load_state_dict(model_state_dict)
                logger.info(f"Successfully loaded trained model parameters for job {job_id}")
                return model_template
            except Exception as e:
                logger.error(f"Failed to deserialize model state dict for job {job_id}: {e}")
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve model for job {job_id}: {e}")
            return None
        
    def serialize_optimizer(self, opt):
        cls_name = type(opt).__name__
        kwargs = {}
        for k, v in opt.defaults.items():
            kwargs[k] = v
        return {"class": cls_name, "kwargs": kwargs}

class Reporter:
    def __init__(self, sio, job_id, rank):
        """
        sio: socketio.Client instance
        job_id: str, unique job identifier
        rank: int, process rank
        """
        self.sio = sio
        self.job_id = job_id
        self.rank = rank

    def log(self, *args):
        """
        Log arbitrary messages locally (print) and optionally to the server as debug logs.
        """
        message = " ".join(map(str, args))
        print(message)  # Always print locally

        if self.rank == 0 and self.sio is not None and self.sio.connected:
            def emit_async():
                try:
                    self.sio.emit(
                        "debug_log",
                        {
                            "job_id": self.job_id,
                            "rank": self.rank,
                            "message": message
                        }
                    )
                except Exception as e:
                    print(f"[Reporter debug_log emit error] {e}")
            threading.Thread(target=emit_async, daemon=True).start()

    def log_metrics(self, **kwargs):
        """
        Log structured metrics as a dictionary, returned for local logging and sent to the server.
        Args:
            **kwargs: Arbitrary key-value pairs (e.g., epoch, training_loss, eval_loss, accuracy).
        Returns:
            dict: The metrics dictionary, including job_id and rank.
        """
        metrics = {"job_id": self.job_id, "rank": self.rank}
        metrics.update(kwargs)  # Add user-provided metrics

        print(f"[Rank {self.rank}] Metrics: {metrics}")

        if self.rank == 0 and self.sio is not None and self.sio.connected:
            def emit_async():
                try:
                    self.sio.emit("training_progress", metrics)
                except Exception as e:
                    print(f"[Reporter training_progress emit error] {e}")
            threading.Thread(target=emit_async, daemon=True).start()

        return metrics