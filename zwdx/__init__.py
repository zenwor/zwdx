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

import logging

logger = logging.getLogger("zwdx")
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

class ZWDX:
    def __init__(self, server_url):
        self.server_url = server_url

    def submit_job(self, model, data_loader_func, parallelism = "DDP", memory_required = 0):
        try:
            model_bytes = cloudpickle.dumps(model)
            data_loader_bytes = cloudpickle.dumps(data_loader_func)
            job = {
                "model_bytes": base64.b64encode(model_bytes).decode("utf-8"),
                "data_loader_bytes": base64.b64encode(data_loader_bytes).decode("utf-8"),
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
                        logger.info(f"Job {job_id}, epoch {update['epoch']}: loss={update['loss']}")
                    last_progress_count = len(result["progress"])
                    time.sleep(2)
                elif result["status"] == "complete":
                    logger.info(f"Job {job_id} completed: final_loss={result['results']['final_loss']}")
                    return result
                else:
                    logger.error(f"Error retrieving results: {result['message']}")
                    return result
        except Exception as e:
            logger.error(f"Job submission failed: {e}")
            return {"status": "error", "message": str(e)}