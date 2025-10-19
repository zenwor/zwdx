import uuid
import time
from zwdx.server.db import Database
import logging

logger = logging.getLogger(__name__)

class Job:
    def __init__(self, model_payload, room_token, parallelism="DDP", 
                 memory_required=0, data_loader_payload=None, train_payload=None,
                 eval_payload=None, data_fetch_payload=None, optimizer_bytes=None,
                 persist: bool = True, timeout: int = 3600):
        self.job_id = str(uuid.uuid4())
        self.model_payload = model_payload
        self.room_token = room_token
        self.parallelism = parallelism
        self.memory_required = memory_required
        self.data_loader_payload = data_loader_payload
        self.train_payload = train_payload
        self.eval_payload = eval_payload
        self.data_fetch_payload = data_fetch_payload
        self.optimizer_bytes = optimizer_bytes
        
        # Job execution state
        self.selected_clients = []
        self.master_addr = None
        self.world_size = 0
        
        # Results tracking
        self.complete = False
        self.progress = []
        self.results = None
        self.model_state_dict_bytes = None
        self.created_at = time.time()
        self.completed_at = None
        self.status = "pending"
        self.timeout = timeout
        self.last_progress_time = time.time()
        
        if persist:
            self._persist_to_db()

    # ------------------ DB Persistence ------------------

    def is_timed_out(self):
        """Check if job has exceeded timeout without progress."""
        if self.status not in ["running", "pending"]:
            return False
    
    def update_progress_time(self):
        """Update last progress timestamp."""
        self.last_progress_time = time.time()
    
    def _persist_to_db(self):
        """Persist or update this job in MongoDB."""
        db = Database.instance()
        doc = self.to_dict(include_payloads=True)
        
        # Use upsert to handle both insert and update
        db.jobs.update_one(
            {"job_id": self.job_id},
            {"$set": doc},
            upsert=True
        )
        logger.debug(f"Persisted job {self.job_id} with status {self.status}")

    def mark_complete(self, final_loss, model_state_dict_bytes):
        """Mark job as complete and persist to database."""
        self.complete = True
        self.status = "complete"
        self.results = {"final_loss": final_loss}
        self.model_state_dict_bytes = model_state_dict_bytes
        self.completed_at = time.time()
        
        self._persist_to_db()
        logger.info(f"Job {self.job_id} marked complete with loss {final_loss}")

    def mark_failed(self, reason=None):
        """Mark job as failed and persist to database."""
        self.status = "failed"
        self.complete = True  # Mark as complete (done, but failed)
        if reason:
            if self.results is None:
                self.results = {}
            self.results["failure_reason"] = reason
        self.completed_at = time.time()
        
        self._persist_to_db()
        logger.warning(f"Job {self.job_id} marked failed: {reason}")

    def update_status(self, new_status):
        """Update job status and persist."""
        old_status = self.status
        self.status = new_status
        self._persist_to_db()
        logger.info(f"Job {self.job_id} status: {old_status} -> {new_status}")

    def add_progress(self, progress_data):
        """Add progress data and persist to database."""
        self.progress.append(progress_data)
        self._persist_to_db()
        logger.debug(f"Added progress to job {self.job_id}")

    def set_execution_config(self, master_addr, world_size):
        """Set execution configuration after client selection."""
        self.master_addr = master_addr
        self.world_size = world_size
        self.status = "running"
        self._persist_to_db()
        logger.info(f"Job {self.job_id} configured: master={master_addr}, world_size={world_size}")

    def to_dict(self, include_payloads=True):
        """
        Convert job to dictionary for serialization.
        
        Args:
            include_payloads: If False, exclude large payloads (useful for API responses)
        """
        base_dict = {
            "job_id": self.job_id,
            "room_token": self.room_token,
            "parallelism": self.parallelism,
            "memory_required": self.memory_required,
            "status": self.status,
            "progress": self.progress,
            "results": self.results,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "world_size": self.world_size,
            "master_addr": self.master_addr,
            "complete": self.complete,
        }
        
        if include_payloads:
            base_dict.update({
                "model_payload": self.model_payload,
                "data_loader_payload": self.data_loader_payload,
                "train_payload": self.train_payload,
                "eval_payload": self.eval_payload,
                "data_fetch_payload": self.data_fetch_payload,
                "optimizer_bytes": self.optimizer_bytes,
                "model_state_dict_bytes": self.model_state_dict_bytes,
            })
        
        return base_dict

    @staticmethod
    def from_dict(data):
        """Reconstruct Job from database document."""
        job = Job.__new__(Job)  # bypass __init__

        # Core fields
        job.job_id = data.get("job_id")
        job.room_token = data.get("room_token")
        job.parallelism = data.get("parallelism", "DDP")
        job.memory_required = data.get("memory_required", 0)

        # Payloads (stored as-is in MongoDB)
        job.model_payload = data.get("model_payload")
        job.data_loader_payload = data.get("data_loader_payload")
        job.train_payload = data.get("train_payload")
        job.eval_payload = data.get("eval_payload")
        job.data_fetch_payload = data.get("data_fetch_payload")
        job.optimizer_bytes = data.get("optimizer_bytes")
        job.model_state_dict_bytes = data.get("model_state_dict_bytes")

        # Execution state
        job.selected_clients = []  # Not persisted, will be repopulated on restart
        job.master_addr = data.get("master_addr")
        job.world_size = data.get("world_size", 0)

        # Tracking
        job.complete = data.get("complete", False)
        job.progress = data.get("progress", [])
        job.results = data.get("results")
        job.created_at = data.get("created_at", time.time())
        job.completed_at = data.get("completed_at")
        job.status = data.get("status", "pending")

        return job