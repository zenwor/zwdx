import dill
import uuid
from flask import request, jsonify
from flask_socketio import emit
import time
import base64

class Job:
    def __init__(self, model_payload, room_token, parallelism="DDP", 
                 memory_required=0, data_loader_payload=None, train_payload=None,
                 eval_payload=None, data_fetch_payload=None, optimizer_bytes=None):
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
    
    def mark_complete(self, final_loss, model_state_dict_bytes):
        """Mark job as complete with results."""
        self.complete = True
        self.results = {"final_loss": final_loss}
        self.model_state_dict_bytes = model_state_dict_bytes
        self.completed_at = time.time()
    
    def add_progress(self, progress_data):
        """Add progress update."""
        self.progress.append(progress_data)
    
    def to_dict(self):
        """Serialize job state for API responses."""
        return {
            "job_id": self.job_id,
            "status": "complete" if self.complete else "pending",
            "progress": self.progress,
            "results": self.results,
            "model_state_dict_bytes": self.model_state_dict_bytes,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "world_size": self.world_size,
        }