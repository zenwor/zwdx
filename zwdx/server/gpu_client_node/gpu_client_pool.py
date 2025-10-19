import time
import threading

from zwdx.server.gpu_client_node import GPUClientNode

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class GPUClientPool:
    def __init__(self):
        self.clients = {}  # client_id -> ClientNode
        
        self._cleanup_thread = None
        self._cleanup_timeout = 15
        self._cleanup_interval = 5
        
        self.start_cleanup()
        
    def add_client(self, client_node_or_entry):
        client_node = client_node_or_entry
        if isinstance(client_node_or_entry, dict):
            client_node = GPUClientNode(**client_node_or_entry)
        
        self.clients[client_node.id] = client_node
        logger.info(f"Added client {client_node.id} to pool.")

    def remove_client(self, client_id):
        """Remove client and fail any running jobs."""
        if client_id in self.clients:
            client = self.clients[client_id]
           
            # Check if client was busy with a job
            if client.is_busy():
                logger.warning(f"Client {client_id} disconnected while busy - failing associated jobs")
                self._fail_client_jobs(client)
           
            from zwdx.server.server import Server
            server = Server.instance()
            server.room_pool.remove_client(client_id)
           
            del self.clients[client_id]
            logger.info(f"Removed client {client_id} from pool.")

    def get_by_sid(self, sid):
        for client in self.clients.values():
            if client.sid == sid:
                return client
        return None

    def get_by_id(self, client_id):
        return self.clients.get(client_id)

    def all_clients(self):
        return list(self.clients.values())

    def update_client(self, client_id, **kwargs):
        client = self.get_by_id(client_id)
        if client:
            client.update(**kwargs)
            return client
        return None
    
    def cleanup_clients(self):
        while True:
            now = time.time()
            to_remove = []
            
            for cid, c in self.clients.items():
                if not c.connected or (now - c.last_seen) > self._cleanup_timeout:
                    to_remove.append(cid)
            
            for cid in to_remove:
                logger.info(f"Removing disconnected/timed-out client {cid}")
                self.remove_client(cid)
                
            time.sleep(self._cleanup_interval)
            
    def start_cleanup(self):
        if self._cleanup_thread is None:
            self._cleanup_thread = threading.Thread(target=self.cleanup_clients, daemon=True)
            self._cleanup_thread.start()
    
    def _fail_client_jobs(self, client):
        """Fail all jobs that this client was working on."""
        from zwdx.server.server import Server
        server = Server.instance()
        
        # Find all running jobs that include this client
        for job in server.job_pool.get_jobs_by_status("running"):
            if any(c.id == client.id for c in job.selected_clients):
                logger.error(f"Job {job.job_id} failed due to client {client.id} disconnect")
                job.mark_failed(f"Client {client.id} disconnected during training")
                
                # Mark all other clients in this job as free
                for c in job.selected_clients:
                    if c.id != client.id:
                        c.mark_free()
    
    def __iter__(self):
        return iter(self.clients.values())