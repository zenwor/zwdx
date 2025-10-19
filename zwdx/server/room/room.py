from zwdx.server.db import Database

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Room:
    def __init__(self, token, jobs = None):
        self.token = token
        self.clients = set()
        self.jobs = set(jobs) if jobs else set()
        
    def add_client(self, client_id):
        self.clients.add(client_id)

    def remove_client(self, client_id):
        self.clients.discard(client_id)

    def has_client(self, client_id):
        return client_id in self.clients

    def client_count(self):
        return len(self.clients)
    
    def select_clients_for_job(self, memory_required):
        if self.client_count() == 0:
            logger.warning(f"No clients in room {self.token} with GPU info")
            return None

        from zwdx.server.server import Server
        server = Server.instance()
        
        eligible_clients = []
        total_memory = 0
        for client_id in self.clients:
            client = server.client_pool.get_by_id(client_id)
            
            # Prevent silent fail
            if client is None:
                logger.warning(f"Client {client_id} not found in pool (may have disconnected)")
                continue
            
            if client.is_busy():
                continue
            
            client_mem = sum([gpu.get("memory", 0) for gpu in client.gpus])
            if client_mem == 0:
                logger.info(f"Skipping client {client_id} — no GPU memory reported")
                continue

            eligible_clients.append(client)
            total_memory += client_mem

            if total_memory >= memory_required:
                for i, client in enumerate(eligible_clients):
                    client.rank = i
                return eligible_clients

        logger.error(f"Not enough cumulative GPU memory in room {self.token} ({total_memory} < {memory_required})")
        return None

    def add_job(self, job):
        self.jobs.add(job.job_id)
        
        # Persist
        db = Database.instance()
        db.rooms.update_one(
            {"room_token": self.token},
            {"$addToSet": {"jobs": job.job_id}}
        )
        
        logger.info(f"Added job {job.job_id} to room {self.token} in DB")
    
    def remove_job(self, job):
        self.jobs.discard(job.job_id)

        # Persist to database
        db = Database.instance()
        db.rooms.update_one(
            {"room_token": self.token},
            {"$pull": {"jobs": job.job_id}}
        )

        logger.info(f"Removed job {job.job_id} from room {self.token} in DB")    

    def has_job(self, job):
        return job.job_id in self.jobs
    
    def job_count(self):
        return len(self.jobs)
    
    @staticmethod
    def from_dict(data):
        return Room(
            token=data.get("room_token"),
            jobs=set(data.get("jobs", []))
        )
    
    def to_dict(self):
        return {
            "room_token": self.token,
            "jobs": list(self.jobs)
        }