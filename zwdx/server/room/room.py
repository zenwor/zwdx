import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Room:
    def __init__(self, token):
        self.token = token
        self.clients = set()

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
            logger.warning(f"No clients in room {self.room_token} with GPU info")
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

        logger.error(f"Not enough cumulative GPU memory in room {self.room_token} ({total_memory} < {self.memory_required})")
        return None