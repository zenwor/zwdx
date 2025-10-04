import time
import threading

from zwdx.server.client import ClientNode

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ClientPool:
    def __init__(self):
        self.clients = {}  # client_id -> ClientNode
        
        self._cleanup_thread = None
        self._cleanup_timeout = 15
        self._cleanup_interval = 5
        
        self.start_cleanup()
        
    def add_client(self, client_node_or_entry):
        client_node = client_node_or_entry
        if isinstance(client_node_or_entry, dict):
            client_node = ClientNode(**client_node_or_entry)
        
        self.clients[client_node.id] = client_node
        logger.info(f"Added client {client_node.id} to pool.")

    def remove_client(self, client_id):
        if client_id in self.clients:
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
            to_remove = [cid for cid, c in self.clients.items() if not c.connected]
            for cid in to_remove:
                logger.info(f"Removing disconnected client {cid}")
                del self.clients[cid]
            time.sleep(self._cleanup_interval)
        
    def start_cleanup(self):
        if self._cleanup_thread is None:
            self._cleanup_thread = threading.Thread(target=self.cleanup_clients, daemon=True)
            self._cleanup_thread.start()
    
    def __iter__(self):
        return iter(self.clients.values())