import time

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ClientNode:
    def __init__(self, sid, id, ip=None, gpus=None, room_token=None, last_seen=None):
        self.sid = sid
        self.id = id
        self.ip = ip
        self.gpus = gpus or []
        self.room_token = room_token

        self.busy = False
        
        self.connected = True
        self.last_seen = None
        self.heartbeat()

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        self.heartbeat()

    def set_ip(self, ip):
        self.ip = ip
        self.heartbeat()

    def set_room(self, room_token):
        self.room_token = room_token
        self.heartbeat()

    def set_gpus(self, gpus):
        self.gpus = gpus
        self.heartbeat()

    def heartbeat(self):
        self.last_seen = time.time()

    def mark_disconnected(self):
        if self.connected:
            self.connected = False
            self.last_seen = time.time()
            logger.info(f"Client {self.id} marked disconnected")
    
    def update_activity(self):
        self.last_seen = time.time()

    def mark_busy(self):
        self.busy = True
        logger.info(f"Client {self.id} marked as busy")
        self.heartbeat()

    def mark_free(self):
        self.busy = False
        logger.info(f"Client {self.id} marked as free")
        self.heartbeat()

    def is_busy(self):
        return self.busy

    def is_available(self):
        return self.connected and not self.busy

    def to_dict(self):
        return {
            "sid": self.sid,
            "id": self.id,
            "ip": self.ip,
            "gpus": self.gpus,
            "room_token": self.room_token,
            "last_seen": self.last_seen,
            "busy": self.busy
        }

    def __repr__(self):
        status = "busy" if self.busy else "free"
        return f"<ClientNode {self.id} sid={self.sid} room={self.room_token} status={status}>"