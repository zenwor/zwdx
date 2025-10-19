import secrets
import logging

from .room import Room
from zwdx.server.db import Database

logger = logging.getLogger("server")

class RoomPool:
    def __init__(self):
        self.rooms = {}  # token -> Room
        self.client_to_room = {}  # client_id -> token

        self.populate()

    def populate(self):
        db = Database.instance()
        rooms_cursor = db.db.rooms.find({})

        for room_doc in rooms_cursor:
            room = Room.from_dict(room_doc)
            self.rooms[room.token] = room

        logger.info(f"Loaded {len(self.rooms)} rooms from database.")
        
    def create_room(self):
        while True:
            token = secrets.token_hex(8)
            if token not in self.rooms:
                break
            
        new_room = Room(token)
        self.rooms[token] = new_room
        logger.info(f"Created new room: {token}")
        
        # Persist
        db = Database.instance()
        db.add_room(new_room.to_dict())
        return token

    def assign_client(self, client_id, token=None):
        room = self.rooms.get(token)

        room.add_client(client_id)
        self.client_to_room[client_id] = token
        logger.info(f"Client {client_id} joined room {token or 'public'}")
        return token

    def remove_client(self, client_id):
        token = self.client_to_room.pop(client_id, None)
        if token in self.rooms:
            self.rooms[token].remove_client(client_id)

    def get_clients_in_room(self, token):
        room = self.rooms.get(token)
        if room:
            return list(room.clients)
        return []

    def get_room_of_client(self, client_id):
        return self.client_to_room.get(client_id)
    
    def get_room_by_token(self, room_token):
        return self.rooms.get(room_token, None)