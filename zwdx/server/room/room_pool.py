import secrets
import logging

from .room import Room

logger = logging.getLogger("server")

class RoomPool:
    def __init__(self):
        self.rooms = {}  # token -> Room
        self.client_to_room = {}  # client_id -> token

        # Create a public room
        public = Room(token=None)
        self.rooms[None] = public

    def create_room(self):
        while True:
            token = secrets.token_hex(8)
            if token not in self.rooms:
                break
        self.rooms[token] = Room(token)
        logger.info(f"Created new room: {token}")
        return token

    def assign_client(self, client_id, token=None):
        # If token is invalid, assign to public room (token=None)
        room = self.rooms.get(token)
        if room is None:
            room = self.rooms[None]
            token = None

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