import time
import secrets
from flask import request, jsonify
from zwdx.server import globals as g
from flask_socketio import emit

rooms = {"public": set()}      # token -> set of client IDs
client_rooms = {}              # client_id -> token

def create_room():
    """Generate a unique room token and register it."""
    token = secrets.token_hex(8)
    rooms[token] = set()
    g.logger.info(f"Created new room: {token}")
    return token


def assign_client_to_room(client_id, token=None):
    """Add a client to a room (or to the public pool if no token)."""
    if token and token in rooms:
        rooms[token].add(client_id)
        client_rooms[client_id] = token
        g.logger.info(f"Client {client_id} joined room {token}")
        return {"status": "joined_room", "room": token}
    else:
        rooms["public"].add(client_id)
        client_rooms[client_id] = "public"
        g.logger.info(f"Client {client_id} joined public pool")
        return {"status": "joined_public"}


def get_clients_in_room(token):
    """Return list of clients currently in the room."""
    return list(rooms.get(token, []))


def register_room_routes(app, socketio):
    @app.route("/create_room", methods=["POST"])
    def create_room_route():
        """Endpoint: Create a new private room (returns a token)."""
        try:
            token = create_room()
            return jsonify({"status": "success", "token": token}), 200
        except Exception as e:
            g.logger.error(f"Error in create_room: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/list_rooms", methods=["GET"])
    def list_rooms_route():
        """Endpoint: List all active rooms and connected clients."""
        try:
            result = {token: len(clients) for token, clients in rooms.items()}
            return jsonify({"status": "success", "rooms": result}), 200
        except Exception as e:
            g.logger.error(f"Error in list_rooms: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/room_clients/<token>", methods=["GET"])
    def room_clients_route(token):
        """Endpoint: List all clients connected to a specific room."""
        try:
            clients = get_clients_in_room(token)
            return jsonify({"status": "success", "clients": clients}), 200
        except Exception as e:
            g.logger.error(f"Error in room_clients: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @socketio.on("register_client")
    def handle_register_client(data):
        client_id = data.get("client_id")
        token = data.get("room_token")
        hostname = data.get("hostname", client_id)  # fallback
        gpus = data.get("gpus", [])
        sid = request.sid
        ip = request.remote_addr

        if not client_id:
            g.logger.error("register_client missing client_id")
            emit("registration_ack", {"status": "error", "message": "missing client_id"})
            return

        # Assign client to a room
        result = assign_client_to_room(client_id, token)
        emit("registration_ack", result)

        # Update existing client
        for c in g.registered_clients:
            if c["sid"] == sid:
                c.update({
                    "last_seen": time.time(),
                    "client_id": client_id,
                    "room_token": token or "public",
                    "hostname": hostname,
                    "gpus": gpus or c.get("gpus", [])
                })
                g.logger.info(f"Updated existing client {client_id} with GPUs: {len(c['gpus'])}")
                return

        # Add new client
        g.registered_clients.append({
            "sid": sid,
            "client_id": client_id,
            "hostname": hostname,
            "ip": ip,
            "last_seen": time.time(),
            "gpus": gpus,
            "room_token": token or "public"
        })
        g.logger.info(f"Registered new client {client_id} with GPUs: {len(gpus)} into room {token or 'public'}")

    g.logger.info("Room routes registered.")