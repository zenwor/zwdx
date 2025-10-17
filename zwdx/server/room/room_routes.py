from flask import request, jsonify
from flask_socketio import emit
import time, uuid
from zwdx.server.server import Server

def register_room_routes():
    server = Server.instance()
    app = server.app
    socketio = server.socketio
    
    @app.route("/create_room", methods=["POST"])
    def create_room_route():
        try:
            token = server.room_pool.create_room()
            return jsonify({"status": "success", "token": token})
        except Exception as e:
            server.logger.error(f"Error creating room: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/list_rooms", methods=["GET"])
    def list_rooms_route():
        try:
            rooms_info = {token: room.client_count() for token, room in server.room_pool.rooms.items()}
            return jsonify({"status": "success", "rooms": rooms_info})
        except Exception as e:
            server.logger.error(f"Error listing rooms: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/room_clients/<token>", methods=["GET"])
    def room_clients_route(token):
        try:
            clients = server.room_pool.get_clients_in_room(token)
            return jsonify({"status": "success", "clients": clients})
        except Exception as e:
            server.logger.error(f"Error fetching room clients: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @server.app.route("/room_jobs/<room_token>", methods=["GET"])
    def room_jobs(room_token):
        """Return all jobs for a specific room."""
        jobs_in_room = [
            job.to_dict() for job in server.job_pool.jobs.values()
            if job.room_token == room_token
        ]
        return jsonify({"status": "success", "jobs": jobs_in_room})

    @socketio.on("register_client")
    def handle_register_client(data):
        client_id = str(uuid.uuid4())
        sid = request.sid
        ip = request.remote_addr
        room_token = data.get("room_token")
        gpus = data.get("gpus", [])

        # assign room
        assigned_token = server.room_pool.assign_client(client_id, room_token)

        # create client entry and register in ClientPool
        client_entry = {
            "sid": sid,
            "id": client_id,
            "ip": ip,
            "last_seen": time.time(),
            "gpus": gpus,
            "room_token": assigned_token
        }

        server.client_pool.add_client(client_entry)
        server.logger.info(f"Registered new client {client_id} in room {assigned_token}")
        emit("registration_ack", {"status": "ok", "client_id": client_id, "sid": sid, "room": assigned_token})

    server.logger.info("Room routes registered.")