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

    @app.route("/room_jobs/<room_token>", methods=["GET"])
    def room_jobs(room_token):
        """Get all jobs for a specific room (optimized for UI display)."""
        room = server.room_pool.get_room_by_token(room_token)
        if room is None:
            return jsonify({"status": "error", "message": "Room not found"}), 404
        
        jobs_in_room = []
        for job_id in room.jobs:
            job = server.job_pool.get_job(job_id)
            if job is None:
                server.logger.warning(f"Job {job_id} in room {room_token} not found in JobPool")
            else:
                # Return job info WITHOUT large binary payloads
                jobs_in_room.append({
                    "job_id": job.job_id,
                    "status": job.status,
                    "progress": job.progress,
                    "results": job.results,
                    "created_at": job.created_at,
                    "completed_at": job.completed_at,
                    "world_size": job.world_size,
                    "parallelism": job.parallelism,
                    "memory_required": job.memory_required,
                    "complete": job.complete
                })
        
        return jsonify({"status": "success", "jobs": jobs_in_room})

    @app.route("/check_room/<token>", methods=["GET"])
    def check_room_route(token):
        try:
            room = server.room_pool.get_room_by_token(token)
            
            if room is not None:
                return jsonify({"status": "success", "exists": True})
            else:
                return jsonify({"status": "success", "exists": False})
        except Exception as e:
            server.logger.error(f"Error checking room {token}: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

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