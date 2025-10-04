try:
    import dill as pickle_module
    print("Server using dill for serialization")
except ImportError:
    import cloudpickle as pickle_module
    print("Server using cloudpickle for serialization")
    
import uuid
from flask import request, jsonify
from zwdx.server import globals as g
from flask_socketio import emit
import base64

from zwdx.server.rooms import rooms

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def select_clients_for_job(memory_required, room_token):
    eligible_clients = []
    total_memory = 0

    # Filter clients in the correct room
    clients_in_room = [c for c in g.registered_clients if c.get("room_token") == room_token and "gpus" in c]

    if not clients_in_room:
        logger.warning(f"No clients in room {room_token} with GPU info")
        return None

    # Sort clients if you want deterministic order (optional)
    clients_in_room.sort(key=lambda c: c.get("client_id"))

    for client in clients_in_room:
        client_mem = sum([gpu.get("memory", 0) for gpu in client["gpus"]])
        if client_mem == 0:
            logger.info(f"Skipping client {client['client_id']} — no GPU memory reported")
            continue

        eligible_clients.append(client)
        total_memory += client_mem

        if total_memory >= memory_required:
            logger.info(f"Selected clients for job in room {room_token}: {[c['client_id'] for c in eligible_clients]}")
            return eligible_clients

    logger.error(f"Not enough cumulative GPU memory in room {room_token} ({total_memory} < {memory_required})")
    return None

def submit_job_route(app):
    @app.route("/submit_job", methods=["POST"])
    def submit_job():
        try:
            job = request.json
            if job is None:
                return jsonify({"status": "error", "message": "No JSON data received"}), 400

            parallelism = job.get("parallelism", "DDP")
            model_payload = job.get("model_payload")
            if model_payload is None:
                return jsonify({"status": "error", "message": "model_payload is required"}), 400

            memory_required = job.get("memory_required", 0)
            room_token = job.get("room_token")
            g.logger.info(f"Received job submission for room {room_token} requiring {memory_required} bytes")

            # --- Select clients ---
            selected_clients = select_clients_for_job(memory_required, room_token)
            if selected_clients is None:
                return jsonify({"status": "error", "message": "Not enough memory within the requested room"}), 400
                
            # Assign ranks and master address
            for i, client in enumerate(selected_clients):
                client["rank"] = i
            master_addr = selected_clients[0]["ip"]

            world_size = len(selected_clients)
            job_id = str(uuid.uuid4())
            g.job_results[job_id] = {
                "job_id": job_id,
                "progress": [],
                "complete": False,
                "results": {},
                "model_state_dict_bytes": None
            }

            g.logger.info(f"Submitting job {job_id} to {world_size} clients, parallelism={parallelism}")
            g.logger.info(f"Selected clients: {[c['client_id'] for c in selected_clients]}")

            for client in selected_clients:
                g.socketio.emit(
                    "assign_rank",
                    {"rank": client["rank"], "world_size": world_size, "master_addr": master_addr},
                    room=client["sid"]
                )
                g.socketio.emit(
                    "start_training",
                    {
                        "parallelism": parallelism,
                        "model_payload": model_payload,
                        "data_loader_payload": job.get("data_loader_payload"),
                        "train_payload": job.get("train_payload"),
                        "eval_payload": job.get("eval_payload"),
                        "data_fetch_payload": job.get("data_fetch_payload"),
                        "optimizer_bytes": job.get("optimizer_bytes"),
                        "master_port": g.master_port,
                        "job_id": job_id,
                        "room_token": room_token,
                    },
                    room=client["sid"]
                )

            return jsonify({"status": "job started", "world_size": world_size, "job_id": job_id})

        except Exception as e:
            g.logger.error(f"Error in submit_job: {e}")
            import traceback
            g.logger.error(traceback.format_exc())
            return jsonify({"status": "error", "message": str(e)}), 500

def register_socketio_handlers(socketio):
    @socketio.on("training_progress")
    def handle_training_progress(data):
        job_id = data.get("job_id")
        rank = data.get("rank")
        if job_id is None or rank is None:
            g.logger.error(f"Invalid training_progress data: {data}")
            return

        log_message = f"Job {job_id}, rank {rank}: {data}"
        g.logger.info(log_message)

        if job_id in g.job_results:
            g.job_results[job_id]["progress"].append(data)

    @socketio.on("debug_log")
    def handle_debug_log(data):
        job_id = data.get("job_id")
        rank = data.get("rank")
        message = data.get("message")
        if job_id is None or rank is None or message is None:
            g.logger.error(f"Invalid debug_log data: {data}")
            return
        g.logger.info(f"Job {job_id}, rank {rank}: {message}")

    @socketio.on("training_done")
    def handle_training_done(data):
        job_id = data.get("job_id")
        final_loss = data.get("final_loss")
        model_state_dict_bytes = data.get("model_state_dict_bytes")
        g.logger.info(f"Received final results for job {job_id}: loss={final_loss}")
        if job_id in g.job_results:
            g.job_results[job_id]["results"] = {"final_loss": final_loss}
            g.job_results[job_id]["model_state_dict_bytes"] = model_state_dict_bytes
            g.job_results[job_id]["complete"] = True

def register_job_routes(app, socketio):
    submit_job_route(app)
    register_socketio_handlers(socketio)