# job.py
import uuid
from flask import request, jsonify
from zwdx.server import globals as g
from flask_socketio import emit

# -------- Job route --------
def select_clients_for_job(memory_required):
    """
    Pick the first N clients whose GPU memory adds up >= memory_required.
    Each client has a 'gpus' list with 'memory' in bytes.
    """
    selected = []
    total_mem = 0
    for client in g.registered_clients:
        # sum all GPU memory in this client
        client_mem = sum([gpu["memory"] for gpu in client["gpus"]])
        selected.append(client)
        total_mem += client_mem
        if total_mem >= memory_required:
            return selected
    return None  # not enough memory


# ----------------- Job submission route -----------------
def submit_job_route(app):
    @app.route("/submit_job", methods=["POST"])
    def submit_job():
        job = request.json
        parallelism = job.get("parallelism", "DDP")
        model_bytes = job["model_bytes"]
        data_loader_bytes = job["data_loader_bytes"]
        memory_required = job.get("memory_required", 0)  # MB

        # ----------------- Allocate clients -----------------
        selected_clients = select_clients_for_job(memory_required)
        if selected_clients is None:
            return jsonify({"status": "error", "message": "Not enough GPU memory available"})

        print("Selected clients:", selected_clients)

        # ----------------- Assign ranks per-job -----------------
        for i, client in enumerate(selected_clients):
            client["rank"] = i
        master_addr = selected_clients[0]["ip"]

        world_size = len(selected_clients)
        job_id = str(uuid.uuid4())
        g.job_results[job_id] = {"progress": [], "complete": False, "results": {}}

        g.logger.info(f"Submitting job {job_id} to {world_size} clients, parallelism={parallelism}")

        # ----------------- Emit assign_rank + start_training -----------------
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
                    "model_bytes": model_bytes,
                    "data_loader_bytes": data_loader_bytes,
                    "master_port": g.master_port,
                    "job_id": job_id,
                },
                room=client["sid"]
            )

        return jsonify({"status": "job started", "world_size": world_size, "job_id": job_id})

# -------- SocketIO handlers --------
def register_socketio_handlers(socketio):
    @socketio.on("training_progress")
    def handle_training_progress(data):
        job_id = data["job_id"]
        epoch = data["epoch"]
        loss = data["loss"]
        g.logger.info(f"Job {job_id}, epoch {epoch}: loss={loss}")
        if job_id in g.job_results:
            g.job_results[job_id]["progress"].append({"epoch": epoch, "loss": loss})

    @socketio.on("training_done")
    def handle_training_done(data):
        job_id = data["job_id"]
        final_loss = data["final_loss"]
        g.logger.info(f"Received final results for job {job_id}: loss={final_loss}")
        if job_id in g.job_results:
            g.job_results[job_id]["results"] = {"final_loss": final_loss}
            g.job_results[job_id]["complete"] = True

# -------- Main registration function --------
def register_job_routes(app, socketio):
    submit_job_route(app)
    register_socketio_handlers(socketio)
