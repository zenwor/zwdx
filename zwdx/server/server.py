from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import time
import threading
import logging
import uuid

from zwdx.utils import getenv

FLASK_HOST = getenv("FLASK_HOST")
FLASK_PORT = int(getenv("FLASK_PORT"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

registered_clients = []
master_port = "29500"
job_results = {}


def cleanup_clients():
    while True:
        now = time.time()
        timeout = 15
        registered_clients[:] = [c for c in registered_clients if now - c["last_seen"] < timeout]
        time.sleep(5)


threading.Thread(target=cleanup_clients, daemon=True).start()


@socketio.on("connect")
def handle_connect():
    logger.info(f"Client connected: {request.sid}")


@socketio.on("register")
def handle_register(data):
    sid = request.sid
    hostname = data["hostname"]
    gpus = data["gpus"]
    client_ip = data["client_ip"]

    logger.info(f"Registering client: {hostname}, IP: {client_ip}, GPUs: {len(gpus)}")

    for c in registered_clients:
        if c["hostname"] == hostname:
            c.update({"sid": sid, "gpus": gpus, "ip": client_ip, "last_seen": time.time()})
            break
    else:
        registered_clients.append(
            {"sid": sid, "hostname": hostname, "gpus": gpus, "ip": client_ip, "last_seen": time.time()}
        )

    world_size = len(registered_clients)
    for i, c in enumerate(registered_clients):
        c["rank"] = i

    if world_size > 0:
        master_addr = registered_clients[0]["ip"]
        logger.info(f"Master addr: {master_addr}, world_size: {world_size}")
        for c in registered_clients:
            emit(
                "assign_rank",
                {"rank": c["rank"], "world_size": world_size, "master_addr": master_addr},
                room=c["sid"],
            )


@app.route("/heartbeat", methods=["POST"])
def heartbeat():
    hostname = request.json.get("hostname")
    for c in registered_clients:
        if c["hostname"] == hostname:
            c["last_seen"] = time.time()
            break
    return {"status": "ok"}


@app.route("/submit_job", methods=["POST"])
def submit_job():
    job = request.json
    parallelism = job.get("parallelism", "DDP")
    model_bytes = job["model_bytes"]
    data_loader_bytes = job["data_loader_bytes"]

    world_size = len(registered_clients)
    if world_size == 0:
        return jsonify({"status": "error", "message": "No clients registered"})

    job_id = str(uuid.uuid4())
    job_results[job_id] = {"progress": [], "complete": False, "results": {}}

    logger.info(f"Submitting job {job_id} with parallelism: {parallelism}, world_size: {world_size}")
    socketio.emit(
        "start_training",
        {
            "parallelism": parallelism,
            "model_bytes": model_bytes,
            "data_loader_bytes": data_loader_bytes,
            "master_port": master_port,
            "job_id": job_id,
        },
    )
    return jsonify({"status": "job started", "world_size": world_size, "job_id": job_id})


@socketio.on("training_progress")
def handle_training_progress(data):
    job_id = data["job_id"]
    epoch = data["epoch"]
    loss = data["loss"]
    logger.info(f"Job {job_id}, epoch {epoch}: loss={loss}")
    if job_id in job_results:
        job_results[job_id]["progress"].append({"epoch": epoch, "loss": loss})


@socketio.on("training_done")
def handle_training_done(data):
    job_id = data["job_id"]
    final_loss = data["final_loss"]
    logger.info(f"Received final results for job {job_id}: loss={final_loss}")
    if job_id in job_results:
        job_results[job_id]["results"] = {"final_loss": final_loss}
        job_results[job_id]["complete"] = True


@app.route("/get_results/<job_id>", methods=["GET"])
def get_results(job_id):
    if job_id not in job_results:
        return jsonify({"status": "error", "message": "Job not found"})
    job_data = job_results[job_id]
    response = {
        "status": "complete" if job_data["complete"] else "pending",
        "progress": job_data["progress"],
        "results": job_data["results"],
    }
    if job_data["complete"]:
        del job_results[job_id]
    return jsonify(response)


if __name__ == "__main__":
    logger.info(f"Starting server on {FLASK_HOST}:{FLASK_PORT}")
    socketio.run(app, host=FLASK_HOST, port=FLASK_PORT, allow_unsafe_werkzeug=True)
