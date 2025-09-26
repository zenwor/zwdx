# app.py
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import time
import logging
from zwdx.utils import getenv
from zwdx.server import globals as g
from zwdx.server.job import register_job_routes

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
g.logger = logging.getLogger(__name__)

# Flask + SocketIO
g.app = Flask(__name__)
g.socketio = SocketIO(g.app, cors_allowed_origins="*", logger=False, engineio_logger=False)

# Global lists/dicts (redundant but explicit)
g.registered_clients = []
g.job_results = {}

# -------- Client cleanup thread --------
def cleanup_clients():
    while True:
        now = time.time()
        timeout = 15
        g.registered_clients[:] = [c for c in g.registered_clients if now - c["last_seen"] < timeout]
        time.sleep(5)

threading.Thread(target=cleanup_clients, daemon=True).start()

# -------- SocketIO events --------
@g.socketio.on("connect")
def handle_connect():
    g.logger.info(f"Client connected: {request.sid}")

@g.socketio.on("register")
def handle_register(data):
    sid = request.sid
    hostname = data["hostname"]
    gpus = data["gpus"]
    client_ip = data["client_ip"]

    g.logger.info(f"Registering client: {hostname}, IP: {client_ip}, GPUs: {len(gpus)}")

    # Update existing client or add new
    for c in g.registered_clients:
        if c["hostname"] == hostname:
            c.update({"sid": sid, "gpus": gpus, "ip": client_ip, "last_seen": time.time()})
            break
    else:
        g.registered_clients.append(
            {"sid": sid, "hostname": hostname, "gpus": gpus, "ip": client_ip, "last_seen": time.time()}
        )

    # Just log current clients; do NOT assign global ranks here
    g.logger.info(f"Currently registered clients: {[c['hostname'] for c in g.registered_clients]}")

# -------- Heartbeat --------
@g.app.route("/heartbeat", methods=["POST"])
def heartbeat():
    hostname = request.json.get("hostname")
    for c in g.registered_clients:
        if c["hostname"] == hostname:
            c["last_seen"] = time.time()
            break
    return {"status": "ok"}

# -------- Job routes --------
register_job_routes(g.app, g.socketio)

# -------- Get results route --------
@g.app.route("/get_results/<job_id>", methods=["GET"])
def get_results(job_id):
    if job_id not in g.job_results:
        return jsonify({"status": "error", "message": "Job not found"})
    job_data = g.job_results[job_id]
    response = {
        "status": "complete" if job_data["complete"] else "pending",
        "progress": job_data["progress"],
        "results": job_data["results"],
    }
    if job_data["complete"]:
        del g.job_results[job_id]
    return jsonify(response)

# -------- Main entrypoint --------
if __name__ == "__main__":
    g.logger.info(f"Starting server on {g.FLASK_HOST}:{g.FLASK_PORT}")
    g.socketio.run(g.app, host=g.FLASK_HOST, port=g.FLASK_PORT, allow_unsafe_werkzeug=True)
