from flask import request, jsonify

def register_core_routes():
    from zwdx.server.server import Server
    server = Server.instance()
    
    @server.app.route("/heartbeat", methods=["POST"])
    def heartbeat():
        client_id = request.json.get("client_id")
        client = server.client_pool.get_by_id(client_id)
        if client:
            client.heartbeat()
        return {"status": "ok"}
    
    @server.socketio.on("disconnect")
    def handle_disconnect():
        sid = request.sid
        client = server.client_pool.get_by_sid(sid)
        if client:
            client.mark_disconnected()
    
    @server.app.route("/get_results/<job_id>", methods=["GET"])
    def get_results(job_id):
        job = server.job_pool.get_job(job_id)
        if not job:
            return jsonify({"status": "error", "message": "Job not found"}), 404
        return jsonify(job.to_dict())
    
    @server.app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "clients": len(server.client_pool.get_all_clients()),
            "rooms": len(server.room_pool.rooms),
            "jobs": len(server.job_pool.jobs)
        })