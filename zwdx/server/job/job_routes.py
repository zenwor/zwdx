from flask import request, jsonify
from zwdx.server.server import Server
from zwdx.server.job import Job
import logging

logger = logging.getLogger(__name__)

def register_job_routes():
    server = Server.instance()
    app = server.app
    socketio = server.socketio
    
    @app.route("/submit_job", methods=["POST"])
    def submit_job():
        try:
            job_data = request.json
            if not job_data:
                return jsonify({"status": "error", "message": "No JSON data received"}), 400

            # Create a Job object
            job = Job(
                model_payload=job_data.get("model_payload"),
                room_token=job_data.get("room_token"),
                parallelism=job_data.get("parallelism", "DDP"),
                memory_required=job_data.get("memory_required", 0),
                data_loader_payload=job_data.get("data_loader_payload"),
                train_payload=job_data.get("train_payload"),
                eval_payload=job_data.get("eval_payload"),
                data_fetch_payload=job_data.get("data_fetch_payload"),
                optimizer_bytes=job_data.get("optimizer_bytes")
            )

            if not job.model_payload:
                return jsonify({"status": "error", "message": "model_payload is required"}), 400

            # Get room
            room_requested = server.room_pool.get_room_by_token(job.room_token)
            if not room_requested:
                return jsonify({"status": "error", "message": "Room not found"}), 404
            
            # Select clients
            job_clients = room_requested.select_clients_for_job(job.memory_required)
            if not job_clients:
                job.mark_failed("Not enough memory within the requested room")
                return jsonify({"status": "error", "message": "Not enough memory within the requested room"}), 400

            job.selected_clients = job_clients
            
            # Set execution config and update status
            job.set_execution_config(
                master_addr=job_clients[0].ip,
                world_size=len(job_clients)
            )

            # Add to Room and JobPool
            server.job_pool.add_job(job)
            room_requested.add_job(job)
            
            # Emit training start messages
            for client in job.selected_clients:
                server.socketio.emit(
                    "assign_rank",
                    {
                        "rank": client.rank, 
                        "world_size": job.world_size, 
                        "master_addr": job.master_addr
                    },
                    room=client.sid,
                )
            
            for client in job.selected_clients:
                logger.info(f"Emitting start_training to client.sid={client.sid}")
                server.socketio.emit(
                    "start_training",
                    {
                        "parallelism": job.parallelism,
                        "model_payload": job.model_payload,
                        "data_loader_payload": job.data_loader_payload,
                        "train_payload": job.train_payload,
                        "eval_payload": job.eval_payload,
                        "data_fetch_payload": job.data_fetch_payload,
                        "optimizer_bytes": job.optimizer_bytes,
                        "master_port": server.master_port,
                        "job_id": job.job_id,
                        "room_token": job.room_token,
                    },
                    room=client.sid,
                )
                
                client.mark_busy()

            return jsonify({
                "status": "job started", 
                "world_size": job.world_size, 
                "job_id": job.job_id
            })

        except Exception as e:
            logger.error(f"Error in submit_job: {e}", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/get_results/<job_id>", methods=["GET"])
    def get_results(job_id):
        """Get results for a specific job (for UI)."""
        try:
            job = server.job_pool.get_job(job_id)
            if not job:
                return jsonify({"status": "error", "message": "Job not found"}), 404

            # Ensure defaults if some fields are missing
            parallelism = getattr(job, "parallelism", "DDP") or "DDP"
            world_size = getattr(job, "world_size", 0) or 0
            status = getattr(job, "status", "pending") or "pending"

            return jsonify({
                "status": "success",
                "job_id": job.job_id,
                "job_status": status,
                "progress": getattr(job, "progress", []),
                "results": getattr(job, "results", None),
                "created_at": getattr(job, "created_at", None),
                "completed_at": getattr(job, "completed_at", None),
                "world_size": world_size,
                "complete": getattr(job, "complete", False),
                "parallelism": parallelism,
            })

        except Exception as e:
            logger.error(f"Error fetching results for job {job_id}: {e}", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500

    # SocketIO handlers
    @socketio.on("debug_log")
    def handle_debug_log(data):
        job_id = data.get("job_id")
        rank = data.get("rank")
        message = data.get("message")
        if not job_id or rank is None or message is None:
            logger.error(f"Invalid debug_log data: {data}")
            return
        logger.info(f"Job {job_id}, rank {rank}: {message}")

    @socketio.on("training_done")
    def handle_training_done(data):
        job_id = data.get("job_id")
        final_loss = data.get("final_loss")
        model_state_dict_bytes = data.get("model_state_dict_bytes")
        
        logger.info(f"Received final results for job {job_id}: loss={final_loss}")
        
        job = server.job_pool.get_job(job_id)
        if job:
            job.mark_complete(final_loss, model_state_dict_bytes)
            
            # Mark all job clients as free
            for client in job.selected_clients:
                client.mark_free()
        else:
            logger.error(f"Job {job_id} not found in pool")
        
    @socketio.on("training_progress")
    def handle_training_progress(data):
        job_id = data.get("job_id")
        job = server.job_pool.get_job(job_id)
        if job:
            job.add_progress(data)
            job.update_progress_time()
            logger.info(f"Job {job_id}, rank {data.get('rank')}: progress updated")
        else:
            logger.warning(f"Received progress for unknown job {job_id}")
    
    @socketio.on("training_failed")
    def handle_training_failed(data):
        """Handle training failure from client."""
        job_id = data.get("job_id")
        reason = data.get("reason", "Unknown error")
        
        logger.error(f"Job {job_id} failed: {reason}")
        
        job = server.job_pool.get_job(job_id)
        if job:
            job.mark_failed(reason)
            
            # Mark all job clients as free
            for client in job.selected_clients:
                client.mark_free()
        else:
            logger.error(f"Job {job_id} not found in pool")