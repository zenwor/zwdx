from zwdx.server.db import Database
from zwdx.server.job import Job
import time
import logging
import threading

logger = logging.getLogger(__name__)

class JobPool:
    def __init__(self):
        self.jobs = {}  # job_id -> Job
        self.populate()
        self._monitor_thread = None
        self.start_job_monitor()
        
    def populate(self):
        """Load all jobs from database and handle crash recovery."""
        db = Database.instance()
        logger.info("JobPool: starting populate()")
        count = 0
        failed_count = 0
       
        for job_doc in db.jobs.find({}):
            job = Job.from_dict(job_doc)
           
            # Mark incomplete jobs as failed on server restart
            if job.status in ["pending", "running"] and not job.complete:
                old_status = job.status
                job.mark_failed(f"Server restart - job was {old_status}")
                failed_count += 1
                logger.warning(f"Job {job.job_id} was '{old_status}', marked as 'failed' in DB")
           
            self.add_job(job)
            count += 1
       
        logger.info(f"JobPool: loaded {count} jobs ({failed_count} marked as failed due to server restart)")
           
           
    def start_job_monitor(self):
        """Start background thread to monitor job timeouts."""
        if self._monitor_thread is None:
            self._monitor_thread = threading.Thread(target=self._monitor_jobs, daemon=True)
            self._monitor_thread.start()
            logger.info("Started job timeout monitor")
    
    def _monitor_jobs(self):
        """Monitor running jobs for timeouts."""
        while True:
            try:
                for job in self.get_jobs_by_status("running"):
                    if job.is_timed_out():
                        logger.error(f"Job {job.job_id} timed out after {job.timeout}s")
                        job.mark_failed(f"Job timed out after {job.timeout}s without progress")
                        
                        # Free clients
                        for client in job.selected_clients:
                            client.mark_free()
            except Exception as e:
                logger.error(f"Error in job monitor: {e}")
            
            time.sleep(10)  # Check every 10 seconds
    
    def add_job(self, job):
        """Add job to pool."""
        self.jobs[job.job_id] = job
   
    def get_job(self, job_id):
        """Get job by ID."""
        return self.jobs.get(job_id)
   
    def remove_job(self, job_id):
        """Remove job from pool (optional cleanup)."""
        return self.jobs.pop(job_id, None)
   
    def get_all_jobs(self):
        """Get all jobs in pool."""
        return list(self.jobs.values())
    
    def get_jobs_by_status(self, status):
        """Get all jobs with a specific status."""
        return [job for job in self.jobs.values() if job.status == status]
    
    def get_jobs_by_room(self, room_token):
        """Get all jobs for a specific room."""
        return [job for job in self.jobs.values() if job.room_token == room_token]