from zwdx.server.job import Job

class JobPool:
    def __init__(self):
        self.jobs = {}  # job_id -> Job
    
    def add_job(self, job):
        self.jobs[job.job_id] = job
    
    def get_job(self, job_id):
        return self.jobs.get(job_id)
    
    def remove_job(self, job_id):
        """Remove completed job (optional cleanup)."""
        return self.jobs.pop(job_id, None)
    
    def get_all_jobs(self):
        return list(self.jobs.values())