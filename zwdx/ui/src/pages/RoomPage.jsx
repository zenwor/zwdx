import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { getRoomJobs, getJobResults, deleteJob } from "../api";
import JobProgressGraph from "../components/JobProgressGraph.jsx";
import JobDetails from "../components/JobDetails.jsx";
import "../styles/RoomPage.css";

export function formatJobStatus(status) {
  switch (status.toLowerCase()) {
    case "complete":
    case "done":
      return "✅ Complete";
    case "failed":
      return "❌ Failed";
    case "running":
    case "in_progress":
      return "🚀 Running";
    default:
      return `❔ ${status}`;
  }
}

export default function RoomPage() {
  const { token: roomToken } = useParams();
  const [jobs, setJobs] = useState([]);
  const [selectedJob, setSelectedJob] = useState(null);
  const [jobResults, setJobResults] = useState(null);

  useEffect(() => {
    console.log("RoomPage mounted with roomToken:", roomToken);
  }, [roomToken]);

  // Fetch jobs
  useEffect(() => {
    if (!roomToken) return;

    async function fetchJobs() {
      try {
        const jobsList = await getRoomJobs(roomToken);
        const cleanedJobs = jobsList.map((job) => ({
          job_id: job.job_id,
          status: typeof job.status === "string" ? job.status : String(job.status),
          progress: job.progress || [],
          created_at: job.created_at,
          completed_at: job.completed_at,
          world_size: typeof job.world_size === "number" ? job.world_size : null,
          parallelism: job.parallelism,
        }));
        console.log("Cleaned jobs:", cleanedJobs);
        setJobs(cleanedJobs);
      } catch (err) {
        console.error("Error fetching jobs:", err);
      }
    }

    fetchJobs();
    const interval = setInterval(fetchJobs, 5000);
    return () => clearInterval(interval);
  }, [roomToken]);

  // Fetch selected job results
  useEffect(() => {
    if (!selectedJob) return;

    setJobResults(null);
    const interval = setInterval(async () => {
      try {
        const results = await getJobResults(selectedJob.job_id);
        setJobResults(results);
      } catch (err) {
        console.error("Error fetching job results:", err);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [selectedJob]);

  return (
    <div className="room-page-container">
        <div className="jobs-sidebar">
        <h2>Jobs</h2>
        {jobs.length === 0 ? (
            <p>No jobs yet in this room.</p>
        ) : (
        <ul>
          {jobs.map((job) => (
            <li key={job.job_id} className="job-item">
              <div className="job-entry">
                <button
                  className={selectedJob?.job_id === job.job_id ? "job-btn active" : "job-btn"}
                  onClick={() => setSelectedJob(job)}
                >
                  <span>{`Job ${job.job_id.slice(0, 8)}`}</span>
                  <span className="job-status">{formatJobStatus(job.status)}</span>
                </button>
                <button
                  className="delete-job-btn"
                  title="Delete Job"
                  onClick={async (e) => {
                    e.stopPropagation();
                    if (!window.confirm("Are you sure you want to delete this job?")) return;
                    try {
                      await deleteJob(job.job_id);
                      setJobs((prev) => prev.filter((j) => j.job_id !== job.job_id));
                      if (selectedJob?.job_id === job.job_id) setSelectedJob(null);
                    } catch (err) {
                      console.error("Failed to delete job:", err);
                    }
                  }}
                >
                  🗑️
                </button>
              </div>
            </li>
          ))}
        </ul>
        )}
        </div>
        <div className="job-results-panel">
          {selectedJob ? (
            <>
              <h3>Results for Job {selectedJob.job_id.slice(0, 8)}</h3>
              <JobDetails job={selectedJob} jobResults={jobResults} />
              <JobProgressGraph progress={jobResults?.progress || selectedJob.progress} />
            </>
          ) : (
            <p>Select a job to see its results.</p>
          )}
        </div>
    </div>
  );
}
