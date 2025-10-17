import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { getRoomJobs, getJobResults } from "../api";
import JobProgressGraph from "../components/JobProgressGraph.jsx";
import "../styles/RoomPage.css"; // optional: for styling


export default function RoomPage() {
  const { token: roomToken } = useParams();
  const [jobs, setJobs] = useState([]);
  const [selectedJob, setSelectedJob] = useState(null);
  const [jobResults, setJobResults] = useState(null);

  // Debug
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
          progress: job.progress || [], // fix: keep progress array
          created_at: job.created_at,
          completed_at: job.completed_at,
          world_size: typeof job.world_size === "number" ? job.world_size : null,
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
                <li key={job.job_id}>
                <button
                    className={selectedJob?.job_id === job.job_id ? "job-btn active" : "job-btn"}
                    onClick={() => setSelectedJob(job)}
                >
                    {`Job ${job.job_id.slice(0, 8)} — ${job.status}`}
                </button>
                </li>
            ))}
            </ul>
        )}
        </div>
        <div className="job-results-panel">
            {selectedJob ? (
                <>
                <h3>Results for Job {selectedJob.job_id.slice(0, 8)}</h3>
                <JobProgressGraph progress={jobResults?.progress || selectedJob.progress} />
                </>
            ) : (
                <p>Select a job to see its results.</p>
            )}
        </div>
    </div>
  );
}
