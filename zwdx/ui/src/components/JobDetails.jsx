import React from "react";
import "../styles/JobDetails.css";

export default function JobDetails({ job, jobResults }) {
  if (!job) return null;

  const formatTimestamp = (ts) => ts ? new Date(ts * 1000).toLocaleString() : "-";
  console.log(job)
  
  return (
    <div className="job-details">
      <h4>Job Details</h4>
      <table>
        <tbody>
          <tr>
            <td>Job ID</td>
            <td>{job.job_id}</td>
          </tr>
          <tr>
            <td>Status</td>
            <td>{job.status}</td>
          </tr>
          <tr>
            <td>World Size</td>
            <td>{job.world_size ?? "-"}</td>
          </tr>
          <tr>
            <td>Parallelism</td>
            <td>{job.parallelism ?? "-"}</td>
          </tr>
          <tr>
            <td>Created At</td>
            <td>{formatTimestamp(job.created_at)}</td>
          </tr>
          <tr>
            <td>Completed At</td>
            <td>{formatTimestamp(job.completed_at)}</td>
          </tr>
          {jobResults?.final_loss !== undefined && (
            <tr>
              <td>Final Loss</td>
              <td>{jobResults.final_loss}</td>
            </tr>
          )}
          {jobResults?.failure_reason && (
            <tr>
              <td>Failure Reason</td>
              <td>{jobResults.failure_reason}</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
