const API_BASE = "http://172.25.114.124:4461"; 

export async function createRoom() {
  const res = await fetch(`${API_BASE}/create_room`, { method: "POST" });
  const data = await res.json();
  if (data.status === "success") return data.token;
  throw new Error(data.message || "Failed to create room");
}

export async function getJobResults(job_id) {
  const res = await fetch(`${API_BASE}/get_results/${job_id}`);
  const data = await res.json();
  if (data.status === "error") throw new Error(data.message);
  return data;
}

export async function getRoomJobs(roomToken) {
  console.log("Fetching jobs for room:", roomToken);
  const res = await fetch(`${API_BASE}/room_jobs/${roomToken}`);
  console.log("Response status:", res.status);
  const data = await res.json();
  console.log("Jobs response:", data);
  if (data.status === "error") throw new Error(data.message);
  return data.jobs;
}

export async function checkRoomExists(roomToken) {
  const res = await fetch(`${API_BASE}/check_room/${roomToken}`);
  const data = await res.json();
  if (data.status === "error") throw new Error(data.message);
  return data.exists;
}

export async function deleteJob(job_id) {
  try {
    const res = await fetch(`${API_BASE}/delete_job/${job_id}`, { method: "DELETE" });
    const data = await res.json();
    if (data.status === "error") throw new Error(data.message);
    return data;
  } catch (err) {
    console.error("Failed to delete job:", err);
    throw err;
  }
}