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
  console.log("Fetching jobs for room:", roomToken); // 👈 add this line
  const res = await fetch(`${API_BASE}/room_jobs/${roomToken}`);
  console.log("Response status:", res.status); // 👈 add this line
  const data = await res.json();
  console.log("Jobs response:", data); // 👈 add this line
  if (data.status === "error") throw new Error(data.message);
  return data.jobs;
}