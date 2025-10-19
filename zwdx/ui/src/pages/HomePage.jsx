import React, { useState } from "react";
import { createRoom, checkRoomExists } from "../api";
import "../styles/HomePage.css";
import { useNavigate } from "react-router-dom";

export default function HomePage() {
  const [roomToken, setRoomToken] = useState("");
  const navigate = useNavigate();

  const handleGenerateRoom = async () => {
    try {
      const token = await createRoom();
      setRoomToken(token);
      navigate(`/room/${token}`);
    } catch (err) {
      alert("Error creating room: " + err.message);
    }
  };

  const handleJoinRoom = async () => {
    if (!roomToken) {
      alert("Please enter a room token");
      return;
    }

    try {
      const exists = await checkRoomExists(roomToken);
      if (exists) {
        navigate(`/room/${roomToken}`);
      } else {
        alert("Room does not exist. Please check the token or create a new one.");
      }
    } catch (err) {
      alert("Error checking room: " + err.message);
    }
  };

  return (
    <div className="home-container">
      <div className="header-and-input">
        <h1>
          Welcome to 
          <span className="pixel-title vertical-adjust" data-text="zwdx">
              zwdx
              {/* <span className="emoji">🚀</span> */}
          </span>
        </h1>

        <div className="input-section">
          <div className="input-with-button">
            <input
              type="text"
              id="roomToken"
              placeholder="Enter your room token"
              value={roomToken}
              onChange={(e) => setRoomToken(e.target.value)}
            />
            <button className="button join-btn" onClick={handleJoinRoom}>
              Join
            </button>
          </div>

          <div className="generate-section">
            <button className="button" onClick={handleGenerateRoom}>
              Don't have one? Generate a room token!
            </button>
          </div>
        
        </div>
     </div>
    </div>
  );
}
