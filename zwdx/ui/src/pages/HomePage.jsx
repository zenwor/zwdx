import React, { useState } from "react";
import { createRoom } from "../api";
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

  const handleJoinRoom = () => {
    if (!roomToken) {
      alert("Please enter a room token");
      return;
    }
    navigate(`/room/${roomToken}`);
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
