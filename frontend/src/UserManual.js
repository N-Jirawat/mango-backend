import React from "react";
import { useNavigate } from "react-router-dom";
import "./css/UserManual.css";

function UserManual() {
  const navigate = useNavigate();

  return (
    <div className="user-manual-container">
      <h1>📘 คู่มือการใช้งาน</h1>
      <div className="manual-choice">
        <button
          className="manual-button"
          onClick={() => navigate("/usermanualmobile")}
        >
          📱 คู่มือสำหรับใช้งานบนโทรศัพท์มือถือ
        </button>
        <button
          className="manual-button"
          onClick={() => navigate("/usermanualpc")}
        >
          💻 คู่มือสำหรับใช้งานบนเครื่องคอมพิวเตอร์
        </button>
      </div>
    </div>
  );
}

export default UserManual;
