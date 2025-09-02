import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { sendPasswordResetEmail } from "firebase/auth";
import { auth } from "../firebaseConfig";
import "../css/forgotpassword.css";

function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const validateEmail = (email) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);

  const handleReset = async (e) => {
    e.preventDefault();
    setMessage("");
    setError("");
    setLoading(true);

    if (!email.trim()) {
      setError("กรุณากรอกอีเมล");
      setLoading(false);
      return;
    }

    if (!validateEmail(email)) {
      setError("รูปแบบอีเมลไม่ถูกต้อง");
      setLoading(false);
      return;
    }

    try {
      const actionCodeSettings = {
        url: "https://mangoleafanalyzer.onrender.com/reset-password",
        handleCodeInApp: true,
      };

      // ห่อ sendPasswordResetEmail ด้วย Promise และ timeout
      await new Promise((resolve, reject) => {
        const timeoutId = setTimeout(() => {
          reject(new Error("การส่งลิงก์รีเซ็ตรหัสผ่านใช้เวลานานเกินไป กรุณาลองใหม่"));
        }, 10000); // Timeout 10 วินาที

        sendPasswordResetEmail(auth, email, actionCodeSettings)
          .then(() => {
            clearTimeout(timeoutId);
            resolve();
          })
          .catch((err) => {
            clearTimeout(timeoutId);
            reject(err);
          });
      });

      setMessage("✅ ส่งลิงก์รีเซ็ตรหัสผ่านไปยังอีเมลแล้ว กรุณาตรวจสอบกล่องจดหมายหรือโฟลเดอร์สแปม");
      setTimeout(() => navigate("/login"), 5000); // Redirect ไปหน้า login หลัง 5 วินาที
    } catch (err) {
      console.error("Reset password error:", err);
      if (err.code === "auth/user-not-found") {
        setError("ไม่พบบัญชีผู้ใช้ที่เชื่อมโยงกับอีเมลนี้");
      } else if (err.code === "auth/invalid-email") {
        setError("รูปแบบอีเมลไม่ถูกต้อง");
      } else if (err.code === "auth/too-many-requests") {
        setError("มีการร้องขอมากเกินไป กรุณารอสักครู่แล้วลองใหม่");
      } else if (err.message.includes("การส่งลิงก์รีเซ็ตรหัสผ่านใช้เวลานานเกินไป")) {
        setError(err.message);
      } else {
        setError("เกิดข้อผิดพลาด โปรดลองใหม่ภายหลัง");
      }
      setLoading(false);
    }
  };

  return (
    <div className="forgot-password-container">
      <div className="forgot-header">
        <button
          onClick={() => navigate("/login")}
          className="back-button"
          disabled={loading}
        >
          ⬅️ หน้าหลัก
        </button>
      </div>
      <h2>ลืมรหัสผ่าน</h2>
      <form onSubmit={handleReset}>
        <input
          type="email"
          placeholder="กรอกอีเมล เพื่อรีเซ็ตรหัสผ่าน"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          disabled={loading}
        />
        <button type="submit" disabled={loading}>
          {loading ? "กำลังส่ง..." : "ส่งลิงก์รีเซ็ตรหัสผ่าน"}
        </button>
      </form>
      {message && (
        <div className="success-container">
          <p className="success-text">{message}</p>
          <p className="redirect-text">กำลังนำคุณไปยังหน้าเข้าสู่ระบบ...</p>
        </div>
      )}
      {error && <p className="error-text">{error}</p>}
    </div>
  );
}

export default ForgotPasswordPage;