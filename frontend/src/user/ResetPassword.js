import { useState, useEffect } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { auth } from "../firebaseConfig";
import { confirmPasswordReset, verifyPasswordResetCode } from "firebase/auth";
import "../css/resetpassword.css";  // ✅ import css ที่เราแยกออกมา

function ResetPasswordPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [newPassword, setNewPassword] = useState("");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [oobCodeValid, setOobCodeValid] = useState(false);

  const oobCode = searchParams.get("oobCode"); // Firebase ส่ง code มาใน URL

  useEffect(() => {
    if (!oobCode) {
      setError("ลิงก์ไม่ถูกต้อง");
      return;
    }
    verifyPasswordResetCode(auth, oobCode)
      .then(() => setOobCodeValid(true))
      .catch(() => setError("ลิงก์หมดอายุหรือไม่ถูกต้อง"));
  }, [oobCode]);

  const validatePassword = (pwd) => {
    const regex = /^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{6,}$/;
    return regex.test(pwd);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    if (!validatePassword(newPassword)) {
      setError("รหัสผ่านต้องมีอย่างน้อย 6 ตัว และประกอบด้วย ตัวอักษร + ตัวเลข อย่างน้อยอย่างละ 1 ตัว");
      return;
    }

    try {
      await confirmPasswordReset(auth, oobCode, newPassword);
      setMessage("เปลี่ยนรหัสผ่านสำเร็จ! กำลังพากลับไปหน้าเข้าสู่ระบบ...");
      setTimeout(() => navigate("/login"), 3000);
    } catch (err) {
      console.error(err);
      setError("เกิดข้อผิดพลาด โปรดลองใหม่");
    }
  };

  if (!oobCodeValid) return <p>{error || "กำลังตรวจสอบลิงก์..."}</p>;

  return (
    <div className="reset-password-container">
      <h2>ตั้งรหัสผ่านใหม่</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="password"
          placeholder="รหัสผ่านใหม่"
          value={newPassword}
          onChange={(e) => setNewPassword(e.target.value)}
          required
        />
        <button type="submit">ตั้งรหัสผ่านใหม่</button>
      </form>
      {error && <p className="error-text">{error}</p>}
      {message && <p className="success-text">{message}</p>}
    </div>
  );
}

export default ResetPasswordPage;
