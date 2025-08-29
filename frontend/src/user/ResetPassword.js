import { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { auth } from "../firebaseConfig";
import { confirmPasswordReset, verifyPasswordResetCode } from "firebase/auth";
import "../css/resetpassword.css";

function ResetPasswordPage() {
    const [searchParams] = useSearchParams();
    const [newPassword, setNewPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");
    const [error, setError] = useState("");
    const [message, setMessage] = useState("");
    const [oobCodeValid, setOobCodeValid] = useState(false);
    const [completed, setCompleted] = useState(false);

    const [passwordValidation, setPasswordValidation] = useState({
        hasMinLength: false,
        hasLetter: false,
        hasNumber: false
    });

    const oobCode = searchParams.get("oobCode");

    const [showPassword, setShowPassword] = useState(false);
    const [showConfirmPassword, setShowConfirmPassword] = useState(false);

    useEffect(() => {
        if (!oobCode) {
            setError("ลิงก์ไม่ถูกต้อง");
            return;
        }
        verifyPasswordResetCode(auth, oobCode)
            .then(() => setOobCodeValid(true))
            .catch(() => setError("ลิงก์หมดอายุหรือไม่ถูกต้อง"));
    }, [oobCode]);

    // ตรวจสอบรหัสผ่าน realtime
    useEffect(() => {
        setPasswordValidation({
            hasMinLength: newPassword.length >= 6,
            hasLetter: /[a-zA-Z]/.test(newPassword),
            hasNumber: /[0-9]/.test(newPassword)
        });
    }, [newPassword]);

    const validatePassword = (pwd) =>
        pwd.length >= 6 && /[a-zA-Z]/.test(pwd) && /[0-9]/.test(pwd);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError("");

        if (!validatePassword(newPassword)) {
            setError("รหัสผ่านต้องมีอย่างน้อย 6 ตัว และประกอบด้วยตัวอักษร + ตัวเลข");
            return;
        }

        if (newPassword !== confirmPassword) {
            setError("รหัสผ่านไม่ตรงกัน");
            return;
        }

        try {
            await confirmPasswordReset(auth, oobCode, newPassword);
            setMessage("✅ เปลี่ยนรหัสผ่านสำเร็จ! โปรดใช้รหัสผ่านใหม่นี้ในการเข้าสู่ระบบครั้งถัดไป");
            setCompleted(true);
        } catch (err) {
            console.error(err);
            setError("เกิดข้อผิดพลาด โปรดลองใหม่");
        }
    };

    // ฟังก์ชันสำหรับ toggle การแสดงรหัสผ่าน
    const togglePasswordVisibility = () => {
        setShowPassword(!showPassword);
    };

    const toggleConfirmPasswordVisibility = () => {
        setShowConfirmPassword(!showConfirmPassword);
    };

    if (!oobCodeValid) return <p className="loading-text">{error || "กำลังตรวจสอบลิงก์..."}</p>;

    return (
        <div className="reset-password-container">
            {!completed ? (
                <>
                    <h2>ตั้งค่ารหัสผ่านใหม่</h2>
                    <form onSubmit={handleSubmit} className="reset-form">
                        <div className="password-input-wrapper">
                            <input
                                type={showPassword ? "text" : "password"}
                                placeholder="รหัสผ่านใหม่"
                                value={newPassword}
                                onChange={(e) => setNewPassword(e.target.value)}
                                required
                                className="password-input"
                            />
                            <button
                                type="button"
                                onClick={togglePasswordVisibility}
                                className="password-toggle-btn"
                            >
                                <img
                                    src={showPassword ? "/img/hide.png" : "/img/view.png"}
                                    alt={showPassword ? "hide" : "view"}
                                    className="password-toggle-icon"
                                />
                            </button>
                        </div>

                        <div className="password-input-wrapper">
                            <input
                                type={showConfirmPassword ? "text" : "password"}
                                placeholder="ยืนยันรหัสผ่านใหม่"
                                value={confirmPassword}
                                onChange={(e) => setConfirmPassword(e.target.value)}
                                required
                                className="password-input"
                            />
                            <button
                                type="button"
                                onClick={toggleConfirmPasswordVisibility}
                                className="password-toggle-btn"
                            >
                                <img
                                    src={showConfirmPassword ? "/img/hide.png" : "/img/view.png"}
                                    alt={showConfirmPassword ? "hide" : "view"}
                                    className="password-toggle-icon"
                                />
                            </button>
                        </div>

                        <div className="password-validation">
                            <div className={`validation-item ${passwordValidation.hasMinLength ? "valid" : "invalid"}`}>
                                {passwordValidation.hasMinLength ? "✅" : "❌"} อย่างน้อย 6 ตัวอักษร
                            </div>
                            <div className={`validation-item ${passwordValidation.hasLetter ? "valid" : "invalid"}`}>
                                {passwordValidation.hasLetter ? "✅" : "❌"} มีตัวอักษรอย่างน้อย 1 ตัว
                            </div>
                            <div className={`validation-item ${passwordValidation.hasNumber ? "valid" : "invalid"}`}>
                                {passwordValidation.hasNumber ? "✅" : "❌"} มีตัวเลขอย่างน้อย 1 ตัว
                            </div>
                        </div>

                        <button type="submit" className="submit-btn">ยืนยัน</button>
                    </form>
                    {error && <p className="error-text">{error}</p>}
                </>
            ) : (
                <p className="success-text">{message}</p>
            )}
        </div>
    );
}

export default ResetPasswordPage;