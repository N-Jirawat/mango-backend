import { useState } from "react";
import { useNavigate } from "react-router-dom";  // <-- import useNavigate
import { sendPasswordResetEmail } from "firebase/auth";
import { auth, db } from "../firebaseConfig";
import { collection, query, where, getDocs } from "firebase/firestore";
import "../css/forgotpassword.css";

function ForgotPasswordPage() {
    const [input, setInput] = useState("");
    const [message, setMessage] = useState("");
    const [error, setError] = useState("");
    const navigate = useNavigate();  // <-- ใช้ useNavigate

    const isEmail = (input) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(input);

    const findEmailByUsername = async (username) => {
        const usersRef = collection(db, "users");
        const q = query(usersRef, where("username", "==", username));
        const snapshot = await getDocs(q);
        if (!snapshot.empty) {
            return snapshot.docs[0].data().email;
        }
        return null;
    };

    const handleReset = async (e) => {
        e.preventDefault();
        setMessage("");
        setError("");

        let emailToUse = input;
        try {
            if (!isEmail(input)) {
                const foundEmail = await findEmailByUsername(input);
                if (!foundEmail) {
                    setError("ไม่พบชื่อผู้ใช้นี้ในระบบ");
                    return;
                }
                emailToUse = foundEmail;
            }

            await sendPasswordResetEmail(auth, emailToUse);
            setMessage("ส่งลิงก์รีเซ็ตรหัสผ่านไปยังอีเมลแล้ว กรุณาตรวจสอบกล่องจดหมาย");

            // รอ 8 วินาที แล้วพาไปหน้า login
            setTimeout(() => {
                navigate("/login");
            }, 8000);

        } catch (err) {
            console.error(err);
            if (err.code === "auth/user-not-found") {
                setError("ไม่พบบัญชีผู้ใช้นี้");
            } else if (err.code === "auth/invalid-email") {
                setError("อีเมลไม่ถูกต้อง");
            } else {
                setError("เกิดข้อผิดพลาด โปรดลองใหม่ภายหลัง");
            }
        }
    };

    return (
        <div className="forgot-password-container">
            <div className="forgot-header">
                <button onClick={() => navigate("/login")} className="back-button">
                    ⬅️ หน้าหลัก
                </button>
            </div>
            <h2>ลืมรหัสผ่าน</h2>
            <form onSubmit={handleReset}>
                <input
                    type="text"
                    placeholder="ชื่อผู้ใช้หรืออีเมล"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    required
                />
                <button type="submit">ส่งลิงก์รีเซ็ตรหัสผ่าน</button>
            </form>
            {message && <p style={{ color: "green" }}>{message}</p>}
            {error && <p style={{ color: "red" }}>{error}</p>}
        </div>
    );
}

export default ForgotPasswordPage;
