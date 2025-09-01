import { signInWithEmailAndPassword } from "firebase/auth";
import { doc, getDoc, setDoc } from "firebase/firestore";
import { auth, db } from "./firebaseConfig";
import { useNavigate } from "react-router-dom";
import { useState } from "react";
import { Link } from "react-router-dom";

function LoginPage() {
  const [loginInput, setLoginInput] = useState(""); // username หรือ email
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  const isEmail = (input) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(input);

  // lookup email ผ่าน Flask backend
  const findEmailByUsername = async (username) => {
    try {
      const res = await fetch("https://render-backend-mu.vercel.app/find_email_by_username", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username }),
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        console.error("Backend error:", data);
        alert(data.error || "เกิดข้อผิดพลาดจากเซิร์ฟเวอร์");
        return null;
      }

      return data.email;
    } catch (error) {
      console.error("Network or fetch error:", error);
      alert("ไม่สามารถเชื่อมต่อ backend ได้");
      return null;
    }
  };

  // สร้าง /users/{uid} ถ้ายังไม่มี
  const ensureUserDocExists = async (user, username = "") => {
    const docRef = doc(db, "users", user.uid);
    const docSnap = await getDoc(docRef);

    if (!docSnap.exists()) {
      await setDoc(docRef, {
        uid: user.uid,
        email: user.email,
        username: username,
        role: "user",
        tel: "",
        address: "",
        district: "",
        province: "",
        subdistrict: "",
        village: "",
        fullName: "",
        createdAt: new Date()
      });
    }
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      let emailToUse = loginInput;
      let usernameFromLookup = "";

      if (!isEmail(loginInput)) {
        const foundEmail = await findEmailByUsername(loginInput);
        if (!foundEmail) {
          alert("ไม่พบชื่อผู้ใช้นี้ในระบบ");
          return;
        }
        emailToUse = foundEmail;
        usernameFromLookup = loginInput;
      }

      // login Firebase Auth ด้วย email + password
      const userCredential = await signInWithEmailAndPassword(auth, emailToUse, password);
      const loggedInUser = userCredential.user;

      await ensureUserDocExists(loggedInUser, usernameFromLookup);

      const userDocRef = doc(db, "users", loggedInUser.uid);
      const userDocSnap = await getDoc(userDocRef);

      if (userDocSnap.exists()) {
        const userData = userDocSnap.data();
        if (userData.role === "admin") navigate("/admin-dashboard");
        else navigate("/");
      } else {
        alert("เกิดข้อผิดพลาด ไม่พบข้อมูลผู้ใช้หลัง login");
      }
    } catch (error) {
      console.error("Login error:", error.message);
      if (error.code === "auth/user-not-found") alert("ไม่พบผู้ใช้นี้ในระบบ");
      else if (error.code === "auth/wrong-password") alert("รหัสผ่านไม่ถูกต้อง");
      else if (error.code === "auth/invalid-email") alert("รูปแบบอีเมลไม่ถูกต้อง");
      else alert("ชื่อผู้ใช้/อีเมลหรือรหัสผ่านไม่ถูกต้อง");
    }
  };

  return (
    <div className="login-container">
      <h2>เข้าสู่ระบบ</h2>
      <form onSubmit={handleLogin}>
        <input
          type="text"
          placeholder="ชื่อผู้ใช้หรืออีเมล"
          value={loginInput}
          onChange={(e) => setLoginInput(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="รหัสผ่าน"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        <button type="submit">เข้าสู่ระบบ</button>

        <div className="login-footer-links">
          <Link to="/signup" className="footer-link">สมัครสมาชิก</Link>
          <Link to="/forgot-password" className="footer-link">ลืมรหัสผ่าน?</Link>
        </div>
      </form>
    </div>
  );
}

export default LoginPage;
