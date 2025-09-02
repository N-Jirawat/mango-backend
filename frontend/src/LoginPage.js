import { signInWithEmailAndPassword } from "firebase/auth";
import { doc, getDoc, setDoc } from "firebase/firestore";
import { auth, db } from "./firebaseConfig";
import { useNavigate } from "react-router-dom";
import { useState } from "react";
import { Link } from "react-router-dom";

function LoginPage() {
  const [loginInput, setLoginInput] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  const isEmail = (input) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(input);

  const findEmailByUsername = async (username) => {
    try {
      const res = await fetch("https://render-backend-mu.vercel.app/find_email_by_username", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username }),
      });

      const data = await res.json();
      if (!res.ok) {
        console.error("Backend error:", data);
        alert(data.error || "เกิดข้อผิดพลาดจากเซิร์ฟเวอร์");
        return null;
      }

      if (!data.email) {
        alert("ไม่พบชื่อผู้ใช้นี้ในระบบ");
        return null;
      }

      return data.email;
    } catch (error) {
      console.error("Network or fetch error:", error);
      alert("ไม่สามารถเชื่อมต่อ backend ได้");
      return null;
    }
  };

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
        createdAt: new Date(),
      });
    }
  };

  const handleLogin = async (e) => {
    e.preventDefault();

    if (!loginInput || !password) {
      alert("กรุณากรอกชื่อผู้ใช้/อีเมลและรหัสผ่าน");
      return;
    }

    try {
      let emailToUse = loginInput;
      let usernameFromLookup = "";

      if (!isEmail(loginInput)) {
        const foundEmail = await findEmailByUsername(loginInput);
        if (!foundEmail) {
          return;
        }
        emailToUse = foundEmail;
        usernameFromLookup = loginInput;
      }

      const userCredential = await signInWithEmailAndPassword(auth, emailToUse, password);
      const loggedInUser = userCredential.user;

      await ensureUserDocExists(loggedInUser, usernameFromLookup);

      const userDocRef = doc(db, "users", loggedInUser.uid);
      const userDocSnap = await getDoc(userDocRef);

      if (userDocSnap.exists()) {
        const userData = userDocSnap.data();
        if (userData.role === "admin") {
          navigate("/admin-dashboard");
        } else {
          navigate("/");
        }
      } else {
        alert("ไม่พบข้อมูลผู้ใช้หลังล็อกอิน");
      }
    } catch (error) {
      console.error("Login error:", error);
      if (error.code === "auth/invalid-credential" || error.code === "auth/wrong-password" || error.code === "auth/user-not-found") {
        alert("ชื่อผู้ใช้/อีเมลหรือรหัสผ่านไม่ถูกต้อง");
      } else if (error.code === "auth/invalid-email") {
        alert("รูปแบบอีเมลไม่ถูกต้อง");
      } else if (error.code === "auth/too-many-requests") {
        alert("มีการพยายามล็อกอินมากเกินไป กรุณารอสักครู่แล้วลองใหม่");
      } else {
        alert("เกิดข้อผิดพลาดในการล็อกอิน: " + error.message);
      }
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
          <Link to="/signup" className="footer-link">
            สมัครสมาชิก
          </Link>
          <Link to="/forgot-password" className="footer-link">
            ลืมรหัสผ่าน?
          </Link>
        </div>
      </form>
    </div>
  );
}

export default LoginPage;