import { signInWithEmailAndPassword } from "firebase/auth";
import { doc, getDoc, setDoc } from "firebase/firestore";
import { auth, db } from "./firebaseConfig";
import { useNavigate } from "react-router-dom";
import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import "./css/login.css";

function LoginPage() {
  const [loginInput, setLoginInput] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [loginAttempts, setLoginAttempts] = useState(0);
  const [lockoutTime, setLockoutTime] = useState(0);
  const navigate = useNavigate();

  const isEmail = (input) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(input);

  const delayTimes = [0, 2000, 5000, 10000, 20000]; // หน่วง 0, 2, 5, 10, 20 วินาที

  // นับถอยหลัง lockoutTime ทุกวินาที
  useEffect(() => {
    if (lockoutTime > 0) {
      const timer = setInterval(() => {
        setLockoutTime((prevTime) => {
          if (prevTime <= 1000) {
            return 0;
          }
          return prevTime - 1000;
        });
      }, 1000);
      return () => clearInterval(timer);
    }
  }, [lockoutTime]);

  const findEmailByUsername = async (username) => {
    try {
      const response = await fetch("https://render-backend-mu.vercel.app/find_email_by_username", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        body: JSON.stringify({ username }),
        signal: AbortSignal.timeout(10000), // Timeout 10 วินาที
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "เกิดข้อผิดพลาดจากเซิร์ฟเวอร์");
      }

      const data = await response.json();
      if (!data.email) {
        throw new Error("ไม่พบชื่อบัญชีนี้ในระบบ");
      }

      return data.email;
    } catch (error) {
      if (error.name === "TimeoutError") {
        throw new Error("การค้นหาชื่อบัญชีใช้เวลานานเกินไป กรุณาลองใหม่");
      } else if (error.message.includes("ไม่พบชื่อบัญชีนี้ในระบบ")) {
        throw new Error(error.message);
      }
      throw new Error("ไม่สามารถเชื่อมต่อ backend ได้ อาจเป็นปัญหาเครือข่ายหรือ CORS");
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
    setLoading(true);

    if (lockoutTime > 0) {
      alert(`กรุณารอ ${Math.ceil(lockoutTime / 1000)} วินาทีก่อนลองใหม่`);
      setLoading(false);
      return;
    }

    if (!loginInput || !password) {
      alert("กรุณากรอกชื่อบัญชีหรืออีเมลและรหัสผ่าน");
      setLoading(false);
      return;
    }

    try {
      let emailToUse = loginInput;
      let usernameFromLookup = "";

      if (!isEmail(loginInput)) {
        emailToUse = await findEmailByUsername(loginInput);
        if (!emailToUse) {
          setLoginAttempts((prevAttempts) => prevAttempts + 1);
          setLockoutTime(delayTimes[Math.min(loginAttempts + 1, delayTimes.length - 1)]);
          alert(`ชื่อบัญชีหรือรหัสผ่านไม่ถูกต้อง กรุณารอ ${Math.ceil(delayTimes[Math.min(loginAttempts + 1, delayTimes.length - 1)] / 1000)} วินาทีก่อนลองใหม่`);
          setLoading(false);
          return;
        }
        usernameFromLookup = loginInput;
      }

      const userCredential = await new Promise((resolve, reject) => {
        const timeoutId = setTimeout(() => {
          reject(new Error("การล็อกอินใช้เวลานานเกินไป กรุณาลองใหม่"));
        }, 10000);

        signInWithEmailAndPassword(auth, emailToUse, password)
          .then((credential) => {
            clearTimeout(timeoutId);
            resolve(credential);
          })
          .catch((err) => {
            clearTimeout(timeoutId);
            reject(err);
          });
      });

      const loggedInUser = userCredential.user;
      await ensureUserDocExists(loggedInUser, usernameFromLookup);

      const userDocRef = doc(db, "users", loggedInUser.uid);
      const userDocSnap = await getDoc(userDocRef);

      if (userDocSnap.exists()) {
        const userData = userDocSnap.data();
        setLoginAttempts(0);
        setLockoutTime(0);
        if (userData.role === "admin") {
          navigate("/admin-dashboard");
        } else {
          navigate("/");
        }
      } else {
        alert("ไม่พบข้อมูลผู้ใช้หลังล็อกอิน");
      }
    } catch (error) {
      setLoginAttempts((prevAttempts) => prevAttempts + 1);
      setLockoutTime(delayTimes[Math.min(loginAttempts + 1, delayTimes.length - 1)]);
      alert(`ชื่อบัญชีหรือรหัสผ่านไม่ถูกต้อง กรุณารอ ${Math.ceil(delayTimes[Math.min(loginAttempts + 1, delayTimes.length - 1)] / 1000)} วินาทีก่อนลองใหม่`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-container">
      <h2>เข้าสู่ระบบ</h2>
      <form onSubmit={handleLogin}>
        <input
          type="text"
          placeholder="ชื่อบัญชีหรืออีเมล"
          value={loginInput}
          onChange={(e) => setLoginInput(e.target.value)}
          required
          disabled={loading || lockoutTime > 0}
        />
        <input
          type="password"
          placeholder="รหัสผ่าน"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          disabled={loading || lockoutTime > 0}
        />
        <button type="login-submit" disabled={loading || lockoutTime > 0}>
          {loading ? "กำลังล็อกอิน..." : lockoutTime > 0 ? `รอ ${Math.ceil(lockoutTime / 1000)} วินาที` : "เข้าสู่ระบบ"}
        </button>
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