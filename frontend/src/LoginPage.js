import { signInWithEmailAndPassword } from "firebase/auth";
import { doc, getDoc, collection, query, where, getDocs } from "firebase/firestore";
import { auth, db } from "./firebaseConfig";
import { useNavigate } from "react-router-dom";
import { useState } from "react";
import { Link } from "react-router-dom";

function LoginPage() {
  const [loginInput, setLoginInput] = useState(""); // เปลี่ยนจาก email เป็น loginInput
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  // ฟังก์ชันเช็คว่าเป็นอีเมลหรือไม่
  const isEmail = (input) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(input);
  };

  // ฟังก์ชันค้นหาอีเมลจากชื่อผู้ใช้
  const findEmailByUsername = async (username) => {
    try {
      const usersRef = collection(db, "users");
      const q = query(usersRef, where("username", "==", username));
      const querySnapshot = await getDocs(q);

      if (!querySnapshot.empty) {
        const userDoc = querySnapshot.docs[0];
        return userDoc.data().email;
      }
      return null;
    } catch (error) {
      console.error("Error finding user by username:", error);
      return null;
    }
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      let emailToUse = loginInput;

      // ถ้าไม่ได้ใส่อีเมล ให้ค้นหาอีเมลจากชื่อผู้ใช้
      if (!isEmail(loginInput)) {
        const foundEmail = await findEmailByUsername(loginInput);
        if (!foundEmail) {
          alert("ไม่พบชื่อผู้ใช้นี้ในระบบ");
          return;
        }
        emailToUse = foundEmail;
      }

      // 1. ล็อกอินผ่าน Firebase Auth
      const userCredential = await signInWithEmailAndPassword(auth, emailToUse, password);
      const loggedInUser = userCredential.user;

      // 2. ดึงข้อมูลผู้ใช้จาก Firestore
      const docRef = doc(db, "users", loggedInUser.uid);
      const docSnap = await getDoc(docRef);

      if (docSnap.exists()) {
        const userData = docSnap.data();

        // 3. เช็กว่า role เป็น admin หรือไม่
        if (userData.role === "admin") {
          navigate("/admin-dashboard");
        } else {
          navigate("/user-dashboard");
        }
      } else {
        alert("ไม่พบข้อมูลผู้ใช้ในฐานข้อมูล");
      }

    } catch (error) {
      console.error("Login error:", error.message);
      // ปรับข้อความ error ให้เหมาะสม
      if (error.code === "auth/user-not-found") {
        alert("ไม่พบผู้ใช้นี้ในระบบ");
      } else if (error.code === "auth/wrong-password") {
        alert("รหัสผ่านไม่ถูกต้อง");
      } else if (error.code === "auth/invalid-email") {
        alert("รูปแบบอีเมลไม่ถูกต้อง");
      } else {
        alert("ชื่อผู้ใช้/อีเมลหรือรหัสผ่านไม่ถูกต้อง");
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

        {/* ใส่ div ห่อไว้เพื่อจัด flex */}
        <div className="login-footer-links">
          <Link to="/signup" className="footer-link">สมัครสมาชิก</Link>
          <Link to="/forgot-password" className="footer-link">ลืมรหัสผ่าน?</Link>
        </div>
      </form>
    </div>
  );
}

export default LoginPage;