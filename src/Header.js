// Header.js (เปลี่ยนปุ่มล็อกอินเป็นชื่อบัญชี)
import React, { useState, useEffect } from "react";
import { getAuth, onAuthStateChanged, signOut } from "firebase/auth";
import { getDoc, doc } from "firebase/firestore";
import { db } from "./firebaseConfig";
import { useNavigate } from "react-router-dom";

function Header() {
  const [username, setUsername] = useState(null);
  const auth = getAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      if (user) {
        const userDoc = await getDoc(doc(db, "users", user.uid));
        if (userDoc.exists()) {
          setUsername(userDoc.data().username);
        }
      } else {
        setUsername(null);
      }
    });
    return () => unsubscribe();
  }, [auth]);

  const handleLogout = async () => {
    await signOut(auth);
    navigate("/");
  };

  return (
    <nav>
      {username ? (
        <div>
          <button onClick={() => navigate("/profile")}>{username}</button>
          <button onClick={handleLogout}>ออกจากระบบ</button>
        </div>
      ) : (
        <button onClick={() => navigate("/login")}>ลงชื่อเข้าใช้</button>
      )}
    </nav>
  );
}

export default Header;