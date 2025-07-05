import React, { useState, useEffect } from "react";
import { getAuth, onAuthStateChanged } from "firebase/auth";
import { doc, getDoc } from "firebase/firestore";
import { db } from "../firebaseConfig"; // ตรวจสอบ path ให้ถูก
import { useNavigate } from "react-router-dom";

function UserDashboard() {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);

  useEffect(() => {
    const auth = getAuth();
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      if (currentUser) {
        const docRef = doc(db, "users", currentUser.uid);
        const docSnap = await getDoc(docRef);

        if (docSnap.exists()) {
          const userData = docSnap.data();
          if (userData.role === "user") {
            // ถ้าเป็นแอดมินให้ไปหน้า admin-dashboard
            navigate("/admin-dashboard");
          } else {
            setUser(currentUser); // เป็นผู้ใช้ทั่วไป
          }
        } else {
          console.error("ไม่พบข้อมูลผู้ใช้ใน Firestore");
          navigate("/login");
        }
      } else {
        navigate("/login");
      }
    });

    return () => unsubscribe();
  }, [navigate]);

  return (
    <div>
      <h2>Dashboard ผู้ใช้</h2>
      <p>ยินดีต้อนรับ, {user ? user.email : ""}</p>
    </div>
  );
}

export default UserDashboard;
