import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { db } from "../firebaseConfig";
import { collection, getDocs } from "firebase/firestore";
import { getAuth, onAuthStateChanged } from "firebase/auth";

function AccountManagement() {
  const navigate = useNavigate();  // ใช้ navigate เพื่อเปลี่ยนเส้นทาง
  const auth = getAuth();  // ดึงข้อมูล authentication จาก Firebase
  const [loading, setLoading] = useState(true);  // สถานะการโหลดข้อมูล
  const [usersList, setUsersList] = useState([]);  // รายการผู้ใช้ทั้งหมด

  // ตรวจสอบสถานะของผู้ใช้ว่าเป็นผู้ใช้ที่ลงชื่อเข้าใช้งานหรือไม่
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      if (currentUser) {
        // หากมีผู้ใช้ที่เข้าสู่ระบบแล้ว ให้ดึงข้อมูลรายการผู้ใช้
        await fetchUsersList();
      } else {
        // หากไม่มีผู้ใช้ที่เข้าสู่ระบบ ให้เปลี่ยนเส้นทางไปยังหน้า login
        navigate("/login");
      }
    });
    return () => unsubscribe(); // ลบการติดตามเมื่อ component ถูกลบ
  }, [auth, navigate]);

  // ฟังก์ชันในการดึงข้อมูลรายการผู้ใช้จาก Firestore
  const fetchUsersList = async () => {
    setLoading(true);  // ตั้งค่าการโหลดข้อมูลเป็น true
    try {
      const usersRef = collection(db, "users");  // อ้างอิงถึง collection "users" ใน Firestore
      const snapshot = await getDocs(usersRef);  // ดึงข้อมูลจาก Firestore
      const users = snapshot.docs.map((doc) => ({
        id: doc.id,  // รหัสผู้ใช้
        ...doc.data(),  // ข้อมูลของผู้ใช้
      }));
      setUsersList(users);  // เก็บข้อมูลใน state
    } catch (error) {
      console.error("เกิดข้อผิดพลาดในการดึงรายชื่อผู้ใช้:", error);
    } finally {
      setLoading(false);  // ตั้งค่าการโหลดข้อมูลเป็น false เมื่อเสร็จสิ้น
    }
  };

  // หากกำลังโหลดข้อมูลจะแสดงข้อความ "กำลังโหลด..."
  if (loading) {
    return (
      <div className="manage-container">
        <h2 className="title">กำลังโหลด...</h2>
      </div>
    );
  }

  return (
    <div className="manage-container">
      <h2 className="title">บัญชีผู้ใช้</h2>

      {/* แสดงรายการผู้ใช้ */}
      <div className="users-list">
        {usersList.length > 0 ? (
          usersList.map((user) => (
            <div
              key={user.id}
              className="user-card"
              onClick={() => navigate(`/userdetails/${user.id}`)} // ✅ เชื่อมกับ UserDetails
            >
              <h3 className="user-name">{user.fullName}</h3> {/* แสดงชื่อผู้ใช้ */}
            </div>
          ))
        ) : (
          <p className="no-users">ยังไม่มีผู้ใช้ในระบบ</p>  // หากไม่มีผู้ใช้แสดงข้อความนี้
        )}
      </div>

      {/* กลุ่มปุ่มเพื่อไปยังหน้าอื่น */}
      <div className="button-group">
        <button className="btn btn-gray" onClick={() => navigate("/")}>
          ⬅️ กลับ
        </button>
        <button className="btn btn-green" onClick={() => navigate("/signup")}>
          ➕ เพิ่มสมาชิก
        </button>
      </div>
    </div>
  );
}

export default AccountManagement;
