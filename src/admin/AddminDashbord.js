import React, { useState, useEffect } from "react";
import { db } from "../firebaseConfig";
import { collection, getDocs } from "firebase/firestore";
import { Link } from "react-router-dom";
import "../css/addmindashbord.css"; // ไฟล์นี้ชื่อเดิม ถ้าอยากเปลี่ยนชื่อไฟล์ ต้องเปลี่ยนตรงนี้ด้วยนะ

function AdminDashboard() {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      await getDocs(collection(db, "users"));  // ไม่เก็บผลลัพธ์
      setLoading(false);
    };

    fetchData();
  }, []);

  if (loading) {
    return <p>กำลังโหลดข้อมูล...</p>;
  }

  return (
    <div className="admin-manage-container-dashbord">
      <h2 className="title-dashbord">เลือกการจัดการ</h2>
      <div className="button-wrapper-dashbord">
        <div className="button-container-dashbord">
          <Link to="/accountmanagement" className="admin-link-dashbord">
            <button className="admin-button-dashbord">จัดการบัญชีผู้ใช้</button>
          </Link>
        </div>
        <div className="button-container-dashbord">
          <Link to="/mango" className="admin-link-dashbord">
            <button className="admin-button-dashbord">จัดการข้อมูลโรคใบมะม่วง</button>
          </Link>
        </div>
      </div>
    </div>
  );
}

export default AdminDashboard;
