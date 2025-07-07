import React, { useEffect, useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { getDoc, doc } from "firebase/firestore";
import { db } from "../firebaseConfig";
import { getAuth, onAuthStateChanged } from "firebase/auth";
import "../css/UserDetails.css";

function UserDetails() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [userInfo, setUserInfo] = useState(null);
  const [role, setRole] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const auth = getAuth();

    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      if (!currentUser) {
        navigate("/login");
        return;
      }

      const currentUserDoc = await getDoc(doc(db, "users", currentUser.uid));
      const currentUserData = currentUserDoc.data();

      if (!currentUserData) {
        alert("ไม่พบข้อมูลผู้ใช้ปัจจุบัน");
        navigate("/login");
        return;
      }

      setRole(currentUserData.role);

      const userDoc = await getDoc(doc(db, "users", id));
      if (userDoc.exists()) {
        setUserInfo(userDoc.data());
      } else {
        alert("ไม่พบข้อมูลผู้ใช้");
      }

      setLoading(false);
    });

    return () => unsubscribe();
  }, [id, navigate]);

  const handleBack = () => {
    // สมมติ role มีค่า "admin" หรือ "user"
    if (role === "admin") {
      navigate("/accountmanagement");
    } else {
      navigate("/");
    }
  };

  const roleName = (role) => {
    switch (role) {
      case "admin": return "ผู้ดูแลระบบ";
      case "user": return "สมาชิก";
      default: return "ไม่ระบุบทบาท";
    }
  };

  if (loading) return <p>กำลังโหลดข้อมูล...</p>;

  if (!userInfo) return <p>ไม่พบข้อมูลผู้ใช้</p>;

  return (
    <div className="card">
      <h2>
        รายละเอียดข้อมูล (
        <span className="role-green">
          {roleName(userInfo.role)}
        </span>
        )
      </h2>
      <p><strong>ชื่อบัญชีผู้ใช้:</strong> {userInfo.username || "-"}</p>
      <p><strong>ชื่อเต็ม:</strong> {userInfo.fullName || "-"}</p>
      <p><strong>ที่อยู่:</strong>
        {userInfo.address ? `${userInfo.address}, บ้าน ${userInfo.village || "-"}, ตำบล ${userInfo.subdistrict || "-"}, อำเภอ ${userInfo.district || "-"}, จังหวัด ${userInfo.province || "-"}` : "-"}
      </p>
      <p><strong>เบอร์โทร:</strong> {userInfo.tel ? userInfo.tel.toString() : "-"}</p>
      <p><strong>อีเมล:</strong> {userInfo.email || "-"}</p>
      <div className="link-container">
        <Link to={`/staticsuser`}>
          <span>ข้อมูลการใช้งาน</span>
        </Link>
      </div>

      <div className="button-container">
        <button onClick={handleBack}>⬅️ กลับ</button>
        <Link to={`/edituser/${id}`}>
          <button>✏️ แก้ไขข้อมูล</button>
        </Link>
      </div>
    </div>
  );
}

export default UserDetails;
