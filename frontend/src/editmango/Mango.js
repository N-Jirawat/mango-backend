import React, { useState, useEffect } from "react";
import { db } from "../firebaseConfig";
import { collection, getDocs } from "firebase/firestore";
import { useNavigate } from "react-router-dom";
import "../css/mango.css";

function Mango() {
  const navigate = useNavigate();
  const [diseases, setDiseases] = useState([]);
  const [loading, setLoading] = useState(true);  // เพิ่มสถานะการโหลด
  const [error, setError] = useState(null);  // เพิ่มสถานะข้อผิดพลาด

  // ดึงข้อมูลโรคจาก Firestore เมื่อหน้าโหลด
  useEffect(() => {
    const fetchDiseases = async () => {
      try {
        const querySnapshot = await getDocs(collection(db, "mango_diseases"));
        const data = querySnapshot.docs.map((doc) => ({
          id: doc.id,
          ...doc.data(),
        }));
        setDiseases(data);
      } catch (error) {
        console.error("Error fetching diseases:", error);
        setError("เกิดข้อผิดพลาดในการโหลดข้อมูลโรค");
      } finally {
        setLoading(false);  // การโหลดเสร็จสิ้น
      }
    };

    fetchDiseases();
  }, []);

  // แสดงรายการโรคมะม่วงหรือสถานะการโหลด
  if (loading) {
    return <div>กำลังโหลดข้อมูล...</div>;
  }

  if (error) {
    return <div style={{ color: "red" }}>{error}</div>;
  }

  return (
    <div className="card">
      <h2>รายการโรคมะม่วง</h2>
      <button
        onClick={() => navigate("/addMango")}
        style={{ marginBottom: "10px" }}
      >
        ➕ เพิ่มข้อมูล
      </button>

      <div className="disease-list">
        {diseases.length === 0 ? (
          <p>ไม่มีข้อมูลโรคมะม่วงในระบบ</p>
        ) : (
          diseases.map((disease) => (
            <div
              key={disease.id}
              className="disease-item"
              onClick={() => navigate(`/editmango/${disease.id}`)}  // ปรับให้ URL ถูกต้อง
            >
              <h3>{disease.diseaseName}</h3>
              <p>{disease.symptoms}</p>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default Mango;
