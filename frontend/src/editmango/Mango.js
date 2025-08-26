import React, { useState, useEffect } from "react";
import { db } from "../firebaseConfig";
import { collection, getDocs} from "firebase/firestore";
import { useNavigate } from "react-router-dom";
import "../css/mango.css";

function Mango() {
  const navigate = useNavigate();
  const [diseases, setDiseases] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // ดึงข้อมูลโรคจาก Firestore เมื่อหน้าโหลด
  useEffect(() => {
    const fetchDiseases = async () => {
      try {
        // 1. ดึงข้อมูลจาก MangoDisease collection
        const mangoDiseaseSnapshot = await getDocs(collection(db, "MangoDisease"));
        
        // 2. ดึงข้อมูลรูปภาพจาก ImageMango collection
        const imageMangoSnapshot = await getDocs(collection(db, "ImageMango"));
        const imageMap = {};
        
        // สร้าง map ของรูปภาพตาม DiseaseID
        imageMangoSnapshot.docs.forEach((doc) => {
          const imageData = doc.data();
          imageMap[imageData.DiseaseID] = {
            id: doc.id,
            ImgPath: imageData.ImgPath,
            DateUploadImg: imageData.DateUploadImg
          };
        });

        // 3. รวมข้อมูลโรคกับรูปภาพ
        const diseasesData = mangoDiseaseSnapshot.docs.map((doc) => {
          const diseaseData = doc.data();
          const imageInfo = imageMap[doc.id] || {};
          
          return {
            id: doc.id,
            DiseaseName: diseaseData.DiseaseName || "",
            Style: diseaseData.Style || "",
            Treatment: diseaseData.Treatment || "",
            Protection: diseaseData.Protection || "",
            UpdateAt: diseaseData.UpdateAt,
            ImgID: diseaseData.ImgID || "",
            // ข้อมูลรูปภาพ
            ImgPath: imageInfo.ImgPath || "",
            DateUploadImg: imageInfo.DateUploadImg
          };
        });

        // เรียงลำดับตามวันที่อัปเดตล่าสุด
        diseasesData.sort((a, b) => {
          if (a.UpdateAt && b.UpdateAt) {
            return b.UpdateAt.toDate() - a.UpdateAt.toDate();
          }
          return 0;
        });

        setDiseases(diseasesData);
      } catch (error) {
        console.error("Error fetching diseases:", error);
        setError("เกิดข้อผิดพลาดในการโหลดข้อมูลโรค");
      } finally {
        setLoading(false);
      }
    };

    fetchDiseases();
  }, []);

  // แสดงรายการโรคมะม่วงหรือสถานะการโหลด
  if (loading) {
    return (
      <div className="card">
        <div style={{ textAlign: "center", padding: "20px" }}>
          กำลังโหลดข้อมูล...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div style={{ color: "red", textAlign: "center", padding: "20px" }}>
          {error}
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <h2>รายการโรคมะม่วง</h2>
      <button
        onClick={() => navigate("/addMango")}
        style={{ marginBottom: "20px" }}
        className="add-button"
      >
        ➕ เพิ่มข้อมูลโรคมะม่วง
      </button>

      <div className="disease-list">
        {diseases.length === 0 ? (
          <div style={{ textAlign: "center", padding: "40px" }}>
            <p>ไม่มีข้อมูลโรคมะม่วงในระบบ</p>
            <p>คลิกปุ่ม "เพิ่มข้อมูลโรคมะม่วง" เพื่อเริ่มต้นเพิ่มข้อมูล</p>
          </div>
        ) : (
          diseases.map((disease) => (
            <div
              key={disease.id}
              className="disease-item"
              onClick={() => navigate(`/mangodetail/${disease.id}`)}
            >
              {/* แสดงรูปภาพถ้ามี */}
              {disease.ImgPath && (
                <div className="disease-image-container">
                  <img 
                    src={disease.ImgPath} 
                    alt={disease.DiseaseName}
                    className="disease-thumbnail"
                    onError={(e) => {
                      e.target.style.display = 'none';
                    }}
                  />
                </div>
              )}
              
              <div className="disease-info">
                <h3>{disease.DiseaseName}</h3>
                <p className="disease-symptoms">
                  <strong>อาการ:</strong> {disease.Style}
                </p>
                <p className="disease-meta">
                  {disease.UpdateAt && (
                    <small>
                      อัปเดตล่าสุด: {disease.UpdateAt.toDate().toLocaleDateString('th-TH')}
                    </small>
                  )}
                </p>
              </div>
              
              {/* ปุ่มแก้ไขแยกจากการคลิกหลัก */}
              <div className="disease-actions">
                <button
                  className="edit-quick-btn"
                  onClick={(e) => {
                    e.stopPropagation(); // ป้องกันไม่ให้เรียก onClick ของ parent
                    navigate(`/editmango/${disease.id}`);
                  }}
                  title="แก้ไขข้อมูล"
                >
                  ✏️
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default Mango;