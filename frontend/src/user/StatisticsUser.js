import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getAuth, onAuthStateChanged } from "firebase/auth";
import { getFirestore, collection, query, where, getDocs } from "firebase/firestore";
import "../css/StatisticsUser.css";

function StatisticsUser() {
  const navigate = useNavigate();
  const [loginCount, setLoginCount] = useState(0);
  const [analysisCount, setAnalysisCount] = useState(0);
  const [diseaseStats, setDiseaseStats] = useState({});
  const [lastActive, setLastActive] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const db = getFirestore();
    const auth = getAuth();

    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      if (!user) {
        alert("กรุณาเข้าสู่ระบบก่อนใช้งาน");
        navigate("/login");
        return;
      }

      try {
        // 1. จำนวนครั้งที่เข้าสู่ระบบ
        try {
          const loginQuery = query(collection(db, "loginLogs"), where("uid", "==", user.uid));
          const loginSnapshot = await getDocs(loginQuery);
          setLoginCount(loginSnapshot.size);
        } catch (error) {
          console.log("ไม่พบข้อมูลการเข้าสู่ระบบ:", error);
          setLoginCount(0);
        }

        // 2. จำนวนภาพที่วิเคราะห์แล้ว + 3. สัดส่วนโรค (จาก prediction_results)
        try {
          const predictionQuery = query(
            collection(db, "prediction_results"),
            where("userId", "==", user.uid)
          );
          const predictionSnapshot = await getDocs(predictionQuery);

          setAnalysisCount(predictionSnapshot.size);

          const diseaseMap = {};
          let latestDate = null;

          predictionSnapshot.forEach((doc) => {
            const data = doc.data();
            const disease = data.diseaseName || data.predictedClass || "ไม่ระบุโรค";
            
            // จัดการ timestamp
            let createdAt = null;
            if (data.timestamp?.seconds) {
              createdAt = new Date(data.timestamp.seconds * 1000);
            } else if (data.timestamp?.toDate) {
              createdAt = data.timestamp.toDate();
            } else if (data.createdAt?.toDate) {
              createdAt = data.createdAt.toDate();
            }

            diseaseMap[disease] = (diseaseMap[disease] || 0) + 1;

            if (!latestDate || (createdAt && createdAt > latestDate)) {
              latestDate = createdAt;
            }
          });

          setDiseaseStats(diseaseMap);
          setLastActive(latestDate?.toLocaleString("th-TH"));

        } catch (predictionError) {
          console.log("ไม่พบข้อมูลการวิเคราะห์จาก prediction_results:", predictionError);
          
          // ลองดึงจาก analysisHistory แทน (fallback)
          try {
            const analysisQuery = query(
              collection(db, "analysisHistory"),
              where("uid", "==", user.uid)
            );
            const analysisSnapshot = await getDocs(analysisQuery);

            setAnalysisCount(analysisSnapshot.size);

            const diseaseMap = {};
            let latestDate = null;

            analysisSnapshot.forEach((doc) => {
              const data = doc.data();
              const disease = data.predictedClass || data.diseaseName || "ไม่ระบุโรค";
              const createdAt = data.createdAt?.toDate ? data.createdAt.toDate() : null;

              diseaseMap[disease] = (diseaseMap[disease] || 0) + 1;

              if (!latestDate || (createdAt && createdAt > latestDate)) {
                latestDate = createdAt;
              }
            });

            setDiseaseStats(diseaseMap);
            setLastActive(latestDate?.toLocaleString("th-TH"));

          } catch (analysisError) {
            console.log("ไม่พบข้อมูลการวิเคราะห์จาก analysisHistory:", analysisError);
            setAnalysisCount(0);
            setDiseaseStats({});
            setLastActive(null);
          }
        }

      } catch (error) {
        console.error("เกิดข้อผิดพลาดในการโหลดข้อมูลสถิติ:", error);
      } finally {
        setLoading(false);
      }
    });

    return () => unsubscribe();
  }, [navigate]);

  if (loading) {
    return (
      <div className="user-manual-container">
        <h2>📊 ข้อมูลสถิติการใช้งาน</h2>
        <p>กำลังโหลดข้อมูล...</p>
      </div>
    );
  }

  return (
    <div className="user-manual-container">
      <h2>📊 ข้อมูลสถิติการใช้งาน</h2>

      <div className="statistics-section">
        <div className="stat-box">
          <h3>🔐 จำนวนครั้งที่เข้าสู่ระบบ</h3>
          <div className="stat-number">{loginCount}</div>
          <p className="stat-unit">ครั้ง</p>
        </div>

        <div className="stat-box">
          <h3>🔍 จำนวนภาพที่วิเคราะห์แล้ว</h3>
          <div className="stat-number">{analysisCount}</div>
          <p className="stat-unit">ภาพ</p>
        </div>

        <div className="stat-box">
          <h3>📈 สัดส่วนโรคที่พบ</h3>
          {Object.keys(diseaseStats).length > 0 ? (
            <div className="disease-stats">
              {Object.entries(diseaseStats)
                .sort(([,a], [,b]) => b - a) // เรียงจากมากไปน้อย
                .map(([disease, count]) => {
                  const percentage = ((count / analysisCount) * 100).toFixed(1);
                  return (
                    <div key={disease} className="disease-item">
                      <div className="disease-info">
                        <span className="disease-name">{disease}</span>
                        <span className="disease-count">{count} ครั้ง ({percentage}%)</span>
                      </div>
                      <div className="disease-bar">
                        <div 
                          className="disease-fill" 
                          style={{width: `${percentage}%`}}
                        ></div>
                      </div>
                    </div>
                  );
                })}
            </div>
          ) : (
            <p className="no-data">ยังไม่มีการวิเคราะห์</p>
          )}
        </div>

        <div className="stat-box">
          <h3>⏰ วันที่ใช้งานล่าสุด</h3>
          <p className="last-active">{lastActive || "ยังไม่มีการใช้งาน"}</p>
        </div>
      </div>

      <div className="action-buttons">
        <button 
          className="nav-button"
          onClick={() => navigate('/history')}
        >
          📋 ดูประวัติการวิเคราะห์
        </button>
        <button 
          className="nav-button"
          onClick={() => navigate('/')}
        >
          🏠 กลับหน้าหลัก
        </button>
      </div>
    </div>
  );
}

export default StatisticsUser;