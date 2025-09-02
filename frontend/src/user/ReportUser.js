import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getAuth, onAuthStateChanged } from "firebase/auth";
import { getFirestore, collection, query, where, getDocs, doc, getDoc } from "firebase/firestore";
import "../css/ReportUser.css";

function ReportUser() {
  const navigate = useNavigate();
  const [userData, setUserData] = useState({});
  const [analysisCount, setAnalysisCount] = useState(0);
  const [diseaseStats, setDiseaseStats] = useState({});
  const [mostFrequentDisease, setMostFrequentDisease] = useState("-");
  const [lastActive, setLastActive] = useState(null);
  const [loading, setLoading] = useState(true);

  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  useEffect(() => {
    const fetchData = async () => {
      const db = getFirestore();
      const auth = getAuth();

      onAuthStateChanged(auth, async (user) => {
        if (!user) {
          alert("กรุณาเข้าสู่ระบบก่อนใช้งาน");
          navigate("/login");
          return;
        }

        // ข้อมูลพื้นฐานจาก auth
        const baseUserData = {
          uid: user.uid,
          fullName: user.displayName || "ไม่ระบุชื่อ",
          email: user.email || "ไม่ระบุอีเมล",
        };

        // ดึงข้อมูลเพิ่มเติมจาก Firestore
        const userDocRef = doc(db, "users", user.uid);
        const userDocSnap = await getDoc(userDocRef);

        if (userDocSnap.exists()) {
          const firestoreUserData = userDocSnap.data();
          // รวมข้อมูลจาก auth + firestore
          setUserData({ ...baseUserData, ...firestoreUserData });
        } else {
          setUserData(baseUserData);
        }

        try {

          // Prediction Results
          // Prediction Results
          const predictionQuery = query(
            collection(db, "AnalysisHistory"),
            where("userId", "==", user.uid)
          );
          const predictionSnapshot = await getDocs(predictionQuery);

          const filteredPredictionDocs = predictionSnapshot.docs.filter((doc) => {
            const data = doc.data();
            const ts = data.UpdateAt?.toDate?.();
            if (!ts) return false;
            return (
              (!startDate || ts >= new Date(startDate)) &&
              (!endDate || ts <= new Date(endDate + "T23:59:59"))
            );
          });

          if (filteredPredictionDocs.length === 0)
            throw new Error("ไม่มีข้อมูลใน AnalysisHistory");

          const diseaseMap = {};
          let latestDate = null;

          filteredPredictionDocs.forEach((doc) => {
            const data = doc.data();
            const disease = data.diseaseName || "ไม่ระบุโรค";
            const createdAt = data.UpdateAt?.toDate?.();

            diseaseMap[disease] = (diseaseMap[disease] || 0) + 1;

            if (!latestDate || (createdAt && createdAt > latestDate)) {
              latestDate = createdAt;
            }
          });

          setAnalysisCount(filteredPredictionDocs.length);
          setDiseaseStats(diseaseMap);
          setLastActive(latestDate?.toLocaleString("th-TH"));

          const mostFrequent = Object.entries(diseaseMap).sort(
            ([, a], [, b]) => b - a
          )[0];
          if (mostFrequent) {
            setMostFrequentDisease(
              `${mostFrequent[0]} (${mostFrequent[1]} ครั้ง)`
            );
          }

        } catch (error) {
          try {
            const analysisQuery = query(collection(db, "AnalysisHistory"), where("uid", "==", user.uid));
            const analysisSnapshot = await getDocs(analysisQuery);

            const filteredDocs = analysisSnapshot.docs.filter((doc) => {
              const data = doc.data();
              const createdAt = data.UpdateAt?.toDate?.();
              if (!createdAt) return false;
              return (
                (!startDate || createdAt >= new Date(startDate)) &&
                (!endDate || createdAt <= new Date(endDate + "T23:59:59"))
              );
            });

            const diseaseMap = {};
            let latestDate = null;

            filteredDocs.forEach((doc) => {
              const data = doc.data();
              const disease = data.predictedClass || data.diseaseName || "ไม่ระบุโรค";
              const createdAt = data.createdAt?.toDate?.();

              diseaseMap[disease] = (diseaseMap[disease] || 0) + 1;

              if (!latestDate || (createdAt && createdAt > latestDate)) {
                latestDate = createdAt;
              }
            });

            setAnalysisCount(filteredDocs.length);
            setDiseaseStats(diseaseMap);
            setLastActive(latestDate?.toLocaleString("th-TH"));

            const mostFrequent = Object.entries(diseaseMap).sort(([, a], [, b]) => b - a)[0];
            if (mostFrequent) {
              setMostFrequentDisease(`${mostFrequent[0]} (${mostFrequent[1]} ครั้ง)`);
            }
          } catch (err) {
            setAnalysisCount(0);
            setDiseaseStats({});
            setLastActive(null);
          }
        } finally {
          setLoading(false);
        }
      });
    };

    fetchData();
  }, [startDate, endDate, navigate]);

  if (loading) {
    return (
      <div className="user-manual-container">
        <h2>📊 รายงานการใช้งาน</h2>
        <p>กำลังโหลดข้อมูล...</p>
      </div>
    );
  }

  return (
    <div className="report-user-container">
      <h2>📊 รายงานข้อมูลการใช้งาน</h2>

      <div className="user-info">
        <p><strong>👤 ชื่อเต็ม:</strong> {userData.fullName || "-"}</p>
        <p><strong>📧 อีเมล:</strong> {userData.email || "-"}</p>
        <p><strong>📞 เบอร์โทร:</strong> {userData.tel || "-"}</p>
        <p>
          <strong>📍 ที่อยู่:</strong> {userData.address || "-"} &nbsp;|&nbsp;
          <strong>หมู่บ้าน:</strong> {userData.village || "-"} &nbsp;|&nbsp;
          <strong>ตำบล:</strong> {userData.subdistrict || "-"} &nbsp;|&nbsp;
          <strong>อำเภอ:</strong> {userData.district || "-"} &nbsp;|&nbsp;
          <strong>จังหวัด:</strong> {userData.province || "-"}
        </p>
        <p><strong>🛡️ บทบาท:</strong> {userData.role || "-"}</p>
      </div>

      {/* 🔍 Date Filters */}
      <div className="date-filters">
        <label>
          📅 วันที่เริ่มต้น:
          <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
        </label>
        <label>
          📅 วันที่สิ้นสุด:
          <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
        </label>
      </div>

      <table className="report-table">
        <thead>
          <tr>
            <th>รายการ</th>
            <th>ข้อมูล</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>🔍 จำนวนภาพที่วิเคราะห์แล้ว</td>
            <td>{analysisCount} ภาพ</td>
          </tr>
          <tr>
            <td>📌 โรคที่พบมากที่สุด</td>
            <td>{mostFrequentDisease}</td>
          </tr>
          <tr>
            <td>⏰ วันที่ใช้งานล่าสุด</td>
            <td>{lastActive || "ยังไม่มีการใช้งาน"}</td>
          </tr>
        </tbody>
      </table>

      <h3>📈 รายการโรคที่เคยวิเคราะห์</h3>
      {Object.keys(diseaseStats).length > 0 ? (
        <table className="report-disease-table">
          <thead>
            <tr>
              <th>โรค</th>
              <th>จำนวน (ครั้ง)</th>
              <th>เปอร์เซ็นต์</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(diseaseStats)
              .sort(([, a], [, b]) => b - a)
              .map(([disease, count]) => {
                const percentage = ((count / analysisCount) * 100).toFixed(1);
                return (
                  <tr key={disease}>
                    <td>{disease}</td>
                    <td>{count}</td>
                    <td>{percentage}%</td>
                  </tr>
                );
              })}
          </tbody>
        </table>
      ) : (
        <p>ยังไม่มีข้อมูลการวิเคราะห์</p>
      )}

      <div className="action-buttons">
        <button className="nav-button" onClick={() => navigate("/history")}>
          📋 ดูประวัติการวิเคราะห์
        </button>
        <button className="nav-button green" onClick={() => navigate("/")}>
          🏠 กลับหน้าหลัก
        </button>
      </div>
    </div>
  );
}

export default ReportUser;
