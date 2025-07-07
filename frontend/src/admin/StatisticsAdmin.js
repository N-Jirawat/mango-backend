import React, { useEffect, useState, useMemo } from "react";
import { getFirestore, collection, getDocs } from "firebase/firestore";
import "../css/StatisticsAdmin.css";
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";

function StatisticsAdmin() {
  const [memberCount, setMemberCount] = useState(0); // จำนวนสมาชิกที่ไม่ใช่แอดมิน
  const [imageCount, setImageCount] = useState(0);
  const [memberImageCount, setMemberImageCount] = useState(0); // จำนวนภาพของสมาชิก
  const [diseaseStats, setDiseaseStats] = useState({});
  const [lastUpdated, setLastUpdated] = useState(null);
  const [districtDiseaseMap, setDistrictDiseaseMap] = useState({});
  const [loading, setLoading] = useState(true);
  const [usersMap, setUsersMap] = useState({});

  useEffect(() => {
    const fetchStatistics = async () => {
      const db = getFirestore();
      setLoading(true);

      try {
        const usersSnapshot = await getDocs(collection(db, "users"));
        const usersMapTemp = {};
        const memberIds = new Set(); // เก็บ ID ของสมาชิกที่ไม่ใช่แอดมิน
        let memberCountTemp = 0;

        usersSnapshot.forEach(doc => {
          const user = doc.data();
          const userId = doc.id;
          
          // ถ้าไม่ใช่แอดมิน ให้เก็บ ID ไว้
          if (user.role !== 'admin') {
            memberIds.add(userId);
            memberCountTemp++;
          }
          
          usersMapTemp[userId] = {
            district: user.district || "ไม่ระบุอำเภอ",
            province: user.province || "ไม่ระบุจังหวัด",
            role: user.role || "user"
          };
        });
        
        setUsersMap(usersMapTemp);
        setMemberCount(memberCountTemp); // เฉพาะสมาชิก

        const predictionSnapshot = await getDocs(collection(db, "prediction_results"));
        setImageCount(predictionSnapshot.size); // ภาพทั้งหมด รวมแอดมิน

        const diseaseMap = {};
        const districtMap = {};
        let latestDate = null;
        let memberImageCountTemp = 0;

        predictionSnapshot.forEach((doc) => {
          const data = doc.data();
          const disease = data.diseaseName || data.predictedClass || "ไม่ระบุโรค";
          const userId = data.userId || "ไม่ทราบผู้ใช้";
          const userInfo = usersMapTemp[userId];
          const district = userInfo?.district || "ไม่ระบุอำเภอ";

          // เฉพาะสมาชิกที่ไม่ใช่แอดมิน
          if (memberIds.has(userId)) {
            memberImageCountTemp++; // นับภาพของสมาชิก

            // รวมสถิติโรคเฉพาะสมาชิก
            diseaseMap[disease] = (diseaseMap[disease] || 0) + 1;

            // รวมตามอำเภอเฉพาะสมาชิก
            if (!districtMap[district]) {
              districtMap[district] = {};
            }
            districtMap[district][disease] = (districtMap[district][disease] || 0) + 1;

            // เวลาทำนายล่าสุดของสมาชิก
            let createdAt = null;
            if (data.timestamp?.seconds) {
              createdAt = new Date(data.timestamp.seconds * 1000);
            } else if (data.timestamp?.toDate) {
              createdAt = data.timestamp.toDate();
            }

            if (!latestDate || (createdAt && createdAt > latestDate)) {
              latestDate = createdAt;
            }
          }
        });

        setMemberImageCount(memberImageCountTemp);
        setDiseaseStats(diseaseMap);
        setDistrictDiseaseMap(districtMap);
        setLastUpdated(latestDate?.toLocaleString("th-TH") || "ไม่มีข้อมูล");

      } catch (error) {
        console.error("เกิดข้อผิดพลาดในการดึงข้อมูลสถิติ:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchStatistics();
  }, []);

  // แปลงเป็นข้อมูลกราฟ
  const chartData = useMemo(() => {
    return Object.entries(districtDiseaseMap).map(([district, diseases]) => {
      const matchingUser = Object.values(usersMap).find(u => u.district === district);
      const province = matchingUser?.province || "ไม่ระบุจังหวัด";
      return {
        district,
        locationLabel: `${district}, ${province}`,
        ...diseases,
      };
    });
  }, [districtDiseaseMap, usersMap]);

  if (loading) {
    return <div className="statistics-admin-container">กำลังโหลดข้อมูลสถิติผู้ใช้ทั้งหมด...</div>;
  }

  return (
    <div className="statistics-admin-container">
      <h2>📊 สถิติของสมาชิกทั้งหมดในระบบ</h2>

      <div className="admin-statistics-section">
        <div className="admin-stat-box">
          <h3>👥 จำนวนสมาชิก</h3>
          <div className="admin-stat-number">{memberCount}</div>
          <p className="admin-stat-unit">คน</p>
          <small style={{color: '#666'}}>รวมทั้งหมด: {memberCount} คน</small>
        </div>

        <div className="admin-stat-box">
          <h3>🖼️ จำนวนภาพที่วิเคราะห์</h3>
          <div className="admin-stat-number">{memberImageCount}</div>
          <p className="admin-stat-unit">ภาพ</p>
          <small style={{color: '#666'}}>รวมทั้งหมด: {imageCount} ภาพ</small>
        </div>

        <div className="admin-stat-box">
          <h3>📈 สัดส่วนโรคที่พบ</h3>
          {Object.keys(diseaseStats).length > 0 ? (
            <div className="admin-disease-stats">
              {Object.entries(diseaseStats)
                .sort(([, a], [, b]) => b - a)
                .map(([disease, count]) => {
                  const percentage = memberImageCount > 0 ? ((count / memberImageCount) * 100).toFixed(1) : 0;
                  return (
                    <div key={disease} className="admin-disease-item">
                      <div className="admin-disease-info">
                        <span className="admin-disease-name">{disease}</span>
                        <span className="admin-disease-count">
                          {count} ครั้ง ({percentage}%)
                        </span>
                      </div>
                      <div className="admin-disease-bar">
                        <div
                          className="admin-disease-fill"
                          style={{ width: `${percentage}%` }}
                        ></div>
                      </div>
                    </div>
                  );
                })}
            </div>
          ) : (
            <p className="admin-no-data">ยังไม่มีข้อมูลการวิเคราะห์</p>
          )}
        </div>

        <div className="admin-stat-box">
          <h3>⏰ การทำนายโรคล่าสุด</h3>
          <p className="admin-last-active">{lastUpdated}</p>
        </div>
      </div>

      {chartData.length > 0 ? (
        <div className="district-chart-box">
          <h4>📍 โรคที่พบบ่อยในแต่ละอำเภอ</h4>
          <div className="chart-container">
            <div className="chart-section" style={{ width: '70%', height: '400px' }}>
              <BarChart
                width={600}
                height={400}
                data={chartData}
                margin={{ top: 20, right: 30, bottom: 50, left: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="locationLabel" angle={-15} textAnchor="end" interval={0} />
                <YAxis />
                <Tooltip />
                {Object.keys(diseaseStats).map((disease, idx) => (
                  <Bar
                    key={disease}
                    dataKey={disease}
                    stackId="a"
                    fill={colorFromIndex(idx)}
                  />
                ))}
              </BarChart>
            </div>
            <div className="chart-legend">
              <h5>สีแทนโรค</h5>
              <div className="legend-items">
                {Object.keys(diseaseStats).map((disease, idx) => (
                  <div key={disease} className="legend-item">
                    <div 
                      className="legend-color" 
                      style={{ backgroundColor: colorFromIndex(idx) }}
                    ></div>
                    <span className="legend-text">{disease}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div style={{ margin: '20px 0', padding: '20px', background: '#fff3cd', border: '1px solid #ffeaa7' }}>
          <p>ไม่มีข้อมูลกราฟเพื่อแสดงผل</p>
          <p>chartData: {JSON.stringify(chartData, null, 2)}</p>
        </div>
      )}
    </div>
  );
}

// สุ่มสีอ่อน ๆ ให้แต่ละโรค
function colorFromIndex(index) {
  const colors = [
    "#4CAF50", "#2196F3", "#FFC107", "#FF5722", "#9C27B0", "#00BCD4", "#8BC34A", "#E91E63"
  ];
  return colors[index % colors.length];
}

export default StatisticsAdmin;