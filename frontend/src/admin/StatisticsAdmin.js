import React, { useEffect, useState, useMemo, useCallback } from "react";
import { getFirestore, collection, getDocs } from "firebase/firestore";
import "../css/StatisticsAdmin.css";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
  PieChart, Pie, Cell, LineChart, Line, ResponsiveContainer
} from "recharts";



function StatisticsAdmin() {
  const [diseaseStats, setDiseaseStats] = useState({});
  const [districtDiseaseMap, setDistrictDiseaseMap] = useState({});
  const [loading, setLoading] = useState(true);
  const [usersMap, setUsersMap] = useState({});
  const [allPredictions, setAllPredictions] = useState([]);
  const [dateRange, setDateRange] = useState({ startDate: '', endDate: '' });
  const [timelineData, setTimelineData] = useState([]);

  // ✅ ฟังก์ชันแปลงเดือน (วางไว้นอก useCallback อื่น)
  const formatMonthLabel = useCallback((monthKey) => {
    const [year, month] = monthKey.split('-');
    const monthNames = [
      'ม.ค.', 'ก.พ.', 'มี.ค.', 'เม.ย.', 'พ.ค.', 'มิ.ย.',
      'ก.ค.', 'ส.ค.', 'ก.ย.', 'ต.ค.', 'พ.ย.', 'ธ.ค.'
    ];
    return `${monthNames[parseInt(month) - 1]} ${parseInt(year) + 543}`;
  }, []);

  // ✅ ย้ายฟังก์ชันออกนอก useEffect และอยู่นอกกันกับ formatMonthLabel
  const processStatistics = useCallback((predictions, usersMapTemp) => {
    const diseaseMap = {};
    const districtMap = {};
    const monthlyData = {};

    predictions.forEach((prediction) => {
      const { disease, userId, timestamp } = prediction;
      const userInfo = usersMapTemp[userId];
      const district = userInfo?.district || "ไม่ระบุอำเภอ";

      diseaseMap[disease] = (diseaseMap[disease] || 0) + 1;

      if (!districtMap[district]) {
        districtMap[district] = {};
      }
      districtMap[district][disease] = (districtMap[district][disease] || 0) + 1;

      if (timestamp && timestamp instanceof Date && !isNaN(timestamp.getTime())) {
        const monthKey = `${timestamp.getFullYear()}-${String(timestamp.getMonth() + 1).padStart(2, '0')}`;
        if (!monthlyData[monthKey]) {
          monthlyData[monthKey] = { month: monthKey, count: 0, diseases: {} };
        }
        monthlyData[monthKey].count++;
        monthlyData[monthKey].diseases[disease] = (monthlyData[monthKey].diseases[disease] || 0) + 1;
      }

    });

    setDiseaseStats(diseaseMap);
    setDistrictDiseaseMap(districtMap);

    const timelineArray = Object.values(monthlyData)
      .sort((a, b) => a.month.localeCompare(b.month))
      .map(item => ({
        month: formatMonthLabel(item.month),
        count: item.count,
        ...item.diseases
      }));

    setTimelineData(timelineArray);
  }, [formatMonthLabel]);

  useEffect(() => {
    const fetchStatistics = async () => {
      const db = getFirestore();
      setLoading(true);

      try {
        const usersSnapshot = await getDocs(collection(db, "users"));
        const usersMapTemp = {};

        usersSnapshot.forEach(doc => {
          const user = doc.data();
          const userId = doc.id;

          usersMapTemp[userId] = {
            district: user.district || "ไม่ระบุอำเภอ",
            province: user.province || "ไม่ระบุจังหวัด",
            role: user.role || "user",
            name: user.displayName || user.name || "ไม่ระบุชื่อ"
          };
        });

        setUsersMap(usersMapTemp);

        const predictionSnapshot = await getDocs(collection(db, "prediction_results"));
        const predictionsData = [];

        predictionSnapshot.forEach((doc) => {
          const data = doc.data();
          let createdAt = null;

          if (data.timestamp?.toDate) {
            createdAt = data.timestamp.toDate();
          } else if (data.timestamp?.seconds) {
            createdAt = new Date(data.timestamp.seconds * 1000);
          } else if (data.createdAt) {
            createdAt = new Date(data.createdAt);
          }

          if (createdAt instanceof Date && !isNaN(createdAt)) {
            // ปลอดภัย
          } else {
            createdAt = null; // หลีกเลี่ยงการส่งค่าไม่ถูกต้องไป processStatistics
          }

          predictionsData.push({
            id: doc.id,
            disease: data.diseaseName || data.predictedClass || "ไม่ระบุโรค",
            userId: data.userId || "ไม่ทราบผู้ใช้",
            timestamp: createdAt,
            confidence: data.confidence || 0,
            ...data
          });
        });

        setAllPredictions(predictionsData);
        processStatistics(predictionsData, usersMapTemp);

      } catch (error) {
        console.error("เกิดข้อผิดพลาดในการดึงข้อมูลสถิติ:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchStatistics();
  }, [processStatistics]);

  const handleDateFilter = () => {
    if (!dateRange.startDate && !dateRange.endDate) {
      processStatistics(allPredictions, usersMap);
      return;
    }

    const filtered = allPredictions.filter(prediction => {
      if (!prediction.timestamp) return false;

      const predictionDate = prediction.timestamp;
      const start = dateRange.startDate ? new Date(dateRange.startDate) : null;
      const end = dateRange.endDate ? new Date(dateRange.endDate + 'T23:59:59') : null;

      if (start && predictionDate < start) return false;
      if (end && predictionDate > end) return false;

      return true;
    });

    processStatistics(filtered, usersMap);
  };

  const resetFilter = () => {
    setDateRange({ startDate: '', endDate: '' });
    processStatistics(allPredictions, usersMap);
  };

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

  const pieData = useMemo(() => {
    const totalCount = Object.values(diseaseStats).reduce((sum, val) => sum + val, 0);
    return Object.entries(diseaseStats).map(([disease, count]) => ({
      name: disease,
      value: count,
      percentage: totalCount > 0 ? ((count / totalCount) * 100).toFixed(1) : 0
    }));
  }, [diseaseStats]);

  const colors = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"
  ];

  const getColorFromIndex = (index) => {
    return colors[index % colors.length];
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>กำลังโหลดข้อมูลสถิติ...</p>
      </div>
    );
  }

  return (
    <div className="statistics-admin-container">
      <div className="header-section">
        <h1>Dashboard การวิเคราะห์โรคพืช</h1>
      </div>

      {/* Date Filter Section */}
      <div className="filter-section">
        <div className="filter-container">
          <h3>🔍 กรองข้อมูลตามช่วงเวลา</h3>
          <div className="date-inputs">
            <div className="input-group">
              <label>วันที่เริ่มต้น:</label>
              <input
                type="date"
                value={dateRange.startDate}
                onChange={(e) => setDateRange(prev => ({ ...prev, startDate: e.target.value }))}
                className="date-input"
              />
            </div>
            <div className="input-group">
              <label>วันที่สิ้นสุด:</label>
              <input
                type="date"
                value={dateRange.endDate}
                onChange={(e) => setDateRange(prev => ({ ...prev, endDate: e.target.value }))}
                className="date-input"
              />
            </div>
            <div className="button-group">
              <button onClick={handleDateFilter} className="filter-btn">
                🔍 กรองข้อมูล
              </button>
              <button onClick={resetFilter} className="reset-btn">
                🔄 รีเซ็ต
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="summary-cards">
        <div className="summary-card">
          <div className="card-icon">🦠</div>
          <div className="card-content">
            <h3>ประเภทโรค</h3>
            <div className="card-number">{Object.keys(diseaseStats).length}</div>
            <p>ชนิด</p>
          </div>
        </div>

        <div className="summary-card">
          <div className="card-icon">📍</div>
          <div className="card-content">
            <h3>พื้นที่</h3>
            <div className="card-number">{Object.keys(districtDiseaseMap).length}</div>
            <p>อำเภอ</p>
          </div>
        </div>
      </div>

      {/* Charts Section */}
      <div className="charts-section">
        {/* Disease Statistics */}
        <div className="chart-container">
          <div className="chart-header">
            <h3>📊 สัดส่วนโรคที่พบ</h3>
          </div>
          <div className="chart-content">
            <div className="chart-left">
              {Object.keys(diseaseStats).length > 0 ? (
                <div className="disease-list">
                  {Object.entries(diseaseStats)
                    .sort(([, a], [, b]) => b - a)
                    .map(([disease, count], index) => {
                      const totalCount = Object.values(diseaseStats).reduce((sum, val) => sum + val, 0);
                      const percentage = totalCount > 0 ? ((count / totalCount) * 100).toFixed(1) : 0;
                      return (
                        <div key={disease} className="disease-item">
                          <div className="disease-info">
                            <div className="disease-label">
                              <div
                                className="disease-color"
                                style={{ backgroundColor: getColorFromIndex(index) }}
                              ></div>
                              <span className="disease-name">{disease}</span>
                            </div>
                            <div className="disease-stats">
                              <span className="disease-count">{count} ครั้ง</span>
                              <span className="disease-percentage">({percentage}%)</span>
                            </div>
                          </div>
                          <div className="disease-bar">
                            <div
                              className="disease-fill"
                              style={{
                                width: `${percentage}%`,
                                backgroundColor: getColorFromIndex(index)
                              }}
                            ></div>
                          </div>
                        </div>
                      );
                    })}
                </div>
              ) : (
                <div className="no-data">
                  <div className="no-data-icon">📊</div>
                  <p>ยังไม่มีข้อมูลการวิเคราะห์</p>
                </div>
              )}
            </div>

            {pieData.length > 0 && (
              <div className="chart-right">
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      dataKey="value"
                      label={({ name, percentage }) => `${percentage}%`}
                    >
                      {pieData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={getColorFromIndex(index)} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value, name) => [`${value} ครั้ง`, name]} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        </div>

        {/* Timeline Chart */}
        {timelineData.length > 0 && (
          <div className="chart-container">
            <div className="chart-header">
              <h3>📈 แนวโน้มการวิเคราะห์รายเดือน</h3>
            </div>
            <div className="chart-content">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={timelineData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="count"
                    stroke="#45B7D1"
                    strokeWidth={3}
                    dot={{ fill: '#45B7D1', strokeWidth: 2, r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* District Chart */}
        {chartData.length > 0 && (
          <div className="chart-container full-width">
            <div className="chart-header">
              <h3>📍 การกระจายโรคตามพื้นที่</h3>
            </div>
            <div className="chart-content">
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={chartData} margin={{ top: 20, right: 30, bottom: 60, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="locationLabel"
                    angle={-45}
                    textAnchor="end"
                    interval={0}
                    height={100}
                  />
                  <YAxis />
                  <Tooltip />
                  {Object.keys(diseaseStats).map((disease, idx) => (
                    <Bar
                      key={disease}
                      dataKey={disease}
                      stackId="a"
                      fill={getColorFromIndex(idx)}
                    />
                  ))}
                </BarChart>
              </ResponsiveContainer>

              <div className="chart-legend">
                <h4>สีแทนโรค</h4>
                <div className="legend-grid">
                  {Object.keys(diseaseStats).map((disease, idx) => (
                    <div key={disease} className="legend-item">
                      <div
                        className="legend-color"
                        style={{ backgroundColor: getColorFromIndex(idx) }}
                      ></div>
                      <span className="legend-text">{disease}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {!Object.keys(diseaseStats).length && (
        <div className="empty-state">
          <div className="empty-icon">📊</div>
          <h3>ไม่มีข้อมูลสถิติ</h3>
          <p>ยังไม่มีการวิเคราะห์โรคในระบบ หรือไม่มีข้อมูลในช่วงเวลาที่เลือก</p>
        </div>
      )}
    </div>
  );
}

export default StatisticsAdmin;