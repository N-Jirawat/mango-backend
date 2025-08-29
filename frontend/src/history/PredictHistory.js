import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getAuth, onAuthStateChanged } from "firebase/auth";
import { collection, query, where, getDocs, orderBy } from "firebase/firestore";
import { db } from "../firebaseConfig";
import "../css/PredictHistory.css";

function History() {
  const [allPredictions, setAllPredictions] = useState([]);
  const [filteredPredictions, setFilteredPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [user, setUser] = useState(null);
  const navigate = useNavigate();

  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 4;

  const [searchDisease, setSearchDisease] = useState("");
  const [searchDateFrom, setSearchDateFrom] = useState("");
  const [searchDateTo, setSearchDateTo] = useState("");

  // ฟังก์ชันสำหรับจัดรูปแบบวันที่เป็น YYYY-MM-DD
  const formatDateForInput = (date) => {
    if (!date) return "";
    const d = new Date(date);
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
  };

  // ฟังก์ชันสำหรับจัดการการเปลี่ยนแปลงวันที่
  const handleDateFromChange = (e) => {
    const value = e.target.value;
    console.log("Date From changed:", value);
    setSearchDateFrom(value);
    
    // ถ้าวันที่เริ่มต้นมากกว่าวันที่สิ้นสุด ให้ clear วันที่สิ้นสุด
    if (value && searchDateTo && value > searchDateTo) {
      setSearchDateTo("");
    }
  };

  const handleDateToChange = (e) => {
    const value = e.target.value;
    console.log("Date To changed:", value);
    
    // ตรวจสอบว่าวันที่สิ้นสุดไม่น้อยกว่าวันที่เริ่มต้น
    if (value && searchDateFrom && value < searchDateFrom) {
      alert("วันที่สิ้นสุดต้องไม่น้อยกว่าวันที่เริ่มต้น");
      return;
    }
    
    setSearchDateTo(value);
  };

  useEffect(() => {
    const auth = getAuth();
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      if (currentUser) {
        fetchHistory(currentUser);
      } else {
        setLoading(false);
        setError("กรุณาเข้าสู่ระบบเพื่อดูประวัติ");
      }
    });
    return () => unsubscribe();
  }, []);

  useEffect(() => {
    let filtered = [...allPredictions];
    
    if (searchDisease.trim()) {
      filtered = filtered.filter(prediction => {
        const diseaseNames = [
          prediction.DiseaseName,
          prediction.diseaseName, 
          prediction.disease_name,
          prediction.predictedDisease,
          prediction.predicted_disease,
          prediction.result,
          prediction.prediction
        ];
        
        const validDiseaseName = diseaseNames.find(name => 
          name && typeof name === 'string' && name.trim().length > 0
        );
        
        if (!validDiseaseName) {
          console.log('No valid disease name found for prediction:', prediction.id);
          return false;
        }
        
        return validDiseaseName.toLowerCase().includes(searchDisease.toLowerCase());
      });
    }
    
    // ค้นหาตามช่วงวันที่ - ปรับปรุงการเปรียบเทียบ
    if (searchDateFrom || searchDateTo) {
      filtered = filtered.filter(prediction => {
        if (!prediction.UpdateAt?.seconds) return false;
        
        const predictionDate = new Date(prediction.UpdateAt.seconds * 1000);
        const predictionDateStr = formatDateForInput(predictionDate);
        
        // ถ้ามีทั้ง วันเริ่มต้น และ วันสิ้นสุด
        if (searchDateFrom && searchDateTo) {
          return predictionDateStr >= searchDateFrom && predictionDateStr <= searchDateTo;
        }
        // ถ้ามีแค่วันเริ่มต้น
        else if (searchDateFrom && !searchDateTo) {
          return predictionDateStr >= searchDateFrom;
        }
        // ถ้ามีแค่วันสิ้นสุด
        else if (!searchDateFrom && searchDateTo) {
          return predictionDateStr <= searchDateTo;
        }
        
        return true;
      });
    }
    
    setFilteredPredictions(filtered);
    setCurrentPage(1);
  }, [allPredictions, searchDisease, searchDateFrom, searchDateTo]);

  const fetchHistory = async (currentUser) => {
    try {
      const q = query(
        collection(db, "AnalysisHistory"),
        where("userId", "==", currentUser.uid),
        orderBy("UpdateAt", "desc")
      );
      const querySnapshot = await getDocs(q);
      const historyData = [];
      querySnapshot.forEach((doc) => {
        const data = { id: doc.id, ...doc.data() };
        console.log('Document data:', data);
        historyData.push(data);
      });
      setAllPredictions(historyData);
    } catch (error) {
      console.error("Error fetching with orderBy:", error);
      try {
        const fallbackQuery = query(
          collection(db, "AnalysisHistory"),
          where("userId", "==", currentUser.uid)
        );
        const snapshot = await getDocs(fallbackQuery);
        const fallbackData = [];
        snapshot.forEach((doc) => {
          const data = { id: doc.id, ...doc.data() };
          console.log('Fallback document data:', data);
          fallbackData.push(data);
        });
        fallbackData.sort((a, b) => (b.UpdateAt?.seconds || 0) - (a.UpdateAt?.seconds || 0));
        setAllPredictions(fallbackData);
        setError(null);
      } catch (simpleError) {
        console.error("Fallback error:", simpleError);
        setError("เกิดข้อผิดพลาดในการโหลดประวัติ");
      }
    } finally {
      setLoading(false);
    }
  };

  const getDiseaseName = (prediction) => {
    const diseaseNames = [
      prediction.DiseaseName,
      prediction.diseaseName, 
      prediction.disease_name,
      prediction.predictedDisease,
      prediction.predicted_disease,
      prediction.result,
      prediction.prediction
    ];
    
    return diseaseNames.find(name => 
      name && typeof name === 'string' && name.trim().length > 0
    ) || "ไม่ระบุชื่อโรค";
  };

  const handleGoHome = () => navigate('/');

  const getCurrentPageData = () => {
    const startIndex = (currentPage - 1) * itemsPerPage;
    return filteredPredictions.slice(startIndex, startIndex + itemsPerPage);
  };

  const totalPages = Math.ceil(filteredPredictions.length / itemsPerPage);

  const handlePageChange = (pageNumber) => setCurrentPage(pageNumber);

  const clearSearch = () => {
    setSearchDisease("");
    setSearchDateFrom("");
    setSearchDateTo("");
    console.log("Search cleared");
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>กำลังโหลดข้อมูล...</p>
      </div>
    );
  }
  if (error) return <p style={{ color: 'red' }}>{error}</p>;
  if (!user) return <p>กรุณาเข้าสู่ระบบเพื่อดูประวัติ</p>;

  const currentPageData = getCurrentPageData();

  return (
    <div className="history-container">
      <div className="analy-header">
        <button onClick={handleGoHome} className="back-button">หน้าหลัก</button>
      </div>

      <h2>ประวัติการวิเคราะห์โรค</h2>

      <div className="search-container">
        <h3>ค้นหาข้อมูล</h3>

        <div className="search-field">
          <label htmlFor="disease-search">ค้นหาตามชื่อโรค:</label>
          <input
            id="disease-search"
            type="text"
            value={searchDisease}
            onChange={(e) => setSearchDisease(e.target.value)}
            placeholder="พิมพ์ชื่อโรคที่ต้องการค้นหา..."
          />
        </div>

        <div className="search-field">
          <label>ค้นหาตามช่วงวันที่:</label>
          <div className="date-range-container">
            <div className="date-field">
              <label htmlFor="date-from" style={{fontSize: '12px', color: '#666'}}>จาก:</label>
              <input
                id="date-from"
                type="date"
                value={searchDateFrom}
                onChange={handleDateFromChange}
                max={searchDateTo || undefined}
                style={{
                  WebkitAppearance: 'none',
                  MozAppearance: 'textfield'
                }}
              />
            </div>
            <span style={{alignSelf: 'center', margin: '20px 0 0 0'}}>ถึง</span>
            <div className="date-field">
              <label htmlFor="date-to" style={{fontSize: '12px', color: '#666'}}>ถึง:</label>
              <input
                id="date-to"
                type="date"
                value={searchDateTo}
                onChange={handleDateToChange}
                min={searchDateFrom || undefined}
                max={formatDateForInput(new Date())}
                style={{
                  WebkitAppearance: 'none',
                  MozAppearance: 'textfield'
                }}
              />
            </div>
          </div>
          <div style={{fontSize: '12px', color: '#888', marginTop: '5px'}}>
            💡 สามารถเลือกเฉพาะวันเริ่มต้น หรือ วันสิ้นสุด หรือทั้งคู่ได้
          </div>
        </div>

        <div className="search-buttons">
          <button onClick={clearSearch} className="clear-search-btn">🔄 ล้างการค้นหา</button>
        </div>

        {/* Debug info - แสดงเฉพาะใน development */}
        {process.env.NODE_ENV === 'development' && (
          <div style={{fontSize: '12px', color: '#666', marginTop: '10px', padding: '10px', backgroundColor: '#f0f0f0', borderRadius: '4px'}}>
            Debug: searchDateFrom = "{searchDateFrom}", searchDateTo = "{searchDateTo}"
          </div>
        )}
      </div>

      <div className="search-results-info">
        <p>📊 พบข้อมูล {filteredPredictions.length} รายการ จากทั้งหมด {allPredictions.length} รายการ</p>
        {searchDisease && (
          <p>🔍 ค้นหาโรค: "{searchDisease}"</p>
        )}
        {(searchDateFrom || searchDateTo) && (
          <p>📅 ช่วงวันที่: {
            searchDateFrom && searchDateTo 
              ? `${searchDateFrom} ถึง ${searchDateTo}`
              : searchDateFrom 
                ? `ตั้งแต่ ${searchDateFrom} เป็นต้นไป`
                : `จนถึง ${searchDateTo}`
          }</p>
        )}
      </div>

      {filteredPredictions.length === 0 ? (
        <div className="no-data-message">
          <p>{allPredictions.length === 0
            ? "📋 คุณยังไม่มีประวัติการวิเคราะห์"
            : "🔍 ไม่พบข้อมูลที่ตรงกับการค้นหา"}</p>
          {searchDisease && allPredictions.length > 0 && (
            <div>
              <p>ลองตรวจสอบการสะกดคำ หรือใช้คำค้นหาที่สั้นกว่า</p>
            </div>
          )}
          {(searchDateFrom || searchDateTo) && allPredictions.length > 0 && (
            <p>ลองปรับช่วงวันที่ หรือล้างการค้นหาเพื่อดูข้อมูลทั้งหมด</p>
          )}
          {(searchDisease || searchDateFrom || searchDateTo) && allPredictions.length > 0 && (
            <div>
              <p>ชื่อโรคที่มีในระบบ:</p>
              <ul style={{textAlign: 'left', maxWidth: '300px', margin: '0 auto'}}>
                {Array.from(new Set(
                  allPredictions
                    .map(p => getDiseaseName(p))
                    .filter(name => name !== "ไม่ระบุชื่อโรค")
                )).map((name, index) => (
                  <li key={index}>{name}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      ) : (
        <>
          <div className="history-list">
            {currentPageData.map((prediction) => (
              <div className="history-item" key={prediction.id}>
                <h3>{getDiseaseName(prediction)}</h3>
                {(prediction.imageUrl || prediction.imageBase64) && (
                  <div className="image-container">
                    <img
                      src={prediction.imageUrl || prediction.imageBase64}
                      alt="ภาพที่ใช้วิเคราะห์"
                      className="image-thumbnail"
                      onError={(e) => { e.target.style.display = 'none'; }}
                    />
                  </div>
                )}
                <p><strong>ความมั่นใจ:</strong> {prediction.confidence !== undefined ? prediction.confidence.toFixed(4) + "%" : "ไม่ระบุ"}</p>
                <p><strong>วันที่:</strong> {
                  prediction.UpdateAt?.seconds
                    ? new Date(prediction.UpdateAt.seconds * 1000).toLocaleString("th-TH")
                    : "ไม่ระบุวันที่"
                }</p>
                <button
                  className="view-details-btn"
                  onClick={() => navigate("/historydetail", {
                    state: { docId: prediction.id, ...prediction }
                  })}
                >
                  ดูรายละเอียด
                </button>
              </div>
            ))}
          </div>

          <div className="pagination-container">
            <button
              className="pagination-nav"
              onClick={() => handlePageChange(currentPage - 1)}
              disabled={currentPage === 1}
            >
              ◀️
            </button>

            {Array.from({ length: totalPages }, (_, i) => (
              <button
                key={i + 1}
                className={`pagination-btn ${currentPage === i + 1 ? "active" : ""}`}
                onClick={() => handlePageChange(i + 1)}
              >
                {i + 1}
              </button>
            ))}

            <button
              className="pagination-nav"
              onClick={() => handlePageChange(currentPage + 1)}
              disabled={currentPage === totalPages}
            >
              ▶️
            </button>
          </div>
        </>
      )}
    </div>
  );
}

export default History;