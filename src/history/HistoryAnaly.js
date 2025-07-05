import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getAuth, onAuthStateChanged } from "firebase/auth";
import { collection, query, where, getDocs, orderBy } from "firebase/firestore";
import { db } from "../firebaseConfig";
import "../css/historyanaly.css";

function History() {
  const [allPredictions, setAllPredictions] = useState([]); // เก็บข้อมูลทั้งหมด
  const [filteredPredictions, setFilteredPredictions] = useState([]); // เก็บข้อมูลที่กรองแล้ว
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [user, setUser] = useState(null);
  const navigate = useNavigate();

  // States สำหรับ pagination
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 4;

  // States สำหรับการค้นหา
  const [searchDisease, setSearchDisease] = useState("");
  const [searchDate, setSearchDate] = useState("");

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

  // Auto-filter เมื่อ search criteria เปลี่ยน
  useEffect(() => {
    let filtered = [...allPredictions];

    // กรองตามชื่อโรค
    if (searchDisease.trim()) {
      filtered = filtered.filter(prediction =>
        (prediction.diseaseName || "").toLowerCase().includes(searchDisease.toLowerCase())
      );
    }

    // กรองตามวันที่ (ตรงกับวันที่ที่เลือก)
    if (searchDate) {
      filtered = filtered.filter(prediction => {
        if (!prediction.timestamp?.seconds) return false;
        
        const predictionDate = new Date(prediction.timestamp.seconds * 1000);
        const predictionDateStr = predictionDate.toISOString().split('T')[0];
        
        return predictionDateStr === searchDate;
      });
    }

    setFilteredPredictions(filtered);
    setCurrentPage(1); // รีเซ็ตไปหน้าแรกเมื่อมีการกรอง
  }, [allPredictions, searchDisease, searchDate]);

  const fetchHistory = async (currentUser) => {
    try {
      console.log("Fetching history for user:", currentUser.uid);
      
      const q = query(
        collection(db, "prediction_results"),
        where("userId", "==", currentUser.uid),
        orderBy("timestamp", "desc")
      );

      const querySnapshot = await getDocs(q);
      console.log("Query snapshot size:", querySnapshot.size);
      
      const historyData = [];
      querySnapshot.forEach((doc) => {
        console.log("Document data:", doc.data());
        historyData.push({ id: doc.id, ...doc.data() });
      });

      setAllPredictions(historyData);
      
      if (historyData.length === 0) {
        console.log("No predictions found for user:", currentUser.uid);
      }
      
    } catch (error) {
      console.error("เกิดข้อผิดพลาดในการโหลดประวัติ:", error);
      setError("เกิดข้อผิดพลาดในการโหลดประวัติ: " + error.message);
      
      try {
        console.log("Trying without orderBy...");
        const simpleQuery = query(
          collection(db, "prediction_results"),
          where("userId", "==", currentUser.uid)
        );
        
        const simpleSnapshot = await getDocs(simpleQuery);
        const simpleData = [];
        simpleSnapshot.forEach((doc) => {
          simpleData.push({ id: doc.id, ...doc.data() });
        });
        
        simpleData.sort((a, b) => {
          const aTime = a.timestamp?.seconds || 0;
          const bTime = b.timestamp?.seconds || 0;
          return bTime - aTime;
        });
        
        setAllPredictions(simpleData);
        setError(null);
        console.log("Successfully loaded with simple query:", simpleData.length);
        
      } catch (simpleError) {
        console.error("Simple query also failed:", simpleError);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleGoHome = () => {
    navigate('/');
  };

  // ฟังก์ชันสำหรับ pagination
  const getCurrentPageData = () => {
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    return filteredPredictions.slice(startIndex, endIndex);
  };

  const getTotalPages = () => {
    return Math.ceil(filteredPredictions.length / itemsPerPage);
  };

  const handlePageChange = (pageNumber) => {
    setCurrentPage(pageNumber);
  };

  // ฟังก์ชันล้างการค้นหา
  const clearSearch = () => {
    setSearchDisease("");
    setSearchDate("");
    // การล้าง search จะทำให้ useEffect ทำงานและแสดงข้อมูลทั้งหมด
  };

  if (loading) return <p>กำลังโหลดข้อมูล...</p>;
  
  if (error) return <p style={{color: 'red'}}>{error}</p>;
  
  if (!user) return <p>กรุณาเข้าสู่ระบบเพื่อดูประวัติ</p>;

  const currentPageData = getCurrentPageData();
  const totalPages = getTotalPages();

  return (
    <div className="history-container">
      <div className="analy-header">
        <button onClick={handleGoHome} className="back-button">
          ⬅️ หน้าหลัก
        </button>
      </div>

      <h2>ประวัติการวิเคราะห์โรค</h2>
      
      {/* ส่วนการค้นหา */}
      <div className="search-container">
        <h3>ค้นหาข้อมูล</h3>
        
        {/* ค้นหาตามชื่อโรค */}
        <div className="search-field">
          <label>ค้นหาตามชื่อโรค:</label>
          <input
            type="text"
            value={searchDisease}
            onChange={(e) => setSearchDisease(e.target.value)}
            placeholder="พิมพ์ชื่อโรคที่ต้องการค้นหา..."
          />
        </div>

        {/* ค้นหาตามวันที่ */}
        <div className="search-field">
          <label>ค้นหาตามวันที่:</label>
          <input
            type="date"
            value={searchDate}
            onChange={(e) => setSearchDate(e.target.value)}
          />
        </div>

        {/* ปุ่มล้างการค้นหา */}
        <div className="search-buttons">
          <button onClick={clearSearch} className="clear-search-btn">
            🔄 ล้างการค้นหา
          </button>
        </div>
      </div>

      {/* แสดงจำนวนผลลัพธ์ */}
      <div className="search-results-info">
        <p>📊 พบข้อมูล {filteredPredictions.length} รายการ จากทั้งหมด {allPredictions.length} รายการ</p>
      </div>

      {filteredPredictions.length === 0 ? (
        <div className="no-data-message">
          <p>
            {allPredictions.length === 0 
              ? "📋 คุณยังไม่มีประวัติการวิเคราะห์" 
              : "🔍 ไม่พบข้อมูลที่ตรงกับการค้นหา"
            }
          </p>
        </div>
      ) : (
        <>
          {/* แสดงข้อมูลของหน้าปัจจุบัน */}
          <div className="history-list">
            {currentPageData.map((prediction) => (
              <div className="history-item" key={prediction.id}>
                <h3>{prediction.diseaseName || "ไม่ระบุชื่อโรค"}</h3>
                {(prediction.imageUrl || prediction.imageBase64) && (
                  <div className="image-container">
                    <img
                      src={prediction.imageUrl || prediction.imageBase64}
                      alt="ภาพที่ใช้วิเคราะห์"
                      className="image-thumbnail"
                      onError={(e) => {
                        e.target.style.display = 'none';
                      }}
                    />
                  </div>
                )}
                <p><strong>ความมั่นใจ (confidence):</strong> {Math.round((prediction.confidence || 0) * 100)}%</p>
                <p><strong>ความแม่นยำ (accuracy):</strong> {Math.round((prediction.accuracy || 0) * 100)}%</p>
                <p><strong>วันที่วิเคราะห์:</strong> {
                  prediction.timestamp?.seconds 
                    ? new Date(prediction.timestamp.seconds * 1000).toLocaleString("th-TH")
                    : "ไม่ระบุวันที่"
                }</p>
                <button
                  className="view-details-btn"
                  onClick={() => navigate("/historydetail", {
                    state: {
                      docId: prediction.id,
                      ...prediction
                    }
                  })}
                >
                  ดูรายละเอียด
                </button>
              </div>
            ))}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="pagination-container">
              {/* ปุ่มหน้าก่อนหน้า */}
              <button
                onClick={() => handlePageChange(currentPage - 1)}
                disabled={currentPage === 1}
                className="pagination-nav"
              >
                ← ก่อนหน้า
              </button>

              {/* ปุ่มเลขหน้า */}
              {Array.from({ length: totalPages }, (_, index) => {
                const pageNumber = index + 1;
                return (
                  <button
                    key={pageNumber}
                    onClick={() => handlePageChange(pageNumber)}
                    className={`pagination-btn ${currentPage === pageNumber ? 'active' : ''}`}
                  >
                    {pageNumber}
                  </button>
                );
              })}

              {/* ปุ่มหน้าถัดไป */}
              <button
                onClick={() => handlePageChange(currentPage + 1)}
                disabled={currentPage === totalPages}
                className="pagination-nav"
              >
                ถัดไป →
              </button>
            </div>
          )}

          {/* แสดงข้อมูลสถิติ */}
          <div className="pagination-stats">
            <p>
              📄 หน้า {currentPage} จาก {totalPages} 
              ({((currentPage - 1) * itemsPerPage) + 1}-{Math.min(currentPage * itemsPerPage, filteredPredictions.length)} 
              จาก {filteredPredictions.length} รายการ)
            </p>
          </div>
        </>
      )}
    </div>
  );
}

export default History;