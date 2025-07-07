import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getAuth, onAuthStateChanged } from "firebase/auth";
import { collection, query, where, getDocs, orderBy } from "firebase/firestore";
import { db } from "../firebaseConfig";
import "../css/historyanaly.css";

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

  useEffect(() => {
    let filtered = [...allPredictions];
    if (searchDisease.trim()) {
      filtered = filtered.filter(prediction =>
        (prediction.diseaseName || "").toLowerCase().includes(searchDisease.toLowerCase())
      );
    }
    if (searchDate) {
      filtered = filtered.filter(prediction => {
        if (!prediction.timestamp?.seconds) return false;
        const predictionDate = new Date(prediction.timestamp.seconds * 1000).toISOString().split('T')[0];
        return predictionDate === searchDate;
      });
    }
    setFilteredPredictions(filtered);
    setCurrentPage(1);
  }, [allPredictions, searchDisease, searchDate]);

  const fetchHistory = async (currentUser) => {
    try {
      const q = query(
        collection(db, "prediction_results"),
        where("userId", "==", currentUser.uid),
        orderBy("timestamp", "desc")
      );
      const querySnapshot = await getDocs(q);
      const historyData = [];
      querySnapshot.forEach((doc) => {
        historyData.push({ id: doc.id, ...doc.data() });
      });
      setAllPredictions(historyData);
    } catch (error) {
      try {
        const fallbackQuery = query(
          collection(db, "prediction_results"),
          where("userId", "==", currentUser.uid)
        );
        const snapshot = await getDocs(fallbackQuery);
        const fallbackData = [];
        snapshot.forEach((doc) => {
          fallbackData.push({ id: doc.id, ...doc.data() });
        });
        fallbackData.sort((a, b) => (b.timestamp?.seconds || 0) - (a.timestamp?.seconds || 0));
        setAllPredictions(fallbackData);
        setError(null);
      } catch (simpleError) {
        setError("เกิดข้อผิดพลาดในการโหลดประวัติ");
      }
    } finally {
      setLoading(false);
    }
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
    setSearchDate("");
  };

  if (loading) return <p>กำลังโหลดข้อมูล...</p>;
  if (error) return <p style={{ color: 'red' }}>{error}</p>;
  if (!user) return <p>กรุณาเข้าสู่ระบบเพื่อดูประวัติ</p>;

  const currentPageData = getCurrentPageData();

  return (
    <div className="history-container">
      <div className="analy-header">
        <button onClick={handleGoHome} className="back-button">⬅️ หน้าหลัก</button>
      </div>

      <h2>ประวัติการวิเคราะห์โรค</h2>

      <div className="search-container">
        <h3>ค้นหาข้อมูล</h3>

        <div className="search-field">
          <label>ค้นหาตามชื่อโรค:</label>
          <input
            type="text"
            value={searchDisease}
            onChange={(e) => setSearchDisease(e.target.value)}
            placeholder="พิมพ์ชื่อโรคที่ต้องการค้นหา..."
          />
        </div>

        <div className="search-field">
          <label>ค้นหาตามวันที่:</label>
          <input
            type="date"
            value={searchDate}
            onChange={(e) => setSearchDate(e.target.value)}
          />
        </div>

        <div className="search-buttons">
          <button onClick={clearSearch} className="clear-search-btn">🔄 ล้างการค้นหา</button>
        </div>
      </div>

      <div className="search-results-info">
        <p>📊 พบข้อมูล {filteredPredictions.length} รายการ จากทั้งหมด {allPredictions.length} รายการ</p>
      </div>

      {filteredPredictions.length === 0 ? (
        <div className="no-data-message">
          <p>{allPredictions.length === 0
            ? "📋 คุณยังไม่มีประวัติการวิเคราะห์"
            : "🔍 ไม่พบข้อมูลที่ตรงกับการค้นหา"}</p>
        </div>
      ) : (
        <>
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
                      onError={(e) => { e.target.style.display = 'none'; }}
                    />
                  </div>
                )}
                <p><strong>ความมั่นใจ:</strong> {Math.round((prediction.confidence || 0) * 100)}%</p>
                <p><strong>ความแม่นยำ:</strong> {Math.round((prediction.accuracy || 0) * 100)}%</p>
                <p><strong>วันที่:</strong> {
                  prediction.timestamp?.seconds
                    ? new Date(prediction.timestamp.seconds * 1000).toLocaleString("th-TH")
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