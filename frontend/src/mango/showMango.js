import React, { useEffect, useState } from "react";
import { collection, getDocs, doc, getDoc } from "firebase/firestore";
import { getAuth, onAuthStateChanged } from "firebase/auth";
import { db } from "../firebaseConfig";
import { Link, useNavigate } from "react-router-dom";
import "../css/showmango.css";

// Component สำหรับดึงและแสดงรูปภาพจาก ImageMango
function MangoImage({ imgId, diseaseName, fallbackSrc }) {
  const [imageSrc, setImageSrc] = useState(fallbackSrc);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchImage = async () => {
      if (!imgId) {
        setLoading(false);
        return;
      }

      try {
        const docRef = doc(db, "ImageMango", imgId);
        const docSnap = await getDoc(docRef);
        
        if (docSnap.exists()) {
          const imageData = docSnap.data();
          
          // ลองหลาย field สำหรับ image path
          const imgPath = imageData.ImgPath || imageData.imageUrl || imageData.imgPath;
          if (imgPath) {
            setImageSrc(imgPath);
          }
        }
      } finally {
        setLoading(false);
      }
    };

    fetchImage();
  }, [imgId, diseaseName]);

  return (
    <div style={{ position: 'relative' }}>
      {loading && (
        <div style={{ 
          position: 'absolute', 
          top: '50%', 
          left: '50%', 
          transform: 'translate(-50%, -50%)',
          fontSize: '12px',
          color: '#666'
        }}>
          กำลังโหลด...
        </div>
      )}
      <img 
        src={imageSrc}
        alt={diseaseName || 'โรคมะม่วง'}
        onError={(e) => {
          if (imageSrc !== fallbackSrc) {
            setImageSrc(fallbackSrc);
          }
        }}
        style={{ 
          width: '100%', 
          height: '200px', 
          objectFit: 'cover',
          display: loading ? 'none' : 'block'
        }}
      />
    </div>
  );
}

function ShowMango() {
  const [mangoData, setMangoData] = useState([]);
  const [role, setRole] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  // ฟังก์ชันสำหรับกรองใบมะม่วงปกติออก
  const isNormalMango = (diseaseName) => {
    if (!diseaseName) return false;
    
    const normalKeywords = ['ปกติ', 'normal', 'healthy', 'สุขภาพดี', 'ไม่มีโรค'];
    const lowerDiseaseName = diseaseName.toLowerCase();
    
    return normalKeywords.some(keyword => 
      lowerDiseaseName.includes(keyword.toLowerCase())
    );
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const querySnapshot = await getDocs(collection(db, "MangoDisease"));
        const data = querySnapshot.docs.map((doc) => {
          const docData = doc.data();
          return { id: doc.id, ...docData };
        });
        
        // กรองข้อมูลให้แสดงเฉพาะโรค ไม่แสดงใบมะม่วงปกติ
        const filteredData = data.filter(item => !isNormalMango(item.DiseaseName));
        
        setMangoData(filteredData);
      } finally {
        setLoading(false);
      }
    };

    const fetchUserRole = async () => {
      const auth = getAuth();
      onAuthStateChanged(auth, async (user) => {
        if (user) {
          const docRef = doc(db, "users", user.uid);
          const docSnap = await getDoc(docRef);
          if (docSnap.exists()) {
            setRole(docSnap.data().role);
          }
        } else {
          setLoading(false);
        }
      });
    };

    fetchData();
    fetchUserRole();
  }, []);

  const handleBack = () => {
    if (role === "true" || role === true) {
      navigate("/accountmanagement");
    } else {
      navigate("/");
    }
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>กำลังโหลดข้อมูล...</p>
      </div>
    );
  }

  return (
    <div className="show-mango-container">
      <button onClick={handleBack} className="back-button">
        หน้าหลัก
      </button>
      <h2>รายการโรคใบมะม่วง</h2>
      <div className="mango-card-grid">
        {mangoData.map((item) => (
          <div className="mango-card" key={item.id}>
            {/* ใช้ ImgID เพื่อดึงภาพจากตาราง ImageMango */}
            <div className="image-container">
              <MangoImage 
                imgId={item.ImgID} 
                diseaseName={item.DiseaseName}
                fallbackSrc="/placeholder-image.png"
              />
            </div>
            <h3>{item.DiseaseName || 'ไม่มีชื่อโรค'}</h3>
            <Link to={`/usermangodetail/${item.id}`}>
              <button className="view-details-btn">ดูรายละเอียด</button>
            </Link>
          </div>
        ))}
      </div>
      
      {mangoData.length === 0 && !loading && (
        <div className="no-data">
          <p>ไม่พบข้อมูลโรคมะม่วง</p>
        </div>
      )}
    </div>
  );
}

export default ShowMango;