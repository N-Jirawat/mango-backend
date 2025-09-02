import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { doc, getDoc, collection, getDocs, getFirestore } from "firebase/firestore";
import "../css/usermangodetail.css";

function UserMangoDetail() {
    const { id } = useParams();
    const [mango, setMango] = useState(null);
    const [mangoImages, setMangoImages] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const navigate = useNavigate();

    const db = getFirestore();

    useEffect(() => {
        const fetchMangoData = async () => {
            try {
                setLoading(true);
                setError(null);

                // ดึงข้อมูลโรค
                const docRef = doc(db, "MangoDisease", id);
                const docSnap = await getDoc(docRef);

                if (docSnap.exists()) {
                    const mangoData = { id: docSnap.id, ...docSnap.data() };
                    setMango(mangoData);

                    // ดึงภาพจาก ImageMango
                    const imageSnapshot = await getDocs(collection(db, "ImageMango"));
                    const images = [];
                    
                    imageSnapshot.forEach((docImg) => {
                        const data = docImg.data();
                        if (data.DiseaseID === docSnap.id) {
                            images.push({ id: docImg.id, ...data });
                        }
                    });

                    // เรียงลำดับภาพตามวันที่อัพโหลด (ใหม่สุดก่อน)
                    images.sort((a, b) => {
                        const dateA = a.DateUploadImg?.toDate() || new Date(0);
                        const dateB = b.DateUploadImg?.toDate() || new Date(0);
                        return dateB - dateA;
                    });
                    
                    setMangoImages(images);
                } else {
                    setError("ไม่พบข้อมูลโรคนี้");
                }
            } catch (err) {
                setError("เกิดข้อผิดพลาดในการโหลดข้อมูล กรุณาลองใหม่อีกครั้ง");
            } finally {
                setLoading(false);
            }
        };

        if (id) {
            fetchMangoData();
        }
    }, [id, db]);

    const handleBack = () => {
        navigate("/showmango");
    };

    const handleImageError = (e) => {
        e.target.style.display = 'none';
    };

    const handleImageLoad = (e) => {
        e.target.style.opacity = '1';
    };

    // Loading State
    if (loading) {
        return (
            <div className="container-state">
                <div className="loading-container">
                    <div className="loading-spinner"></div>
                    <div className="loading-message">กำลังโหลดข้อมูล...</div>
                </div>
            </div>
        );
    }

    // Error State
    if (error) {
        return (
            <div className="container-state">
                <div className="user-disease-detail-container">
                    <button onClick={handleBack} className="back-button">
                        ⬅️ กลับ
                    </button>
                    <div className="error-message">
                        <p>{error}</p>
                        <button 
                            onClick={() => window.location.reload()} 
                            className="back-button"
                            style={{marginTop: '10px'}}
                        >
                            🔄 ลองใหม่
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    // No data state
    if (!mango) {
        return (
            <div className="container-state">
                <div className="user-disease-detail-container">
                    <button onClick={handleBack} className="back-button">
                        ⬅️ กลับ
                    </button>
                    <div className="error-message">ไม่พบข้อมูลโรคนี้</div>
                </div>
            </div>
        );
    }

    return (
        <div className="container-state">
            <div className="user-disease-detail-container">
                <button onClick={handleBack} className="back-button">
                    ⬅ กลับสู่รายการโรค
                </button>
                
                <h3 className="user-namedisease">{mango.DiseaseName || 'ไม่มีชื่อโรค'}</h3>

                <div className="image-section">
                    {mangoImages.length > 0 ? (
                        mangoImages.map((img, idx) => (
                            <img
                                key={`${img.id}-${idx}`}
                                src={img.ImgPath}
                                alt={`${mango.DiseaseName || 'โรคมะม่วง'} รูปที่ ${idx + 1}`}
                                className="user-img-disease"
                                onError={handleImageError}
                                onLoad={handleImageLoad}
                                style={{opacity: 0, transition: 'opacity 0.3s ease'}}
                                loading="lazy"
                            />
                        ))
                    ) : (
                        mango.ImgPath && (
                            <img
                                src={mango.ImgPath}
                                alt={mango.DiseaseName || 'โรคมะม่วง'}
                                className="user-img-disease"
                                onError={handleImageError}
                                onLoad={handleImageLoad}
                                style={{opacity: 0, transition: 'opacity 0.3s ease'}}
                                loading="lazy"
                            />
                        )
                    )}
                </div>

                <div className="user-boxmango">
                    {mango.Style && (
                        <p>
                            <strong>🔍 ลักษณะอาการ:</strong>
                            {mango.Style}
                        </p>
                    )}
                    
                    {mango.Treatment && (
                        <p>
                            <strong>💊 วิธีรักษา:</strong>
                            {mango.Treatment}
                        </p>
                    )}
                    
                    {mango.Protection && (
                        <p>
                            <strong>🛡️ วิธีป้องกัน:</strong>
                            {mango.Protection}
                        </p>
                    )}

                    {!mango.Style && !mango.Treatment && !mango.Protection && (
                        <p className="no-data-message">
                            <strong>ℹ️ ข้อมูล:</strong>
                            ยังไม่มีข้อมูลรายละเอียดสำหรับโรคนี้
                        </p>
                    )}
                </div>
            </div>
        </div>
    );
}

export default UserMangoDetail;