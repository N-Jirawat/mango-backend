import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { db } from "../firebaseConfig";
import { doc, getDoc, deleteDoc, collection, query, where, getDocs } from "firebase/firestore";
import "../css/mangodetail.css";

function MangoDetail() {
    const { id } = useParams(); // ดึงค่า 'id' จาก URL (DiseaseID)
    const navigate = useNavigate();
    const [disease, setDisease] = useState(null); // สถานะเก็บข้อมูลโรคมะม่วง
    const [imageData, setImageData] = useState(null); // สถานะเก็บข้อมูลรูปภาพ
    const [loading, setLoading] = useState(true);
    const [deleting, setDeleting] = useState(false);

    // ฟังก์ชันสำหรับลบรูปภาพจาก Cloudinary
    const deleteFromCloudinary = async (publicId) => {
        const BACKEND_URL = "https://mango-backend-665966382004.asia-southeast1.run.app";
        //const BACKEND_URL = "http://localhost:5000";
        
        const formData = new FormData();
        formData.append("public_id", publicId);

        try {
            const response = await fetch(`${BACKEND_URL}/delete`, {
                method: "POST",
                body: formData,
            });

            const data = await response.json();
            if (data.result === "ok") {
                console.log("Image deleted successfully from Cloudinary.");
            }
        } catch (error) {
            console.error("Error deleting image:", error);
        }
    };

    // useEffect ใช้เพื่อดึงข้อมูลโรคมะม่วงจาก Firestore เมื่อหน้าเพจโหลด
    useEffect(() => {
        const fetchDiseaseDetail = async () => {
            try {
                // 1. ดึงข้อมูลโรคจาก MangoDisease collection
                const mangoDiseaseRef = doc(db, "MangoDisease", id);
                const mangoDiseaseDoc = await getDoc(mangoDiseaseRef);
                
                if (!mangoDiseaseDoc.exists()) {
                    alert("ไม่พบข้อมูลโรคมะม่วง");
                    navigate("/mango");
                    return;
                }

                const mangoDiseaseData = mangoDiseaseDoc.data();
                setDisease({
                    id: mangoDiseaseDoc.id,
                    DiseaseName: mangoDiseaseData.DiseaseName || "",
                    Style: mangoDiseaseData.Style || "",
                    Treatment: mangoDiseaseData.Treatment || "",
                    Protection: mangoDiseaseData.Protection || "",
                    UpdateAt: mangoDiseaseData.UpdateAt,
                    ImgID: mangoDiseaseData.ImgID || ""
                });

                // 2. ดึงข้อมูลรูปภาพจาก ImageMango collection
                const imageQuery = query(
                    collection(db, "ImageMango"), 
                    where("DiseaseID", "==", id)
                );
                const imageSnapshot = await getDocs(imageQuery);
                
                if (!imageSnapshot.empty) {
                    const imageDoc = imageSnapshot.docs[0]; // เอาภาพแรก
                    const imageInfo = {
                        id: imageDoc.id,
                        ...imageDoc.data()
                    };
                    setImageData(imageInfo);
                }

            } catch (error) {
                console.error("เกิดข้อผิดพลาดในการโหลดข้อมูล:", error);
                alert("เกิดข้อผิดพลาดในการโหลดข้อมูล");
                navigate("/mango");
            } finally {
                setLoading(false);
            }
        };

        fetchDiseaseDetail();
    }, [id, navigate]);

    // ฟังก์ชันสำหรับลบข้อมูล
    const handleDelete = async () => {
        if (!window.confirm("คุณแน่ใจหรือไม่ว่าต้องการลบข้อมูลนี้?")) {
            return;
        }

        setDeleting(true);

        try {
            // 1. ลบรูปภาพจาก Cloudinary (ถ้ามี)
            if (imageData && imageData.public_id) {
                await deleteFromCloudinary(imageData.public_id);
            }

            // 2. ลบข้อมูลรูปภาพจาก ImageMango collection (ถ้ามี)
            if (imageData && imageData.id) {
                const imageRef = doc(db, "ImageMango", imageData.id);
                await deleteDoc(imageRef);
            }

            // 3. ลบข้อมูลโรคจาก MangoDisease collection
            const mangoDiseaseRef = doc(db, "MangoDisease", id);
            await deleteDoc(mangoDiseaseRef);

            alert("ลบข้อมูลสำเร็จ");
            navigate("/mango");
        } catch (error) {
            console.error("เกิดข้อผิดพลาดในการลบ:", error);
            alert("เกิดข้อผิดพลาดในการลบข้อมูล");
        } finally {
            setDeleting(false);
        }
    };

    if (loading) {
        return (
            <div className="disease-detail-container">
                <div style={{ textAlign: "center", padding: "40px" }}>
                    กำลังโหลดข้อมูล...
                </div>
            </div>
        );
    }

    if (!disease) {
        return (
            <div className="disease-detail-container">
                <div style={{ textAlign: "center", padding: "40px" }}>
                    ไม่พบข้อมูลโรคมะม่วง
                </div>
            </div>
        );
    }

    return (
        <div className="disease-detail-container">
            <div className="disease-header">
                <h2>{disease.DiseaseName}</h2>
                {disease.UpdateAt && (
                    <p className="update-date">
                        อัปเดตล่าสุด: {disease.UpdateAt.toDate().toLocaleDateString('th-TH', {
                            year: 'numeric',
                            month: 'long',
                            day: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit'
                        })}
                    </p>
                )}
            </div>

            {/* แสดงรูปภาพ */}
            <div className="disease-image-section">
                {imageData && imageData.ImgPath ? (
                    <div className="image-container">
                        <img 
                            src={imageData.ImgPath} 
                            alt={disease.DiseaseName} 
                            className="disease-image"
                            onError={(e) => {
                                e.target.style.display = 'none';
                                e.target.nextSibling.style.display = 'block';
                            }}
                        />
                        <div style={{ display: 'none', textAlign: 'center', padding: '20px', color: '#666' }}>
                            รูปภาพไม่สามารถแสดงได้
                        </div>
                        {imageData.DateUploadImg && (
                            <p className="image-upload-date">
                                อัปโหลดเมื่อ: {imageData.DateUploadImg.toDate().toLocaleDateString('th-TH')}
                            </p>
                        )}
                    </div>
                ) : (
                    <div className="no-image">
                        <p>ไม่มีรูปภาพ</p>
                    </div>
                )}
            </div>

            {/* แสดงรายละเอียดโรค */}
            <div className="disease-details">
                <div className="detail-section">
                    <h3>ลักษณะอาการ</h3>
                    <p>{disease.Style || "ไม่มีข้อมูล"}</p>
                </div>

                <div className="detail-section">
                    <h3>วิธีรักษา</h3>
                    <p>{disease.Treatment || "ไม่มีข้อมูล"}</p>
                </div>

                <div className="detail-section">
                    <h3>วิธีป้องกัน</h3>
                    <p>{disease.Protection || "ไม่มีข้อมูล"}</p>
                </div>
            </div>

            {/* ปุ่มต่างๆ */}
            <div className="button-group">
                <button 
                    onClick={() => navigate("/mango")} 
                    className="button-back"
                >
                    ⬅️ ย้อนกลับ
                </button>
                
                <button 
                    onClick={() => navigate(`/editmango/${id}`)} 
                    className="edit-btn"
                >
                    ✏️ แก้ไข
                </button>
                
                <button 
                    type="button" 
                    onClick={handleDelete} 
                    className="delete-btn-detail"
                    disabled={deleting}
                >
                    {deleting ? "กำลังลบ..." : "🗑️ ลบข้อมูล"}
                </button>
            </div>
        </div>
    );
}

export default MangoDetail;