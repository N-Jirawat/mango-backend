import React, { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { collection, addDoc } from "firebase/firestore";
import { db } from "../firebaseConfig";
import { getAuth } from "firebase/auth"; // ดึง Firebase Authentication
import { query, where, getDocs } from "firebase/firestore"; // นำเข้า query, where, getDocs
import "../css/resultanaly.css";

function ResultAnaly() {
    const { state } = useLocation();
    const { prediction, confidence, accuracy, imagePreview, imageFile } = state || {};
    const [resultInfo, setResultInfo] = useState(null);
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

    // // ฟังก์ชันสำหรับแปลง blob URL เป็นไฟล์
    // const blobUrlToFile = async (blobUrl) => {
    //     const response = await fetch(blobUrl);
    //     const blob = await response.blob();
    //     const file = new File([blob], "image.jpg", { type: blob.type });
    //     return file;
    // };

    // ฟังก์ชันสำหรับอัปโหลดภาพไปยัง Cloudinary
    const uploadImageToCloudinary = async (file) => {
        if (!file) {
            console.error("No file provided!");
            return null; // คืนค่า null ถ้าไม่มีไฟล์
        }

        const formData = new FormData();
        formData.append("file", file); // ส่งไฟล์จริงไปยัง Cloudinary
        formData.append("upload_preset", "ml_default");
        formData.append("folder", "Result_Analy");
        formData.append("cloud_name", "dsf25dlca");

        try {
            const response = await fetch("https://api.cloudinary.com/v1_1/dsf25dlca/image/upload", {
                method: "POST",
                body: formData, // ส่งไฟล์จริงผ่าน FormData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(`Upload failed with status: ${response.status}, ${data.error.message}`);
            }

            if (data.secure_url) {
                return data.secure_url; // ส่ง URL ของภาพที่อัปโหลด
            } else {
                throw new Error("Upload failed: No URL returned from Cloudinary");
            }
        } catch (error) {
            console.error("Upload failed:", error);
            alert(`เกิดข้อผิดพลาดในการอัปโหลดภาพ: ${error.message}`);
            return null;
        }
    };

    const saveDataToFirestore = async (imageUrl) => {
        const auth = getAuth();
        const user = auth.currentUser;

        if (!user) {
            // แทนที่จะ alert ให้นำทางไปหน้า login พร้อมข้อความ
            navigate("/login", {
                state: {
                    message: "กรุณาเข้าสู่ระบบเพื่อบันทึกข้อมูล",
                    redirectTo: "/resultanaly"
                }
            });
            return;
        }

        try {
            await addDoc(collection(db, "prediction_results"), {
                diseaseName: prediction,
                confidence: confidence,
                accuracy: accuracy,
                symptoms: resultInfo?.symptoms || "ไม่มีข้อมูลอาการ",
                prevention: resultInfo?.prevention || "ไม่มีข้อมูลวิธีการป้องกัน",
                treatment: resultInfo?.treatment || "ไม่มีข้อมูลวิธีการรักษา",
                userId: user.uid,
                timestamp: new Date(),
                imageUrl: imageUrl,
            });
            alert("บันทึกข้อมูลสำเร็จ!");
        } catch (error) {
            console.error("Error saving data:", error);
            alert("เกิดข้อผิดพลาดในการบันทึกข้อมูล");
        }
    };

    useEffect(() => {
        const fetchResult = async () => {
            if (prediction) {
                // สร้าง query เพื่อตรวจสอบว่า diseaseName ตรงกับ prediction หรือไม่
                const q = query(
                    collection(db, "mango_diseases"),
                    where("diseaseName", "==", prediction) // ค้นหาจากฟิลด์ diseaseName
                );

                const querySnapshot = await getDocs(q);
                if (!querySnapshot.empty) {
                    // หากพบเอกสารที่มีชื่อโรคตรงกับ prediction
                    querySnapshot.forEach(doc => {
                        setResultInfo(doc.data()); // ตั้งค่าข้อมูลที่ได้จาก Firestore
                    });
                } else {
                    setResultInfo(null); // หากไม่พบข้อมูล
                }
            }
        };

        fetchResult();
    }, [prediction]); // useEffect จะทำงานเมื่อ prediction เปลี่ยนแปลง

    if (!prediction) return <p className="not-found">ไม่พบข้อมูลการทำนาย</p>;

    const handleSaveData = async () => {
        setLoading(true);

        // ใช้ไฟล์ต้นฉบับแทน blobUrl
        if (!imageFile) {
            alert("ไม่พบไฟล์ภาพต้นฉบับ");
            setLoading(false);
            return;
        }

        const imageUrl = await uploadImageToCloudinary(imageFile);
        if (imageUrl) {
            await saveDataToFirestore(imageUrl);
        }

        setLoading(false);
    };

    const handleGoHome = () => {
        navigate('/');
    };

    // ตรวจสอบสถานะ login สำหรับแสดงปุ่มบันทึก
    const auth = getAuth();
    const isLoggedIn = auth.currentUser !== null;

    return (
        <div className="result-container">
            <div className="result-header">
                <button onClick={handleGoHome} className="back-button">
                    ⬅️ หน้าหลัก
                </button>
            </div>

            <h2 className="title">รายละเอียดโรค</h2>

            {imagePreview && <img src={imagePreview} alt="Uploaded" className="image-preview" />}

            <div className="result-item">
                <strong>ชื่อโรค:</strong> {prediction}
            </div>
            <div className="result-item">
                <strong>ความมั่นใจ (confidence):</strong> {Math.round(confidence * 100)}%
            </div>
            <div className="result-item">
                <strong>ความแม่นยำ (accuracy):</strong> {Math.round(accuracy * 100)}%
            </div>

            {resultInfo ? (
                <>
                    <div className="result-item">
                        <strong>รายละเอียดโรค:</strong> {resultInfo.symptoms || "ไม่มีข้อมูลรายละเอียดโรค"}
                    </div>
                    <div className="result-item">
                        <strong>วิธีป้องกัน:</strong> {resultInfo.prevention || "ไม่มีข้อมูลวิธีการป้องกัน"}
                    </div>
                    <div className="result-item">
                        <strong>วิธีการรักษา:</strong> {resultInfo.treatment || "ไม่มีข้อมูลวิธีการรักษา"}
                    </div>
                    
                    {/* แสดงปุ่มบันทึกเฉพาะเมื่อ login แล้ว */}
                    {isLoggedIn ? (
                        <button onClick={handleSaveData} className="save-btn" disabled={loading}>
                            {loading ? "กำลังบันทึก..." : "บันทึกข้อมูล"}
                        </button>
                    ) : (
                        <div className="login-prompt">
                            <p>หากต้องการบันทึกผลการวิเคราะห์ กรุณาเข้าสู่ระบบ<button 
                                onClick={() => navigate("/login", {
                                    state: {
                                        message: "กรุณาเข้าสู่ระบบเพื่อบันทึกข้อมูล",
                                        redirectTo: "/resultanaly"
                                    }
                                })} 
                                className="login-link-btn"
                            >เข้าสู่ระบบ</button></p>
                        </div>
                    )}
                </>
            ) : (
                <p className="not-found">ไม่พบข้อมูลในระบบ</p>
            )}
        </div>
    );
}

export default ResultAnaly;