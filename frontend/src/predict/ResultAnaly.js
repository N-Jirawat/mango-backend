import React, { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { collection, addDoc } from "firebase/firestore";
import { db } from "../firebaseConfig";
import { getAuth } from "firebase/auth";
import { query, getDocs } from "firebase/firestore";
import "../css/resultanaly.css";

function ResultAnaly() {
    const { state } = useLocation();
    const { prediction, confidence, imagePreview, imageFile } = state || {};
    const [resultInfo, setResultInfo] = useState(null);
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

    // ฟังก์ชันสำหรับอัปโหลดภาพไปยัง Cloudinary
    const uploadImageToCloudinary = async (file) => {
        if (!file) {
            console.error("No file provided!");
            return null;
        }

        const formData = new FormData();
        formData.append("file", file);
        formData.append("upload_preset", "ml_default");
        formData.append("folder", "Result_Analy");
        formData.append("cloud_name", "dsf25dlca");

        try {
            const response = await fetch("https://api.cloudinary.com/v1_1/dsf25dlca/image/upload", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(`Upload failed with status: ${response.status}, ${data.error.message}`);
            }

            if (data.secure_url) {
                return data.secure_url;
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
            navigate("/login", {
                state: {
                    message: "กรุณาเข้าสู่ระบบเพื่อบันทึกข้อมูล",
                    redirectTo: "/resultanaly"
                }
            });
            return;
        }

        try {
            await addDoc(collection(db, "AnalysisHistory"), {
                diseaseName: prediction,
                confidence: confidence,
                Style: resultInfo?.Style || "ไม่มีข้อมูลอาการ",
                Protection: resultInfo?.Protection || "ไม่มีข้อมูลวิธีการป้องกัน",
                Treatment: resultInfo?.Treatment || "ไม่มีข้อมูลวิธีการรักษา",
                userId: user.uid,
                UpdateAt: new Date(),
                imageUrl: imageUrl,
            });
            alert("บันทึกข้อมูลสำเร็จ!");
            navigate("/history");
        } catch (error) {
            alert("เกิดข้อผิดพลาดในการบันทึกข้อมูล");
        }
    };

    // ฟังก์ชันสำหรับจับคู่ชื่อโรค
    const findDiseaseMatch = (searchTerm, documents) => {
        const normalizedSearch = searchTerm.toLowerCase().trim();
        
        // หาแบบตรงกัน 100%
        for (const doc of documents) {
            const data = doc.data();
            if (data.DiseaseName) {
                const diseaseName = data.DiseaseName.toLowerCase().trim();
                
                if (diseaseName === normalizedSearch) {
                    return data;
                }
            }
        }

        // หาแบบมีคำที่ตรงกัน
        for (const doc of documents) {
            const data = doc.data();
            if (data.DiseaseName) {
                const diseaseName = data.DiseaseName.toLowerCase().trim();
                
                // ตรวจสอบว่าชื่อโรคมีคำที่ค้นหาอยู่หรือไม่
                if (diseaseName.includes(normalizedSearch)) {
                    return data;
                }
                
                if (normalizedSearch.includes(diseaseName)) {
                    return data;
                }
                
                // ตรวจสอบคำสำคัญ - แยกคำและเปรียบเทียบ
                const searchWords = normalizedSearch.split(/\s+/);
                const diseaseWords = diseaseName.split(/\s+/);
                
                // ตรวจสอบว่ามีคำที่ตรงกันหรือไม่
                for (const searchWord of searchWords) {
                    for (const diseaseWord of diseaseWords) {
                        if (searchWord === diseaseWord || 
                            searchWord.includes(diseaseWord) || 
                            diseaseWord.includes(searchWord)) {
                            return data;
                        }
                    }
                }
            }
        }

        return null;
    };

    useEffect(() => {
        const fetchResult = async () => {
            if (prediction) {
                try {
                    // ดึงข้อมูลทั้งหมดจาก collection
                    const allDocsQuery = query(collection(db, "MangoDisease"));
                    const allDocsSnapshot = await getDocs(allDocsQuery);

                    if (!allDocsSnapshot.empty) {
                        const allDocs = [];
                        allDocsSnapshot.forEach(doc => {
                            allDocs.push(doc);
                        });

                        // หาข้อมูลที่ตรงกัน
                        const matchedData = findDiseaseMatch(prediction, allDocs);
                        
                        if (matchedData) {
                            setResultInfo(matchedData);
                        } else {
                            // วิธีสำรอง: ค้นหาแบบง่าย ๆ สำหรับ "ใบปกติ"
                            if (prediction.toLowerCase().includes("ใบปกติ") || prediction.toLowerCase().includes("ปกติ")) {
                                const normalLeafData = allDocs.find(doc => {
                                    const data = doc.data();
                                    return data.DiseaseName && 
                                           (data.DiseaseName.includes("ใบปกติ") || 
                                            data.DiseaseName.includes("ปกติ") ||
                                            data.DiseaseName.includes("ใบมะม่วงปกติ"));
                                });
                                
                                if (normalLeafData) {
                                    setResultInfo(normalLeafData.data());
                                } else {
                                    setResultInfo(null);
                                }
                            } else {
                                setResultInfo(null);
                            }
                        }
                    } else {
                        setResultInfo(null);
                    }
                } catch {
                    setResultInfo(null);
                }
            }
        };

        fetchResult();
    }, [prediction]);

    if (!prediction) return <p className="not-found">ไม่พบข้อมูลการวินิจฉัย</p>;

    const handleSaveData = async () => {
        setLoading(true);

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

    const handleGoUpLoad = () => {
        navigate('/predict');
    };

    const auth = getAuth();
    const isLoggedIn = auth.currentUser !== null;

    return (
        <div className="result-container">
            <div className="result-header">
                <button onClick={handleGoUpLoad} className="back-button">
                    ย้อนกลับ
                </button>
            </div>

            <h2 className="title">รายละเอียดโรค</h2>

            {imagePreview && <img src={imagePreview} alt="Uploaded" className="image-preview" />}

            <div className="result-item">
                <strong>ชื่อโรค:</strong> {prediction}
            </div>
            <div className="result-item">
                <strong>ความมั่นใจ (confidence):</strong> {confidence.toFixed(4)}%
            </div>

            {resultInfo ? (
                <>
                    <div className="result-item">
                        <strong>รายละเอียดโรค:</strong> {resultInfo.Style || "ไม่มีข้อมูลรายละเอียดโรค"}
                    </div>
                    <div className="result-item">
                        <strong>วิธีป้องกัน:</strong> {resultInfo.Protection || "ไม่มีข้อมูลวิธีการป้องกัน"}
                    </div>
                    <div className="result-item">
                        <strong>วิธีการรักษา:</strong> {resultInfo.Treatment || "ไม่มีข้อมูลวิธีการรักษา"}
                    </div>

                    {isLoggedIn ? (
                        <button onClick={handleSaveData} className="save-btn" disabled={loading}>
                            {loading ? "กำลังบันทึก..." : "บันทึกข้อมูล"}
                        </button>
                    ) : (
                        <div className="login-prompt">
                            <p>หากต้องการบันทึกผลการวิเคราะห์ กรุณาเข้าสู่ระบบ
                                <button
                                    onClick={() => navigate("/login", {
                                        state: {
                                            message: "กรุณาเข้าสู่ระบบเพื่อบันทึกข้อมูล",
                                            redirectTo: "/resultanaly"
                                        }
                                    })}
                                    className="login-link-btn"
                                >เข้าสู่ระบบ</button>
                            </p>
                        </div>
                    )}
                </>
            ) : (
                <div className="debug-info">
                    <p className="not-found">ไม่พบโรคที่ตรงกับข้อมูลในระบบ</p>
                </div>
            )}
        </div>
    );
}

export default ResultAnaly;