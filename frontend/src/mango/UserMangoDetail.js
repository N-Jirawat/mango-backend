import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { doc, getDoc, collection, getDocs, getFirestore } from "firebase/firestore";
import "../css/usermangodetail.css";

function UserMangoDetail() {
    const { id } = useParams();
    const [mango, setMango] = useState(null);
    const [mangoImages, setMangoImages] = useState([]);
    const [loading, setLoading] = useState(true);
    const navigate = useNavigate();

    const db = getFirestore();

    useEffect(() => {
        const fetchMangoData = async () => {
            try {
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

                    images.sort((a, b) => b.DateUploadImg?.toDate() - a.DateUploadImg?.toDate());
                    setMangoImages(images);
                } else {
                    console.log("ไม่พบเอกสารโรคนี้");
                }
            } catch (err) {
                console.error("เกิดข้อผิดพลาดในการโหลดข้อมูลโรค:", err);
            } finally {
                setLoading(false);
            }
        };

        fetchMangoData();
    }, [id, db]);

    const handleBack = () => navigate("/showmango");

    if (loading) return <p>กำลังโหลดข้อมูล...</p>;
    if (!mango) return <p>ไม่พบข้อมูลโรคนี้</p>;

    return (
        <div className="container-state">
            <div className="user-disease-detail-container">
                <button onClick={handleBack} className="back-button">⬅️ กลับ</button>
                <h3 className="user-namedisease">{mango.DiseaseName}</h3>

                <div className="image-section">
                    {mangoImages.length > 0 ? (
                        mangoImages.map((img, idx) => (
                            <img
                                key={img.id}
                                src={img.ImgPath}
                                alt={`${mango.DiseaseName} ${idx + 1}`}
                                className="user-img-disease"
                                onError={(e) => { e.target.style.display = 'none'; }}
                            />
                        ))
                    ) : (
                        mango.ImgPath && (
                            <img
                                src={mango.ImgPath}
                                alt={mango.DiseaseName}
                                className="user-img-disease"
                                onError={(e) => { e.target.alt = "ไม่สามารถโหลดรูปภาพได้"; }}
                            />
                        )
                    )}
                </div>

                <div className="user-boxmango">
                    <p><strong>ลักษณะอาการ:</strong> {mango.Style}</p>
                    <p><strong>วิธีรักษา:</strong> {mango.Treatment}</p>
                    <p><strong>วิธีป้องกัน:</strong> {mango.Protection}</p>
                </div>
            </div>
        </div>
    );
}

export default UserMangoDetail;
