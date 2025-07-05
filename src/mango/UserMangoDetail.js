import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { doc, getDoc } from "firebase/firestore";
import { db } from "../firebaseConfig";
import "../css/usermangodetail.css";

function UserMangoDetail() {
    const { id } = useParams();
    const [mango, setMango] = useState(null);
    const [loading, setLoading] = useState(true);
    const navigate = useNavigate(); // ✅

    useEffect(() => {
        const fetchMango = async () => {
            try {
                const docRef = doc(db, "mango_diseases", id);
                const docSnap = await getDoc(docRef);
                if (docSnap.exists()) {
                    setMango(docSnap.data());
                } else {
                    console.log("No such document!");
                }
            } catch (error) {
                console.error("Error fetching mango disease:", error);
            } finally {
                setLoading(false);
            }
        };

        fetchMango();
    }, [id]);

    const handleBack = () => {
        navigate("/showmango");
    };

    if (loading) {
        return <div className="disease-detail-container">กำลังโหลดข้อมูล...</div>;
    }

    if (!mango) {
        return <div className="disease-detail-container">ไม่พบข้อมูลโรคนี้</div>;
    }

    return (
        <div className="user-disease-detail-container">
            <button onClick={handleBack} className="back-button">
                ⬅️ กลับ
            </button>
            <h3 className="user-namedisease">{mango.diseaseName}</h3>
            <img
                src={mango.imageUrl}
                alt={mango.diseaseName}
                className="user-img-disease"
            />
            <div className="user-boxmango">
                <p><strong>ลักษณะอาการ:</strong> {mango.symptoms}</p>
                <p><strong>วิธีรักษา:</strong> {mango.treatment}</p>
                <p><strong>วิธีป้องกัน:</strong> {mango.prevention}</p>
            </div>
        </div>
    );
}

export default UserMangoDetail;
