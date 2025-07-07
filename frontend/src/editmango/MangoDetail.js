import React, { useState, useEffect } from "react"; // ใช้ React Hooks
import { useParams, useNavigate } from "react-router-dom"; // ใช้สำหรับดึงพารามิเตอร์จาก URL และนำทาง
import { db } from "../firebaseConfig"; // นำเข้าไฟล์การตั้งค่า Firebase
import { doc, getDoc, deleteDoc } from "firebase/firestore"; // ฟังก์ชันจาก Firebase Firestore สำหรับการดึงและลบข้อมูล
import "../css/mangodetail.css";

function MangoDetail() {
    const { id } = useParams(); // ดึงค่า 'id' จาก URL
    const navigate = useNavigate(); // สร้างฟังก์ชันสำหรับนำทางไปยังหน้าอื่นๆ
    const [disease, setDisease] = useState(null); // สถานะเก็บข้อมูลโรคมะม่วง
    const [loading, setLoading] = useState(true); // สถานะการโหลดข้อมูล

    // useEffect ใช้เพื่อดึงข้อมูลโรคมะม่วงจาก Firestore เมื่อหน้าเพจโหลด
    useEffect(() => {
        const fetchDisease = async () => {
            try {
                const docRef = doc(db, "mango_diseases", id); // สร้างการอ้างอิงเอกสารจาก id ที่ได้จาก URL
                const docSnap = await getDoc(docRef); // ดึงข้อมูลเอกสารจาก Firestore
                if (docSnap.exists()) { // หากเอกสารพบ
                    setDisease(docSnap.data()); // ตั้งค่าข้อมูลโรคมะม่วงในสถานะ
                } else { // หากไม่พบข้อมูล
                    alert("ไม่พบข้อมูลโรคมะม่วง");
                    navigate("/mango"); // นำทางไปหน้ารายการโรคมะม่วง
                }
            } catch (error) {
                console.error("เกิดข้อผิดพลาดในการโหลดข้อมูล:", error); // แสดงข้อผิดพลาดหากเกิด
                alert("เกิดข้อผิดพลาดในการโหลดข้อมูล");
                navigate("/mango"); // นำทางไปหน้ารายการโรคมะม่วง
            } finally {
                setLoading(false); // ปิดสถานะการโหลด
            }
        };

        fetchDisease(); // เรียกฟังก์ชันเพื่อดึงข้อมูล
    }, [id, navigate]); // useEffect จะทำงานเมื่อ id หรือ navigate เปลี่ยนแปลง

    // ฟังก์ชันสำหรับลบข้อมูล
    const handleDelete = async () => {
        if (window.confirm("คุณแน่ใจหรือไม่ว่าต้องการลบข้อมูลนี้?")) { // ถามยืนยันการลบ
            try {
                await deleteDoc(doc(db, "mango_diseases", id)); // ลบเอกสารจาก Firestore
                alert("ลบข้อมูลสำเร็จ");
                navigate("/mango"); // นำทางกลับไปยังหน้ารายการโรคมะม่วง
            } catch (error) {
                console.error("เกิดข้อผิดพลาดในการลบ:", error); // แสดงข้อผิดพลาดหากเกิด
                alert("เกิดข้อผิดพลาดในการลบข้อมูล");
            }
        }
    };

    if (loading) {
        return <p>กำลังโหลดข้อมูล...</p>; // แสดงข้อความขณะโหลดข้อมูล
    }

    if (!disease) {
        return <p>ไม่พบข้อมูลโรคมะม่วง</p>; // หากไม่พบข้อมูลโรคมะม่วง
    }

    return (
        <div className="disease-detail-container">
            <h2>{disease.diseaseName}</h2> {/* แสดงชื่อโรคมะม่วง */}

            {disease.imageUrl ? (
                <img src={disease.imageUrl} alt={disease.diseaseName} className="disease-image" /> // แสดงรูปภาพโรคหากมี
            ) : (
                <p>ไม่มีรูปภาพ</p> // แสดงข้อความหากไม่มีรูปภาพ
            )}

            <p><strong>อาการ:</strong> {disease.symptoms}</p> {/* แสดงอาการ */}
            <p><strong>การรักษา:</strong> {disease.treatment}</p> {/* แสดงวิธีรักษา */}
            <p><strong>การป้องกัน:</strong> {disease.prevention}</p> {/* แสดงวิธีป้องกัน */}

            <div className="button-container">
                <button onClick={() => navigate("/Mango")} className="button-back">
                    ⬅️ ย้อนกลับ
                </button> {/* ปุ่มย้อนกลับไปหน้ารายการโรคมะม่วง */}
                <button onClick={() => navigate(`/editmango/${id}`)} className="edit-btn">แก้ไข</button> {/* ปุ่มแก้ไขข้อมูล */}
                <button onClick={handleDelete} className="delete-btn">ลบ</button> {/* ปุ่มลบข้อมูล */}
            </div>
        </div>
    );
}

export default MangoDetail;
