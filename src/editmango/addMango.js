import React, { useState } from "react";
import { useNavigate } from "react-router-dom"; 
import { db } from "../firebaseConfig"; 
import { collection, addDoc } from "firebase/firestore"; 
import "../css/addmango.css";

function AddMango() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    diseaseName: "",
    symptoms: "",
    treatment: "",
    prevention: "",
  });
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState("");
  const [loading, setLoading] = useState(false);

  // ฟังก์ชันสำหรับอัปโหลดไฟล์ภาพไปยัง Cloudinary
  const uploadToCloudinary = async (file) => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("upload_preset", "ml_default");
    formData.append("folder", "mango_diseases");
    formData.append("cloud_name", "dsf25dlca");

    try {
      const response = await fetch("https://api.cloudinary.com/v1_1/dsf25dlca/image/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (data.secure_url && data.public_id) {
        return { imageUrl: data.secure_url, public_id: data.public_id }; // ส่งกลับ URL และ public_id
      }
      throw new Error("Upload failed");
    } catch (error) {
      console.error("Upload failed:", error);
      return null;
    }
  };

  // ฟังก์ชันเมื่อผู้ใช้เลือกไฟล์ภาพ
  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.type.startsWith("image/")) {
        alert("กรุณาเลือกไฟล์รูปภาพเท่านั้น!");
        return;
      }
      if (file.size > 5 * 1024 * 1024) {
        alert("ไฟล์มีขนาดใหญ่เกินไป! (จำกัด 5MB)");
        return;
      }
      setImage(file);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  // ฟังก์ชันในการอัปเดตข้อมูลใน formData
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  // ฟังก์ชันที่ทำงานเมื่อผู้ใช้กดปุ่ม "บันทึก"
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    if (!image) {
      alert("กรุณาเลือกไฟล์รูปภาพ!");
      setLoading(false);
      return;
    }

    try {
      // เรียกฟังก์ชันอัปโหลดภาพไปยัง Cloudinary
      const uploadData = await uploadToCloudinary(image);
      if (!uploadData) {
        alert("เกิดข้อผิดพลาดในการอัปโหลดภาพไปยัง Cloudinary");
        setLoading(false);
        return;
      }

      // ส่งข้อมูลทั้งหมด (รวมถึง URL และ public_id ของภาพ) ไปยัง Firestore
      await addDoc(collection(db, "mango_diseases"), {
        ...formData,
        imageUrl: uploadData.imageUrl,
        imagePublicId: uploadData.public_id, // เก็บ public_id
      });

      alert("อัปโหลดสำเร็จ!");
      navigate("/mango"); 
      setImage(null); 
      setImagePreview(""); 
      setFormData({
        diseaseName: "",
        symptoms: "",
        treatment: "",
        prevention: "",
      });
    } catch (error) {
      console.error("Error uploading image:", error);
      alert("เกิดข้อผิดพลาดในการบันทึกข้อมูล");
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="disease-detail-container">
      <div className="addmango-header">
        <button onClick={() => navigate("/mango")} className="back-button">
          ⬅️ หน้าหลัก
        </button>
      </div>
      <h2>เพิ่มข้อมูลโรคมะม่วง</h2>
      <form onSubmit={handleSubmit} className="boxmango">
        {Object.keys(formData).map((key) => (
          <div key={key}>
            <label>{key === "diseaseName" ? "ชื่อโรค:" : key === "symptoms" ? "ลักษณะอาการ:" : key === "treatment" ? "วิธีรักษา:" : "วิธีป้องกัน:"}</label>
            <input type="text" name={key} value={formData[key]} onChange={handleChange} required />
          </div>
        ))}
        <label>อัปโหลดรูปภาพ:</label>
        <input type="file" accept="image/*" onChange={handleImageChange} required />
        {imagePreview && <img src={imagePreview} alt="ตัวอย่าง" style={{ width: "200px", marginTop: "10px" }} />}
        <div className="button-container-addmango">
          <button type="submit" disabled={loading}>{loading ? "กำลังบันทึก..." : "บันทึก"}</button>
        </div>
      </form>
    </div>
  );
}

export default AddMango;
