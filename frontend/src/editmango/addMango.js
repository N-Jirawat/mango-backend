import React, { useState } from "react";
import { useNavigate } from "react-router-dom"; 
import { db } from "../firebaseConfig"; 
import { collection, addDoc, serverTimestamp } from "firebase/firestore"; 
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
        return { imageUrl: data.secure_url, public_id: data.public_id };
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
      // 1. อัปโหลดภาพไปยัง Cloudinary ก่อน
      const uploadData = await uploadToCloudinary(image);
      if (!uploadData) {
        alert("เกิดข้อผิดพลาดในการอัปโหลดภาพไปยัง Cloudinary");
        setLoading(false);
        return;
      }

      // 2. บันทึกข้อมูลใน MangoDisease collection ก่อน
      const mangoDiseaseData = {
        DiseaseName: formData.diseaseName,
        Style: formData.symptoms,
        Protection: formData.prevention,
        Treatment: formData.treatment,
        UpdateAt: serverTimestamp(),
        ImgID: "" // จะอัปเดตหลังจากสร้าง ImageMango
      };

      const mangoDiseaseRef = await addDoc(collection(db, "MangoDisease"), mangoDiseaseData);
      const diseaseId = mangoDiseaseRef.id;

      // 3. บันทึกข้อมูลรูปภาพใน ImageMango collection
      const imageMangoData = {
        ImgPath: uploadData.imageUrl,
        DateUploadImg: serverTimestamp(),
        DiseaseID: diseaseId,
        public_id: uploadData.public_id // เก็บ public_id สำหรับการลบภาพ
      };

      const imageMangoRef = await addDoc(collection(db, "ImageMango"), imageMangoData);
      const imgId = imageMangoRef.id;

      // 4. อัปเดต ImgID ใน MangoDisease
      const { updateDoc, doc } = await import("firebase/firestore");
      await updateDoc(doc(db, "MangoDisease", diseaseId), {
        ImgID: imgId
      });

      alert("อัปโหลดสำเร็จ!");
      navigate("/mango"); 
      
      // รีเซ็ตฟอร์ม
      setImage(null); 
      setImagePreview(""); 
      setFormData({
        diseaseName: "",
        symptoms: "",
        treatment: "",
        prevention: "",
      });
    } catch (error) {
      console.error("Error uploading data:", error);
      alert("เกิดข้อผิดพลาดในการบันทึกข้อมูล");
    } finally {
      setLoading(false);
    }
  };

  // ฟังก์ชันสำหรับ render ช่องกรอกข้อมูล
  const renderFormField = (key) => {
    const labels = {
      diseaseName: "ชื่อโรค:",
      symptoms: "ลักษณะอาการ:",
      treatment: "วิธีรักษา:",
      prevention: "วิธีป้องกัน:"
    };

    if (key === "diseaseName") {
      return (
        <div key={key}>
          <label>{labels[key]}</label>
          <input
            type="text"
            name={key}
            value={formData[key]}
            onChange={handleChange}
            required
            style={{
              width: '100%',
              padding: '8px',
              fontSize: '14px',
              borderRadius: '4px',
              border: '1px solid #ddd'
            }}
          />
        </div>
      );
    } else {
      return (
        <div key={key}>
          <label>{labels[key]}</label>
          <textarea
            name={key}
            value={formData[key]}
            onChange={handleChange}
            required
            rows={4}
            style={{
              resize: 'vertical',
              width: '100%',
              padding: '8px',
              fontSize: '14px',
              borderRadius: '4px',
              border: '1px solid #ddd'
            }}
          />
        </div>
      );
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
        {Object.keys(formData).map((key) => renderFormField(key))}
        
        <label>อัปโหลดรูปภาพ:</label>
        <input 
          type="file" 
          accept="image/*" 
          onChange={handleImageChange} 
          required 
        />
        {imagePreview && (
          <img 
            src={imagePreview} 
            alt="ตัวอย่าง" 
            className="image-preview"
          />
        )}
        <div className="button-container-addmango">
          <button type="submit" disabled={loading}>
            {loading ? "กำลังบันทึก..." : "บันทึก"}
          </button>
        </div>
      </form>
    </div>
  );
}

export default AddMango;