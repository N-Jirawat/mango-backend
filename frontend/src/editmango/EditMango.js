import React, { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { db } from "../firebaseConfig";
import { doc, getDoc, updateDoc, deleteDoc } from "firebase/firestore";
import "../css/editmango.css";

function EditMango() {
  const navigate = useNavigate();
  const { id } = useParams();
  const [formData, setFormData] = useState({
    diseaseName: "",
    symptoms: "",
    treatment: "",
    prevention: "",
  });
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState("");
  const [loading, setLoading] = useState(false);
  const [imagePublicId, setImagePublicId] = useState("");
  const [isDeleting, setIsDeleting] = useState(false); // สถานะการลบข้อมูล
  const BACKEND_URL = "https://mango-app-465207.as.r.appspot.com";

  useEffect(() => {
    const fetchMangoData = async () => {
      const mangoRef = doc(db, "mango_diseases", id);
      const mangoDoc = await getDoc(mangoRef);
      if (mangoDoc.exists()) {
        setFormData(mangoDoc.data());
        setImagePreview(mangoDoc.data().imageUrl);
        setImagePublicId(mangoDoc.data().imagePublicId || "");
      } else {
        console.log("No such document!");
      }
    };

    fetchMangoData();
  }, [id]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

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

  const uploadToCloudinary = async (file) => {
    const formData = new FormData();
    formData.append("image", file);
    formData.append("folder", "mango_diseases");

    try {
      const response = await fetch(`${BACKEND_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      return data;
    } catch (error) {
      console.error("Upload failed:", error);
      return null;
    }
  };

  const deleteFromCloudinary = async (publicId) => {
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

  // ฟังก์ชันบันทึกข้อมูล
  // ฟังก์ชันบันทึกข้อมูล
  const handleSubmit = async (e) => {
    e.preventDefault();

    // ข้ามการอัปเดตถ้ากำลังลบข้อมูล
    if (isDeleting) {
      console.log("กำลังลบข้อมูล ข้ามการอัปเดต");
      return;
    }

    setLoading(true);

    try {
      let imageUrl = formData.imageUrl;
      let imagePublicId = formData.imagePublicId;

      if (image) {
        if (imagePublicId) {
          await deleteFromCloudinary(imagePublicId); // ลบภาพเก่าใน Cloudinary
        }

        const uploadedData = await uploadToCloudinary(image);
        if (uploadedData) {
          imageUrl = uploadedData.imageUrl;
          imagePublicId = uploadedData.public_id;
        } else {
          alert("เกิดข้อผิดพลาดในการอัปโหลดภาพ");
          setLoading(false);
          return;
        }
      }

      const mangoRef = doc(db, "mango_diseases", id);
      await updateDoc(mangoRef, { ...formData, imageUrl, imagePublicId });

      alert("อัปเดตข้อมูลสำเร็จ!");
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
      console.error("Error updating data:", error);
      alert("เกิดข้อผิดพลาดในการบันทึกข้อมูล");
    } finally {
      setLoading(false);
    }
  };

  // ฟังก์ชันลบข้อมูล
  const handleDelete = async (e) => {
    e.preventDefault(); // ป้องกัน form submit
    e.stopPropagation(); // ป้องกันการ bubble ไปยัง form

    const confirmation = window.confirm("คุณแน่ใจว่าต้องการลบข้อมูลนี้?");

    if (confirmation) {
      setIsDeleting(true);

      try {
        if (imagePublicId) {
          await deleteFromCloudinary(imagePublicId);
        }

        const mangoRef = doc(db, "mango_diseases", id);
        await deleteDoc(mangoRef);

        alert("ลบข้อมูลสำเร็จ!");
        navigate("/mango");
      } catch (error) {
        console.error("Error deleting document:", error);
        alert("เกิดข้อผิดพลาดในการลบข้อมูล");
      } finally {
        setIsDeleting(false);
      }
    } else {
      console.log("User cancelled delete operation.");
      setIsDeleting(false);
    }
  };

  return (
    <div className="disease-detail-container">
      <h3>แก้ไขข้อมูลโรคมะม่วง</h3>
      <form onSubmit={handleSubmit} className="boxmango">
        <div>
          <label>ชื่อโรค:</label>
          <input
            type="text"
            name="diseaseName"
            value={formData.diseaseName}
            onChange={handleChange}
            required
          />
        </div>

        <div>
          <label>ลักษณะอาการ:</label>
          <input
            type="text"
            name="symptoms"
            value={formData.symptoms}
            onChange={handleChange}
            required
          />
        </div>

        <div>
          <label>วิธีรักษา:</label>
          <input
            type="text"
            name="treatment"
            value={formData.treatment}
            onChange={handleChange}
            required
          />
        </div>

        <div>
          <label>วิธีป้องกัน:</label>
          <input
            type="text"
            name="prevention"
            value={formData.prevention}
            onChange={handleChange}
            required
          />
        </div>

        <div>
          <label>อัปโหลดรูปภาพ:</label>
          <input type="file" accept="image/*" onChange={handleImageChange} />
          {imagePreview && <img src={imagePreview} alt="ตัวอย่าง" style={{ width: "200px", marginTop: "10px" }} />}
        </div>

        <div className="button-container">
          <button type="button" onClick={() => navigate("/mango")}>⬅️ ย้อนกลับ</button>
          <button type="submit" disabled={loading}>
            {loading ? "กำลังบันทึก..." : "บันทึก"}
          </button>
          <button
            className="delete-button"
            type="button"
            onClick={(e) => handleDelete(e)}
          >
            ลบข้อมูล
          </button>

        </div>
      </form>
    </div>
  );
}

export default EditMango;
