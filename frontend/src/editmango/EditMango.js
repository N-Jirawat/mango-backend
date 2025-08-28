import React, { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { db } from "../firebaseConfig";
import { doc, getDoc, updateDoc, deleteDoc, collection, query, where, getDocs, serverTimestamp } from "firebase/firestore";
import "../css/editmango.css";

function EditMango() {
  const navigate = useNavigate();
  const { id } = useParams(); // DiseaseID
  const [formData, setFormData] = useState({
    DiseaseName: "",
    Style: "",
    Treatment: "",
    Protection: "",
  });
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState("");
  const [loading, setLoading] = useState(false);
  const [imageData, setImageData] = useState(null); // เก็บข้อมูลรูปภาพ
  const [isDeleting, setIsDeleting] = useState(false);
  
  //const BACKEND_URL = "https://mango-backend-665966382004.asia-southeast1.run.app";
  const BACKEND_URL = "http://localhost:5000";

  useEffect(() => {
    const fetchMangoData = async () => {
      try {
        // 1. ดึงข้อมูลจาก MangoDisease
        const mangoDiseaseRef = doc(db, "MangoDisease", id);
        const mangoDiseaseDoc = await getDoc(mangoDiseaseRef);
        
        if (mangoDiseaseDoc.exists()) {
          const mangoDiseaseData = mangoDiseaseDoc.data();
          setFormData({
            DiseaseName: mangoDiseaseData.DiseaseName || "",
            Style: mangoDiseaseData.Style || "",
            Treatment: mangoDiseaseData.Treatment || "",
            Protection: mangoDiseaseData.Protection || "",
          });

          // 2. ดึงข้อมูลรูปภาพจาก ImageMango
          const imageQuery = query(
            collection(db, "ImageMango"), 
            where("DiseaseID", "==", id)
          );
          const imageSnapshot = await getDocs(imageQuery);
          
          if (!imageSnapshot.empty) {
            const imageDoc = imageSnapshot.docs[0]; // หาเอาภาพแรก
            const imageInfo = {
              id: imageDoc.id,
              ...imageDoc.data()
            };
            setImageData(imageInfo);
            setImagePreview(imageInfo.ImgPath);
          }
        } else {
          console.log("No such document!");
          alert("ไม่พบข้อมูลที่ต้องการแก้ไข");
          navigate("/mango");
        }
      } catch (error) {
        console.error("Error fetching data:", error);
        alert("เกิดข้อผิดพลาดในการโหลดข้อมูล");
      }
    };

    fetchMangoData();
  }, [id, navigate]);

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
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (isDeleting) {
      console.log("กำลังลบข้อมูล ข้ามการอัปเดต");
      return;
    }

    setLoading(true);

    try {
      // 1. อัปเดตข้อมูลใน MangoDisease
      const mangoDiseaseRef = doc(db, "MangoDisease", id);
      await updateDoc(mangoDiseaseRef, {
        ...formData,
        UpdateAt: serverTimestamp()
      });

      // 2. จัดการรูปภาพ (ถ้ามีการเปลี่ยนแปลง)
      if (image) {
        let newImageUrl = "";
        let newPublicId = "";

        // อัปโหลดรูปใหม่
        const uploadedData = await uploadToCloudinary(image);
        if (uploadedData) {
          newImageUrl = uploadedData.imageUrl;
          newPublicId = uploadedData.public_id;
        } else {
          alert("เกิดข้อผิดพลาดในการอัปโหลดภาพ");
          setLoading(false);
          return;
        }

        // ลบรูปเก่าจาก Cloudinary (ถ้ามี)
        if (imageData && imageData.public_id) {
          await deleteFromCloudinary(imageData.public_id);
        }

        // อัปเดตข้อมูลรูปใน ImageMango
        if (imageData && imageData.id) {
          const imageRef = doc(db, "ImageMango", imageData.id);
          await updateDoc(imageRef, {
            ImgPath: newImageUrl,
            DateUploadImg: serverTimestamp(),
            public_id: newPublicId
          });
        } else {
          // สร้างรายการรูปใหม่ถ้าไม่มี
          const { addDoc } = await import("firebase/firestore");
          const newImageRef = await addDoc(collection(db, "ImageMango"), {
            ImgPath: newImageUrl,
            DateUploadImg: serverTimestamp(),
            DiseaseID: id,
            public_id: newPublicId
          });

          // อัปเดต ImgID ใน MangoDisease
          await updateDoc(mangoDiseaseRef, {
            ImgID: newImageRef.id
          });
        }
      }

      alert("อัปเดตข้อมูลสำเร็จ!");
      navigate("/mango");

    } catch (error) {
      console.error("Error updating data:", error);
      alert("เกิดข้อผิดพลาดในการบันทึกข้อมูล");
    } finally {
      setLoading(false);
    }
  };

  // ฟังก์ชันลบข้อมูล
  const handleDelete = async (e) => {
    e.preventDefault();
    e.stopPropagation();

    const confirmation = window.confirm("คุณแน่ใจว่าต้องการลบข้อมูลนี้?");

    if (confirmation) {
      setIsDeleting(true);

      try {
        // 1. ลบรูปจาก Cloudinary
        if (imageData && imageData.public_id) {
          await deleteFromCloudinary(imageData.public_id);
        }

        // 2. ลบข้อมูลรูปจาก ImageMango collection
        if (imageData && imageData.id) {
          const imageRef = doc(db, "ImageMango", imageData.id);
          await deleteDoc(imageRef);
        }

        // 3. ลบข้อมูลจาก MangoDisease collection
        const mangoDiseaseRef = doc(db, "MangoDisease", id);
        await deleteDoc(mangoDiseaseRef);

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
            name="DiseaseName"
            value={formData.DiseaseName}
            onChange={handleChange}
            required
          />
        </div>

        <div>
          <label>ลักษณะอาการ:</label>
          <input
            type="text"
            name="Style"
            value={formData.Style}
            onChange={handleChange}
            required
          />
        </div>

        <div>
          <label>วิธีรักษา:</label>
          <input
            type="text"
            name="Treatment"
            value={formData.Treatment}
            onChange={handleChange}
            required
          />
        </div>

        <div>
          <label>วิธีป้องกัน:</label>
          <input
            type="text"
            name="Protection"
            value={formData.Protection}
            onChange={handleChange}
            required
          />
        </div>

        <div>
          <label>อัปโหลดรูปภาพ:</label>
          <input type="file" accept="image/*" onChange={handleImageChange} />
          {imagePreview && (
            <img 
              src={imagePreview} 
              alt="ตัวอย่าง" 
              style={{ width: "200px", marginTop: "10px" }} 
            />
          )}
        </div>

        <div className="button-container">
          <button type="button" onClick={() => navigate("/mango")}>
            ⬅️ ย้อนกลับ
          </button>
          <button type="submit" disabled={loading}>
            {loading ? "กำลังบันทึก..." : "บันทึก"}
          </button>
          <button
            className="delete-button"
            type="button"
            onClick={(e) => handleDelete(e)}
            disabled={isDeleting}
          >
            {isDeleting ? "กำลังลบ..." : "ลบข้อมูล"}
          </button>
        </div>
      </form>
    </div>
  );
}

export default EditMango;