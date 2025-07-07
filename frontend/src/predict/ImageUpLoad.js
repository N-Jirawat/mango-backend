import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import "../css/imageupload.css";

function ImageUpload({ setPredictionResult }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isUploaded, setIsUploaded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const navigate = useNavigate();

  // เมื่อเลือกไฟล์
  const handleFileChange = (e) => {
    const selected = e.target.files?.[0];

    // ล้างค่าเก่าก่อน
    if (preview) {
      URL.revokeObjectURL(preview);
    }

    if (!selected) {
      setFile(null);
      setPreview(null);
      setIsUploaded(false);
      return;
    }

    // ตรวจสอบว่าเป็นไฟล์รูปภาพหรือไม่
    if (!selected.type.startsWith('image/')) {
      alert("กรุณาเลือกไฟล์รูปภาพเท่านั้น");
      return;
    }

    // ตรวจสอบขนาดไฟล์ (เช่น ไม่เกิน 10MB)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (selected.size > maxSize) {
      alert("ขนาดไฟล์ใหญ่เกินไป กรุณาเลือกไฟล์ที่มีขนาดไม่เกิน 10MB");
      return;
    }

    try {
      const objectUrl = URL.createObjectURL(selected);
      setFile(selected);
      setPreview(objectUrl);
      setIsUploaded(true);
      setPredictionResult(null);
      setError(null);
    } catch (error) {
      console.error("Error creating object URL:", error);
      alert("เกิดข้อผิดพลาดในการโหลดภาพ กรุณาลองใหม่");
    }
  };

  // เมื่อกดปุ่มทำนาย
  const handleUpload = async () => {
    if (!file) {
      alert('กรุณาเลือกภาพก่อน');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch('https://mango-backend-665966382004.asia-southeast1.run.app/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('ไม่สามารถทำนายได้');
      }

      const data = await response.json();

      if (data.prediction && data.confidence >= 0.5) {
        setPredictionResult(data.prediction);

        // ส่งผลลัพธ์ไปที่หน้า ResultAnaly
        navigate('/resultanaly', {
          state: {
            prediction: data.prediction,
            confidence: data.confidence,
            accuracy: data.accuracy,
            imagePreview: preview,
            imageFile: file, // ✅ เพิ่ม File object ที่ต้องการ
          },
        });
      } else {
        setError(`ไม่พบข้อมูล (ความมั่นใจ: ${Math.round(data.confidence * 100)}%)`);
      }
    } catch (err) {
      console.error('Error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // ล้าง object URL เมื่อ component unmount
  useEffect(() => {
    return () => {
      if (preview && preview.startsWith('blob:')) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  return (
    <div className="container">
      <h2 className="title">อัปโหลดภาพโรคมะม่วง</h2>
      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="file-input"
      />
      <p className='warning'>คำแนะนำ : ควรเป็นภาพของใบมะม่วงที่มีลักษณะโรคชัดเจน</p>
      {preview && (
        <div className="preview-container">
          <img src={preview} alt="Preview" className="preview-image" />
        </div>
      )}
      {isUploaded && (
        <button
          onClick={handleUpload}
          className="button"
          disabled={loading}
        >
          {loading ? 'กำลังทำนาย...' : 'ทำนาย'}
        </button>
      )}
      {error && <p className="error">{error}</p>}
    </div>
  );
}

export default ImageUpload;