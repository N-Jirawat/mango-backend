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
      setError("กรุณาเลือกไฟล์รูปภาพเท่านั้น");
      return;
    }

    // ตรวจสอบขนาดไฟล์ (เช่น ไม่เกิน 5MB เพื่อประสิทธิภาพที่ดีกว่า)
    const maxSize = 5 * 1024 * 1024; // 5MB
    if (selected.size > maxSize) {
      setError("ขนาดไฟล์ใหญ่เกินไป กรุณาเลือกไฟล์ที่มีขนาดไม่เกิน 5MB");
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
      setError("เกิดข้อผิดพลาดในการโหลดภาพ กรุณาลองใหม่");
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('กรุณาเลือกภาพก่อน');
      return;
    }

    setLoading(true);
    setError(null);

    // สร้าง FormData และตรวจสอบขนาดอีกครั้ง
    const formData = new FormData();
    formData.append('image', file);

    // เพิ่ม timeout และ retry logic
    const fetchWithTimeout = async (url, options, timeout = 30000) => {
      const controller = new AbortController();
      const id = setTimeout(() => controller.abort(), timeout);
      
      try {
        const response = await fetch(url, {
          ...options,
          signal: controller.signal
        });
        clearTimeout(id);
        return response;
      } catch (error) {
        clearTimeout(id);
        throw error;
      }
    };

    const maxRetries = 3;
    let lastError;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        console.log(`การพยายามครั้งที่ ${attempt}/${maxRetries}`);
        
        const response = await fetchWithTimeout(
          'https://mango-backend-665966382004.asia-southeast1.run.app/predict',
          {
            method: 'POST',
            body: formData,
            headers: {
              // ไม่ต้องกำหนด Content-Type เพราะ FormData จะทำให้อัตโนมัติ
            }
          },
          30000 // 30 วินาที timeout
        );

        // ตรวจสอบ response status อย่างละเอียด
        if (!response.ok) {
          const errorText = await response.text();
          console.error(`HTTP ${response.status}: ${errorText}`);
          
          if (response.status === 400) {
            throw new Error('ข้อมูลที่ส่งไม่ถูกต้อง กรุณาตรวจสอบรูปภาพอีกครั้ง');
          } else if (response.status === 413) {
            throw new Error('ไฟล์มีขนาดใหญ่เกินไป กรุณาลดขนาดรูปภาพ');
          } else if (response.status === 500) {
            throw new Error('เซิร์ฟเวอร์เกิดข้อผิดพลาด กรุณาลองใหม่ภายหลัง');
          } else if (response.status === 503) {
            throw new Error('บริการไม่พร้อมใช้งานขณะนี้ กรุณาลองใหม่ภายหลัง');
          } else {
            throw new Error(`เกิดข้อผิดพลาด (${response.status}): ${errorText}`);
          }
        }

        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
          throw new Error('การตอบสนองจากเซิร์ฟเวอร์ไม่ถูกต้อง');
        }

        const data = await response.json();
        console.log('Response data:', data);

        // ตรวจสอบโครงสร้างข้อมูลที่ได้รับ
        if (!data || typeof data.confidence === 'undefined') {
          throw new Error('ข้อมูลที่ได้รับจากเซิร์ฟเวอร์ไม่สมบูรณ์');
        }

        if (data.prediction && data.confidence >= 0.5) {
          const result = {
            prediction: data.prediction,
            confidence: Number((data.confidence * 100).toFixed(2)),
          };
          
          setPredictionResult(result);

          navigate('/resultanaly', {
            state: {
              prediction: data.prediction,
              confidence: Number((data.confidence * 100).toFixed(2)),
              imagePreview: preview,
              imageFile: file,
            },
          });
          
          setLoading(false);
          return; // สำเร็จ ออกจาก loop
          
        } else {
          const confidencePercent = (data.confidence * 100).toFixed(2);
          setError(`ไม่สามารถระบุโรคได้อย่างแน่นอน ความมั่นใจ: ${confidencePercent}% กรุณาถ่ายภาพใบมะม่วงที่มีอาการโรคชัดเจนมากขึ้น`);
          setLoading(false);
          return;
        }

      } catch (err) {
        lastError = err;
        console.error(`ความพยายามครั้งที่ ${attempt} ล้มเหลว:`, err);
        
        if (err.name === 'AbortError') {
          lastError = new Error('การเชื่อมต่อใช้เวลานานเกินไป กรุณาลองใหม่');
        }
        
        // ถ้ายังไม่ใช่ครั้งสุดท้าย ให้รอสักพักก่อนลองใหม่
        if (attempt < maxRetries) {
          await new Promise(resolve => setTimeout(resolve, 2000 * attempt)); // รอ 2, 4, 6 วินาที
        }
      }
    }

    // ถ้าทุกครั้งล้มเหลว
    setError(lastError?.message || 'เกิดข้อผิดพลาดไม่ทราบสาเหตุ กรุณาลองใหม่');
    setLoading(false);
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
      <p className='warning'>
        คำแนะนำ: ควรเป็นภาพของใบมะม่วงที่มีลักษณะโรคชัดเจน 
        (ขนาดไฟล์ไม่เกิน 5MB, รองรับ JPG, PNG, WebP)
      </p>
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
          {loading ? 'กำลังวินิจฉัยโรค...' : 'วินิจฉัย'}
        </button>
      )}
      {error && <p className="error">{error}</p>}
    </div>
  );
}

export default ImageUpload;