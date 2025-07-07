import { useLocation, useNavigate } from "react-router-dom";
import { doc, deleteDoc } from "firebase/firestore";
import { db } from "../firebaseConfig";
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import "../css/historyDetail.css";

function HistoryDetail() {
  const { state } = useLocation(); // รับข้อมูลที่ส่งมาจากหน้า HistoryAnaly
  const navigate = useNavigate(); // สำหรับการนำทางไปยังหน้าต่าง ๆ

  if (!state) {
    return <p>ไม่พบข้อมูลการทำนาย</p>;
  }

  const { diseaseName, confidence, accuracy, symptoms, prevention, treatment, timestamp, imageUrl, docId } = state;

  const handleDelete = async () => {
    const confirmDelete = window.confirm("คุณแน่ใจหรือไม่ว่าต้องการลบข้อมูลนี้?");
    if (!confirmDelete) return;

    try {
      // ลบเอกสารจาก Firestore
      await deleteDoc(doc(db, "prediction_results", docId));
      alert("ลบข้อมูลสำเร็จ");
      navigate("/history"); // กลับไปที่หน้า historyanaly หลังจากลบข้อมูลสำเร็จ
    } catch (error) {
      console.error("เกิดข้อผิดพลาดในการลบ:", error);
      alert("เกิดข้อผิดพลาดในการลบข้อมูล");
    }
  };

  const handleGoHome = () => {
    navigate('/history');
  };

  // ฟังก์ชันสำหรับดาวน์โหลด PDF
  const handleDownloadPDF = async () => {
    try {
      // หา element ที่ต้องการแปลงเป็น PDF
      const element = document.querySelector('.details-container');
      if (!element) {
        alert('ไม่สามารถสร้าง PDF ได้');
        return;
      }

      // ซ่อนปุ่มต่าง ๆ ชั่วคราวเพื่อไม่ให้ปรากฏใน PDF
      const buttons = element.querySelectorAll('button');
      const originalDisplay = [];
      buttons.forEach((btn, index) => {
        originalDisplay[index] = btn.style.display;
        btn.style.display = 'none';
      });

      // เก็บค่า style เดิม
      const originalStyles = {
        fontSize: element.style.fontSize,
        fontFamily: element.style.fontFamily,
        lineHeight: element.style.lineHeight
      };

      // ปรับ style ชั่วคราวสำหรับ PDF
      element.style.fontSize = '18px';
      element.style.fontFamily = 'Arial, sans-serif';
      element.style.lineHeight = '1.8';

      // ปรับขนาดหัวข้อ
      const title = element.querySelector('h2');
      const originalTitleStyle = {};
      if (title) {
        originalTitleStyle.fontSize = title.style.fontSize;
        originalTitleStyle.fontWeight = title.style.fontWeight;
        title.style.fontSize = '24px';
        title.style.fontWeight = 'bold';
      }

      // ปรับขนาดรายละเอียด
      const detailItems = element.querySelectorAll('.details-item');
      const originalDetailStyles = [];
      detailItems.forEach((item, index) => {
        originalDetailStyles[index] = {
          fontSize: item.style.fontSize,
          marginBottom: item.style.marginBottom,
          lineHeight: item.style.lineHeight
        };
        item.style.fontSize = '14px';
        item.style.marginBottom = '20px';
        item.style.lineHeight = '1.6';

        const strong = item.querySelector('strong');
        if (strong) {
          originalDetailStyles[index].strongFontSize = strong.style.fontSize;
          strong.style.fontSize = '14px';
        }
      });

      // สร้าง canvas จาก HTML element
      const canvas = await html2canvas(element, {
        scale: 2, // เพิ่มความคมชัด
        useCORS: true, // สำหรับการโหลดรูปภาพจาก URL ภายนอก
        allowTaint: false,
        backgroundColor: '#ffffff'
      });

      // คืนค่า style เดิม
      element.style.fontSize = originalStyles.fontSize;
      element.style.fontFamily = originalStyles.fontFamily;
      element.style.lineHeight = originalStyles.lineHeight;

      if (title) {
        title.style.fontSize = originalTitleStyle.fontSize;
        title.style.fontWeight = originalTitleStyle.fontWeight;
      }

      detailItems.forEach((item, index) => {
        item.style.fontSize = originalDetailStyles[index].fontSize;
        item.style.marginBottom = originalDetailStyles[index].marginBottom;
        item.style.lineHeight = originalDetailStyles[index].lineHeight;

        const strong = item.querySelector('strong');
        if (strong && originalDetailStyles[index].strongFontSize) {
          strong.style.fontSize = originalDetailStyles[index].strongFontSize;
        }
      });

      // แสดงปุ่มกลับมา
      buttons.forEach((btn, index) => {
        btn.style.display = originalDisplay[index];
      });

      // สร้าง PDF
      const pdf = new jsPDF('p', 'mm', 'a4');
      const imgData = canvas.toDataURL('image/png');

      // คำนวณขนาดภาพให้เข้ากับหน้า A4
      // ตั้งค่าระยะขอบ (เช่น 10 มม. ทั้งซ้าย-ขวา และบน-ล่าง)
      const marginX = 10;
      const marginY = 10;
      const pageWidth = 210;
      const pageHeight = 295;
      const imgWidth = pageWidth - marginX * 2;
      const imgHeight = (canvas.height * imgWidth) / canvas.width;
      let heightLeft = imgHeight;
      let position = marginY;

      // เพิ่มหน้าแรก
      pdf.addImage(imgData, 'PNG', marginX, position, imgWidth, imgHeight);
      heightLeft -= pageHeight;

      // เพิ่มหน้าถัดไปหากเนื้อหายาวเกินไป
      while (heightLeft >= 0) {
        position = heightLeft - imgHeight + marginY;
        pdf.addPage();
        pdf.addImage(imgData, 'PNG', marginX, position, imgWidth, imgHeight);
        heightLeft -= pageHeight;
      }

      // ดาวน์โหลดไฟล์ PDF
      const fileName = `รายงานการทำนายโรค_${diseaseName}_${new Date().toLocaleDateString('th-TH')}.pdf`;
      pdf.save(fileName);

    } catch (error) {
      console.error('เกิดข้อผิดพลาดในการสร้าง PDF:', error);
      alert('เกิดข้อผิดพลาดในการสร้าง PDF');
    }
  };

  return (
    <div className="details-container">
      <div className="analyDetail-header">
        <button onClick={handleGoHome} className="back-button">
          ⬅️ หน้าหลัก
        </button>
      </div>
      <h2>รายละเอียดการทำนายโรค</h2>

      {/* แสดงภาพจากการทำนาย */}
      {imageUrl && <img src={imageUrl} alt="Uploaded" className="image-preview" />}

      <div className="details-item">
        <strong>ชื่อโรค:</strong> {diseaseName}
      </div>
      <div className="details-item">
        <strong>ความมั่นใจ (confidence):</strong> {Math.round(confidence * 100)}%
      </div>
      <div className="details-item">
        <strong>ความแม่นยำ (accuracy):</strong>{" "}
        {typeof accuracy === "number" ? `${Math.round(accuracy * 100)}%` : "ไม่มีข้อมูล"}
      </div>
      <div className="details-item">
        <strong>รายละเอียดโรค:</strong> {symptoms || "ไม่มีข้อมูลรายละเอียดโรค"}
      </div>
      <div className="details-item">
        <strong>วิธีป้องกัน:</strong> {prevention || "ไม่มีข้อมูลวิธีการป้องกัน"}
      </div>
      <div className="details-item">
        <strong>วิธีการรักษา:</strong> {treatment || "ไม่มีข้อมูลวิธีการรักษา"}
      </div>

      <div className="details-item"><strong className="t">วันที่วิเคราะห์:</strong> {
        timestamp?.seconds
          ? new Date(timestamp.seconds * 1000).toLocaleString("th-TH")
          : "ไม่ระบุวันที่"
      }</div>

      <button onClick={handleDownloadPDF} className="download-pdf-btn">
          📄 ดาวน์โหลด PDF
        </button>

      <div className="action-buttons">
        <button className="delete-btn" onClick={handleDelete}>
          🗑️ ลบข้อมูล
        </button>
      </div>
    </div>
  );
}

export default HistoryDetail;