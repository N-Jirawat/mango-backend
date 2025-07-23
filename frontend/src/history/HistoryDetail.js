import { useLocation, useNavigate } from "react-router-dom";
import { doc, deleteDoc } from "firebase/firestore";
import { db } from "../firebaseConfig";

import pdfMake from "pdfmake/build/pdfmake";
import pdfFonts from "../PDF/vfs_fonts"; // ไฟล์ฟอนต์ Sarabun ต้องมีในโฟลเดอร์นี้
import "../css/historyDetail.css";

pdfMake.vfs = pdfFonts.pdfMake.vfs;

pdfMake.fonts = {
  Sarabun: {
    normal: "Sarabun-Bold.ttf",
    bold: "Sarabun-Bold.ttf",
    italics: "Sarabun-Bold.ttf",
    bolditalics: "Sarabun-Bold.ttf"
  }
};

function HistoryDetail() {
  const { state } = useLocation(); // รับข้อมูลจากหน้าอื่น
  const navigate = useNavigate();

  if (!state) {
    return <p>ไม่พบข้อมูลการทำนาย</p>;
  }

  const {
    diseaseName,
    confidence,
    accuracy,
    symptoms,
    prevention,
    treatment,
    timestamp,
    imageUrl,
    docId,
  } = state;

  const handleDelete = async () => {
    const confirmDelete = window.confirm("คุณแน่ใจหรือไม่ว่าต้องการลบข้อมูลนี้?");
    if (!confirmDelete) return;

    try {
      await deleteDoc(doc(db, "prediction_results", docId));
      alert("ลบข้อมูลสำเร็จ");
      navigate("/history");
    } catch (error) {
      console.error("เกิดข้อผิดพลาดในการลบ:", error);
      alert("เกิดข้อผิดพลาดในการลบข้อมูล");
    }
  };

  const handleGoHome = () => {
    navigate("/history");
  };

  const handleDownloadPDF = async () => {
    const thaiDate = timestamp?.seconds
      ? new Date(timestamp.seconds * 1000).toLocaleString("th-TH")
      : "ไม่ระบุวันที่";

    let base64Img = null;
    if (imageUrl) {
      try {
        base64Img = await getBase64FromUrl(imageUrl);
      } catch (err) {
        console.error("แปลงรูปภาพเป็น Base64 ไม่สำเร็จ:", err);
        alert("ไม่สามารถโหลดรูปภาพสำหรับ PDF ได้");
      }
    }

    const docDefinition = {
      content: [
        { text: "รายละเอียดการทำนายโรคใบมะม่วง", style: "header" },
        base64Img
          ? {
            image: base64Img,
            width: 200,
            alignment: "center",
            margin: [0, 0, 0, 10],
          }
          : null,
        { text: "ชื่อโรค:", style: "greenLabel" },
        { text: diseaseName, margin: [0, 0, 0, 10] },

        { text: "ความมั่นใจ (Confidence):", style: "greenLabel" },
        { text: `${Math.round(confidence * 100)}%`, margin: [0, 0, 0, 10] },

        { text: "ความแม่นยำ (Accuracy):", style: "greenLabel" },
        {
          text:
            typeof accuracy === "number"
              ? `${Math.round(accuracy * 100)}%`
              : "ไม่มีข้อมูล",
          margin: [0, 0, 0, 10],
        },

        { text: "รายละเอียดโรค:", style: "greenLabel" },
        { text: symptoms || "ไม่มีข้อมูลรายละเอียดโรค", margin: [0, 0, 0, 10] },

        { text: "วิธีป้องกัน:", style: "greenLabel" },
        { text: prevention || "ไม่มีข้อมูลวิธีการป้องกัน", margin: [0, 0, 0, 10] },

        { text: "วิธีการรักษา:", style: "greenLabel" },
        { text: treatment || "ไม่มีข้อมูลวิธีการรักษา", margin: [0, 0, 0, 10] },

        { text: "วันที่วิเคราะห์:", style: "greenLabel" },
        { text: thaiDate },
      ].filter(Boolean),
      defaultStyle: {
        font: "Sarabun",
        fontSize: 14,
      },
      styles: {
        header: { fontSize: 18, bold: true, margin: [0, 0, 0, 10] },
        greenLabel: { color: "green", bold: true, fontSize: 14 },
      },
    };

    pdfMake
      .createPdf(docDefinition)
      .download(
        `รายงานโรค_${diseaseName}_${new Date().toLocaleDateString("th-TH")}.pdf`
      );
  };

  // ฟังก์ชันช่วยแปลง URL เป็น Base64
  async function getBase64FromUrl(imageUrl) {
    const res = await fetch(imageUrl);
    const blob = await res.blob();
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        resolve(reader.result.toString());
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  return (
    <div className="details-container">
      <div className="analyDetail-header">
        <button onClick={handleGoHome} className="back-button">
          ⬅️ หน้าหลัก
        </button>
      </div>
      <h2>รายละเอียดการทำนายโรค</h2>

      {/* แสดงภาพจากการทำนาย */}
      {imageUrl && (
        <img src={imageUrl} alt="Uploaded" className="image-preview" />
      )}

      <div className="details-item">
        <strong>ชื่อโรค:</strong> {diseaseName}
      </div>
      <div className="details-item">
        <strong>ความมั่นใจ (confidence):</strong> {typeof confidence === 'number' ? confidence.toFixed(4) : 'ไม่มีข้อมูล'}%
      </div>
      <div className="details-item">
        <strong>ความแม่นยำ (accuracy):</strong> {typeof accuracy === 'number' ? accuracy.toFixed(4) : 'ไม่มีข้อมูล'}%
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

      <div className="details-item">
        <strong className="t">วันที่วิเคราะห์:</strong>{" "}
        {timestamp?.seconds
          ? new Date(timestamp.seconds * 1000).toLocaleString("th-TH")
          : "ไม่ระบุวันที่"}
      </div>

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
