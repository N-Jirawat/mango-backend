import React from "react";
import "./css/UserManual.css";
import { useNavigate } from "react-router-dom";

function UserManual() {
  const navigate = useNavigate();

  const handleGoHome = () => {
    navigate('/');
  };
  return (
    <div className="user-manual-container">
      <div className="Manual-header">
        <button onClick={handleGoHome} className="back-button">
          ⬅️ หน้าหลัก
        </button>
      </div>
      <h2>📘 คู่มือการใช้งานเว็บ LeafAnalyzer</h2>

      <div className="manual-section">
        <p>🔑 สามารถเข้าใช้งานได้ทั้งแบบไม่เข้าสู่ระบบและแบบเข้าสู่ระบบ โดยแบบเข้าสู่ระบบจะสามารถเก็บผลการวิเคราะห์ไว้ในประวัติการวิเคราะห์ได้</p>
      </div>

      <div className="manual-section">
        <h2>วิธีสมัครสมาชิก</h2>
        <p>1. กดที่ปุ่ม "เข้าสู่ระบบ"</p>
        <img src="img/sign_up1.png" alt="กดเข้าสู่ระบบ" />
        <p>2. กดที่ปุ่ม "สมัครสมาชิก"</p>
        <img src="img/sign_up2.png" alt="กดสมัครสมาชิก" />
        <p>3. กรอกข้อมูลตามที่ระบบต้องการ และ กดที่ปุ่ม "ต่อไป"</p>
        <img src="img/data1.png" alt="กรอกข้อมูล1" />
        <p>4. กรอกข้อมูลเพิ่มเติมตามที่ระบบต้องการ และ กดที่ปุ่ม "บันทึก"</p>
        <img src="img/data2.png" alt="กรอกข้อมูล2" />
        <p>5. หลังจากกดที่ปุ่ม "บันทึก" ระบบจะแจ้งเตือนข้อความด้านบนว่า "สมัครสมาชิกสำเร็จ" แต่ถ้าระบบแจ้งเตือนว่า "ไม่สามารถสมัครสมาชิกได้" ให้ตรวจสอบข้อมูลที่กรอกว่าถูกต้องหรือไม่</p>
        <img src="img/saveuser.png" alt="บันทึกข้อมูลผู้ใช้ใหม่" />
      </div>

      <div className="manual-section">
        <h2>วิธีเข้าสู่ระบบ</h2>
        <p>1. กดที่ปุ่ม "เข้าสู่ระบบ"</p>
        <img src="/img/sign_up1.png" alt="เข้าสู่ระบบ" />
        <p>2. กรอก Email หรือชื่อบัญชีผู้ใช้ และ รหัสผ่าน ตามที่สมัครสมาชิกไว้ และ กดที่ปุ่ม "เข้าสู่ระบบ"</p>
        <img src="/img/sign_in1.png" alt="เข้าสู่ระบบ" />
      </div>

      <div className="manual-section">
        <h2>วิธีอัปโหลดรูปเพื่อวิเคราะห์โรคใบมะม่วง</h2>
        <p>1. กดที่ปุ่ม "อัปโหลดรูปภาพ"</p>
        <img src="/img/upload1.png" alt="อัปโหลดรูป" />
        <p>2. กดที่ปุ่ม "เลือกไฟล์"</p>
        <img src="/img/upload2.png" alt="กดเลือกไฟล์" />
        <p>3. เลือกรูปภาพในเครื่องที่ต้องการวิเคราะห์</p>
        <img src="/img/upload3.png" alt="เลือกรูป" />
        <p>4. กดที่ปุ่ม "ทำนาย" เพื่อเริ่มทำการวิเคราะห์</p>
        <img src="/img/upload4.png" alt="ทำนาย" />
        <p>5. รอดูผลลัพธ์การวิเคราะห์จากระบบ</p>
        <img src="/img/.png" alt="ดูผลลัพธ์" />
        <p>6. สามารถกดที่ปุ่ม "บันทึก" เพื่อเก็บไว้ดูในประวัติการวิเคราะห์ย้อนหลังได้"เฉพาะสมาชิกเท่านั้น"</p>
        <img src="/img/.png" alt="บันทึกผลลัพธ์" />
      </div>
    </div>
  );
}

export default UserManual;
