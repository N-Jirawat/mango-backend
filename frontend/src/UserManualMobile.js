import React from "react";
import "./css/UserManual.css";
import { useNavigate } from "react-router-dom";

function UserManualMobile() {
  const navigate = useNavigate();

  return (
    <div className="user-manual-container">
      <div className="Manual-header">
        <button onClick={() => navigate("/usermanual")} className="back-button">
          ⬅ กลับไปเลือกคู่มือ
        </button>
      </div>
      <h2>📱 คู่มือการใช้งานเว็บ LeafAnalyzer (มือถือ)</h2>

      <h2>วิธีการสมัครสมาชิก</h2>
      <div className="manual-step">
        <p>1. หน้าหลัก</p>
        <img src="/UserManualMobile/Main2.jpg" alt="หน้าหลัก" className="manual-img" />
      </div>

      <div className="manual-step">
        <p>2. คลิก "เข้าสู่ระบบ" ที่มุมขวาบนของหน้าจอ</p>
        <img src="/UserManualMobile/Main2profileedit.png" alt="คลิกเข้าสู่ระบบ" className="manual-img" />
      </div>

      <div className="manual-step">
        <p>3. คลิก "สมัครสมาชิก" เพื่อกรอกข้อมูล</p>
        <img src="/UserManualMobile/Mainsingup2edit.png" alt="คลิกสมัครสมาชิก" className="manual-img" />
      </div>

      <div className="manual-step">
        <p>4. กรอกข้อมูลตามที่ระบบต้องการและคลิก "ต่อไป" เพื่อกรอกข้อมูลเพิ่มเติม</p>
        <img src="/UserManualMobile/Singup2_1edit.png" alt="คลิกต่อไป" className="manual-img" />
      </div>

      <div className="manual-step">
        <p>5. กรอกข้อมูลเพิ่มเติมตามที่ระบบต้องการและคลิก "บันทึก" เพื่อสมัครสมาชิก</p>
        <img src="/UserManualMobile/Singup2_2edit.png" alt="คลิกบันทึก" className="manual-img" />
      </div>

      <div className="manual-step">
        <p>6. หลังจากคลิก "บันทึก" ระบบจะแสดงข้อความ "สมัครสมาชิกสำเร็จ!" และกลับสู่หน้าเข้าสู่ระบบ</p>
        <img src="/UserManualMobile/Singup2done.jpg" alt="สมัครเร็จ" className="manual-img" />
        <p>***หากคลิก "บันทึก" แล้ว ระบบแสดงข้อความแจ้งเตือนข้อผิดพลาด โปรดตรวจสอบให้แน่ใจว่ากรอกข้อมูลถูกต้องและครบถ้วนหรือไม่***</p>
      </div>

      <h2>วิธีการเข้าสู่ระบบ</h2>
      <div className="manual-step">
        <p>1. หน้าหลัก</p>
        <img src="/UserManualMobile/Main2.jpg" alt="หน้าหลักเข้าระบบ" className="manual-img" />
      </div>

      <div className="manual-step">
        <p>2. คลิก "เข้าสู่ระบบ" ที่มุมขวาบนของหน้าจอ</p>
        <img src="/UserManualMobile/Main2profileedit.png" alt="คลิกเข้าสู่ระบบ" className="manual-img" />
      </div>

      <div className="manual-step">
        <p>3. กรอก ชื่อผู้ใช้หรืออีเมล และ รหัสผ่าน ตามที่สมัครไว้กับระบบ และคลิก "เข้าสู่ระบบ"</p>
        <img src="/UserManualMobile/Mainsingin2edit.png" alt="กรอกเพื่อเข้าสู่ระบบ" className="manual-img" />
        <p>***หากคลิก "เข้าสู่ระบบ" แล้ว ระบบแจ้งเตือนรหัสไม่ถูกต้อง โปรดตรวจสอบ ชื่อผู้ใช้หรืออีเมล และ รหัสผ่าน ให้ถูกต้อง***</p>
      </div>

      <div className="manual-step">
        <p>4. เข้าสู่ระบบสำเร็จ</p>
        <img src="/UserManualMobile/Singin2done.jpg" alt="กรอกเข้าระบบเร็จ" className="manual-img" />
      </div>

      <h2>วิธีวิเคราะห์โรคใบมะม่วง</h2>
      <div className="manual-step">
        <p>1. หน้าหลัก</p>
        <img src="/UserManualMobile/Singin2done.jpg" alt="หน้าหลักก่อนวิเคราะห์" className="manual-img" />
      </div>

      <div className="manual-step">
        <p>2. คลิกปุ่ม 3 ขีดด้านซ้ายของหน้าจอ</p>
        <img src="/UserManualMobile/Singin2donechooseedit.png" alt="คลิกหน้าวิเคราะห์" className="manual-img" />
      </div>

      <div className="manual-step">
        <p>3. ระบบจะแสดงเมนูต่างๆ ให้คลิก "วิเคราะห์โรค" เพื่อเข้าสู่หน้าวิเคราะห์โรค</p>
        <img src="/UserManualMobile/Showmenu2edit.png" alt="หน้าวิเคราะห์" className="manual-img" />
      </div>

      <div className="manual-step">
        <p>4. หน้าวิเคราะห์โรค</p>
        <img src="/UserManualMobile/Mainanalyze2.png" alt="คลิกเลือกรูป" className="manual-img" />
      </div>

      <div className="manual-step">
        <p>5. หลังจากเข้าสู่หน้าวิเคราะห์โรคแล้วให้คลิก "เลือกไฟล์" เพื่ออัปโหลดรูปจากในเครื่อง</p>
        <img src="/UserManualMobile/Mainanalyze2edit.png" alt="คลิกเลือกรูป" className="manual-img" />
      </div>

      <div className="manual-step">
        <p>6. เลือกรูปจากในเครื่องเพื่ออัปโหลดเตรียมวิเคราะห์</p>
        <img src="/UserManualMobile/Choose2img.png" alt="เลือกรูป" className="manual-img" />
      </div>

      <div className="manual-step">
        <p>7. คลิก "วินิจฉัย" เพื่อรอระบบวินิจฉัยโรคที่เกิดกับใบมะม่วง</p>
        <img src="/UserManualMobile/Diagnose2edit.png" alt="วินิจฉัย" className="manual-img" />
      </div>

      <div className="manual-step">
        <p>8. ระบบแสดงผลลัพธ์การวินิจฉัยโรคที่เกิดขึ้นกับใบมะม่วง</p>
        <img src="/UserManualMobile/Show2result.png" alt="ผลวินิจฉัย" className="manual-img" />
      </div>

      <div className="manual-step">
        <p>9. สามารถบันทึกผลลัพธ์การวินิจฉัยไว้ดูภายหลังได้ โดยคลิก "บันทึกข้อมูล"</p>
        <p>***การบันทึกผลลัพธ์การวินิจฉัยสามารถทำได้โดยผู้ใช้ที่สมัครสมาชิกเท่านั้น***</p>
        <img src="/UserManualMobile/Save2Showresultedit.png" alt="บันทึกไว้ดู" className="manual-img" />
      </div>

      <div className="manual-step">
        <p>10. ระบบแจ้งเตือนการบันทึกผลลัพธ์การวินิจฉัยสำเร็จและดูผลลัพธ์ที่บันทึกไว้ที่ "ประวัติการวิเคราะห์"</p>
        <img src="/UserManualMobile/Message2notification.png" alt="แจ้งเตือนบันทึก" className="manual-img" />
      </div>
    </div>
  );
}

export default UserManualMobile;
