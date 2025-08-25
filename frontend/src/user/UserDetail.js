import React, { useEffect, useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { getDoc, doc, updateDoc } from "firebase/firestore";
import { db } from "../firebaseConfig";
import { getAuth, onAuthStateChanged } from "firebase/auth";
import "../css/UserDetails.css";

function UserDetails() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [userInfo, setUserInfo] = useState(null);
  const [role, setRole] = useState(null);
  const [loading, setLoading] = useState(true);
  const [editUser, setEditUser] = useState(false);

  const [phoneError, setPhoneError] = useState("");
  const [emailError, setEmailError] = useState("");

  // Province/District/Subdistrict data - แสดงค่าที่มีใน database ด้วย
  const [provinces] = useState([]);
  const [districtList] = useState([]);
  const [subdistrictList] = useState([]);

  const [formData, setFormData] = useState({
    fullName: "",
    address: "",
    village: "",
    province: "",
    district: "",
    subdistrict: "",
    tel: "",
    email: "",
  });

  const validateEmail = (email) => {
    const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return regex.test(email);
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmitUpdate = async () => {
    // ตรวจสอบเบอร์โทร
    const telRegex = /^[0-9]{10}$/;
    if (!telRegex.test(formData.tel)) {
      setPhoneError("กรุณากรอกเบอร์โทรให้ถูกต้อง 10 หลัก");
      return;
    } else {
      setPhoneError("");
    }

    // ตรวจสอบอีเมล
    if (!validateEmail(formData.email)) {
      setEmailError("รูปแบบอีเมลไม่ถูกต้อง");
      return;
    } else {
      setEmailError("");
    }

    try {
      // อัปเดตข้อมูลใน Firebase
      await updateDoc(doc(db, "users", id), formData);
      
      // อัปเดต userInfo state
      setUserInfo(prev => ({
        ...prev,
        ...formData
      }));
      
      // ปิด modal
      setEditUser(false);
      
      alert("อัปเดตข้อมูลสำเร็จ");
    } catch (error) {
      console.error("Error updating user:", error);
      alert("เกิดข้อผิดพลาดในการอัปเดตข้อมูล");
    }
  };

  const openEditModal = () => {
    // เติมข้อมูลปัจจุบันลงในฟอร์ม
    setFormData({
      fullName: userInfo.fullName || "",
      address: userInfo.address || "",
      village: userInfo.village || "",
      province: userInfo.province || "",
      district: userInfo.district || "",
      subdistrict: userInfo.subdistrict || "",
      tel: userInfo.tel || "",
      email: userInfo.email || "",
    });
    setEditUser(true);
  };

  const closeEditModal = () => {
    setEditUser(false);
    setPhoneError("");
    setEmailError("");
  };

  const handleOverlayClick = () => {
    closeEditModal();
  };

  useEffect(() => {
    const auth = getAuth();

    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      if (!currentUser) {
        navigate("/login");
        return;
      }

      const currentUserDoc = await getDoc(doc(db, "users", currentUser.uid));
      const currentUserData = currentUserDoc.data();

      if (!currentUserData) {
        alert("ไม่พบข้อมูลผู้ใช้ปัจจุบัน");
        navigate("/login");
        return;
      }

      setRole(currentUserData.role);

      const userDoc = await getDoc(doc(db, "users", id));
      if (userDoc.exists()) {
        setUserInfo(userDoc.data());
      } else {
        alert("ไม่พบข้อมูลผู้ใช้");
      }

      setLoading(false);
    });

    return () => unsubscribe();
  }, [id, navigate]);

  const handleBack = () => {
    // สมมติ role มีค่า "admin" หรือ "user"
    if (role === "admin") {
      navigate("/accountmanagement");
    } else {
      navigate("/");
    }
  };

  const roleName = (role) => {
    switch (role) {
      case "admin": return "ผู้ดูแลระบบ";
      case "user": return "สมาชิก";
      default: return "ไม่ระบุบทบาท";
    }
  };

  if (loading) return <p>กำลังโหลดข้อมูล...</p>;

  if (!userInfo) return <p>ไม่พบข้อมูลผู้ใช้</p>;

  return (
    <div className="card">
      <h2>
        รายละเอียดข้อมูล (
        <span className="role-green">
          {roleName(userInfo.role)}
        </span>
        )
      </h2>
      <p><strong>ชื่อบัญชีผู้ใช้:</strong> {userInfo.username || "-"}</p>
      <p><strong>ชื่อเต็ม:</strong> {userInfo.fullName || "-"}</p>
      <p><strong>ที่อยู่:</strong>
        {userInfo.address ? `${userInfo.address}, บ้าน ${userInfo.village || "-"}, ตำบล ${userInfo.subdistrict || "-"}, อำเภอ ${userInfo.district || "-"}, จังหวัด ${userInfo.province || "-"}` : "-"}
      </p>
      <p><strong>เบอร์โทร:</strong> {userInfo.tel ? userInfo.tel.toString() : "-"}</p>
      <p><strong>อีเมล:</strong> {userInfo.email || "-"}</p>

      <div className="link-container">
        <Link to="/Reportuser" className="report-link">
          ข้อมูลการใช้งาน
        </Link>
      </div>
      
      <div className="button-container">
        <button onClick={handleBack}>
          กลับ
        </button>
        <button onClick={openEditModal} style={{backgroundColor: '#757575'}}>
          แก้ไขข้อมูล
        </button>
      </div>

      {editUser && (
        <div className="modal-overlay" onClick={handleOverlayClick}>
          <div className="modal-content-edit" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2 className="modal-title">แก้ไขข้อมูลผู้ใช้</h2>
            </div>

            <div className="modal-body">
              <div className="form-group-edit">
                <label>ชื่อ-นามสกุล:</label>
                <input
                  type="text"
                  name="fullName"
                  value={formData.fullName || ""}
                  onChange={handleChange}
                  required
                  className="form-input-edit"
                />
              </div>

              <div className="form-group-edit">
                <label>ที่อยู่:</label>
                <textarea
                  name="address"
                  value={formData.address || ""}
                  onChange={handleChange}
                  required
                  rows="2"
                  className="form-input-edit form-textarea-edit"
                />
              </div>

              <div className="form-group-edit">
                <label>หมู่บ้าน:</label>
                <input
                  type="text"
                  name="village"
                  value={formData.village || ""}
                  onChange={handleChange}
                  className="form-input-edit"
                />
              </div>

              <div className="form-row-edit">
                <div className="form-group-edit">
                  <label>จังหวัด:</label>
                  <select
                    name="province"
                    value={formData.province || ""}
                    onChange={handleChange}
                    required
                    className="form-select-edit"
                  >
                    <option value="">เลือกจังหวัด</option>
                    {/* แสดงข้อมูลปัจจุบันถ้ามี */}
                    {formData.province && !provinces.find(p => p.name_th === formData.province) && (
                      <option value={formData.province}>{formData.province}</option>
                    )}
                    {provinces.map((province) => (
                      <option key={province.id} value={province.name_th}>
                        {province.name_th}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="form-group-edit">
                  <label>อำเภอ:</label>
                  <select
                    name="district"
                    value={formData.district || ""}
                    onChange={handleChange}
                    required
                    className="form-select-edit"
                  >
                    <option value="">เลือกอำเภอ</option>
                    {/* แสดงข้อมูลปัจจุบันถ้ามี */}
                    {formData.district && !districtList.find(d => d.name_th === formData.district) && (
                      <option value={formData.district}>{formData.district}</option>
                    )}
                    {districtList.map((districtItem) => (
                      <option key={districtItem.id} value={districtItem.name_th}>
                        {districtItem.name_th}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="form-group-edit">
                  <label>ตำบล:</label>
                  <select
                    name="subdistrict"
                    value={formData.subdistrict || ""}
                    onChange={handleChange}
                    required
                    className="form-select-edit"
                  >
                    <option value="">เลือกตำบล</option>
                    {/* แสดงข้อมูลปัจจุบันถ้ามี */}
                    {formData.subdistrict && !subdistrictList.find(s => s.name_th === formData.subdistrict) && (
                      <option value={formData.subdistrict}>{formData.subdistrict}</option>
                    )}
                    {subdistrictList.map((subItem) => (
                      <option key={subItem.id} value={subItem.name_th}>
                        {subItem.name_th}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="form-row-edit contact-row">
                <div className="form-group-edit">
                  <label>เบอร์โทรศัพท์:</label>
                  <input
                    type="tel"
                    name="tel"
                    value={formData.tel || ""}
                    onChange={handleChange}
                    placeholder="0812345678"
                    className={`form-input-edit ${phoneError ? "error" : ""}`}
                  />
                  {phoneError && <p className="error-message">{phoneError}</p>}
                </div>

                <div className="form-group-edit">
                  <label>อีเมล:</label>
                  <input
                    type="email"
                    name="email"
                    value={formData.email || ""}
                    onChange={handleChange}
                    placeholder="example@example.com"
                    required
                    className={`form-input-edit ${emailError ? "error" : ""}`}
                  />
                  {emailError && <p className="error-message">{emailError}</p>}
                </div>
              </div>

              <div className="form-buttons-edit">
                <button
                  type="button"
                  onClick={closeEditModal}
                  className="btn-edit-user btn-gray"
                >
                  ยกเลิก
                </button>
                <button
                  type="button"
                  onClick={handleSubmitUpdate}
                  className="btn-edit-user btn-green"
                >
                  บันทึกการเปลี่ยนแปลง
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

export default UserDetails;