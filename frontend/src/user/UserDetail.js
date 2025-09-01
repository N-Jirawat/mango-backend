import React, { useEffect, useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import {
  getDocs,
  collection,
  query,
  where,
  addDoc,     // สำหรับเพิ่มเอกสาร
  updateDoc,  // สำหรับอัปเดตเอกสาร
  doc,        // สำหรับอ้างอิงเอกสาร
  getDoc,      // สำหรับดึงเอกสาร
} from "firebase/firestore";

import { db } from "../firebaseConfig";
import { getAuth, onAuthStateChanged, updatePassword, reauthenticateWithCredential, EmailAuthProvider } from "firebase/auth";
import "../css/UserDetails.css";

import provincesData from "../่json/thai_provinces.json";
import districtsData from "../่json/thai_amphures.json";
import subdistrictsData from "../่json/thai_tambons.json";

// กำหนด URL ของ backend API
const BACKEND_URL = "https://render-backend-ftkg.onrender.com";

function UserDetails() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [userInfo, setUserInfo] = useState(null);
  const [role, setRole] = useState(null);
  const [loading, setLoading] = useState(true);
  const [editUser, setEditUser] = useState(false);
  const [changePassword, setChangePassword] = useState(false);

  const [phoneError, setPhoneError] = useState("");
  const [emailError, setEmailError] = useState("");

  const [provinces] = useState(provincesData);
  const [districtList, setDistrictList] = useState([]);
  const [subdistrictList, setSubdistrictList] = useState([]);

  // เพิ่ม state สำหรับแสดง/ซ่อนรหัสผ่าน
  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  // Password change states
  const [passwordData, setPasswordData] = useState({
    currentPassword: "",
    newPassword: "",
    confirmPassword: ""
  });
  const [passwordErrors, setPasswordErrors] = useState({
    currentPassword: "",
    newPassword: "",
    confirmPassword: ""
  });

  // เพิ่ม state สำหรับแสดงสถานะการตรวจสอบรหัสผ่าน
  const [passwordValidation, setPasswordValidation] = useState({
    hasMinLength: false,
    hasLetter: false,
    hasNumber: false
  });

  // เพิ่ม state สำหรับเก็บอีเมลเดิม
  const [originalEmail, setOriginalEmail] = useState("");

  // 👉 เวลามีการเลือก จังหวัด
  const handleProvinceChange = (e) => {
    const selectedProvince = e.target.value;
    setFormData((prev) => ({
      ...prev,
      province: selectedProvince,
      district: "",
      subdistrict: ""
    }));

    // หา province_id
    const province = provinces.find((p) => p.name_th === selectedProvince);
    if (province) {
      const filteredDistricts = districtsData.filter(
        (d) => d.province_id === province.id
      );
      setDistrictList(filteredDistricts);
      setSubdistrictList([]); // reset
    }
  };

  // 👉 เวลามีการเลือก อำเภอ
  const handleDistrictChange = (e) => {
    const selectedDistrict = e.target.value;
    setFormData((prev) => ({
      ...prev,
      district: selectedDistrict,
      subdistrict: ""
    }));

    // หา district_id
    const district = districtsData.find((d) => d.name_th === selectedDistrict);
    if (district) {
      const filteredSubdistricts = subdistrictsData.filter(
        (s) => s.amphure_id === district.id
      );
      setSubdistrictList(filteredSubdistricts);
    }
  };

  // 👉 เวลามีการเลือก ตำบล
  const handleSubdistrictChange = (e) => {
    setFormData((prev) => ({
      ...prev,
      subdistrict: e.target.value
    }));
  };

  const fetchUserReports = async (uid) => {
    try {
      const q = query(collection(db, "AnalysisHistory"), where("userId", "==", uid));
      const snapshot = await getDocs(q);

      return snapshot.docs.map(doc => ({
        AnalysisID: doc.id,
        DiseaseID: doc.data().diseaseId || null,
        DateReUser: doc.data().timestamp?.toDate() || null,
      }));
    } catch (err) {
      console.error("Error fetching user reports:", err);
      return [];
    }
  };

  useEffect(() => {
    const auth = getAuth();

    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      if (!currentUser) {
        navigate("/login");
        return;
      }

      try {
        // ดึงรายงานการวิเคราะห์ของผู้ใช้
        const reports = await fetchUserReports(currentUser.uid);

        if (reports.length === 0) return; // ถ้าไม่มี report อะไร

        // เพิ่มเอกสารใหม่ทุกครั้ง
        await addDoc(collection(db, "ReportDataUser"), {
          UserID: currentUser.uid,
          DateReUser: new Date(),
          AnalysisReports: reports // array ของ {AnalysisID, DiseaseID, DateReUser}
        });

        console.log("เพิ่ม ReportDataUser ใหม่เรียบร้อย");
      } catch (err) {
        console.error("เกิดข้อผิดพลาดในการบันทึก ReportDataUser:", err);
      }
    });

    return () => unsubscribe();
  }, [id, navigate]);

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

  const validatePassword = (password) => {
    // ตรวจสอบความยาวอย่างน้อย 8 หลัก
    if (password.length < 8) {
      return { isValid: false, message: "รหัสผ่านต้องมีอย่างน้อย 8 ตัว!" };
    }

    // ตรวจสอบว่ามีตัวอักษรอย่างน้อย 1 ตัว
    const hasLetter = /[a-zA-Z]/.test(password);
    if (!hasLetter) {
      return { isValid: false, message: "รหัสผ่านต้องมีตัวอักษรอย่างน้อย 1 ตัว!" };
    }

    // ตรวจสอบว่ามีตัวเลขอย่างน้อย 1 ตัว
    const hasNumber = /[0-9]/.test(password);
    if (!hasNumber) {
      return { isValid: false, message: "รหัสผ่านต้องมีตัวเลขอย่างน้อย 1 ตัว!" };
    }

    return { isValid: true, message: "" };
  };

  // ฟังก์ชันสำหรับตรวจสอบอีเมลซ้ำผ่าน backend
  const checkEmailExists = async (email, excludeUid = null) => {
    try {
      const response = await fetch(`${BACKEND_URL}/check_email`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, exclude_uid: excludeUid })
      });
      
      const result = await response.json();
      return result.exists;
    } catch (error) {
      console.error('Error checking email:', error);
      return false;
    }
  };

  // ฟังก์ชันสำหรับอัปเดตอีเมลผ่าน backend
  const updateEmailInFirebase = async (uid, newEmail) => {
    try {
      const response = await fetch(`${BACKEND_URL}/update_email`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ uid, new_email: newEmail })
      });
      
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.error || 'Failed to update email');
      }
      
      return result;
    } catch (error) {
      console.error('Error updating email:', error);
      throw error;
    }
  };

  // ฟังก์ชันสำหรับ toggle การแสดงรหัสผ่าน
  const toggleCurrentPasswordVisibility = () => {
    setShowCurrentPassword(!showCurrentPassword);
  };

  const toggleNewPasswordVisibility = () => {
    setShowNewPassword(!showNewPassword);
  };

  const toggleConfirmPasswordVisibility = () => {
    setShowConfirmPassword(!showConfirmPassword);
  };

  const handleChange = (e) => {
    const { name, value } = e.target;

    // สำหรับฟิลด์ที่ไม่ใช่ tel ใช้ logic เดิม
    if (name !== 'tel') {
      setFormData((prev) => ({
        ...prev,
        [name]: value === "-" ? "" : value, // ไม่เก็บ "-" ลง state
      }));
    }
    // สำหรับ tel จะใช้ handlePhoneInput แทน
  };

  const handlePasswordChange = (e) => {
    const { name, value } = e.target;
    setPasswordData((prev) => ({
      ...prev,
      [name]: value,
    }));

    // Clear errors when user types
    setPasswordErrors((prev) => ({
      ...prev,
      [name]: "",
    }));

    // ตรวจสอบรหัสผ่านใหม่ real-time
    if (name === "newPassword") {
      setPasswordValidation({
        hasMinLength: value.length >= 8,
        hasLetter: /[a-zA-Z]/.test(value),
        hasNumber: /[0-9]/.test(value)
      });
    }
  };

  const handleSubmitPasswordChange = async () => {
    // Reset errors
    setPasswordErrors({
      currentPassword: "",
      newPassword: "",
      confirmPassword: ""
    });

    // Validate form
    let hasErrors = false;

    if (!passwordData.currentPassword) {
      setPasswordErrors(prev => ({ ...prev, currentPassword: "กรุณากรอกรหัสผ่านเดิม" }));
      hasErrors = true;
    }

    if (!passwordData.newPassword) {
      setPasswordErrors(prev => ({ ...prev, newPassword: "กรุณากรอกรหัสผ่านใหม่" }));
      hasErrors = true;
    } else {
      const passwordValidationResult = validatePassword(passwordData.newPassword);
      if (!passwordValidationResult.isValid) {
        setPasswordErrors(prev => ({ ...prev, newPassword: passwordValidationResult.message }));
        hasErrors = true;
      }
    }

    if (!passwordData.confirmPassword) {
      setPasswordErrors(prev => ({ ...prev, confirmPassword: "กรุณายืนยันรหัสผ่านใหม่" }));
      hasErrors = true;
    } else if (passwordData.newPassword !== passwordData.confirmPassword) {
      setPasswordErrors(prev => ({ ...prev, confirmPassword: "รหัสผ่านใหม่ไม่ตรงกัน" }));
      hasErrors = true;
    }

    if (hasErrors) return;

    try {
      const auth = getAuth();
      const user = auth.currentUser;

      if (!user || !user.email) {
        alert("ไม่สามารถระบุผู้ใช้งานได้");
        return;
      }

      // Re-authenticate user with current password
      const credential = EmailAuthProvider.credential(user.email, passwordData.currentPassword);
      await reauthenticateWithCredential(user, credential);

      // Update password
      await updatePassword(user, passwordData.newPassword);

      // Close modal and reset form
      setChangePassword(false);
      setPasswordData({
        currentPassword: "",
        newPassword: "",
        confirmPassword: ""
      });

      alert("เปลี่ยนรหัสผ่านสำเร็จ");
    } catch (error) {
      console.error("Error changing password:", error);

      if (error.code === 'auth/wrong-password') {
        setPasswordErrors(prev => ({ ...prev, currentPassword: "รหัสผ่านเดิมไม่ถูกต้อง" }));
      } else if (error.code === 'auth/weak-password') {
        setPasswordErrors(prev => ({ ...prev, newPassword: "รหัสผ่านใหม่ไม่ปลอดภัยเพียงพอ" }));
      } else {
        alert("เกิดข้อผิดพลาดในการเปลี่ยนรหัสผ่าน: " + error.message);
      }
    }
  };

  // แก้ไขการตรวจสอบในฟังก์ชัน handleSubmitUpdate
  const handleSubmitUpdate = async () => {
    // ตรวจสอบเบอร์โทร - อัปเดตเงื่อนไขการตรวจสอบ
    if (formData.tel && formData.tel !== "-") {
      // ตรวจสอบว่าเป็นตัวเลขทั้งหมดและมีความยาว 10 หลัก
      const telRegex = /^[0-9]{10}$/;
      if (!telRegex.test(formData.tel)) {
        setPhoneError("กรุณากรอกเบอร์โทรให้ถูกต้อง 10 หลัก (ตัวเลขเท่านั้น)");
        return;
      } else {
        setPhoneError("");
      }
    } else {
      setPhoneError("");
    }

    // ตรวจสอบอีเมล
    if (!validateEmail(formData.email)) {
      setEmailError("รูปแบบอีเมลไม่ถูกต้อง");
      return;
    }

    // ตรวจสอบว่าอีเมลเปลี่ยนแล้วมีคนใช้แล้วหรือไม่
    if (formData.email !== originalEmail) {
      const emailExists = await checkEmailExists(formData.email, id);
      if (emailExists) {
        setEmailError("อีเมลนี้ถูกใช้แล้ว");
        return;
      }
    }

    setEmailError("");

    try {
      // อัปเดตข้อมูลใน Firestore
      await updateDoc(doc(db, "users", id), formData);

      // ถ้าอีเมลเปลี่ยนแปลง ให้อัปเดตใน Firebase Auth ด้วย
      if (formData.email !== originalEmail) {
        try {
          await updateEmailInFirebase(id, formData.email);
          console.log("อัปเดตอีเมลใน Firebase Auth สำเร็จ");
        } catch (error) {
          console.error("Error updating email in Firebase Auth:", error);
          // หากอัปเดต Auth ไม่สำเร็จ แต่ Firestore สำเร็จแล้ว
          // ควร rollback Firestore หรือแจ้งเตือนผู้ใช้
          alert("อัปเดตข้อมูลสำเร็จ แต่อาจมีปัญหากับการอัปเดตอีเมลในระบบยืนยันตัวตน");
        }
      }

      setUserInfo(prev => ({ ...prev, ...formData }));
      setOriginalEmail(formData.email); // อัปเดตอีเมลเดิม
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

    // เก็บอีเมลเดิมไว้เปรียบเทียบ
    setOriginalEmail(userInfo.email || "");

    // 👉 preload districts ตาม province ที่มีใน Firebase
    if (userInfo.province) {
      const province = provinces.find((p) => p.name_th === userInfo.province);
      if (province) {
        const filteredDistricts = districtsData.filter(
          (d) => d.province_id === province.id
        );
        setDistrictList(filteredDistricts);

        // 👉 preload subdistricts ตาม district ที่มีใน Firebase
        if (userInfo.district) {
          const district = filteredDistricts.find(
            (d) => d.name_th === userInfo.district
          );
          if (district) {
            const filteredSubdistricts = subdistrictsData.filter(
              (s) => s.amphure_id === district.id
            );
            setSubdistrictList(filteredSubdistricts);
          }
        }
      }
    }

    setEditUser(true);
  };

  const openPasswordModal = () => {
    setPasswordData({
      currentPassword: "",
      newPassword: "",
      confirmPassword: ""
    });
    setPasswordErrors({
      currentPassword: "",
      newPassword: "",
      confirmPassword: ""
    });
    setPasswordValidation({
      hasMinLength: false,
      hasLetter: false,
      hasNumber: false
    });
    setShowCurrentPassword(false);
    setShowNewPassword(false);
    setShowConfirmPassword(false);
    setChangePassword(true);
  };

  const closeEditModal = () => {
    setEditUser(false);
    setPhoneError("");
    setEmailError("");
  };

  const closePasswordModal = () => {
    setChangePassword(false);
    setPasswordData({
      currentPassword: "",
      newPassword: "",
      confirmPassword: ""
    });
    setPasswordErrors({
      currentPassword: "",
      newPassword: "",
      confirmPassword: ""
    });
    setPasswordValidation({
      hasMinLength: false,
      hasLetter: false,
      hasNumber: false
    });
    setShowCurrentPassword(false);
    setShowNewPassword(false);
    setShowConfirmPassword(false);
  };

  const handleOverlayClick = () => {
    closeEditModal();
  };

  const handlePasswordOverlayClick = () => {
    closePasswordModal();
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
      navigate("/");
    } else {
      navigate("/");
    }
  };

  // ฟังก์ชันใหม่สำหรับจัดการการป้อนเบอร์โทร
  const handlePhoneInput = (e) => {
    const value = e.target.value;

    // อนุญาตเฉพาะตัวเลข
    const numericValue = value.replace(/[^0-9]/g, '');

    // จำกัดไม่เกิน 10 หลัก
    const limitedValue = numericValue.slice(0, 10);

    // อัปเดต formData
    setFormData(prev => ({
      ...prev,
      tel: limitedValue
    }));

    // ลบ error message เมื่อผู้ใช้พิมพ์
    if (phoneError) {
      setPhoneError("");
    }
  };

  const handleDeleteOwnAccount = async () => {
    const confirmDelete = window.confirm("คุณแน่ใจหรือไม่ว่าต้องการลบบัญชีของคุณ?");
    if (!confirmDelete) return;

    const auth = getAuth();
    const user = auth.currentUser;

    if (user) {
      try {
        // ใช้ backend API สำหรับลบบัญชี
        const response = await fetch(`${BACKEND_URL}/delete_user`, {
          method: 'DELETE',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ uid: user.uid })
        });

        const result = await response.json();
        
        if (!response.ok) {
          throw new Error(result.error || 'Failed to delete account');
        }

        alert("บัญชีและข้อมูลถูกลบเรียบร้อยแล้ว");
        
        // redirect ไปหน้า login
        window.location.href = "/login";
      } catch (error) {
        console.error("Error deleting account:", error);
        alert(`ไม่สามารถลบบัญชี: ${error.message}`);
      }
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
      <p><strong>ชื่อบัญชี :</strong> {userInfo.username || "-"}</p>
      <p><strong>ชื่อ-นามสกุล :</strong> {userInfo.fullName || "-"}</p>
      <p><strong>ที่อยู่ :</strong>
        {userInfo.address ? `${userInfo.address}, บ้าน ${userInfo.village || "-"}, ตำบล ${userInfo.subdistrict || "-"}, อำเภอ ${userInfo.district || "-"}, จังหวัด ${userInfo.province || "-"}` : "-"}
      </p>
      <p><strong>เบอร์โทร :</strong> {userInfo.tel ? userInfo.tel.toString() : "-"}</p>
      <p><strong>อีเมล :</strong> {userInfo.email || "-"}</p>

      <div className="link-container">
        <button
          onClick={openPasswordModal}
          className="report-link"
          style={{
            background: 'none',
            border: 'none',
            color: '#007bff',
            textDecoration: 'underline',
            cursor: 'pointer',
            fontSize: 'inherit',
            padding: 0,
            marginLeft: '10px'
          }}
        >
          เปลี่ยนรหัสผ่าน
        </button>
        <Link to="/Reportuser" className="report-link">
          ข้อมูลการใช้งาน
        </Link>
      </div>

      <div className="button-container">
        <button onClick={openEditModal} style={{ backgroundColor: '#757575' }}>
          แก้ไขข้อมูล
        </button>
        <button onClick={handleBack}>
          หน้าหลัก
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

              {/* จังหวัด */}
              <div className="form-group-edit">
                <label>จังหวัด:</label>
                <select
                  name="province"
                  value={formData.province || ""}
                  onChange={handleProvinceChange}
                  className="form-select-edit"
                >
                  <option value="">เลือกจังหวัด</option>
                  {provinces.map((province) => (
                    <option key={province.id} value={province.name_th}>
                      {province.name_th}
                    </option>
                  ))}
                </select>
              </div>

              {/* อำเภอ */}
              <div className="form-group-edit">
                <label>อำเภอ:</label>
                <select
                  name="district"
                  value={formData.district || ""}
                  onChange={handleDistrictChange}
                  className="form-select-edit"
                  disabled={!formData.province}
                >
                  <option value="">เลือกอำเภอ</option>
                  {districtList.map((district) => (
                    <option key={district.id} value={district.name_th}>
                      {district.name_th}
                    </option>
                  ))}
                </select>
              </div>

              {/* ตำบล */}
              <div className="form-group-edit">
                <label>ตำบล:</label>
                <select
                  name="subdistrict"
                  value={formData.subdistrict || ""}
                  onChange={handleSubdistrictChange}
                  className="form-select-edit"
                  disabled={!formData.district}
                >
                  <option value="">เลือกตำบล</option>
                  {subdistrictList.map((sub) => (
                    <option key={sub.id} value={sub.name_th}>
                      {sub.name_th}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-row-edit contact-row">
                <div className="form-group-edit">
                  <label>เบอร์โทรศัพท์:</label>
                  <input
                    type="tel"
                    name="tel"
                    value={formData.tel && formData.tel.trim() !== "" ? formData.tel : "-"}
                    onChange={handlePhoneInput}
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

              <button
                onClick={handleDeleteOwnAccount}
                className="report-link"
                style={{
                  background: 'none',
                  border: 'none',
                  color: '#ff0000ff',
                  textDecoration: 'underline',
                  cursor: 'pointer',
                  fontSize: 'inherit',
                  padding: 0,
                  marginLeft: '10px',
                  marginBottom: '20px'
                }}
              >
                ลบบัญชี
              </button>

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

      {changePassword && (
        <div className="modal-overlay" onClick={handlePasswordOverlayClick}>
          <div className="modal-content-edit" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2 className="modal-title">เปลี่ยนรหัสผ่าน</h2>
            </div>

            <div className="modal-body">
              <div className="form-group-edit">
                <label>รหัสผ่านเดิม:</label>
                <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
                  <input
                    type={showCurrentPassword ? "text" : "password"}
                    name="currentPassword"
                    value={passwordData.currentPassword}
                    onChange={handlePasswordChange}
                    placeholder="รหัสผ่านเดิม"
                    className={`form-input-edit ${passwordErrors.currentPassword ? "error" : ""}`}
                    style={{ paddingRight: '40px', fontSize: '14px' }}
                  />
                  <button
                    type="button"
                    onClick={toggleCurrentPasswordVisibility}
                    style={{
                      position: 'absolute',
                      right: '10px',
                      top: '50%',
                      transform: 'translateY(-50%)',
                      background: 'none',
                      border: 'none',
                      cursor: 'pointer',
                      padding: '0',
                      zIndex: 10,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      width: '24px',
                      height: '24px'
                    }}
                  >
                    <img
                      src={showCurrentPassword ? "/img/hide.png" : "/img/view.png"}
                      alt={showCurrentPassword ? "hide" : "view"}
                      style={{ width: '20px', height: '20px', marginBottom: '20px' }}
                    />
                  </button>
                </div>
                {passwordErrors.currentPassword && (
                  <p className="error-message">{passwordErrors.currentPassword}</p>
                )}
              </div>

              <div className="form-group-edit">
                <label>รหัสผ่านใหม่:</label>
                <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
                  <input
                    type={showNewPassword ? "text" : "password"}
                    name="newPassword"
                    value={passwordData.newPassword}
                    onChange={handlePasswordChange}
                    placeholder="รหัสผ่านใหม่"
                    className={`form-input-edit ${passwordErrors.newPassword ? "error" : ""}`}
                    style={{ paddingRight: '40px', fontSize: '14px' }}
                  />
                  <button
                    type="button"
                    onClick={toggleNewPasswordVisibility}
                    style={{
                      position: 'absolute',
                      right: '10px',
                      top: '50%',
                      transform: 'translateY(-50%)',
                      background: 'none',
                      border: 'none',
                      cursor: 'pointer',
                      padding: '0',
                      zIndex: 10,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      width: '24px',
                      height: '24px'
                    }}
                  >
                    <img
                      src={showNewPassword ? "/img/hide.png" : "/img/view.png"}
                      alt={showNewPassword ? "hide" : "view"}
                      style={{ width: '20px', height: '20px', marginBottom: '20px' }}
                    />
                  </button>
                </div>

                <p style={{ display: 'flex', fontSize: "12px", color: "#666", margin: "5px 0" }}>
                  *รหัสผ่านอย่างน้อย 8 ตัว ต้องมีตัวอักษรและตัวเลขอย่างน้อยตัวละ 1 ตัว
                </p>

                {/* แสดงสถานะการตรวจสอบรหัสผ่าน */}
                {passwordData.newPassword && (
                  <div style={{ margin: "10px 0", fontSize: "12px" }}>
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      margin: "3px 0",
                      color: passwordValidation.hasMinLength ? "#28a745" : "#dc3545"
                    }}>
                      <span style={{ marginRight: "5px" }}>
                        {passwordValidation.hasMinLength ? "✅" : "❌"}
                      </span>
                      อย่างน้อย 8 ตัวอักษร
                    </div>
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      margin: "3px 0",
                      color: passwordValidation.hasLetter ? "#28a745" : "#dc3545"
                    }}>
                      <span style={{ marginRight: "5px" }}>
                        {passwordValidation.hasLetter ? "✅" : "❌"}
                      </span>
                      มีตัวอักษรอย่างน้อย 1 ตัว (a-z, A-Z)
                    </div>
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      margin: "3px 0",
                      color: passwordValidation.hasNumber ? "#28a745" : "#dc3545"
                    }}>
                      <span style={{ marginRight: "5px" }}>
                        {passwordValidation.hasNumber ? "✅" : "❌"}
                      </span>
                      มีตัวเลขอย่างน้อย 1 ตัว (0-9)
                    </div>
                  </div>
                )}

                {passwordErrors.newPassword && (
                  <p className="error-message">{passwordErrors.newPassword}</p>
                )}
              </div>

              <div className="form-group-edit">
                <label>ยืนยันรหัสผ่านใหม่:</label>
                <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
                  <input
                    type={showConfirmPassword ? "text" : "password"}
                    name="confirmPassword"
                    value={passwordData.confirmPassword}
                    onChange={handlePasswordChange}
                    placeholder="ยืนยันรหัสผ่านใหม่"
                    className={`form-input-edit ${passwordErrors.confirmPassword ? "error" : ""}`}
                    style={{ paddingRight: '40px', fontSize: '14px' }}
                  />
                  <button
                    type="button"
                    onClick={toggleConfirmPasswordVisibility}
                    style={{
                      position: 'absolute',
                      right: '10px',
                      top: '50%',
                      transform: 'translateY(-50%)',
                      background: 'none',
                      border: 'none',
                      cursor: 'pointer',
                      padding: '0',
                      zIndex: 10,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      width: '24px',
                      height: '24px'
                    }}
                  >
                    <img
                      src={showConfirmPassword ? "/img/hide.png" : "/img/view.png"}
                      alt={showConfirmPassword ? "hide" : "view"}
                      style={{ width: '20px', height: '20px', marginBottom: '20px' }}
                    />
                  </button>
                </div>
                {passwordErrors.confirmPassword && (
                  <p className="error-message">{passwordErrors.confirmPassword}</p>
                )}
              </div>

              <div className="form-buttons-edit">
                <button
                  type="button"
                  onClick={closePasswordModal}
                  className="btn-edit-user btn-gray"
                >
                  ยกเลิก
                </button>
                <button
                  type="button"
                  onClick={handleSubmitPasswordChange}
                  className="btn-edit-user btn-green"
                >
                  เปลี่ยนรหัสผ่าน
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