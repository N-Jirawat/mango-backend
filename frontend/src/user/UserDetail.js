import React, { useEffect, useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { getAuth, signInWithCustomToken, onAuthStateChanged } from "firebase/auth";
import { getDocs, collection, query, where, addDoc, doc, getDoc, updateDoc } from "firebase/firestore";
import { db } from "../firebaseConfig";
import "../css/UserDetails.css";

import provincesData from "../่json/thai_provinces.json";
import districtsData from "../่json/thai_amphures.json";
import subdistrictsData from "../่json/thai_tambons.json";

const BACKEND_URL = "https://render-backend-mu.vercel.app";

function UserDetails() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [userInfo, setUserInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [editModal, setEditModal] = useState(false);
  const [phoneError, setPhoneError] = useState("");
  const [emailError, setEmailError] = useState("");
  const [provinces] = useState(provincesData);
  const [districtList, setDistrictList] = useState([]);
  const [subdistrictList, setSubdistrictList] = useState([]);
  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [passwordData, setPasswordData] = useState({
    currentPassword: "",
    newPassword: "",
    confirmPassword: "",
  });
  const [passwordErrors, setPasswordErrors] = useState({
    currentPassword: "",
    newPassword: "",
    confirmPassword: "",
  });
  const [passwordValidation, setPasswordValidation] = useState({
    hasMinLength: false,
    hasLetter: false,
    hasNumber: false,
  });
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
  const [originalEmail, setOriginalEmail] = useState("");
  const [changePassword, setChangePassword] = useState(false);
  const [changeEmail, setChangeEmail] = useState(false);

  // Handle province change
  const handleProvinceChange = (e) => {
    const selectedProvince = e.target.value;
    setFormData((prev) => ({
      ...prev,
      province: selectedProvince,
      district: "",
      subdistrict: "",
    }));

    const province = provinces.find((p) => p.name_th === selectedProvince);
    if (province) {
      const filteredDistricts = districtsData.filter(
        (d) => d.province_id === province.id
      );
      setDistrictList(filteredDistricts);
      setSubdistrictList([]);
    }
  };

  // Handle district change
  const handleDistrictChange = (e) => {
    const selectedDistrict = e.target.value;
    setFormData((prev) => ({
      ...prev,
      district: selectedDistrict,
      subdistrict: "",
    }));

    const district = districtsData.find((d) => d.name_th === selectedDistrict);
    if (district) {
      const filteredSubdistricts = subdistrictsData.filter(
        (s) => s.amphure_id === district.id
      );
      setSubdistrictList(filteredSubdistricts);
    }
  };

  // Handle subdistrict change
  const handleSubdistrictChange = (e) => {
    setFormData((prev) => ({
      ...prev,
      subdistrict: e.target.value,
    }));
  };

  // Fetch user reports
  const fetchUserReports = async (uid) => {
    try {
      const q = query(collection(db, "AnalysisHistory"), where("userId", "==", uid));
      const snapshot = await getDocs(q);
      return snapshot.docs.map((doc) => ({
        AnalysisID: doc.id,
        DiseaseID: doc.data().diseaseId || null,
        DateReUser: doc.data().timestamp?.toDate() || null,
      }));
    } catch (err) {
      console.error("Error fetching user reports:", err);
      return [];
    }
  };

  // Initialize user data
  useEffect(() => {
    const auth = getAuth();
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      if (!currentUser) {
        navigate("/login");
        return;
      }

      try {
        const currentUserDoc = await getDoc(doc(db, "users", currentUser.uid));
        const currentUserData = currentUserDoc.data();

        if (!currentUserData) {
          alert("ไม่พบข้อมูลผู้ใช้ปัจจุบัน");
          navigate("/login");
          return;
        }

        const userDoc = await getDoc(doc(db, "users", id));
        if (userDoc.exists()) {
          const userData = userDoc.data();
          setUserInfo(userData);
          setOriginalEmail(userData.email || "");
        } else {
          alert("ไม่พบข้อมูลผู้ใช้");
        }

        setLoading(false);
      } catch (error) {
        console.error("Error fetching user data:", error);
        alert("เกิดข้อผิดพลาดในการโหลดข้อมูล");
        setLoading(false);
      }
    });

    return () => unsubscribe();
  }, [id, navigate]);

  // Save user reports to ReportDataUser
  useEffect(() => {
    const auth = getAuth();
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      if (!currentUser) return;

      try {
        const reports = await fetchUserReports(currentUser.uid);
        if (reports.length === 0) return;

        await addDoc(collection(db, "ReportDataUser"), {
          UserID: currentUser.uid,
          DateReUser: new Date(),
          AnalysisReports: reports,
        });
        console.log("เพิ่ม ReportDataUser ใหม่เรียบร้อย");
      } catch (err) {
        console.error("เกิดข้อผิดพลาดในการบันทึก ReportDataUser:", err);
      }
    });

    return () => unsubscribe();
  }, [id]);

  // Toggle password visibility
  const toggleCurrentPasswordVisibility = () => setShowCurrentPassword(!showCurrentPassword);
  const toggleNewPasswordVisibility = () => setShowNewPassword(!showNewPassword);
  const toggleConfirmPasswordVisibility = () => setShowConfirmPassword(!showConfirmPassword);

  // Handle form input changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    if (name !== "tel") {
      setFormData((prev) => ({
        ...prev,
        [name]: value === "-" ? "" : value,
      }));
    }
    if (emailError && name === "email") setEmailError("");
    
    // Check if email has changed
    if (name === "email") {
      setChangeEmail(value !== originalEmail);
    }
  };

  // Handle phone input
  const handlePhoneInput = (e) => {
    const value = e.target.value.replace(/[^0-9]/g, "").slice(0, 10);
    setFormData((prev) => ({ ...prev, tel: value }));
    if (phoneError) setPhoneError("");
  };

  // Handle password input changes
  const handlePasswordChange = (e) => {
    const { name, value } = e.target;
    setPasswordData((prev) => ({ ...prev, [name]: value }));
    setPasswordErrors((prev) => ({ ...prev, [name]: "" }));

    if (name === "newPassword") {
      setPasswordValidation({
        hasMinLength: value.length >= 8,
        hasLetter: /[a-zA-Z]/.test(value),
        hasNumber: /[0-9]/.test(value),
      });
    }
  };

  // Validate form data
  const validateForm = () => {
    let isValid = true;
    const errors = { ...passwordErrors };

    // Validate phone number
    if (formData.tel && !/^0[0-9]{9}$/.test(formData.tel)) {
      setPhoneError("เบอร์โทรศัพท์ต้องเป็นตัวเลข 10 หลักและขึ้นต้นด้วย 0");
      isValid = false;
    } else {
      setPhoneError("");
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (formData.email && !emailRegex.test(formData.email)) {
      setEmailError("รูปแบบอีเมลไม่ถูกต้อง");
      isValid = false;
    } else {
      setEmailError("");
    }

    // Validate password if changing password
    if (changePassword) {
      // Remove current password validation since backend doesn't verify it
      if (!passwordData.newPassword) {
        errors.newPassword = "กรุณากรอกรหัสผ่านใหม่";
        isValid = false;
      } else if (!passwordValidation.hasMinLength || !passwordValidation.hasLetter || !passwordValidation.hasNumber) {
        errors.newPassword = "รหัสผ่านไม่ตรงตามเงื่อนไข";
        isValid = false;
      }

      if (passwordData.newPassword !== passwordData.confirmPassword) {
        errors.confirmPassword = "รหัสผ่านไม่ตรงกัน";
        isValid = false;
      }
    }

    // Validate current password if email changed (but not if just changing password since backend doesn't verify it)
    if (changeEmail && !changePassword && !passwordData.currentPassword) {
      errors.currentPassword = "กรุณากรอกรหัสผ่านเดิมเพื่อยืนยันตัวตน";
      isValid = false;
    }

    setPasswordErrors(errors);
    return isValid;
  };

  const auth = getAuth();

  // Handle form submission (user info and/or password)
  const handleSubmitUpdate = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    setLoading(true);

    try {
      // Update user info in Firestore first
      const userDocRef = doc(db, "users", id);
      await updateDoc(userDocRef, {
        fullName: formData.fullName || "",
        address: formData.address || "",
        village: formData.village || "",
        province: formData.province || "",
        district: formData.district || "",
        subdistrict: formData.subdistrict || "",
        tel: formData.tel || "",
        email: formData.email || "",
      });

      // Update local userInfo state
      setUserInfo(prev => ({
        ...prev,
        fullName: formData.fullName || "",
        address: formData.address || "",
        village: formData.village || "",
        province: formData.province || "",
        district: formData.district || "",
        subdistrict: formData.subdistrict || "",
        tel: formData.tel || "",
        email: formData.email || "",
      }));

      // ========== อัปเดตอีเมล ==========
      if (changeEmail) {
        try {
          const response = await fetch(`${BACKEND_URL}/update_email`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
              uid: id, 
              new_email: formData.email
            }),
          });

          const result = await response.json();
          if (!response.ok) {
            throw new Error(result.error || "Failed to update email");
          }

          alert("เปลี่ยนอีเมลเรียบร้อยแล้ว");
          setOriginalEmail(formData.email);
        } catch (emailError) {
          console.error("Email update error:", emailError);
          alert("เกิดข้อผิดพลาดในการเปลี่ยนอีเมล: " + emailError.message);
          // Reset email to original if update failed
          setFormData(prev => ({ ...prev, email: originalEmail }));
          setUserInfo(prev => ({ ...prev, email: originalEmail }));
        }
      }

      // ========== อัปเดตรหัสผ่าน ==========
      if (changePassword) {
        try {
          // Try using fetch first
          const response = await fetch(`${BACKEND_URL}/update_password`, {
            method: "POST",
            headers: { 
              "Content-Type": "application/json",
              "Access-Control-Request-Method": "POST",
              "Access-Control-Request-Headers": "Content-Type"
            },
            body: JSON.stringify({ 
              uid: id, 
              new_password: passwordData.newPassword
            }),
          });

          const result = await response.json();
          if (!response.ok) {
            throw new Error(result.error || "Failed to update password");
          }

          if (result.id_token) {
            await signInWithCustomToken(auth, result.id_token);
            console.log("Password updated and re-authenticated with new token");
          }

          alert("เปลี่ยนรหัสผ่านเรียบร้อยแล้ว");
        } catch (passwordError) {
          console.error("Password update error:", passwordError);
          
          // If CORS error, show specific message
          if (passwordError.message.includes("fetch") || passwordError.message.includes("CORS")) {
            alert("เกิดข้อผิดพลาดการเชื่อมต่อกับเซิร์ฟเวอร์ (CORS)\n\nกรุณาลอง:\n1. รีเฟรชหน้าแล้วลองใหม่\n2. หรือติดต่อผู้ดูแลระบบ\n\nสำหรับ Development: ให้เปิด browser ด้วย --disable-web-security");
          } else {
            alert("เกิดข้อผิดพลาดในการเปลี่ยนรหัสผ่าน: " + passwordError.message);
          }
        }
      }

      if (!changeEmail && !changePassword) {
        alert("อัปเดตข้อมูลเรียบร้อยแล้ว");
      }

      setChangeEmail(false);
      setChangePassword(false);
      setPasswordData({ currentPassword: "", newPassword: "", confirmPassword: "" });
      setEditModal(false);
      
    } catch (error) {
      console.error("Error in handleSubmitUpdate:", error);
      alert("เกิดข้อผิดพลาด: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  // Open edit modal
  const openEditModal = () => {
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
    setOriginalEmail(userInfo.email || "");
    setPasswordData({
      currentPassword: "",
      newPassword: "",
      confirmPassword: "",
    });
    setPasswordErrors({
      currentPassword: "",
      newPassword: "",
      confirmPassword: "",
    });
    setPasswordValidation({
      hasMinLength: false,
      hasLetter: false,
      hasNumber: false,
    });
    setPhoneError("");
    setEmailError("");
    setChangeEmail(false);

    // Preload districts and subdistricts only if not in password change mode
    if (!changePassword && userInfo.province) {
      const province = provinces.find((p) => p.name_th === userInfo.province);
      if (province) {
        const filteredDistricts = districtsData.filter(
          (d) => d.province_id === province.id
        );
        setDistrictList(filteredDistricts);
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
    } else {
      setDistrictList([]);
      setSubdistrictList([]);
    }

    setEditModal(true);
  };

  // Close edit modal
  const closeEditModal = () => {
    setEditModal(false);
    setPhoneError("");
    setEmailError("");
    setPasswordErrors({
      currentPassword: "",
      newPassword: "",
      confirmPassword: "",
    });
    setPasswordData({
      currentPassword: "",
      newPassword: "",
      confirmPassword: "",
    });
    setChangePassword(false);
    setChangeEmail(false);
  };

  // Handle delete account
  const handleDeleteOwnAccount = async () => {
    const confirmDelete = window.confirm("คุณแน่ใจหรือไม่ว่าต้องการลบบัญชีของคุณ?");
    if (!confirmDelete) return;

    const auth = getAuth();
    const user = auth.currentUser;

    if (user) {
      try {
        const response = await fetch(`${BACKEND_URL}/delete_user`, {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ uid: user.uid }),
        });

        const result = await response.json();
        if (!response.ok) {
          throw new Error(result.error || "Failed to delete account");
        }

        alert("บัญชีและข้อมูลถูกลบเรียบร้อยแล้ว");
        window.location.href = "/login";
      } catch (error) {
        console.error("Error deleting account:", error);
        alert(`ไม่สามารถลบบัญชี: ${error.message}`);
      }
    }
  };

  // Role name mapping
  const roleName = (role) => {
    switch (role) {
      case "admin":
        return "ผู้ดูแลระบบ";
      case "user":
        return "สมาชิก";
      default:
        return "ไม่ระบุบทบาท";
    }
  };

  if (loading) return <p>กำลังโหลดข้อมูล...</p>;
  if (!userInfo) return <p>ไม่พบข้อมูลผู้ใช้</p>;

  return (
    <div className="card">
      <h2>
        รายละเอียดข้อมูล (<span className="role-green">{roleName(userInfo.role)}</span>)
      </h2>
      <p><strong>ชื่อบัญชี :</strong> {userInfo.username || "-"}</p>
      <p><strong>ชื่อ-นามสกุล :</strong> {userInfo.fullName || "-"}</p>
      <p>
        <strong>ที่อยู่ :</strong>{" "}
        {userInfo.address
          ? `${userInfo.address}, บ้าน ${userInfo.village || "-"}, ตำบล ${userInfo.subdistrict || "-"
          }, อำเภอ ${userInfo.district || "-"}, จังหวัด ${userInfo.province || "-"}`
          : "-"}
      </p>
      <p><strong>เบอร์โทร :</strong> {userInfo.tel ? userInfo.tel : "-"}</p>
      <p><strong>อีเมล :</strong> {userInfo.email || "-"}</p>

      <div className="link-container">
        <button
          onClick={() => {
            console.log("Change password button clicked");
            setChangePassword(true);
            openEditModal();
          }}
          className="report-link"
          style={{
            background: "none",
            border: "none",
            color: "#007bff",
            textDecoration: "underline",
            cursor: "pointer",
            fontSize: "inherit",
            padding: 0,
            marginLeft: "10px",
          }}
        >
          เปลี่ยนรหัสผ่าน
        </button>
        <Link to="/Reportuser" className="report-link">
          ข้อมูลการใช้งาน
        </Link>
      </div>

      <div className="button-container">
        <button onClick={openEditModal} style={{ backgroundColor: "#757575" }}>
          แก้ไขข้อมูล
        </button>
        <button onClick={() => navigate("/")}>หน้าหลัก</button>
      </div>

      {editModal && (
        <div className="modal-overlay" onClick={closeEditModal}>
          <div className="modal-content-edit" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2 className="modal-title">
                {changePassword ? "เปลี่ยนรหัสผ่าน" : "แก้ไขข้อมูลผู้ใช้"}
              </h2>
            </div>

            <div className="modal-body">
              {!changePassword && (
                <>
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
                        value={formData.tel && formData.tel.trim() !== "" ? formData.tel : ""}
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
                </>
              )}

              {changePassword && (
                <>
                  <div className="form-group-edit">
                    <label>รหัสผ่านใหม่:</label>
                    <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
                      <input
                        type={showNewPassword ? "text" : "password"}
                        name="newPassword"
                        value={passwordData.newPassword}
                        onChange={handlePasswordChange}
                        placeholder="รหัสผ่านใหม่"
                        className={`form-input-edit ${passwordErrors.newPassword ? "error" : ""}`}
                        style={{ paddingRight: "40px", fontSize: "14px" }}
                      />
                      <button
                        type="button"
                        onClick={toggleNewPasswordVisibility}
                        style={{
                          position: "absolute",
                          right: "10px",
                          top: "50%",
                          transform: "translateY(-50%)",
                          background: "none",
                          border: "none",
                          cursor: "pointer",
                          padding: "0",
                          zIndex: 10,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          width: "24px",
                          height: "24px",
                        }}
                      >
                        <img
                          src={showNewPassword ? "/img/hide.png" : "/img/view.png"}
                          alt={showNewPassword ? "hide" : "view"}
                          style={{ width: "20px", height: "20px", marginBottom: "20px" }}
                        />
                      </button>
                    </div>
                    <p style={{ display: "flex", fontSize: "12px", color: "#666", margin: "5px 0" }}>
                      *รหัสผ่านอย่างน้อย 8 ตัว ต้องมีตัวอักษรและตัวเลขอย่างน้อยตัวละ 1 ตัว
                    </p>
                    {passwordData.newPassword && (
                      <div style={{ margin: "10px 0", fontSize: "12px" }}>
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            margin: "3px 0",
                            color: passwordValidation.hasMinLength ? "#28a745" : "#dc3545",
                          }}
                        >
                          <span style={{ marginRight: "5px" }}>
                            {passwordValidation.hasMinLength ? "✅" : "❌"}
                          </span>
                          อย่างน้อย 8 ตัวอักษร
                        </div>
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            margin: "3px 0",
                            color: passwordValidation.hasLetter ? "#28a745" : "#dc3545",
                          }}
                        >
                          <span style={{ marginRight: "5px" }}>
                            {passwordValidation.hasLetter ? "✅" : "❌"}
                          </span>
                          มีตัวอักษรอย่างน้อย 1 ตัว (a-z, A-Z)
                        </div>
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            margin: "3px 0",
                            color: passwordValidation.hasNumber ? "#28a745" : "#dc3545",
                          }}
                        >
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
                    <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
                      <input
                        type={showConfirmPassword ? "text" : "password"}
                        name="confirmPassword"
                        value={passwordData.confirmPassword}
                        onChange={handlePasswordChange}
                        placeholder="ยืนยันรหัสผ่านใหม่"
                        className={`form-input-edit ${passwordErrors.confirmPassword ? "error" : ""}`}
                        style={{ paddingRight: "40px", fontSize: "14px" }}
                      />
                      <button
                        type="button"
                        onClick={toggleConfirmPasswordVisibility}
                        style={{
                          position: "absolute",
                          right: "10px",
                          top: "50%",
                          transform: "translateY(-50%)",
                          background: "none",
                          border: "none",
                          cursor: "pointer",
                          padding: "0",
                          zIndex: 10,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          width: "24px",
                          height: "24px",
                        }}
                      >
                        <img
                          src={showConfirmPassword ? "/img/hide.png" : "/img/view.png"}
                          alt={showConfirmPassword ? "hide" : "view"}
                          style={{ width: "20px", height: "20px", marginBottom: "20px" }}
                        />
                      </button>
                    </div>
                    {passwordErrors.confirmPassword && (
                      <p className="error-message">{passwordErrors.confirmPassword}</p>
                    )}
                  </div>
                </>
              )}

              {!changePassword && changeEmail && (
                <div className="form-group-edit">
                  <label>กรุณากรอกรหัสผ่านเดิมเพื่อยืนยันตัวตน:</label>
                  <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
                    <input
                      type={showCurrentPassword ? "text" : "password"}
                      name="currentPassword"
                      value={passwordData.currentPassword}
                      onChange={handlePasswordChange}
                      placeholder="รหัสผ่านเดิม"
                      className={`form-input-edit ${passwordErrors.currentPassword ? "error" : ""}`}
                      style={{ paddingRight: "40px", fontSize: "14px" }}
                    />
                    <button
                      type="button"
                      onClick={toggleCurrentPasswordVisibility}
                      style={{
                        position: "absolute",
                        right: "10px",
                        top: "50%",
                        transform: "translateY(-50%)",
                        background: "none",
                        border: "none",
                        cursor: "pointer",
                        padding: "0",
                        zIndex: 10,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        width: "24px",
                        height: "24px",
                      }}
                    >
                      <img
                        src={showCurrentPassword ? "/img/hide.png" : "/img/view.png"}
                        alt={showCurrentPassword ? "hide" : "view"}
                        style={{ width: "20px", height: "20px", marginBottom: "20px" }}
                      />
                    </button>
                  </div>
                  {passwordErrors.currentPassword && (
                    <p className="error-message">{passwordErrors.currentPassword}</p>
                  )}
                </div>
              )}

              {!changePassword && (
                <button
                  onClick={handleDeleteOwnAccount}
                  className="report-link"
                  style={{
                    background: "none",
                    border: "none",
                    color: "#ff0000",
                    textDecoration: "underline",
                    cursor: "pointer",
                    fontSize: "inherit",
                    padding: 0,
                    marginLeft: "10px",
                    marginBottom: "20px",
                  }}
                >
                  ลบบัญชี
                </button>
              )}

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
                  disabled={loading}
                >
                  {loading ? "กำลังบันทึก..." : "บันทึกการเปลี่ยนแปลง"}
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