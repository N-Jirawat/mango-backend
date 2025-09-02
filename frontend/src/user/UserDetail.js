import React, { useEffect, useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { getAuth, updatePassword, signInWithEmailAndPassword, signOut } from "firebase/auth";
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
  const [requiresReLogin, setRequiresReLogin] = useState(false); // เพิ่ม state เพื่อจัดการ re-login

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
    const unsubscribe = auth.onAuthStateChanged(async (currentUser) => {
      if (!currentUser) {
        navigate("/login");
        return;
      }

      try {
        const userDoc = await getDoc(doc(db, "users", id));
        if (userDoc.exists()) {
          setUserInfo(userDoc.data());
          setFormData(userDoc.data());
          setOriginalEmail(userDoc.data().email || "");
        } else {
          alert("ไม่พบข้อมูลผู้ใช้");
        }
      } catch (err) {
        console.error("Error fetching user data:", err);
        alert("เกิดข้อผิดพลาดในการโหลดข้อมูล");
      } finally {
        setLoading(false);
      }
    });

    return () => unsubscribe();
  }, [id, navigate]);

  // Save user reports to ReportDataUser
  useEffect(() => {
    const auth = getAuth();
    const unsubscribe = auth.onAuthStateChanged(async (currentUser) => {
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

    if (!changePassword && formData.tel && !/^0[0-9]{9}$/.test(formData.tel)) {
      setPhoneError("เบอร์โทรศัพท์ต้องเป็นตัวเลข 10 หลักและขึ้นต้นด้วย 0");
      isValid = false;
    } else {
      setPhoneError("");
    }

    if (!changePassword) {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (formData.email && !emailRegex.test(formData.email)) {
        setEmailError("รูปแบบอีเมลไม่ถูกต้อง");
        isValid = false;
      } else {
        setEmailError("");
      }
    }

    if (changePassword) {
      if (!passwordData.currentPassword) {
        errors.currentPassword = "กรุณากรอกรหัสผ่านเดิม";
        isValid = false;
      }

      if (!passwordData.newPassword) {
        errors.newPassword = "กรุณากรอกรหัสผ่านใหม่";
        isValid = false;
      } else if (!passwordValidation.hasMinLength || !passwordValidation.hasLetter || !passwordValidation.hasNumber) {
        errors.newPassword = "รหัสผ่านไม่ตรงตามเงื่อนไข";
        isValid = false;
      }

      if (!passwordData.confirmPassword) {
        errors.confirmPassword = "กรุณายืนยันรหัสผ่านใหม่";
        isValid = false;
      } else if (passwordData.newPassword !== passwordData.confirmPassword) {
        errors.confirmPassword = "รหัสผ่านไม่ตรงกัน";
        isValid = false;
      }
    }

    if (!changePassword && changeEmail && !passwordData.currentPassword) {
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

    if (!validateForm()) return;

    setLoading(true);

    try {
      console.log("Starting update process...", { changePassword, changeEmail });

      // Update Firestore data if not only changing password
      if (!changePassword || changeEmail) {
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
        setUserInfo((prev) => ({ ...prev, ...formData }));
        console.log("Firestore data updated");
      }

      // Update password using Firebase Client SDK
      if (changePassword) {
        console.log("Attempting to update password...");
        const user = auth.currentUser;
        if (!user) {
          throw new Error("ผู้ใช้ไม่ได้ล็อกอิน");
        }

        // Re-authenticate user with the latest email
        try {
          const emailToUse = changeEmail && formData.email ? formData.email : user.email;
          await signInWithEmailAndPassword(auth, emailToUse, passwordData.currentPassword);
          console.log("Re-authentication successful");
        } catch (authError) {
          console.error("Re-authentication error:", authError);
          if (authError.code === "auth/wrong-password") {
            throw new Error("รหัสผ่านเดิมไม่ถูกต้อง");
          } else if (authError.code === "auth/user-not-found" || authError.code === "auth/invalid-email") {
            throw new Error("อีเมลไม่ถูกต้องหรือไม่พบผู้ใช้ กรุณาลองใหม่");
          } else if (authError.code === "auth/too-many-requests") {
            throw new Error("มีการร้องขอมากเกินไป กรุณารอสักครู่แล้วลองใหม่");
          }
          throw new Error("เกิดข้อผิดพลาดในการยืนยันตัวตน: " + authError.message);
        }

        // Update password
        try {
          await updatePassword(user, passwordData.newPassword);
          console.log("Password updated successfully");
          alert("เปลี่ยนรหัสผ่านเรียบร้อยแล้ว");
        } catch (updateError) {
          console.error("Password update error:", updateError);
          if (updateError.code === "auth/requires-recent-login") {
            setRequiresReLogin(true);
            throw new Error("ต้องล็อกอินใหม่เพื่อเปลี่ยนรหัสผ่าน กรุณาลองอีกครั้ง");
          }
          throw new Error("เกิดข้อผิดพลาดในการเปลี่ยนรหัสผ่าน: " + updateError.message);
        }
      }

      // Update email via backend
      if (changeEmail) {
        console.log("Attempting to update email...");
        try {
          const response = await fetch(`${BACKEND_URL}/update_email`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "Accept": "application/json",
            },
            body: JSON.stringify({
              uid: id,
              new_email: formData.email,
              current_password: passwordData.currentPassword,
            }),
            signal: AbortSignal.timeout(30000),
          });

          if (!response.ok) {
            const errorText = await response.text();
            try {
              const errorJson = JSON.parse(errorText);
              throw new Error(errorJson.error || "Failed to update email");
            } catch {
              throw new Error(`HTTP ${response.status}: Failed to update email`);
            }
          }

          const result = await response.json();
          console.log("Email update successful:", result);
          alert("เปลี่ยนอีเมลเรียบร้อยแล้ว");
          setOriginalEmail(formData.email);
          // Force re-login after email change to refresh session
          await signOut(auth);
          alert("กรุณาล็อกอินใหม่ด้วยอีเมลใหม่");
          navigate("/login");
        } catch (fetchError) {
          console.error("Email update error:", fetchError);
          if (fetchError.name === "TimeoutError") {
            throw new Error("การเปลี่ยนอีเมลใช้เวลานานเกินไป กรุณาลองใหม่อีกครั้ง");
          } else if (fetchError.name === "TypeError" && fetchError.message.includes("Failed to fetch")) {
            throw new Error("ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์เพื่อเปลี่ยนอีเมลได้ อาจเป็นปัญหา CORS หรือเครือข่าย");
          } else {
            throw fetchError;
          }
        }
      }

      if (!changeEmail && !changePassword) {
        alert("อัปเดตข้อมูลเรียบร้อยแล้ว");
      }

      // Reset state
      setChangeEmail(false);
      setChangePassword(false);
      setPasswordData({ currentPassword: "", newPassword: "", confirmPassword: "" });
      setPasswordErrors({ currentPassword: "", newPassword: "", confirmPassword: "" });
      setPasswordValidation({ hasMinLength: false, hasLetter: false, hasNumber: false });
      setEditModal(false);
      setRequiresReLogin(false);
      console.log("Update process completed successfully");

    } catch (error) {
      console.error("Error updating user:", error);
      if (error.message.includes("รหัสผ่านเดิมไม่ถูกต้อง")) {
        alert("รหัสผ่านเดิมไม่ถูกต้อง กรุณาตรวจสอบและลองใหม่");
      } else if (error.message.includes("ต้องล็อกอินใหม่")) {
        alert("ต้องล็อกอินใหม่เพื่อดำเนินการ กรุณาล็อกอินด้วยอีเมลและรหัสผ่านของคุณ");
        navigate("/login");
      } else if (error.message.includes("CORS")) {
        alert("ปัญหาการเชื่อมต่อ CORS: " + error.message + "\nกรุณาแจ้งผู้ดูแลระบบ");
      } else if (error.message.includes("password")) {
        alert("เกิดข้อผิดพลาดในการเปลี่ยนรหัสผ่าน: " + error.message);
      } else if (error.message.includes("email")) {
        alert("เกิดข้อผิดพลาดในการเปลี่ยนอีเมล: " + error.message);
        setFormData((prev) => ({ ...prev, email: originalEmail }));
        setUserInfo((prev) => ({ ...prev, email: originalEmail }));
      } else if (error.message.includes("เชื่อมต่อ")) {
        alert("ปัญหาการเชื่อมต่อ: " + error.message + "\nกรุณาตรวจสอบการเชื่อมต่ออินเทอร์เน็ตและลองใหม่");
      } else {
        alert("เกิดข้อผิดพลาด: " + error.message);
      }
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
    setRequiresReLogin(false);

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
    setRequiresReLogin(false);
  };

  // Handle delete account
  const handleDeleteOwnAccount = async () => {
    const confirmDelete = window.confirm("คุณแน่ใจหรือไม่ว่าต้องการลบบัญชีของคุณ?");
    if (!confirmDelete) return;

    const currentPassword = window.prompt("กรุณากรอกรหัสผ่านเพื่อยืนยันการลบบัญชี:");
    if (!currentPassword) return;

    const user = auth.currentUser;
    if (user) {
      try {
        // Re-authenticate before deletion
        await signInWithEmailAndPassword(auth, user.email, currentPassword);
        
        const response = await fetch(`${BACKEND_URL}/delete_user`, {
          method: "DELETE",
          headers: {
            "Content-Type": "application/json",
            "Accept": "application/json",
          },
          body: JSON.stringify({ uid: user.uid }),
          signal: AbortSignal.timeout(30000),
        });

        if (!response.ok) {
          const errorText = await response.text();
          let errorMessage = "Failed to delete account";
          try {
            const errorJson = JSON.parse(errorText);
            errorMessage = errorJson.error || errorMessage;
          } catch {
            errorMessage = `HTTP ${response.status}: ${errorText}`;
          }
          throw new Error(errorMessage);
        }

        alert("บัญชีและข้อมูลถูกลบเรียบร้อยแล้ว");
        navigate("/login");
      } catch (error) {
        console.error("Error deleting account:", error);
        if (error.code === "auth/wrong-password") {
          alert("รหัสผ่านไม่ถูกต้อง กรุณาลองใหม่");
        } else if (error.code === "auth/too-many-requests") {
          alert("มีการร้องขอมากเกินไป กรุณารอสักครู่แล้วลองใหม่");
        } else if (error.message.includes("Failed to fetch")) {
          alert("ไม่สามารถลบบัญชี: ปัญหาการเชื่อมต่อ CORS หรือเครือข่าย กรุณาลองใหม่");
        } else {
          alert(`ไม่สามารถลบบัญชี: ${error.message}`);
        }
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
          ? `${userInfo.address}, บ้าน ${userInfo.village || "-"}, ตำบล ${userInfo.subdistrict || "-"}, อำเภอ ${userInfo.district || "-"}, จังหวัด ${userInfo.province || "-"}`
          : "-"}
      </p>
      <p><strong>เบอร์โทร :</strong> {userInfo.tel ? userInfo.tel : "-"}</p>
      <p><strong>อีเมล :</strong> {userInfo.email || "-"}</p>

      {requiresReLogin && (
        <p style={{ color: "red", fontWeight: "bold" }}>
          ต้องล็อกอินใหม่เพื่อดำเนินการเปลี่ยนรหัสผ่าน กรุณาล็อกอินด้วยอีเมลและรหัสผ่านของคุณ
        </p>
      )}

      <div className="link-container">
        <button
          onClick={() => {
            console.log("Change password button clicked");
            setChangePassword(true);
            setChangeEmail(false);
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
                    <label>รหัสผ่านเดิม:</label>
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