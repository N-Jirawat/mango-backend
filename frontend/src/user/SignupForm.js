import React, { useState, useEffect, useCallback } from "react";
import { db } from "../firebaseConfig";
import { collection, query, where, getDocs, setDoc, doc } from "firebase/firestore";
import { getAuth, createUserWithEmailAndPassword, signOut, onAuthStateChanged } from "firebase/auth";
import { useNavigate } from "react-router-dom";

import provincesData from "../่json/thai_provinces.json";
import districtsData from "../่json/thai_amphures.json";
import subdistrictsData from "../่json/thai_tambons.json";

function SignupForm() {
  const auth = getAuth();
  const navigate = useNavigate();

  const [step, setStep] = useState(1);
  const [loading, setLoading] = useState(false);
  const [currentUserRole, setCurrentUserRole] = useState(null);

  // State สำหรับการแสดง/ซ่อนรหัสผ่าน
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  // State สำหรับการตรวจสอบรหัสผ่าน
  const [passwordValidation, setPasswordValidation] = useState({
    hasMinLength: false,
    hasLetter: false,
    hasNumber: false,
  });

  // State สำหรับตรวจสอบข้อมูลซ้ำ
  const [duplicateCheck, setDuplicateCheck] = useState({
    username: { isDuplicate: false, isChecking: false },
    email: { isDuplicate: false, isChecking: false },
  });

  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    confirmPassword: "",
  });

  const [userInfo, setUserInfo] = useState({
    fullName: "",
    address: "",
    village: "",
    subdistrict: "",
    district: "",
    province: "",
    tel: "",
  });

  const [provinces, setProvinces] = useState([]);
  const [districts, setDistricts] = useState([]);
  const [subdistricts, setSubdistricts] = useState([]);

  // Email validation function
  const validateEmail = useCallback((email) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email), []);

  // API functions with timeout
  const checkUsername = useCallback(async (username) => {
    try {
      const response = await fetch("https://render-backend-mu.vercel.app/check_username", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        body: JSON.stringify({ username }),
        signal: AbortSignal.timeout(10000), // Timeout 10 วินาที
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }

      const data = await response.json();
      return data.exists || false;
    } catch (error) {
      if (error.name === "TimeoutError") {
        alert("การตรวจสอบชื่อบัญชีใช้เวลานานเกินไป กรุณาลองใหม่");
      } else if (error.message.includes("Failed to fetch")) {
        alert("ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์เพื่อตรวจสอบชื่อบัญชี อาจเป็นปัญหา CORS หรือเครือข่าย");
      }
      return false;
    }
  }, []);

  const checkEmail = useCallback(async (email) => {
    try {
      const response = await fetch("https://render-backend-mu.vercel.app/check_email", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        body: JSON.stringify({ email }),
        signal: AbortSignal.timeout(10000), // Timeout 10 วินาที
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }

      const data = await response.json();
      return data.exists || false;
    } catch (error) {
      if (error.name === "TimeoutError") {
        alert("การตรวจสอบอีเมลใช้เวลานานเกินไป กรุณาลองใหม่");
      } else if (error.message.includes("Failed to fetch")) {
        alert("ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์เพื่อตรวจสอบอีเมล อาจเป็นปัญหา CORS หรือเครือข่าย");
      }
      return false;
    }
  }, []);

  // Duplicate check functions
  const checkDuplicateUsername = useCallback(
    async (username) => {
      if (!username.trim()) {
        setDuplicateCheck((prev) => ({
          ...prev,
          username: { isDuplicate: false, isChecking: false },
        }));
        return;
      }

      setDuplicateCheck((prev) => ({
        ...prev,
        username: { isDuplicate: false, isChecking: true },
      }));

      try {
        const exists = await checkUsername(username);
        setDuplicateCheck((prev) => ({
          ...prev,
          username: { isDuplicate: exists, isChecking: false },
        }));
      } catch (error) {
        setDuplicateCheck((prev) => ({
          ...prev,
          username: { isDuplicate: false, isChecking: false },
        }));
      }
    },
    [checkUsername]
  );

  const checkDuplicateEmail = useCallback(
    async (email) => {
      if (!email.trim() || !validateEmail(email)) {
        setDuplicateCheck((prev) => ({
          ...prev,
          email: { isDuplicate: false, isChecking: false },
        }));
        return;
      }

      setDuplicateCheck((prev) => ({
        ...prev,
        email: { isDuplicate: false, isChecking: true },
      }));

      try {
        const exists = await checkEmail(email);
        setDuplicateCheck((prev) => ({
          ...prev,
          email: { isDuplicate: exists, isChecking: false },
        }));
      } catch (error) {
        setDuplicateCheck((prev) => ({
          ...prev,
          email: { isDuplicate: false, isChecking: false },
        }));
      }
    },
    [checkEmail, validateEmail]
  );

  useEffect(() => {
    setProvinces(provincesData);
  }, []);

  // ตรวจสอบ role ของผู้ใช้ปัจจุบัน
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      if (user) {
        try {
          const userDoc = await getDocs(query(collection(db, "users"), where("uid", "==", user.uid)));
          if (!userDoc.empty) {
            const userData = userDoc.docs[0].data();
            setCurrentUserRole(userData.role);
          }
        } catch (error) {
          setCurrentUserRole("user");
        }
      } else {
        setCurrentUserRole(null);
      }
    });

    return () => unsubscribe();
  }, [auth]);

  useEffect(() => {
    if (userInfo.province) {
      const filteredDistricts = districtsData.filter(
        (district) => district.province_id === Number(userInfo.province)
      );
      setDistricts(filteredDistricts);
      setUserInfo((prev) => ({ ...prev, district: "", subdistrict: "" }));
      setSubdistricts([]);
    } else {
      setDistricts([]);
      setSubdistricts([]);
    }
  }, [userInfo.province]);

  useEffect(() => {
    if (userInfo.district) {
      const filteredSubdistricts = subdistrictsData.filter(
        (subdistrict) => subdistrict.amphure_id === Number(userInfo.district)
      );
      setSubdistricts(filteredSubdistricts);
      setUserInfo((prev) => ({ ...prev, subdistrict: "" }));
    } else {
      setSubdistricts([]);
    }
  }, [userInfo.district]);

  // Debounce สำหรับการตรวจสอบ username
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (formData.username) {
        checkDuplicateUsername(formData.username);
      } else {
        setDuplicateCheck((prev) => ({
          ...prev,
          username: { isDuplicate: false, isChecking: false },
        }));
      }
    }, 500);

    return () => clearTimeout(timeoutId);
  }, [formData.username, checkDuplicateUsername]);

  // Debounce สำหรับการตรวจสอบ email
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (formData.email) {
        checkDuplicateEmail(formData.email);
      } else {
        setDuplicateCheck((prev) => ({
          ...prev,
          email: { isDuplicate: false, isChecking: false },
        }));
      }
    }, 500);

    return () => clearTimeout(timeoutId);
  }, [formData.email, checkDuplicateEmail]);

  const handleChange = (e) => {
    const { name, value } = e.target;

    if (name === "username") {
      const noSpaceValue = value.replace(/\s/g, "");
      setFormData({ ...formData, [name]: noSpaceValue });
    } else {
      setFormData({ ...formData, [name]: value });

      if (name === "password") {
        setPasswordValidation({
          hasMinLength: value.length >= 8,
          hasLetter: /[a-zA-Z]/.test(value),
          hasNumber: /[0-9]/.test(value),
        });
      }
    }
  };

  const handleUserInfoChange = (e) => {
    const { name, value } = e.target;

    if (name === "tel") {
      if (!/^\d*$/.test(value)) return;
      if (value.length > 10) return;
    }
    setUserInfo({ ...userInfo, [name]: value });
  };

  const togglePasswordVisibility = () => setShowPassword(!showPassword);
  const toggleConfirmPasswordVisibility = () => setShowConfirmPassword(!showConfirmPassword);

  const validatePassword = (password) => {
    if (password.length < 8) {
      return { isValid: false, message: "รหัสผ่านต้องมีอย่างน้อย 8 ตัว!" };
    }

    const hasLetter = /[a-zA-Z]/.test(password);
    if (!hasLetter) {
      return { isValid: false, message: "รหัสผ่านต้องมีตัวอักษรอย่างน้อย 1 ตัว!" };
    }

    const hasNumber = /[0-9]/.test(password);
    if (!hasNumber) {
      return { isValid: false, message: "รหัสผ่านต้องมีตัวเลขอย่างน้อย 1 ตัว!" };
    }

    return { isValid: true, message: "" };
  };

  const validateStep1 = () => {
    const { username, email, password, confirmPassword } = formData;

    if (!username.trim()) {
      alert("กรุณากรอกชื่อบัญชี");
      return false;
    }

    if (duplicateCheck.username.isChecking) {
      alert("กรุณารอการตรวจสอบชื่อบัญชี");
      return false;
    }

    if (duplicateCheck.username.isDuplicate) {
      alert("ชื่อบัญชีนี้ถูกใช้งานแล้ว กรุณาเลือกชื่อใหม่");
      return false;
    }

    if (!email.trim()) {
      alert("กรุณากรอกอีเมล");
      return false;
    }

    if (!validateEmail(email)) {
      alert("รูปแบบอีเมลไม่ถูกต้อง");
      return false;
    }

    if (duplicateCheck.email.isChecking) {
      alert("กรุณารอการตรวจสอบอีเมล");
      return false;
    }

    if (duplicateCheck.email.isDuplicate) {
      alert("อีเมลนี้ถูกใช้งานแล้ว กรุณาใช้อีเมลอื่น");
      return false;
    }

    if (!password.trim()) {
      alert("กรุณากรอกรหัสผ่าน");
      return false;
    }

    const passwordCheck = validatePassword(password);
    if (!passwordCheck.isValid) {
      alert(passwordCheck.message);
      return false;
    }

    if (!confirmPassword.trim()) {
      alert("กรุณายืนยันรหัสผ่าน");
      return false;
    }

    if (password !== confirmPassword) {
      alert("รหัสผ่านไม่ตรงกัน");
      return false;
    }

    return true;
  };

  const validateStep2 = () => {
    if (!userInfo.fullName.trim()) {
      alert("กรุณากรอกชื่อ-นามสกุล");
      return false;
    }
    if (userInfo.tel && !/^0[0-9]{9}$/.test(userInfo.tel)) {
      alert("เบอร์โทรศัพท์ต้องเป็นตัวเลข 10 หลักและขึ้นต้นด้วย 0");
      return false;
    }
    return true;
  };

  const handleNextStep = () => {
    if (validateStep1()) {
      setStep(2);
    }
  };

  const resetForm = () => {
    setFormData({ username: "", password: "", confirmPassword: "", email: "" });
    setUserInfo({
      fullName: "",
      address: "",
      village: "",
      subdistrict: "",
      district: "",
      province: "",
      tel: "",
    });
    setStep(1);
    setDuplicateCheck({
      username: { isDuplicate: false, isChecking: false },
      email: { isDuplicate: false, isChecking: false },
    });
    setPasswordValidation({
      hasMinLength: false,
      hasLetter: false,
      hasNumber: false,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateStep2()) return;

    if (duplicateCheck.username.isDuplicate || duplicateCheck.email.isDuplicate) {
      alert("พบข้อมูลซ้ำ กรุณาตรวจสอบและแก้ไข");
      setStep(1);
      return;
    }

    setLoading(true);

    try {
      // เช็คจำนวน users เพื่อกำหนด role (สำหรับ admin คนแรก)
      let role = "user";
      try {
        const allUsers = await getDocs(collection(db, "users"));
        const isFirstUser = allUsers.empty;
        role = isFirstUser ? "admin" : "user";
      } catch (error) {
        role = "user";
      }

      // สร้าง Firebase Auth account
      const userCredential = await createUserWithEmailAndPassword(auth, formData.email, formData.password);
      const user = userCredential.user;

      // สร้าง user document ใน Firestore
      const docRef = doc(collection(db, "users"), user.uid);
      await setDoc(docRef, {
        uid: user.uid,
        username: formData.username,
        email: formData.email,
        fullName: userInfo.fullName,
        address: userInfo.address || "",
        village: userInfo.village || "",
        subdistrict: subdistricts.find((s) => s.id === Number(userInfo.subdistrict))?.name_th || "",
        district: districts.find((d) => d.id === Number(userInfo.district))?.name_th || "",
        province: provinces.find((p) => p.id === Number(userInfo.province))?.name_th || "",
        tel: userInfo.tel || "", 
        role: role,
      });

      if (currentUserRole === "admin") {
        alert("เพิ่มสมาชิกสำเร็จ!");
        navigate("/admin-dashboard");
      } else {
        await signOut(auth);
        alert("สมัครสมาชิกสำเร็จ!");
        navigate("/login");
      }

      resetForm();
    } catch (error) {
      let errorMessage = "ไม่สามารถสมัครสมาชิกได้!";
      if (error.code === "auth/email-already-in-use") {
        errorMessage = "อีเมลนี้ถูกใช้งานแล้ว กรุณาใช้อีเมลอื่น";
        setDuplicateCheck((prev) => ({
          ...prev,
          email: { isDuplicate: true, isChecking: false },
        }));
        setStep(1);
      } else if (error.code === "auth/weak-password") {
        errorMessage = "รหัสผ่านไม่ปลอดภัย กรุณาใช้รหัสผ่านที่แข็งแกร่งกว่านี้";
        setStep(1);
      } else if (error.code === "auth/invalid-email") {
        errorMessage = "รูปแบบอีเมลไม่ถูกต้อง";
        setStep(1);
      } else if (error.code === "auth/too-many-requests") {
        errorMessage = "มีการร้องขอมากเกินไป กรุณารอสักครู่แล้วลองใหม่";
        setStep(1);
      }

      alert(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {step === 1 && (
        <div className="card">
          <h3>{currentUserRole === "admin" ? "เพิ่มสมาชิกใหม่" : "สมัครสมาชิก"}</h3>

          {/* Username field with validation */}
          <div style={{ position: "relative" }}>
            <input
              style={{ fontSize: "14px" }}
              type="text"
              name="username"
              placeholder="ชื่อบัญชี :"
              value={formData.username}
              onChange={handleChange}
              disabled={loading}
            />
            {duplicateCheck.username.isChecking && (
              <div style={{ fontSize: "12px", color: "#007bff", marginTop: "2px" }}>
                🔍 กำลังตรวจสอบ...
              </div>
            )}
            {formData.username && !duplicateCheck.username.isChecking && (
              <div
                style={{
                  fontSize: "12px",
                  color: duplicateCheck.username.isDuplicate ? "#dc3545" : "#28a745",
                  marginTop: "2px",
                }}
              >
                {duplicateCheck.username.isDuplicate
                  ? "❌ ชื่อบัญชีนี้ถูกใช้งานแล้ว"
                  : "✅ ชื่อบัญชีใช้งานได้"}
              </div>
            )}
          </div>

          {/* Email field with validation */}
          <div style={{ position: "relative" }}>
            <input
              style={{ fontSize: "14px" }}
              type="email"
              name="email"
              placeholder="อีเมล : เช่น Test000@gmail.com"
              value={formData.email}
              onChange={handleChange}
              disabled={loading}
            />
            {duplicateCheck.email.isChecking && (
              <div style={{ fontSize: "12px", color: "#007bff", marginTop: "2px" }}>
                🔍 กำลังตรวจสอบ...
              </div>
            )}
            {formData.email && validateEmail(formData.email) && !duplicateCheck.email.isChecking && (
              <div
                style={{
                  fontSize: "12px",
                  color: duplicateCheck.email.isDuplicate ? "#dc3545" : "#28a745",
                  marginTop: "2px",
                }}
              >
                {duplicateCheck.email.isDuplicate
                  ? "❌ อีเมลนี้ถูกใช้งานแล้ว"
                  : "✅ อีเมลใช้งานได้"}
              </div>
            )}
            {formData.email && !validateEmail(formData.email) && (
              <div style={{ fontSize: "12px", color: "#dc3545", marginTop: "2px" }}>
                ❌ รูปแบบอีเมลไม่ถูกต้อง
              </div>
            )}
          </div>

          {/* Password field with toggle */}
          <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
            <input
              type={showPassword ? "text" : "password"}
              name="password"
              placeholder="รหัสผ่าน :"
              value={formData.password}
              onChange={handleChange}
              style={{
                width: "100%",
                paddingRight: "40px",
                boxSizing: "border-box",
                border: "1px solid #ccc",
                borderRadius: "4px",
                padding: "10px",
                fontSize: "14px",
              }}
              disabled={loading}
            />
            <button
              type="button"
              onClick={togglePasswordVisibility}
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
              disabled={loading}
            >
              <img
                src={showPassword ? "/img/hide.png" : "/img/view.png"}
                alt={showPassword ? "hide" : "view"}
                style={{ width: "20px", height: "20px", marginBottom: "20px" }}
              />
            </button>
          </div>

          <p style={{ fontSize: "12px", color: "#666", margin: "5px 0" }}>
            *รหัสผ่านต้องมีอย่างน้อย 8 ตัวอักษร และต้องประกอบด้วยตัวอักษรและตัวเลขอย่างน้อยอย่างละ 1 ตัว
          </p>

          {/* Password validation indicators */}
          {formData.password && (
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

          {/* Confirm Password field with toggle */}
          <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
            <input
              type={showConfirmPassword ? "text" : "password"}
              name="confirmPassword"
              placeholder="ยืนยันรหัสผ่าน"
              value={formData.confirmPassword}
              onChange={handleChange}
              style={{
                width: "100%",
                paddingRight: "40px",
                boxSizing: "border-box",
                border: "1px solid #ccc",
                borderRadius: "4px",
                padding: "10px",
                fontSize: "14px",
              }}
              disabled={loading}
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
              disabled={loading}
            >
              <img
                src={showConfirmPassword ? "/img/hide.png" : "/img/view.png"}
                alt={showConfirmPassword ? "hide" : "view"}
                style={{ width: "20px", height: "20px", marginBottom: "20px" }}
              />
            </button>
          </div>

          {/* Password match indicator */}
          {formData.confirmPassword && (
            <div
              style={{
                fontSize: "12px",
                color: formData.password === formData.confirmPassword ? "#28a745" : "#dc3545",
                marginTop: "2px",
              }}
            >
              {formData.password === formData.confirmPassword
                ? "✅ รหัสผ่านตรงกัน"
                : "❌ รหัสผ่านไม่ตรงกัน"}
            </div>
          )}

          <button
            className="next-button"
            onClick={handleNextStep}
            disabled={loading || duplicateCheck.username.isChecking || duplicateCheck.email.isChecking}
            style={{
              background:
                loading || duplicateCheck.username.isChecking || duplicateCheck.email.isChecking
                  ? "#ccc"
                  : "#007bff",
              cursor:
                loading || duplicateCheck.username.isChecking || duplicateCheck.email.isChecking
                  ? "not-allowed"
                  : "pointer",
            }}
          >
            {loading || duplicateCheck.username.isChecking || duplicateCheck.email.isChecking
              ? "กำลังตรวจสอบ..."
              : "ถัดไป →"}
          </button>
        </div>
      )}

      {step === 2 && (
        <div className="card">
          <h3>ข้อมูลเพิ่มเติม</h3>

          <input
            style={{ fontSize: "14px" }}
            type="text"
            name="fullName"
            placeholder="ชื่อ-นามสกุล :"
            value={userInfo.fullName}
            onChange={handleUserInfoChange}
            required
            disabled={loading}
          />
          <p style={{ display: 'flex',fontSize: "12px", color: "#666", margin: "5px 0" }}>
            *จำเป็นต้องกรอกชื่อ
          </p>

          <input
            style={{ fontSize: "14px" }}
            type="text"
            name="address"
            placeholder="ที่อยู่ : เช่น 55/5 หรือ บ้านเลขที่ 55 หมู่ 5"
            value={userInfo.address}
            onChange={handleUserInfoChange}
            disabled={loading}
          />

          <input
            style={{ fontSize: "14px" }}
            type="text"
            name="village"
            placeholder="ชื่อหมู่บ้าน : เช่น กำเนิดเพขร"
            value={userInfo.village}
            onChange={handleUserInfoChange}
            disabled={loading}
          />

          <div className="location-container">
            <select
              style={{ fontSize: "14px" }}
              name="province"
              value={userInfo.province}
              onChange={handleUserInfoChange}
              disabled={loading}
            >
              <option value="">เลือกจังหวัด</option>
              {provinces.map((province) => (
                <option key={province.id} value={province.id}>
                  {province.name_th}
                </option>
              ))}
            </select>

            <select
              style={{ fontSize: "14px" }}
              name="district"
              value={userInfo.district}
              onChange={handleUserInfoChange}
              disabled={!userInfo.province || loading}
            >
              <option value="">เลือกอำเภอ</option>
              {districts.map((district) => (
                <option key={district.id} value={district.id}>
                  {district.name_th}
                </option>
              ))}
            </select>

            <select
              style={{ fontSize: "14px" }}
              name="subdistrict"
              value={userInfo.subdistrict}
              onChange={handleUserInfoChange}
              disabled={!userInfo.district || loading}
            >
              <option value="">เลือกตำบล</option>
              {subdistricts.map((subdistrict) => (
                <option key={subdistrict.id} value={subdistrict.id}>
                  {subdistrict.name_th}
                </option>
              ))}
            </select>
          </div>

          <input
            style={{ fontSize: "14px" }}
            type="text"
            name="tel"
            placeholder="หมายเลขโทรศัพท์ :"
            value={userInfo.tel}
            onChange={handleUserInfoChange}
            maxLength="10"
            pattern="[0-9]*"
            disabled={loading}
          />

          <div className="button-save-signin">
            <button onClick={() => setStep(1)} disabled={loading}>
              ⬅ ย้อนกลับ
            </button>

            <button
              onClick={handleSubmit}
              disabled={loading}
              className="save-button"
              style={{
                background: loading ? "#ccc" : "#28a745",
                cursor: loading ? "not-allowed" : "pointer",
              }}
            >
              {loading
                ? "กำลังบันทึก..."
                : currentUserRole === "admin"
                ? "เพิ่มสมาชิก ✅"
                : "บันทึก ✅"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default SignupForm;