import React, { useState, useEffect } from "react";
import { db } from "../firebaseConfig"; // ✅ ลบ auth ออก
import { collection, query, where, getDocs, setDoc, doc } from "firebase/firestore";
import { getAuth, createUserWithEmailAndPassword, signOut, onAuthStateChanged } from "firebase/auth";
import { useNavigate } from "react-router-dom";  // เพิ่มการนำเข้า useNavigate

import provincesData from "../่json/thai_provinces.json";
import districtsData from "../่json/thai_amphures.json";
import subdistrictsData from "../่json/thai_tambons.json";

function SignupForm() {
  const auth = getAuth();
  const navigate = useNavigate(); // เพิ่มการใช้งาน useNavigate

  const [step, setStep] = useState(1);
  const [loading, setLoading] = useState(false);
  const [currentUserRole, setCurrentUserRole] = useState(null); // เพิ่มการเก็บ role ของผู้ใช้ปัจจุบัน

  // เพิ่ม state สำหรับการแสดง/ซ่อนรหัสผ่าน
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  // เพิ่ม state สำหรับแสดงสถานะการตรวจสอบรหัสผ่าน
  const [passwordValidation, setPasswordValidation] = useState({
    hasMinLength: false,
    hasLetter: false,
    hasNumber: false
  });

  // เพิ่ม state สำหรับตรวจสอบการซ้ำของข้อมูล
  const [duplicateCheck, setDuplicateCheck] = useState({
    username: { isDuplicate: false, isChecking: false },
    email: { isDuplicate: false, isChecking: false }
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
          console.error("Error fetching user role:", error);
          setCurrentUserRole("user"); // default to user if error
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
    } else {
      setSubdistricts([]);
    }
  }, [userInfo.district]);

  // ฟังก์ชันตรวจสอบชื่อบัญชีซ้ำ
  const checkDuplicateUsername = async (username) => {
    if (!username.trim()) {
      setDuplicateCheck(prev => ({
        ...prev,
        username: { isDuplicate: false, isChecking: false }
      }));
      return;
    }

    setDuplicateCheck(prev => ({
      ...prev,
      username: { ...prev.username, isChecking: true }
    }));

    try {
      const usersRef = collection(db, "users");
      const q = query(usersRef, where("username", "==", username));
      const querySnapshot = await getDocs(q);

      setDuplicateCheck(prev => ({
        ...prev,
        username: { isDuplicate: !querySnapshot.empty, isChecking: false }
      }));
    } catch (error) {
      console.error("Error checking username:", error);
      setDuplicateCheck(prev => ({
        ...prev,
        username: { isDuplicate: false, isChecking: false }
      }));
    }
  };

  // ฟังก์ชันตรวจสอบอีเมลซ้ำ
  const checkDuplicateEmail = async (email) => {
    if (!email.trim()) {
      setDuplicateCheck(prev => ({
        ...prev,
        email: { isDuplicate: false, isChecking: false }
      }));
      return;
    }

    setDuplicateCheck(prev => ({
      ...prev,
      email: { ...prev.email, isChecking: true }
    }));

    try {
      const usersRef = collection(db, "users");
      const q = query(usersRef, where("email", "==", email));
      const querySnapshot = await getDocs(q);

      setDuplicateCheck(prev => ({
        ...prev,
        email: { isDuplicate: !querySnapshot.empty, isChecking: false }
      }));
    } catch (error) {
      console.error("Error checking email:", error);
      setDuplicateCheck(prev => ({
        ...prev,
        email: { isDuplicate: false, isChecking: false }
      }));
    }
  };

  // เพิ่ม debounce สำหรับการตรวจสอบ
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (formData.username) {
        checkDuplicateUsername(formData.username);
      }
    }, 500); // รอ 500ms หลังจากหยุดพิมพ์

    return () => clearTimeout(timeoutId);
  }, [formData.username]);

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (formData.email) {
        checkDuplicateEmail(formData.email);
      }
    }, 500); // รอ 500ms หลังจากหยุดพิมพ์

    return () => clearTimeout(timeoutId);
  }, [formData.email]);

  const handleChange = (e) => {
    const { name, value } = e.target;

    // ป้องกันช่องว่างในชื่อบัญชี
    if (name === "username") {
      // ลบช่องว่างทั้งหมด
      const noSpaceValue = value.replace(/\s/g, "");
      setFormData({ ...formData, [name]: noSpaceValue });
    } else {
      setFormData({ ...formData, [name]: value });

      // ตรวจสอบรหัสผ่าน real-time
      if (name === "password") {
        setPasswordValidation({
          hasMinLength: value.length >= 8,
          hasLetter: /[a-zA-Z]/.test(value),
          hasNumber: /[0-9]/.test(value)
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
    setUserInfo({ ...userInfo, [e.target.name]: e.target.value });
  };

  // ฟังก์ชันสำหรับ toggle การแสดงรหัสผ่าน
  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  const toggleConfirmPasswordVisibility = () => {
    setShowConfirmPassword(!showConfirmPassword);
  };

  const validateEmail = (email) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);

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

  // ฟังก์ชันตรวจสอบความสมบูรณ์ของ Step 1
  const validateStep1 = () => {
    const { username, email, password, confirmPassword } = formData;

    if (!username.trim()) {
      alert("กรุณากรอกชื่อบัญชี");
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

  // ฟังก์ชันตรวจสอบความสมบูรณ์ของ Step 2
  const validateStep2 = () => {
    if (!userInfo.fullName.trim()) {
      alert("กรุณากรอกชื่อ-นามสกุล");
      return false;
    }
    return true;
  };

  // ฟังก์ชันสำหรับไปขั้นตอนถัดไป
  const handleNextStep = () => {
    if (validateStep1()) {
      setStep(2);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // ตรวจสอบ Step 2 ก่อนส่ง
    if (!validateStep2()) {
      return;
    }

    // ตรวจสอบการซ้ำอีกครั้งก่อนส่งข้อมูล
    if (duplicateCheck.username.isDuplicate) {
      alert("ชื่อบัญชีนี้ถูกใช้งานแล้ว กรุณาเลือกชื่อใหม่");
      setStep(1);
      return;
    }

    if (duplicateCheck.email.isDuplicate) {
      alert("อีเมลนี้ถูกใช้งานแล้ว กรุณาใช้อีเมลอื่น");
      setStep(1);
      return;
    }

    setLoading(true);

    try {
      // ตรวจสอบการซ้ำอีกครั้ง (double check)
      const usersRef = collection(db, "users");
      const usernameQuery = query(usersRef, where("username", "==", formData.username));
      const emailQuery = query(usersRef, where("email", "==", formData.email));

      const [usernameSnapshot, emailSnapshot] = await Promise.all([
        getDocs(usernameQuery),
        getDocs(emailQuery)
      ]);

      if (!usernameSnapshot.empty) {
        alert("ชื่อบัญชีนี้ถูกใช้งานแล้ว!");
        setStep(1);
        setLoading(false);
        return;
      }

      if (!emailSnapshot.empty) {
        alert("อีเมลนี้ถูกใช้งานแล้ว!");
        setStep(1);
        setLoading(false);
        return;
      }

      const allUsers = await getDocs(usersRef);
      const isFirstUser = allUsers.empty;
      const role = isFirstUser ? "admin" : "user";

      const userCredential = await createUserWithEmailAndPassword(auth, formData.email, formData.password);
      const user = userCredential.user;

      const docRef = doc(usersRef, user.uid);
      await setDoc(docRef, {
        uid: user.uid,
        username: formData.username,
        email: formData.email,
        fullName: userInfo.fullName,
        address: userInfo.address,
        village: userInfo.village,
        subdistrict: subdistricts.find(s => s.id === Number(userInfo.subdistrict))?.name_th || "",
        district: districts.find(d => d.id === Number(userInfo.district))?.name_th || "",
        province: provinces.find(p => p.id === Number(userInfo.province))?.name_th || "",
        tel: userInfo.tel.startsWith("") ? userInfo.tel : "0" + userInfo.tel,
        role: role,
      });

      // ตรวจสอบว่าผู้ใช้ปัจจุบันเป็นแอดมินหรือไม่
      if (currentUserRole === "admin") {
        // แอดมินไม่ต้องออกจากระบบ - กลับไปหน้า AccountManagement
        alert("เพิ่มสมาชิกสำเร็จ!");
        navigate("/admin-dashboard");
        // Reset form
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
        // Reset duplicate check
        setDuplicateCheck({
          username: { isDuplicate: false, isChecking: false },
          email: { isDuplicate: false, isChecking: false }
        });
      } else {
        // คนปกติที่มาสมัครสมาชิก - ออกจากระบบและไปหน้า login
        await signOut(auth);
        navigate("/login");
        alert("สมัครสมาชิกสำเร็จ!");
        // Reset form
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
        // Reset duplicate check
        setDuplicateCheck({
          username: { isDuplicate: false, isChecking: false },
          email: { isDuplicate: false, isChecking: false }
        });
      }
    } catch (error) {
      console.error("เกิดข้อผิดพลาด:", error);

      // แปลข้อความ error จาก Firebase
      let errorMessage = "ไม่สามารถสมัครสมาชิกได้!";
      if (error.code === 'auth/email-already-in-use') {
        errorMessage = "อีเมลนี้ถูกใช้งานแล้ว กรุณาใช้อีเมลอื่น";
        setStep(1);
      } else if (error.code === 'auth/weak-password') {
        errorMessage = "รหัสผ่านไม่ปลอดภัย กรุณาใช้รหัสผ่านที่แข็งแกร่งกว่านี้";
        setStep(1);
      } else if (error.code === 'auth/invalid-email') {
        errorMessage = "รูปแบบอีเมลไม่ถูกต้อง";
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
          <input style={{ fontSize: '14px' }} type="text" name="username" placeholder="ชื่อบัญชี :" value={formData.username} onChange={handleChange} />
          <input style={{ fontSize: '14px' }} type="email" name="email" placeholder="อีเมล : เช่น Test000@gmail.com" value={formData.email} onChange={handleChange} />

          {/* Password field with toggle */}
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            <input
              type={showPassword ? "text" : "password"}
              name="password"
              placeholder="รหัสผ่าน :"
              value={formData.password}
              onChange={handleChange}
              style={{
                width: '100%',
                paddingRight: '40px',
                boxSizing: 'border-box',
                border: '1px solid #ccc',
                borderRadius: '4px',
                padding: '10px',
                fontSize: '14px'
              }}
            />
            <button
              type="button"
              onClick={togglePasswordVisibility}
              style={{
                position: 'absolute',
                right: '10px',
                top: '50%',
                transform: 'translateY(-90%)',
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
                src={showPassword ? "/img/hide.png" : "/img/view.png"}
                alt={showPassword ? "hide" : "view"}
                style={{ width: '20px', height: '20px' }}
              />
            </button>
          </div>

          <p style={{ display: 'flex', fontSize: "12px", color: "#666", margin: "5px 0" }}>
            *รหัสผ่านต้องมีอย่างน้อย 8 ตัวอักษร และต้องประกอบด้วยตัวอักษรและตัวเลขอย่างน้อยอย่างละ 1 ตัว
          </p>

          {/* แสดงสถานะการตรวจสอบรหัสผ่าน */}
          {formData.password && (
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

          {/* Confirm Password field with toggle */}
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            <input
              type={showConfirmPassword ? "text" : "password"}
              name="confirmPassword"
              placeholder="ยืนยันรหัสผ่าน"
              value={formData.confirmPassword}
              onChange={handleChange}
              style={{
                width: '100%',
                paddingRight: '40px',
                boxSizing: 'border-box',
                border: '1px solid #ccc',
                borderRadius: '4px',
                padding: '10px',
                fontSize: '14px'
              }}
            />
            <button
              type="button"
              onClick={toggleConfirmPasswordVisibility}
              style={{
                position: 'absolute',
                right: '10px',
                top: '50%',
                transform: 'translateY(-90%)',
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
                style={{ width: '20px', height: '20px' }}
              />
            </button>
          </div>

          <button className="next-button" onClick={handleNextStep} disabled={loading}>
            ถัดไป →
          </button>
        </div>
      )}

      {step === 2 && (
        <div className="card">
          <h3>ข้อมูลเพิ่มเติม</h3>

          <input style={{ fontSize: '14px' }} type="text" name="fullName" placeholder="ชื่อ-นามสกุล :" value={userInfo.fullName} onChange={handleUserInfoChange} />
          <p style={{ display: 'flex', fontSize: "12px", color: "#666", margin: "5px 0" }}>
            *จำเป็นต้องกรอกชื่อ
          </p>
          <input style={{ fontSize: '14px' }} type="text" name="address" placeholder="ที่อยู่ : เช่น 55/5 หรือ บ้านเลขที่ 55 หมู่ 5" value={userInfo.address} onChange={handleUserInfoChange} />
          <input style={{ fontSize: '14px' }} type="text" name="village" placeholder="ชื่อหมู่บ้าน : เช่น กำเนิดเพขร" value={userInfo.village} onChange={handleUserInfoChange} />
          <div className="location-container">
            <select style={{ fontSize: '14px' }} name="province" value={userInfo.province} onChange={handleUserInfoChange}>
              <option value="">เลือกจังหวัด</option>
              {provinces.map((province) => (
                <option key={province.id} value={province.id}>{province.name_th}</option>
              ))}
            </select>

            <select style={{ fontSize: '14px' }} name="district" value={userInfo.district} onChange={handleUserInfoChange}>
              <option value="">เลือกอำเภอ</option>
              {districts.map((district) => (
                <option key={district.id} value={district.id}>{district.name_th}</option>
              ))}
            </select>

            <select style={{ fontSize: '14px' }} name="subdistrict" value={userInfo.subdistrict} onChange={handleUserInfoChange}>
              <option value="">เลือกตำบล</option>
              {subdistricts.map((subdistrict) => (
                <option key={subdistrict.id} value={subdistrict.id}>{subdistrict.name_th}</option>
              ))}
            </select>
          </div>
          <input
            style={{ fontSize: '14px' }}
            type="text"
            name="tel"
            placeholder="หมายเลขโทรศัพท์ :"
            value={userInfo.tel}
            onChange={handleUserInfoChange}
            maxLength="10"
          />
          <div className="button-save-signin">
            <button onClick={() => setStep(1)}>⬅ ย้อนกลับ</button>

            <button
              onClick={handleSubmit}
              disabled={loading}
              className="save-button"
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