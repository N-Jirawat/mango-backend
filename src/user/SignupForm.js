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

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleUserInfoChange = (e) => {
    const { name, value } = e.target;

    if (name === "tel") {
      if (!/^\d*$/.test(value)) return;
      if (value.length > 10) return;
    }
    setUserInfo({ ...userInfo, [e.target.name]: e.target.value });
  };

  const validateEmail = (email) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
  
    if (formData.password !== formData.confirmPassword) {
      alert("รหัสผ่านไม่ตรงกัน!");
      setLoading(false);
      return;
    }
  
    if (!validateEmail(formData.email)) {
      alert("รูปแบบอีเมลไม่ถูกต้อง!");
      setLoading(false);
      return;
    }
  
    try {
      const usersRef = collection(db, "users");
      const q = query(usersRef, where("username", "==", formData.username));
      const querySnapshot = await getDocs(q);
  
      if (!querySnapshot.empty) {
        alert("ชื่อบัญชีนี้ถูกใช้งานแล้ว!");
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
        tel: userInfo.tel.startsWith("0") ? userInfo.tel : "0" + userInfo.tel,
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
      }
    } catch (error) {
      console.error("เกิดข้อผิดพลาด:", error);
      alert("ไม่สามารถสมัครสมาชิกได้!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {step === 1 && (
        <div className="card">
          <h3>{currentUserRole === "admin" ? "เพิ่มสมาชิกใหม่" : "สมัครสมาชิก"}</h3>
          <input type="text" name="username" placeholder="ชื่อบัญชี :" value={formData.username} onChange={handleChange} />
          <input type="email" name="email" placeholder="อีเมล :" value={formData.email} onChange={handleChange} />
          <input type="password" name="password" placeholder="รหัสผ่าน :" value={formData.password} onChange={handleChange} />
          <input type="password" name="confirmPassword" placeholder="ยืนยันรหัสผ่าน" value={formData.confirmPassword} onChange={handleChange} />
          <button onClick={() => setStep(2)} disabled={loading}>ต่อไป ➡️</button>
        </div>
      )}

      {step === 2 && (
        <div className="card">
          <h3>ข้อมูลเพิ่มเติม</h3>
          <input type="text" name="fullName" placeholder="ชื่อผู้ใช้ :" value={userInfo.fullName} onChange={handleUserInfoChange} />
          <input type="text" name="address" placeholder="ที่อยู่ :" value={userInfo.address} onChange={handleUserInfoChange} />
          <input type="text" name="village" placeholder="หมู่บ้าน :" value={userInfo.village} onChange={handleUserInfoChange} />
          <div className="location-container">
            <select name="province" value={userInfo.province} onChange={handleUserInfoChange}>
              <option value="">เลือกจังหวัด</option>
              {provinces.map((province) => (
                <option key={province.id} value={province.id}>{province.name_th}</option>
              ))}
            </select>

            <select name="district" value={userInfo.district} onChange={handleUserInfoChange}>
              <option value="">เลือกอำเภอ</option>
              {districts.map((district) => (
                <option key={district.id} value={district.id}>{district.name_th}</option>
              ))}
            </select>

            <select name="subdistrict" value={userInfo.subdistrict} onChange={handleUserInfoChange}>
              <option value="">เลือกตำบล</option>
              {subdistricts.map((subdistrict) => (
                <option key={subdistrict.id} value={subdistrict.id}>{subdistrict.name_th}</option>
              ))}
            </select>
          </div>
          <input
            type="text"
            name="tel"
            placeholder="หมายเลขโทรศัพท์ :"
            value={userInfo.tel}
            onChange={handleUserInfoChange}
            maxLength="10"
          />
          <div className="button-container">
            <button onClick={() => setStep(1)}>⬅️ ย้อนกลับ</button>
            <button onClick={handleSubmit} disabled={loading}>
              {loading ? "กำลังบันทึก..." : currentUserRole === "admin" ? "เพิ่มสมาชิก ✅" : "บันทึก ✅"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default SignupForm;