import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from 'react-router-dom';
import { db } from "../firebaseConfig";
import { doc, getDoc, updateDoc, deleteDoc } from "firebase/firestore";
import provincesData from "../่json/thai_provinces.json"; // ข้อมูลจังหวัด
import districtsData from "../่json/thai_amphures.json"; // ข้อมูลอำเภอ
import subdistrictsData from "../่json/thai_tambons.json"; // ข้อมูลตำบล
import { getAuth, onAuthStateChanged, signOut } from "firebase/auth";

function EditUser() {
    const { id } = useParams();
    const navigate = useNavigate();
    const [loading, setLoading] = useState(false);
    const [userInfo, setUserInfo] = useState({
        fullName: "",
        address: "",
        village: "",
        subdistrict: "",
        district: "",
        province: "",
        tel: "",
    });

    const [districtList, setDistrictList] = useState([]);
    const [subdistrictList, setSubdistrictList] = useState([]);
    const [role, setRole] = useState(null);
    const auth = getAuth();

    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
            if (currentUser) {
                try {
                    const userDoc = await getDoc(doc(db, "users", currentUser.uid));
                    const userData = userDoc.data();
                    setRole(userData?.role);

                    // ป้องกันผู้ใช้ทั่วไปแก้ไขบัญชีคนอื่น
                    if (userData?.role !== "admin" && currentUser.uid !== id) {
                        alert("คุณไม่มีสิทธิ์เข้าถึงหน้านี้");
                        navigate("/");
                    }
                } catch (error) {
                    console.error("โหลด role ไม่สำเร็จ:", error);
                }
            } else {
                navigate("/login");
            }
        });

        return () => unsubscribe();
    }, [auth, navigate, id]);

    useEffect(() => {
        const fetchUser = async () => {
            setLoading(true);
            try {
                const docRef = doc(db, "users", id);
                const docSnap = await getDoc(docRef);
                if (docSnap.exists()) {
                    setUserInfo(docSnap.data());
                } else {
                    alert("ไม่พบข้อมูลผู้ใช้");
                    navigate("/accountmanagement");
                }
            } catch (error) {
                console.error("เกิดข้อผิดพลาดในการโหลดข้อมูล:", error);
            } finally {
                setLoading(false);
            }
        };

        fetchUser();
    }, [id, navigate]);

    useEffect(() => {
        if (userInfo.province) {
            const filteredDistricts = districtsData.filter(
                d => d.province_id === provincesData.find(p => p.name_th === userInfo.province)?.id
            );
            setDistrictList(filteredDistricts);
        } else {
            setDistrictList([]);
        }
        setSubdistrictList([]);
    }, [userInfo.province]);

    useEffect(() => {
        if (userInfo.district) {
            const filteredSubdistricts = subdistrictsData.filter(
                s => s.amphure_id === districtsData.find(d => d.name_th === userInfo.district)?.id
            );
            setSubdistrictList(filteredSubdistricts);
        } else {
            setSubdistrictList([]);
        }
    }, [userInfo.district]);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setUserInfo(prev => ({
            ...prev,
            [name]: value,
            ...(name === "province" ? { district: "", subdistrict: "" } : {}),
            ...(name === "district" ? { subdistrict: "" } : {})
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        // ตรวจสอบเฉพาะช่องที่จำเป็น (ในกรณีนี้เหลือเฉพาะชื่อ)
        if (!userInfo.fullName.trim()) {
            alert("กรุณากรอกชื่อ-นามสกุล");
            return;
        }

        // ตรวจสอบเบอร์โทรถ้ามีการกรอก
        if (userInfo.tel && !/^[0-9]{10}$/.test(userInfo.tel)) {
            alert("กรุณากรอกเบอร์โทรศัพท์ให้ถูกต้อง (10 หลัก)");
            return;
        }

        setLoading(true);
        try {
            const docRef = doc(db, "users", id);
            await updateDoc(docRef, {
                ...userInfo,
                tel: String(userInfo.tel || ""), // แปลงเป็น string หรือเก็บเป็นค่าว่าง
            });

            alert("ข้อมูลถูกอัปเดตเรียบร้อย!");
            if (role === "admin") {
                navigate("/accountmanagement");
            } else {
                navigate("/");
            }
        } catch (error) {
            console.error("ไม่สามารถอัปเดตข้อมูลได้:", error);
            alert("เกิดข้อผิดพลาดในการอัปเดตข้อมูล");
        } finally {
            setLoading(false);
        }
    };

    const handleDeleteAccount = async () => {
        if (window.confirm("คุณแน่ใจว่าจะลบบัญชีผู้ใช้?")) {
            setLoading(true);
            try {
                await deleteDoc(doc(db, "users", id));

                const currentUser = auth.currentUser;
                if (currentUser && currentUser.uid === id) {
                    await signOut(auth);
                    alert("บัญชีของคุณถูกลบแล้ว");
                    navigate("/");
                } else {
                    alert("บัญชีถูกลบเรียบร้อยแล้ว");
                    navigate("/accountmanagement");
                }
            } catch (error) {
                console.error("ไม่สามารถลบบัญชีได้:", error);
                alert("เกิดข้อผิดพลาดในการลบบัญชี");
            } finally {
                setLoading(false);
            }
        }
    };

    const handleBack = () => {
        if (role === "admin") {
            navigate(`/userdetails/${id}`);
        } else {
            navigate("/");
        }
    };

    if (loading) {
        return (
            <div className="loading-spinner" style={{ textAlign: "center", padding: "20px" }}>
                <strong>กำลังโหลดข้อมูล...</strong>
            </div>
        );
    }

    return (
        <div className="edit-user">
            <h2>แก้ไขข้อมูลผู้ใช้</h2>
            <form onSubmit={handleSubmit}>
                <input 
                    type="text" 
                    name="fullName" 
                    placeholder="ชื่อ-นามสกุล *" 
                    value={userInfo.fullName} 
                    onChange={handleChange} 
                    required
                />
                <input 
                    type="text" 
                    name="address" 
                    placeholder="ที่อยู่" 
                    value={userInfo.address} 
                    onChange={handleChange} 
                />
                <input 
                    type="text" 
                    name="village" 
                    placeholder="หมู่บ้าน" 
                    value={userInfo.village} 
                    onChange={handleChange} 
                />

                <select name="province" value={userInfo.province || ""} onChange={handleChange}>
                    <option value="">เลือกจังหวัด</option>
                    {provincesData.map(p => (
                        <option key={p.id} value={p.name_th}>{p.name_th}</option>
                    ))}
                </select>

                <select name="district" value={userInfo.district || ""} onChange={handleChange} disabled={!districtList.length}>
                    <option value="">เลือกอำเภอ</option>
                    {districtList.map(d => (
                        <option key={d.id} value={d.name_th}>{d.name_th}</option>
                    ))}
                </select>

                <select name="subdistrict" value={userInfo.subdistrict || ""} onChange={handleChange} disabled={!subdistrictList.length}>
                    <option value="">เลือกตำบล</option>
                    {subdistrictList.map(s => (
                        <option key={s.id} value={s.name_th}>{s.name_th}</option>
                    ))}
                </select>

                <input
                    type="tel"
                    name="tel"
                    placeholder="หมายเลขโทรศัพท์ (10 หลัก)"
                    value={userInfo.tel}
                    onChange={handleChange}
                    pattern="[0-9]{10}"
                    title="กรุณากรอกเบอร์โทรศัพท์ 10 หลัก"
                />

                <div style={{ display: "flex", gap: "10px", marginTop: "10px" }}>
                    <button type="button" onClick={handleBack} style={{ backgroundColor: "gray", color: "white" }}>ย้อนกลับ</button>
                    <button type="button" onClick={handleDeleteAccount} style={{ backgroundColor: "red", color: "white" }}>ลบบัญชี</button>
                    <button type="submit">บันทึก</button>
                </div>
            </form>
            
            <div style={{ marginTop: "15px", fontSize: "14px", color: "#666" }}>
                <p>หมายเหตุ: เฉพาะช่องที่มี * เท่านั้นที่จำเป็นต้องกรอก</p>
            </div>
        </div>
    );
}

export default EditUser;