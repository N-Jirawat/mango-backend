import React, { useState, useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import { getAuth, signOut } from "firebase/auth";
import { FaCog } from "react-icons/fa";
import { db } from "./firebaseConfig";
import { doc, onSnapshot } from "firebase/firestore";
import "./css/profile.css";

function ProfileButton({ user, onUserUpdate }) {
  const [showDropdown, setShowDropdown] = useState(false);
  const [currentUser, setCurrentUser] = useState(user);
  const dropdownRef = useRef(null);
  const profileBtnRef = useRef(null);
  const auth = getAuth();

  const handleSignOut = () => {
    signOut(auth).then(() => {
      setShowDropdown(false);
      window.location.reload(); // รีเฟรชหน้าเว็บ
    }).catch((error) => {
      console.error("Sign out error:", error);
    });
  };

  // ฟังก์ชันสำหรับแสดงที่อยู่
  const formatAddress = () => {
    const addressParts = [];
    
    if (currentUser.address && currentUser.address.trim()) {
      addressParts.push(currentUser.address.trim());
    }
    
    if (currentUser.village && currentUser.village.trim()) {
      addressParts.push(`บ้าน ${currentUser.village.trim()}`);
    }
    
    if (currentUser.subdistrict && currentUser.subdistrict.trim()) {
      addressParts.push(`ตำบล ${currentUser.subdistrict.trim()}`);
    }
    
    if (currentUser.district && currentUser.district.trim()) {
      addressParts.push(`อำเภอ ${currentUser.district.trim()}`);
    }
    
    if (currentUser.province && currentUser.province.trim()) {
      addressParts.push(`จังหวัด ${currentUser.province.trim()}`);
    }
    
    return addressParts.length > 0 ? addressParts.join(", ") : "ไม่ได้ระบุ";
  };

  // ฟังการเปลี่ยนแปลงข้อมูลผู้ใช้แบบ real-time
  useEffect(() => {
    if (user && user.uid) {
      const userDocRef = doc(db, "users", user.uid);
      const unsubscribe = onSnapshot(userDocRef, (doc) => {
        if (doc.exists()) {
          const userData = doc.data();
          setCurrentUser({
            ...user,
            ...userData
          });
          
          // ส่งข้อมูลที่อัปเดตกลับไปให้ parent component
          if (onUserUpdate) {
            onUserUpdate({
              ...user,
              ...userData
            });
          }
        }
      }, (error) => {
        console.error("Error listening to user data:", error);
      });

      return () => unsubscribe();
    }
  }, [user, onUserUpdate]);

  // อัปเดต currentUser เมื่อ user prop เปลี่ยน
  useEffect(() => {
    setCurrentUser(user);
  }, [user]);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target) &&
        profileBtnRef.current &&
        !profileBtnRef.current.contains(event.target)
      ) {
        setShowDropdown(false);
      }
    };

    document.addEventListener("click", handleClickOutside);
    return () => document.removeEventListener("click", handleClickOutside);
  }, []);

  if (!currentUser) {
    return (
      <button className="profile-button">
        <Link to="/login">
          <img src="/img/user.png" alt="Profile Icon" />
          <span>เข้าสู่ระบบ</span>
        </Link>
      </button>
    );
  }

  return (
    <div style={{ position: "relative" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        {currentUser.role === "admin" && (
          <Link to="/admin-dashboard">
            <button className="settings-button">
              <FaCog size={20} />
            </button>
          </Link>
        )}

        <button
          className="profile-button"
          onClick={() => setShowDropdown(!showDropdown)}
          ref={profileBtnRef}
        >
          <img src="/img/user.png" alt="Profile Icon" />
          <span>{currentUser.username}</span>
        </button>
      </div>

      {showDropdown && (
        <div className="dropdown-menu" ref={dropdownRef}>
          <Link to={`/userdetails/${currentUser.uid}`}>
            <span>แก้ไขรายละเอียด</span>
          </Link>
          
          <p><strong>ชื่อเต็ม:</strong> {currentUser.fullName || "ไม่ได้ระบุ"}</p>
          
          <p><strong>ที่อยู่:</strong> {formatAddress()}</p>
          
          <p><strong>โทรศัพท์:</strong> {currentUser.tel && currentUser.tel.trim() ? currentUser.tel : "ไม่ได้ระบุ"}</p>
          
          <p><strong>อีเมล:</strong> {currentUser.email || "ไม่ได้ระบุ"}</p>
          
          <button onClick={handleSignOut}>ออกจากระบบ</button>
        </div>
      )}
    </div>
  );
}

export default ProfileButton;