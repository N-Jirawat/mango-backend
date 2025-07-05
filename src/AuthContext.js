import React, { createContext, useState, useContext, useEffect } from "react";
import { getAuth, onAuthStateChanged } from "firebase/auth";

// สร้าง context
const AuthContext = createContext();

// custom hook เพื่อใช้ใน components อื่นๆ
export const useAuth = () => useContext(AuthContext);

// Provider ที่จะครอบแอป
export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const auth = getAuth();

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser); // ตั้งค่า user เมื่อมีการล็อกอิน
    });
    return () => unsubscribe(); // ยกเลิกเมื่อ component ถูกทำลาย
  }, [auth]);

  return (
    <AuthContext.Provider value={{ user }}>
      {children} {/* ส่งข้อมูลผู้ใช้ให้กับ components ที่ต้องการ */}
    </AuthContext.Provider>
  );
};
