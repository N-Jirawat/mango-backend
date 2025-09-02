import { Routes, Route, useNavigate } from "react-router-dom";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import React, { useEffect, useState } from "react";
import { getAuth, onAuthStateChanged } from "firebase/auth";
import { doc, getDoc } from "firebase/firestore";
import { db } from "./firebaseConfig";

// Components
import MainLayout from "./MainLayout";

import ResetPasswordPage from "./user/ResetPassword";

// CSS
import "./css/allstyle.css";
import "./css/location.css";
import "./css/addminmanage.css";
import "./App.css";
import "./css/manage.css";
import "./css/login.css";

function App() {
  const [currentUser, setCurrentUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const auth = getAuth();
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      try {
        if (user) {
          const docRef = doc(db, "users", user.uid);
          const docSnap = await getDoc(docRef);
          if (docSnap.exists()) {
            setCurrentUser({
              ...docSnap.data(),
              email: user.email,
              uid: user.uid
            });
          } else {
            setCurrentUser(null);
          }
        } else {
          setCurrentUser(null);
        }
      } catch (error) {
        setCurrentUser(null);
      } finally {
        setLoading(false);
      }
    });

    return () => unsubscribe();
  }, []);

  const handleProtectedNav = (path) => {
    if (!currentUser) {
      navigate("/login", {
        state: {
          message: "กรุณาเข้าสู่ระบบก่อนใช้งาน",
          redirectTo: path
        }
      });
    } else {
      navigate(path);
    }
  };

  const navigateToHome = () => {
    navigate("/");
  };

  // แสดง loading ระหว่างรอ auth check
  if (loading) {
    return (
      <div className="loading-container">
        <div>กำลังโหลด...</div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <Routes>
        {/* Route ที่ไม่ต้องการ MainLayout */}
        <Route path="/reset-password" element={<ResetPasswordPage />} />
        
        {/* Routes ที่ใช้ MainLayout */}
        <Route 
          path="/*" 
          element={
            <MainLayout 
              currentUser={currentUser}
              handleProtectedNav={handleProtectedNav}
              navigateToHome={navigateToHome}
            />
          }
        />
      </Routes>

      <ToastContainer
        position="top-right"
        autoClose={3000}
        hideProgressBar={false}
        newestOnTop={false}
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
      />
    </div>
  );
}

export default App;