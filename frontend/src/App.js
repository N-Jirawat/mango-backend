import { Routes, Route, useNavigate } from "react-router-dom";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import React, { useEffect, useState } from "react";
import { getAuth, onAuthStateChanged } from "firebase/auth";
import { doc, getDoc } from "firebase/firestore";
import { db } from "./firebaseConfig";
import PrivateRoute from "./privateRoute";

// Components
import EditMango from "./editmango/EditMango";
import SignupForm from "./user/SignupForm";
import Mango from "./editmango/Mango";
import Home from "./Home";
import AddMango from "./editmango/addMango";
import MangoDetail from "./editmango/MangoDetail";
import LoginPage from "./LoginPage";
import AccountManagement from "./admin/AccountManagement";
import AddminDashbord from "./admin/AddminDashbord";
import UserDashboard from "./user/UserDashbord";
import EditUser from "./user/EditUser";
import UserDetails from "./user/UserDetail";
import ProfileButton from "./ProfileButton";
import ShowMango from "./mango/showMango";
import UserMangoDetail from "./mango/UserMangoDetail";
import PredictPage from "./predict/PredictButton";
import ImageUpload from "./predict/ImageUpLoad";
import ResultAnaly from "./predict/ResultAnaly";
import History from "./history/HistoryAnaly";
import HistoryDetail from "./history/HistoryDetail";
import UserManual from "./UserManual";
import ResponsiveNav from "./ResponsiveNav";
import ForgotPasswordPage from "./user/ForgotPasswordPage";
import StaticsUser from "./user/StatisticsUser";
import StatisticsAdmin from "./admin/StatisticsAdmin";

// CSS
import "./css/allstyle.css";
import "./css/location.css";
import "./css/addminmanage.css";
import "./App.css";
import "./css/edituser.css";
import "./css/manage.css";
import "./css/login.css";

function App() {
  const [currentUser, setCurrentUser] = useState(null);
  const [loading, setLoading] = useState(true); // เพิ่ม loading state
  const navigate = useNavigate();

  const handleProtectedNav = (path) => {
    if (!currentUser) {
      // ไม่แสดง toast ที่นี่ ให้แสดงในหน้า login แทน
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
              uid: user.uid // เพิ่ม uid ด้วย
            });
          } else {
            console.warn("User document not found");
            setCurrentUser(null);
          }
        } else {
          setCurrentUser(null);
        }
      } catch (error) {
        console.error("Error fetching user data:", error);
        setCurrentUser(null);
      } finally {
        setLoading(false); // หยุด loading เมื่อเสร็จ
      }
    });

    return () => unsubscribe();
  }, []);

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
      <header>
        <div className="logo">
          <img src="/img/leaf.png" alt="Logo" />
          <h1>LeafAnalyzer</h1>
        </div>
        <div className="profile-buttons">
          <ProfileButton user={currentUser} />
        </div>
      </header>

      <ResponsiveNav
        currentUser={currentUser}
        handleProtectedNav={handleProtectedNav}
      />

      <main className="main-content">
        <Routes>
          {/* Public Routes */}
          <Route
            path="/"
            element={
              <Home
                currentUser={currentUser}
                onProtectedNav={handleProtectedNav}
              />
            }
          />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/signup" element={<SignupForm />} />
          <Route path="/showmango" element={<ShowMango />} />
          <Route path="/usermangodetail/:id" element={<UserMangoDetail />} />
          <Route path="/usermanual" element={<UserManual />} />
          <Route path="/forgot-password" element={<ForgotPasswordPage />} />
          <Route path="/predict" element={<PredictPage />} />
          <Route path="/resultanaly" element={<ResultAnaly />} />

          {/* Protected Admin Routes */}
          <Route
            path="/admin-dashboard"
            element={
              <PrivateRoute
                currentUser={currentUser}
                requiredRole="admin"
                element={<AddminDashbord />}
              />
            }
          />
          <Route
            path="/accountmanagement"
            element={
              <PrivateRoute
                currentUser={currentUser}
                requiredRole="admin"
                element={<AccountManagement />}
              />
            }
          />
          <Route
            path="/mango"
            element={
              <PrivateRoute
                currentUser={currentUser}
                requiredRole="admin"
                element={<Mango />}
              />
            }
          />
          <Route
            path="/addmango"
            element={
              <PrivateRoute
                currentUser={currentUser}
                requiredRole="admin"
                element={<AddMango />}
              />
            }
          />
          <Route
            path="/editmango/:id"
            element={
              <PrivateRoute
                currentUser={currentUser}
                requiredRole="admin"
                element={<EditMango />}
              />
            }
          />
          <Route
            path="/mangodetail/:id"
            element={
              <PrivateRoute
                currentUser={currentUser}
                requiredRole="admin"
                element={<MangoDetail />}
              />
            }
          />
          <Route
            path="/statisticsadmin"
            element={
              <PrivateRoute
                currentUser={currentUser}
                requiredRole="admin"
                element={<StatisticsAdmin />}
              />
            }
          />

          {/* Protected User Routes */}
          <Route
            path="/user-dashboard"
            element={
              <PrivateRoute
                currentUser={currentUser}
                element={<UserDashboard />}
              />
            }
          />
          <Route
            path="/edituser/:id"
            element={
              <PrivateRoute
                currentUser={currentUser}
                element={<EditUser />}
              />
            }
          />
          <Route
            path="/userdetails/:id"
            element={
              <PrivateRoute
                currentUser={currentUser}
                element={<UserDetails />}
              />
            }
          />
          <Route path="/staticsuser" element={<StaticsUser />} />
          
          {/* Protected Routes for All Authenticated Users */}
          <Route
            path="/imageupload"
            element={
              <PrivateRoute
                currentUser={currentUser}
                element={<ImageUpload />}
              />
            }
          />
          <Route
            path="/history"
            element={
              <PrivateRoute
                currentUser={currentUser}
                element={<History />}
              />
            }
          />
          <Route
            path="/historydetail"
            element={
              <PrivateRoute
                currentUser={currentUser}
                element={<HistoryDetail />}
              />
            }
          />
        </Routes>
      </main>

      <footer>LeafAnalyzer &copy; 2025</footer>

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