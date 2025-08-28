import { Routes, Route } from "react-router-dom";
import ProfileButton from "./ProfileButton";
import ResponsiveNav from "./ResponsiveNav";
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
import UserDetails from "./user/UserDetail";
import ShowMango from "./mango/showMango";
import UserMangoDetail from "./mango/UserMangoDetail";
import PredictPage from "./predict/PredictButton";
import ImageUpload from "./predict/ImageUpLoad";
import ResultAnaly from "./predict/ResultAnaly";
import PrefictHistory from "./history/PredictHistory"
import HistoryDetail from "./history/HistoryDetail";
import UserManual from "./UserManual";
import ForgotPasswordPage from "./user/ForgotPasswordPage";
import ReportUser from "./user/ReportUser";
import StatisticsAdmin from "./admin/StatisticsAdmin";
import ReportAdmin from "./admin/ReportAdmin";
import UserManualMobile from "./UserManualMobile";
import UserManualPC from "./UserManualPC";

export default function MainLayout({ currentUser, handleProtectedNav, navigateToHome }) {
  return (
    <>
      <header>
        <div className="logo" onClick={navigateToHome} style={{ cursor: 'pointer' }}>
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
          <Route path="/usermanualmobile" element={<UserManualMobile />} />
          <Route path="/usermanualpc" element={<UserManualPC />} />

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
                element={<StatisticsAdmin />}
              />
            }
          />
          <Route
            path="/Reportadmin"
            element={
              <PrivateRoute
                currentUser={currentUser}
                requiredRole="admin"
                element={<ReportAdmin />}
              />
            }
          />

          {/* Protected User Routes */}
          <Route
            path="/userdetails/:id"
            element={
              <PrivateRoute
                currentUser={currentUser}
                element={<UserDetails />}
              />
            }
          />
          <Route path="/Reportuser" element={<ReportUser />} />

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
                element={<PrefictHistory />}
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
    </>
  );
}