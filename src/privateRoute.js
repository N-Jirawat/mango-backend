// PrivateRoute.js
import { Navigate } from 'react-router-dom';

const PrivateRoute = ({ element, currentUser, requiredRole }) => {
  if (!currentUser) {
    return <Navigate to="/login" replace />;
  }

  if (requiredRole && currentUser.role !== requiredRole) {
    return <Navigate to="/" replace />;
  }

  return element; // ใช้ element แทนการเรียก Component
};

export default PrivateRoute;
