import React, { useEffect, useState } from "react";
import { collection, getDocs, doc, getDoc } from "firebase/firestore";
import { getAuth, onAuthStateChanged } from "firebase/auth";
import { db } from "../firebaseConfig";
import { Link, useNavigate } from "react-router-dom";
import "../css/showmango.css";


function ShowMango() {
  const [mangoData, setMangoData] = useState([]);
  const [role, setRole] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchData = async () => {
      const querySnapshot = await getDocs(collection(db, "mango_diseases"));
      const data = querySnapshot.docs.map((doc) => ({ id: doc.id, ...doc.data() }));
      setMangoData(data);
    };

    const fetchUserRole = async () => {
      const auth = getAuth();
      onAuthStateChanged(auth, async (user) => {
        if (user) {
          const docRef = doc(db, "users", user.uid);
          const docSnap = await getDoc(docRef);
          if (docSnap.exists()) {
            setRole(docSnap.data().role);
          }
        }
      });
    };

    fetchData();
    fetchUserRole();
  }, []);

  const handleBack = () => {
    if (role === "true" || role === true) {
      navigate("/accountmanagement");
    } else {
      navigate("/");
    }
  };

  return (
    <div className="show-mango-container">
      <button onClick={handleBack} className="back-button">
        ⬅️ หน้าหลัก
      </button>
      <h2>รายการโรคใบมะม่วง</h2>
      <div className="mango-card-grid">
        {mangoData.map((item) => (
          <div className="mango-card" key={item.id}>
            <img src={item.imageUrl} alt={item.diseaseName} />
            <h3>{item.diseaseName}</h3>
            <Link to={`/usermangodetail/${item.id}`}>
              <button className="view-details-btn">ดูรายละเอียด</button>
            </Link>
          </div>
        ))}
      </div>
    </div>
  );
}

export default ShowMango;
