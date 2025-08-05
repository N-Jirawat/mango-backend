import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { db } from "../firebaseConfig";
import { collection, getDocs, doc, deleteDoc, updateDoc } from "firebase/firestore";
import { getAuth, onAuthStateChanged } from "firebase/auth";

import provinces from "../่json/thai_provinces.json";
import districts from "../่json/thai_amphures.json";
import subdistricts from "../่json/thai_tambons.json";

function AccountManagement() {
  const navigate = useNavigate();
  const auth = getAuth();

  const [loading, setLoading] = useState(true);
  const [usersList, setUsersList] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const usersPerPage = 10;

  const [editUser, setEditUser] = useState(null);
  const [formData, setFormData] = useState({});
  const [districtList, setDistrictList] = useState([]);
  const [subdistrictList, setSubdistrictList] = useState([]);
  const [phoneError, setPhoneError] = useState("");
  const [dropdownOpenId, setDropdownOpenId] = useState(null);

  const toggleDropdown = (userId) => {
    setDropdownOpenId(prevId => prevId === userId ? null : userId);
  };

  useEffect(() => {
    const closeOnClickOutside = () => setDropdownOpenId(null);
    document.addEventListener('click', closeOnClickOutside);
    return () => document.removeEventListener('click', closeOnClickOutside);
  }, []);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      if (currentUser) {
        await fetchUsersList();
      } else {
        navigate("/login");
      }
    });
    return () => unsubscribe();
  }, [auth, navigate]);

  const fetchUsersList = async () => {
    setLoading(true);
    try {
      const usersRef = collection(db, "users");
      const snapshot = await getDocs(usersRef);
      const users = snapshot.docs.map((doc) => ({
        id: doc.id,
        ...doc.data(),
      }));
      setUsersList(users);
    } catch (error) {
      console.error("เกิดข้อผิดพลาดในการดึงรายชื่อผู้ใช้:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteUser = async (userId) => {
    const confirmDelete = window.confirm("คุณแน่ใจหรือไม่ว่าต้องการลบผู้ใช้นี้?");
    if (!confirmDelete) return;

    try {
      await deleteDoc(doc(db, "users", userId));
      setUsersList((prev) => prev.filter((user) => user.id !== userId));
    } catch (error) {
      console.error("เกิดข้อผิดพลาดในการลบผู้ใช้:", error);
      alert("ไม่สามารถลบผู้ใช้ได้");
    }
  };

  const openEditModal = (user) => {
    setEditUser(user);
    setFormData({ ...user });
    setPhoneError("");

    const province = provinces.find((p) => p.name_th === user.province);
    if (province) {
      const filteredDistricts = districts.filter((d) => d.province_id === province.id);
      setDistrictList(filteredDistricts);

      const district = filteredDistricts.find((d) => d.name_th === user.district);
      if (district) {
        const filteredSubdistricts = subdistricts.filter((t) => t.amphure_id === district.id);
        setSubdistrictList(filteredSubdistricts);
      } else {
        setSubdistrictList([]);
      }
    } else {
      setDistrictList([]);
      setSubdistrictList([]);
    }
  };

  const closeEditModal = () => {
    setEditUser(null);
    setFormData({});
    setDistrictList([]);
    setSubdistrictList([]);
    setPhoneError("");
  };

  const handleOverlayClick = (e) => {
    if (e.target === e.currentTarget) {
      closeEditModal();
    }
  };

  const handleProvinceChange = (provinceName) => {
    const selectedProvince = provinces.find((p) => p.name_th === provinceName);
    if (!selectedProvince) return;

    const filteredDistricts = districts.filter((d) => d.province_id === selectedProvince.id);
    setDistrictList(filteredDistricts);
    setSubdistrictList([]);

    setFormData((prev) => ({
      ...prev,
      province: provinceName,
      district: '',
      subdistrict: '',
    }));
  };

  const handleDistrictChange = (districtName) => {
    const selectedDistrict = districts.find((d) => d.name_th === districtName);
    if (!selectedDistrict) return;

    const filteredTambons = subdistricts.filter((t) => t.amphure_id === selectedDistrict.id);
    setSubdistrictList(filteredTambons);

    setFormData((prev) => ({
      ...prev,
      district: districtName,
      subdistrict: '',
    }));
  };

  const validatePhone = (phone) => {
    const phoneRegex = /^[0-9]{10}$/;
    return phoneRegex.test(phone);
  };

  const handleChange = (e) => {
    const { name, value } = e.target;

    if (name === "tel") {
      const numericValue = value.replace(/[^0-9]/g, '').slice(0, 10);
      setFormData((prev) => ({ ...prev, [name]: numericValue }));

      if (numericValue && !validatePhone(numericValue)) {
        setPhoneError("เบอร์โทรศัพท์ต้องเป็นตัวเลข 10 หลัก");
      } else {
        setPhoneError("");
      }
    } else if (name === "province") {
      handleProvinceChange(value);
    } else if (name === "district") {
      handleDistrictChange(value);
    } else {
      setFormData((prev) => ({ ...prev, [name]: value }));
    }
  };

  const handleSubmitUpdate = async () => {
    if (formData.tel && !validatePhone(formData.tel)) {
      setPhoneError("เบอร์โทรศัพท์ต้องเป็นตัวเลข 10 หลัก");
      return;
    }

    if (!formData.province || !formData.district || !formData.subdistrict) {
      alert("กรุณาเลือกจังหวัด อำเภอ และตำบล");
      return;
    }

    try {
      const updateData = { ...formData };
      delete updateData.username;

      await updateDoc(doc(db, "users", editUser.id), updateData);
      await fetchUsersList();
      closeEditModal();
      alert("อัปเดตข้อมูลเรียบร้อยแล้ว");
    } catch (error) {
      console.error("เกิดข้อผิดพลาดในการอัปเดต:", error);
      alert("เกิดข้อผิดพลาดในการอัปเดต");
    }
  };

  const indexOfLastUser = currentPage * usersPerPage;
  const indexOfFirstUser = indexOfLastUser - usersPerPage;
  const currentUsers = usersList.slice(indexOfFirstUser, indexOfLastUser);
  const totalPages = Math.ceil(usersList.length / usersPerPage);

  const goToPage = (pageNumber) => {
    if (pageNumber >= 1 && pageNumber <= totalPages) {
      setCurrentPage(pageNumber);
    }
  };

  if (loading) {
    return <div className="manage-container"><h2 className="title">กำลังโหลด...</h2></div>;
  }

  return (
    <div className="manage-container">
      <h2 className="title">บัญชีผู้ใช้</h2>
      <button className="btn btn-green btn-add-user" onClick={() => navigate("/signup")}>
        ➕ เพิ่มสมาชิก
      </button>

      <div class="table-responsive">
        <table className="user-table">
          <thead>
            <tr>
              <th>ลำดับ</th>
              <th>บัญชี</th>
              <th>อีเมล</th>
              <th>ชื่อ-นามสกุล</th>
              <th>ที่อยู่</th>
              <th>หมู่บ้าน</th>
              <th>อำเภอ</th>
              <th>ตำบล</th>
              <th>จังหวัด</th>
              <th>เบอร์โทร</th>
              <th>การจัดการ</th>
            </tr>
          </thead>
          <tbody>
            {currentUsers.length > 0 ? (
              currentUsers.map((user, index) => (
                <tr key={user.id}>
                  <td>{indexOfFirstUser + index + 1}</td>
                  <td>{user.username || "-"}</td>
                  <td>{user.email || "-"}</td>
                  <td>{user.fullName || "-"}</td>
                  <td>{user.address || "-"}</td>
                  <td>{user.village || "-"}</td>
                  <td>{user.district || "-"}</td>
                  <td>{user.subdistrict || "-"}</td>
                  <td>{user.province || "-"}</td>
                  <td>{user.tel || "-"}</td>
                  <td>
                    {/* สำหรับจอใหญ่ */}
                    <div className="btn-edit desktop-only">
                      <button className="btn btn-green" onClick={() => openEditModal(user)}>✏️ แก้ไข</button>
                      <button className="btn btn-gray" onClick={() => handleDeleteUser(user.id)}>🗑 ลบ</button>
                    </div>

                    {/* สำหรับจอเล็ก */}
                    <div className="action-menu mobile-only">
                      <button className="btn btn-gray more-btn" onClick={(e) => { e.stopPropagation(); toggleDropdown(user.id); }}>⋯</button>
                      {dropdownOpenId === user.id && (
                        <div className="dropdown-menu" onClick={(e) => e.stopPropagation()}>
                          <button className="dropdown-item-blue" onClick={() => openEditModal(user)}>✏️ แก้ไข</button>
                          <button className="dropdown-item" onClick={() => handleDeleteUser(user.id)}>🗑 ลบ</button>
                        </div>
                      )}
                    </div>
                  </td>


                </tr>
              ))
            ) : (
              <tr><td colSpan="11" className="no-users">ยังไม่มีผู้ใช้ในระบบ</td></tr>
            )}
          </tbody>
        </table> </div>

      {totalPages > 1 && (
        <div className="pagination">
          <button className="btn-next" onClick={() => goToPage(currentPage - 1)} disabled={currentPage === 1}>
            ⬅️
          </button>

          {(() => {
            const pages = [];

            if (totalPages <= 4) {
              for (let i = 1; i <= totalPages; i++) {
                pages.push(
                  <button
                    key={i}
                    className={`btn-next ${currentPage === i ? "btn-green-next" : ""}`}
                    onClick={() => goToPage(i)}
                    disabled={currentPage === i}
                  >
                    {i}
                  </button>
                );
              }
            } else {
              pages.push(
                <button
                  key={1}
                  className={`btn-next ${currentPage === 1 ? "btn-green-next" : ""}`}
                  onClick={() => goToPage(1)}
                  disabled={currentPage === 1}
                >
                  1
                </button>
              );

              if (currentPage > 3) {
                pages.push(
                  <span key="start-ellipsis" className="btn-green-next">
                    ...
                  </span>
                );
              }

              const start = Math.max(2, currentPage - 1);
              const end = Math.min(totalPages - 1, currentPage + 1);

              for (let i = start; i <= end; i++) {
                pages.push(
                  <button
                    key={i}
                    className={`btn-next ${currentPage === i ? "btn-green-next" : ""}`}
                    onClick={() => goToPage(i)}
                    disabled={currentPage === i}
                  >
                    {i}
                  </button>
                );
              }

              if (currentPage < totalPages - 2) {
                pages.push(
                  <span key="end-ellipsis" className="btn-green-next">
                    ...
                  </span>
                );
              }

              pages.push(
                <button
                  key={totalPages}
                  className={`btn-next ${currentPage === totalPages ? "btn-green-next" : ""}`}
                  onClick={() => goToPage(totalPages)}
                  disabled={currentPage === totalPages}
                >
                  {totalPages}
                </button>
              );
            }

            return pages;
          })()}

          <button className="btn-next" onClick={() => goToPage(currentPage + 1)} disabled={currentPage === totalPages}>
            ➡️
          </button>
        </div>
      )}


      {editUser && (
        <div className="modal-overlay" onClick={handleOverlayClick}>
          <div className="modal-content-edit">
            <div className="modal-header">
              <h2 className="modal-title">แก้ไขข้อมูลผู้ใช้</h2>
            </div>

            <div className="modal-body">
              <div className="form-group-edit">
                <label>ชื่อ-นามสกุล:</label>
                <input type="text" name="fullName" value={formData.fullName || ""} onChange={handleChange} required className="form-input-edit" />
              </div>

              <div className="form-group-edit">
                <label>ที่อยู่:</label>
                <textarea name="address" value={formData.address || ""} onChange={handleChange} required rows="2" className="form-input-edit form-textarea-edit" />
              </div>

              <div className="form-group-edit">
                <label>หมู่บ้าน:</label>
                <input type="text" name="village" value={formData.village || ""} onChange={handleChange} className="form-input-edit" />
              </div>

              <div className="form-row-edit">
                <div className="form-group-edit">
                  <label>จังหวัด: </label>
                  <select name="province" value={formData.province || ""} onChange={handleChange} required className="form-select-edit">
                    <option value="">เลือกจังหวัด</option>
                    {provinces.map((province) => (
                      <option key={province.id} value={province.name_th}>{province.name_th}</option>
                    ))}
                  </select>
                </div>

                <div className="form-group-edit">
                  <label>อำเภอ: </label>
                  <select name="district" value={formData.district || ""} onChange={handleChange} required disabled={!districtList.length} className="form-select-edit">
                    <option value="">เลือกอำเภอ</option>
                    {districtList.map((districtItem) => (
                      <option key={districtItem.id} value={districtItem.name_th}>{districtItem.name_th}</option>
                    ))}
                  </select>
                </div>

                <div className="form-group-edit">
                  <label>ตำบล: </label>
                  <select name="subdistrict" value={formData.subdistrict || ""} onChange={handleChange} required disabled={!subdistrictList.length} className="form-select-edit">
                    <option value="">เลือกตำบล</option>
                    {subdistrictList.map((subItem) => (
                      <option key={subItem.id} value={subItem.name_th}>{subItem.name_th}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="form-group-edit">
                <label>เบอร์โทรศัพท์:</label>
                <input type="tel" name="tel" value={formData.tel || ""} onChange={handleChange} placeholder="0812345678" className={`form-input-edit ${phoneError ? 'error' : ''}`} />
                {phoneError && <p className="error-message">{phoneError}</p>}
              </div>

              <div className="form-buttons-edit">
                <button type="button" onClick={closeEditModal} className="btn btn-gray">ยกเลิก</button>
                <button type="button" onClick={handleSubmitUpdate} className="btn btn-green">บันทึกการเปลี่ยนแปลง</button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default AccountManagement;