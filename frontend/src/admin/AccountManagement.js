import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { db, auth } from "../firebaseConfig";
import { collection, getDocs, doc, updateDoc } from "firebase/firestore";
import { onAuthStateChanged } from "firebase/auth";

import provinces from "../่json/thai_provinces.json";
import districts from "../่json/thai_amphures.json";
import subdistricts from "../่json/thai_tambons.json";

function AccountManagement() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [usersList, setUsersList] = useState([]);
  const [filteredUsers, setFilteredUsers] = useState([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const usersPerPage = 10;

  const [editUser, setEditUser] = useState(null);
  const [formData, setFormData] = useState({});
  const [districtList, setDistrictList] = useState([]);
  const [subdistrictList, setSubdistrictList] = useState([]);
  const [phoneError, setPhoneError] = useState("");
  const [dropdownOpenId, setDropdownOpenId] = useState(null);

  const [dropdownDirection, setDropdownDirection] = useState('drop-down');

  const USER_BACKEND_URL = "https://render-backend-mu.vercel.app";

  // แก้ไข useEffect สำหรับ dropdown แบบ fixed position
  useEffect(() => {
    if (dropdownOpenId) {
      const timer = setTimeout(() => {
        const buttonElement = document.querySelector(`[data-user-id="${dropdownOpenId}"] .more-btn`);
        const dropdownElement = document.querySelector(`[data-user-id="${dropdownOpenId}"] .dropdown-menu`);

        if (buttonElement && dropdownElement) {
          const buttonRect = buttonElement.getBoundingClientRect();
          const windowHeight = window.innerHeight;
          const windowWidth = window.innerWidth;
          const dropdownHeight = 300; // ประมาณ

          // คำนวณพื้นที่ว่าง
          const spaceBelow = windowHeight - buttonRect.bottom;
          const spaceAbove = buttonRect.top;

          let top, right;

          // กำหนดตำแหน่งแนวนอน (ให้ชิดขวา)
          right = windowWidth - buttonRect.right;
          if (right < 0) right = 10; // ป้องกันล้นจอขวา

          // กำหนดตำแหน่งแนวตั้ง
          if (spaceBelow >= 150) {
            // เด้งลง
            top = buttonRect.bottom + 5;
            setDropdownDirection('drop-down');
          } else if (spaceAbove >= 150) {
            // เด้งขึ้น
            top = buttonRect.top - dropdownHeight - 5;
            setDropdownDirection('drop-up');
          } else {
            // พื้นที่ไม่พอทั้งสองด้าน - เลือกด้านที่มีพื้นที่มากกว่า
            if (spaceBelow > spaceAbove) {
              top = buttonRect.bottom + 5;
              setDropdownDirection('drop-down');
            } else {
              top = Math.max(10, buttonRect.top - dropdownHeight - 5);
              setDropdownDirection('drop-up');
            }
          }

          // ป้องกันล้นจอด้านบน
          if (top < 10) top = 10;
          // ป้องกันล้นจอด้านล่าง
          if (top + dropdownHeight > windowHeight - 10) {
            top = windowHeight - dropdownHeight - 10;
          }

          // ตั้งค่าตำแหน่ง
          dropdownElement.style.position = 'fixed';
          dropdownElement.style.top = `${top}px`;
          dropdownElement.style.right = `${right}px`;
          dropdownElement.style.left = 'auto';
          dropdownElement.style.bottom = 'auto';

          console.log('Dropdown positioned at:', {
            top,
            right,
            direction: spaceBelow >= 150 ? 'down' : 'up',
            buttonRect,
            spaceAbove,
            spaceBelow
          });
        }
      }, 10);

      return () => clearTimeout(timer);
    }
  }, [dropdownOpenId]);

  const toggleDropdown = (userId) => {
    if (dropdownOpenId === userId) {
      setDropdownOpenId(null);
    } else {
      setDropdownOpenId(userId);
    }
  };

  useEffect(() => {
    const closeOnClickOutside = () => setDropdownOpenId(null);
    document.addEventListener('click', closeOnClickOutside);
    return () => document.removeEventListener('click', closeOnClickOutside);
  }, []);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (!user) {
        navigate("/login");
      } else {
        fetchUsersList(); // ✅ โหลดข้อมูลผู้ใช้
      }
    });

    return () => unsubscribe();
  }, [navigate]);

  // ฟิลเตอร์ผู้ใช้ตามคำค้นหา
  useEffect(() => {
    if (!searchTerm.trim()) {
      setFilteredUsers(usersList);
    } else {
      const filtered = usersList.filter(user => {
        const username = (user.username || "").toLowerCase();
        const fullName = (user.fullName || "").toLowerCase();
        const search = searchTerm.toLowerCase();
        return username.includes(search) || fullName.includes(search);
      });
      setFilteredUsers(filtered);
    }
    setCurrentPage(1); // รีเซ็ตไปหน้าแรกเมื่อค้นหา
  }, [searchTerm, usersList]);

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
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteUser = async (uid) => {
    const confirmDelete = window.confirm("คุณแน่ใจหรือไม่ว่าต้องการลบผู้ใช้นี้?");
    if (!confirmDelete) return;

    try {
      const response = await fetch(`${USER_BACKEND_URL}/delete_user`, {
        method: "DELETE",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ uid }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(`Error ${response.status}: ${errorData.error || response.statusText}`);
      }

      setUsersList((prev) => prev.filter((user) => user.id !== uid));
      setDropdownOpenId(null);
      alert("ลบผู้ใช้เรียบร้อยแล้ว");

    } catch (error) {
      alert(`ลบผู้ใช้ไม่สำเร็จ: ${error.message}`);
    }
  };

  const handleSearchChange = (e) => {
    setSearchTerm(e.target.value);
  };

  const clearSearch = () => {
    setSearchTerm("");
  };

  const openEditModal = (user) => {
    setEditUser(user);
    setFormData({ ...user });
    setPhoneError("");

    setDropdownOpenId(null);

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
  const currentUsers = filteredUsers.slice(indexOfFirstUser, indexOfLastUser);
  const totalPages = Math.ceil(filteredUsers.length / usersPerPage);

  const goToPage = (pageNumber) => {
    if (pageNumber >= 1 && pageNumber <= totalPages) {
      setCurrentPage(pageNumber);
    }
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>กำลังโหลดข้อมูลสถิติ...</p>
      </div>
    );
  }

  return (
    <div className="manage-container">
      <h2 className="title">บัญชีผู้ใช้ทั้งหมด</h2>

      {/* ปุ่มเพิ่มสมาชิกอยู่กลาง */}
      <div className="add-member-row">
        <button className="btn btn-green btn-add-user" onClick={() => navigate("/signup")}>
          ➕ เพิ่มสมาชิก
        </button>
      </div>

      <div className="info-search-row">
        <div className="total-users">
          จำนวนรายการทั้งหมด: <strong>{usersList.length}</strong> รายการ
          {searchTerm && (
            <span className="filtered-count">{" "}(แสดง {filteredUsers.length} รายการ)</span>
          )}
        </div>

        <div className="search-wrapper">
          <input
            type="text"
            className="search-input"
            placeholder="ค้นหาด้วยชื่อบัญชีหรือชื่อสกุล..."
            value={searchTerm}
            onChange={handleSearchChange}
          />
          {searchTerm && (
            <button
              className="clear-search-btn"
              onClick={clearSearch}
              title="ล้างการค้นหา"
              aria-label="ล้างการค้นหา"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      <div className="table-responsive">
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
                    {/* ส่วน dropdown ใน JSX - แทนที่ในตาราง */}
                    <div className="action-menu mobile-only" data-user-id={user.id}>
                      <button
                        className="btn btn-gray more-btn"
                        onClick={(e) => { e.stopPropagation(); toggleDropdown(user.id); }}
                      >
                        ⋯
                      </button>

                      {dropdownOpenId === user.id && (
                        <div
                          className={`dropdown-menu ${dropdownDirection}`}
                          onClick={(e) => e.stopPropagation()}
                        >
                          <div className="user-details">
                            <div className="user-detail-row">
                              <span className="detail-label">บัญชี :</span>
                              <span className="detail-value">{user.username || '-'}</span>
                            </div>
                            <div className="user-detail-row">
                              <span className="detail-label">ชื่อ-นามสกุล :</span>
                              <span className="detail-value">{user.fullName || '-'}</span>
                            </div>
                            <div className="user-detail-row">
                              <span className="detail-label">อีเมล :</span>
                              <span className="detail-value">{user.email || '-'}</span>
                            </div>
                            <div className="user-detail-row">
                              <span className="detail-label">ที่อยู่ :</span>
                              <span className="detail-value">{user.address || '-'}</span>
                            </div>
                            <div className="user-detail-row">
                              <span className="detail-label">หมู่บ้าน :</span>
                              <span className="detail-value">{user.village || '-'}</span>
                              <span className="detail-label">ตำบล :</span>
                              <span className="detail-value">{user.subdistrict || '-'}</span>
                            </div>
                            <div className="user-detail-row">
                              <span className="detail-label">อำเภอ :</span>
                              <span className="detail-value">{user.district || '-'}</span>
                              <span className="detail-label">จังหวัด :</span>
                              <span className="detail-value">{user.province || '-'}</span>
                            </div>
                            <div className="user-detail-row">
                              <span className="detail-label">เบอร์โทร :</span>
                              <span className="detail-value">{user.tel || '-'}</span>
                            </div>
                          </div>
                          <div className="dropdown-actions">
                            <button
                              className="dropdown-item-blue"
                              onClick={() => openEditModal(user)}
                            >
                              ✏️ แก้ไข
                            </button>
                            <button
                              className="dropdown-item"
                              onClick={() => handleDeleteUser(user.id)}
                            >
                              🗑 ลบ
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="11" className="no-users">
                  {searchTerm ? `ไม่พบผู้ใช้ที่ตรงกับ "${searchTerm}"` : "ยังไม่มีผู้ใช้ในระบบ"}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

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
                <label>ชื่อบัญชี:</label>
                <input type="text" name="username" value={formData.username || ""} onChange={handleChange} required className="form-input-edit" disabled />
              </div>

              <div className="form-group-edit">
                <label>ชื่อ-นามสกุล:</label>
                <input type="text" name="fullName" value={formData.fullName || ""} onChange={handleChange} required className="form-input-edit" />
              </div>

              <div className="form-group-edit">
                <label>อีเมล:</label>
                <input type="text" name="email" value={formData.email || ""} onChange={handleChange} required className="form-input-edit" />
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