import React, { useState, useEffect, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import "./css/responsive.css";

const ResponsiveNav = ({ currentUser, handleProtectedNav }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 769);
  const location = useLocation();
  const navigate = useNavigate();

  // ฟัง event เปลี่ยนขนาดหน้าจอ
  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 769;
      setIsMobile(mobile);
      if (!mobile) setIsMenuOpen(false);  // ปิดเมนูถ้าเป็น desktop
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // ปิดเมนูเมื่อเปลี่ยนหน้า
  useEffect(() => {
    setIsMenuOpen(false);
  }, [location.pathname]);

  // ปิดเมนูเมื่อกด Escape
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape' && isMenuOpen) {
        setIsMenuOpen(false);
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isMenuOpen]);

  // ป้องกันการ scroll เมื่อเปิดเมนูมือถือ
  useEffect(() => {
    if (isMobile && isMenuOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }

    // Cleanup เมื่อ component unmount
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isMobile, isMenuOpen]);

  const toggleMenu = useCallback(() => {
    setIsMenuOpen(prev => !prev);
  }, []);

  const closeMenu = useCallback(() => {
    setIsMenuOpen(false);
  }, []);

  // ตรวจสอบว่าลิงก์ปัจจุบันใช่หรือไม่
  const isActiveLink = (path) => {
    return location.pathname === path;
  };

  const navigationItems = [
    { path: '/', label: 'หน้าหลัก', type: 'link' },
    { path: '/showmango', label: 'โรคใบมะม่วง', type: 'link' },
    { path: '/predict', label: 'อัปโหลดรูปภาพ', type: 'link' },
    { path: '/history', label: 'ประวัติการวิเคราะห์', type: 'protected' },
    { path: '/usermanual', label: 'คู่มือการใช้งาน', type: 'link' }
  ];

  const adminItems = [
    { label: "สถิติผู้ใช้ทั้งหมด", path: "/statisticsadmin" }
  ];

  // ฟังก์ชันสำหรับจัดการการคลิกปุ่ม
  const handleNavClick = useCallback((item) => {
    closeMenu();
    if (item.type === 'protected') {
      handleProtectedNav(item.path);
    } else {
      // ใช้ React Router สำหรับการนำทาง
      navigate(item.path);
    }
  }, [closeMenu, handleProtectedNav, navigate]);

  return (
    <>
      <nav role="navigation" aria-label="Main navigation">
        {/* แสดงปุ่ม hamburger เฉพาะมือถือ */}
        {isMobile && (
          <button
            className={`hamburger-menu ${isMenuOpen ? 'active' : ''}`}
            onClick={toggleMenu}
            aria-label={isMenuOpen ? 'Close navigation menu' : 'Open navigation menu'}
            aria-expanded={isMenuOpen}
            aria-controls="nav-links"
          >
            <div className="hamburger-line"></div>
            <div className="hamburger-line"></div>
            <div className="hamburger-line"></div>
          </button>
        )}

        {/* Navigation Links */}
        <ul
          id="nav-links"
          className={`nav-links ${isMenuOpen ? 'active' : ''}`}
          role="menubar"
          aria-hidden={isMobile && !isMenuOpen}
        >
          {[...navigationItems, ...(currentUser?.role === "admin" ? adminItems : [])].map((item) => (
            <li key={item.path} role="none">
              <button
                className={`nav-fixed-width ${isActiveLink(item.path) ? 'active' : ''}`}
                onClick={() => handleNavClick(item)}
                role="menuitem"
                aria-current={isActiveLink(item.path) ? 'page' : undefined}
              >
                {item.label}
              </button>
            </li>
          ))}
        </ul>

        {/* Overlay for mobile menu */}
        {isMobile && (
          <div
            className={`nav-overlay ${isMenuOpen ? 'active' : ''}`}
            onClick={closeMenu}
            aria-hidden="true"
          ></div>
        )}
      </nav>
    </>
  );
};

export default ResponsiveNav;