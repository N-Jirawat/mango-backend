import React, { useState } from 'react';
import { Carousel } from 'react-responsive-carousel';
import 'react-responsive-carousel/lib/styles/carousel.min.css';
import { useNavigate } from 'react-router-dom';
import './css/homepage.css';

function Home({ currentUser, onProtectedNav }) { // เพิ่ม onProtectedNav prop
    const [currentSlide, setCurrentSlide] = useState(0);
    const totalSlides = 4;
    const navigate = useNavigate();

    const handlePath = (path) => {
        if (onProtectedNav) {
            // ใช้ function จาก App.js
            onProtectedNav(path);
        } else {
            // fallback ถ้าไม่มี onProtectedNav
            if (!currentUser) {
                navigate("/login");
            } else {
                navigate(path);
            }
        }
    };

    const handleNextSlide = () => {
        setCurrentSlide((prev) => (prev + 1) % totalSlides);
    };

    return (
        <div>
            <h1 className="home">Welcome to LeafAnalyzer</h1>
            <p className="home">วิเคราะห์โรคใบมะม่วงจากภาพถ่าย</p>
            <div className='container-button'>
                <button className="navigate-button" onClick={() => handlePath("/predict")}>
                    เริ่มต้นใช้งาน
                </button>
            </div>
            <div>
                <Carousel
                    selectedItem={currentSlide}
                    onChange={(index) => setCurrentSlide(index)}
                    autoPlay={true}
                    infiniteLoop={true}
                    interval={4000}
                    showArrows={false}         // ซ่อนลูกศร
                    showThumbs={false}
                    showStatus={false}
                    swipeable={true}           // เปิดให้เลื่อนภาพด้วย gesture (มือถือ) หรือ mouse drag
                    emulateTouch={true}        // เปิดให้คลิกหรือแตะที่ภาพแล้วเปลี่ยนภาพ
                >
                    <div onClick={handleNextSlide}>
                        <img src="/img/image_Healthy-spot_709.jpg" alt="" />
                    </div>
                    <div onClick={handleNextSlide}>
                        <img src="/img/image_Anthracnose_502.jpg" alt="" />
                    </div>
                    <div onClick={handleNextSlide}>
                        <img src="/img/image_Raised-spot_2773.jpg" alt="" />
                    </div>
                    <div onClick={handleNextSlide}>
                        <img src="/img/image_Sooty-mold_792.jpg" alt="" />
                    </div>
                </Carousel>
            </div>
        </div>
    );
}

export default Home;