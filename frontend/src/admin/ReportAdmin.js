import React, { useEffect, useState } from "react";
import { getFirestore, collection, getDocs } from "firebase/firestore";
import { getAuth, onAuthStateChanged } from "firebase/auth";
import { useNavigate } from "react-router-dom";
import "../css/reportadmin.css";

import pdfMake from "pdfmake/build/pdfmake";
import pdfFonts from "../PDF/vfs_fonts"; // ชี้ไปที่ไฟล์ vfs_fonts.js ที่มี Sarabun-Bold

pdfMake.vfs = pdfFonts.pdfMake.vfs;

pdfMake.fonts = {
    Sarabun: {
        normal: "Sarabun-Bold.ttf",
        bold: "Sarabun-Bold.ttf",
        italics: "Sarabun-Bold.ttf",
        bolditalics: "Sarabun-Bold.ttf"
    }
};

function AdminReport() {
    const [usersReport, setUsersReport] = useState([]);
    const [loading, setLoading] = useState(true);
    const [currentPage, setCurrentPage] = useState(1);
    const usersPerPage = 10;
    const navigate = useNavigate();

    useEffect(() => {
        const auth = getAuth();
        const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
            if (!currentUser) {
                alert("กรุณาเข้าสู่ระบบ");
                navigate("/login");
                return;
            }

            const db = getFirestore();
            const userDoc = await getDocs(collection(db, "users"));
            const currentUserData = userDoc.docs.find(doc => doc.id === currentUser.uid)?.data();

            if (!currentUserData || currentUserData.role !== "admin") {
                alert("คุณไม่มีสิทธิ์เข้าถึงหน้านี้");
                navigate("/");
                return;
            }

            try {
                const usersSnapshot = await getDocs(collection(db, "users"));
                const users = usersSnapshot.docs
                    .map(doc => ({
                        id: doc.id,
                        ...doc.data(),
                    }))
                    .filter(user => user.role !== "admin");

                const diseasesList = ["ใบปกติ", "โรคแอนแทรคโนส", "โรคราดำ", "โรคใบจุดนูน"];

                const reportData = await Promise.all(users.map(async (user) => {
                    const analysisQuery = collection(db, "prediction_results");
                    const analysisSnapshot = await getDocs(analysisQuery);

                    const userAnalyses = analysisSnapshot.docs.filter(d => d.data().userId === user.id);

                    const diseaseCountMap = {};
                    userAnalyses.forEach(doc => {
                        const disease = doc.data().diseaseName || doc.data().predictedClass || "ไม่ระบุโรค";
                        diseaseCountMap[disease] = (diseaseCountMap[disease] || 0) + 1;
                    });

                    const mostFrequentDiseaseEntry = Object.entries(diseaseCountMap).sort((a, b) => b[1] - a[1])[0];

                    const diseaseCounts = {};
                    diseasesList.forEach(diseaseName => {
                        diseaseCounts[diseaseName] = diseaseCountMap[diseaseName] || 0;
                    });

                    return {
                        id: user.id,
                        username: user.username || "-",
                        fullName: user.fullName || "-",
                        email: user.email || "-",
                        analysisCount: userAnalyses.length,
                        mostFrequentDisease: mostFrequentDiseaseEntry
                            ? `${mostFrequentDiseaseEntry[0]} (${mostFrequentDiseaseEntry[1]} ครั้ง)`
                            : "-",
                        diseaseCounts,
                    };
                }));

                setUsersReport(reportData);
            } catch (error) {
                console.error("ไม่สามารถดึงข้อมูลรายงานได้:", error);
            } finally {
                setLoading(false);
            }
        });

        return () => unsubscribe();
    }, [navigate]);

    const sanitize = (value) => {
        if (typeof value === "string") {
            return value.replace(/[^฀-๿a-zA-Z0-9@_.\-\s()]/g, "").trim();
        }
        return value ?? "-";
    };

    const toNumber = (value) => {
        const num = parseFloat(value);
        return isNaN(num) ? 0 : num;
    };

    const generatePDF = () => {
        const docDefinition = {
            pageOrientation: 'landscape',
            content: [
                { text: "รายงานผลการใช้งานของสมาชิก", style: "header" },
                {
                    table: {
                        headerRows: 1,
                        widths: [
                            'auto',
                            '*',
                            '*',
                            '*',      // แก้จาก '1.2*' เป็น '*'
                            'auto',
                            '*',      // แก้จาก '1.2*' เป็น '*'
                            'auto',
                            'auto',
                            'auto',
                            'auto'
                        ],

                        body: [
                            ["ลำดับ", "ชื่อบัญชี", "ชื่อเต็ม", "อีเมล", "จำนวนภาพที่วิเคราะห์", "โรคที่พบมากที่สุด", "ใบปกติ", "โรคแอนแทรคโนส", "โรคราดำ", "โรคใบจุดนูน"],
                            ...usersReport.map((user, index) => [
                                index + 1,
                                sanitize(user.username),
                                sanitize(user.fullName),
                                sanitize(user.email),
                                toNumber(user.analysisCount),
                                sanitize(user.mostFrequentDisease),
                                toNumber(user.diseaseCounts["ใบปกติ"]),
                                toNumber(user.diseaseCounts["โรคแอนแทรคโนส"]),
                                toNumber(user.diseaseCounts["โรคราดำ"]),
                                toNumber(user.diseaseCounts["โรคใบจุดนูน"]),
                            ]),
                        ]
                    }
                }
            ],
            styles: {
                header: { fontSize: 16, bold: true, margin: [0, 0, 0, 10] }
            },
            defaultStyle: {
                font: 'Sarabun',
                fontSize: 9
            }
        };

        pdfMake.createPdf(docDefinition).download("รายงานสมาชิก.pdf");
    }

    const totalPages = Math.ceil(usersReport.length / usersPerPage);
    const indexOfLastUser = currentPage * usersPerPage;
    const indexOfFirstUser = indexOfLastUser - usersPerPage;
    const currentUsers = usersReport.slice(indexOfFirstUser, indexOfLastUser);

    const goToPage = (pageNum) => {
        if (pageNum >= 1 && pageNum <= totalPages) {
            setCurrentPage(pageNum);
        }
    };

    if (loading) return <p>กำลังโหลดรายงาน...</p>;

    return (
        <div className="table-wrapper">
            <div className="admin-report-container">
                <h2>รายงานผลการใช้งานของสมาชิก</h2>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", marginBottom: "10px" }}>
                    <button className="generate-report-btn" onClick={generatePDF}>
                        สร้างรายงานสมาชิก (PDF)
                    </button>
                    
                </div>

                <table className="admin-report-table">
                    <thead>
                        <tr>
                            <th>ลำดับ</th>
                            <th>ชื่อบัญชี</th>
                            <th>ชื่อเต็ม</th>
                            <th>อีเมล</th>
                            <th>จำนวนภาพที่วิเคราะห์</th>
                            <th>โรคที่พบมากที่สุด</th>
                            <th>ใบปกติ</th>
                            <th>โรคแอนแทรคโนส</th>
                            <th>โรคราดำ</th>
                            <th>โรคใบจุดนูน</th>
                        </tr>
                    </thead>
                    <tbody>
                        {currentUsers.length === 0 ? (
                            <tr><td colSpan="10" style={{ textAlign: "center" }}>ยังไม่มีข้อมูล</td></tr>
                        ) : (
                            currentUsers.map((user, index) => (
                                <tr key={user.id}>
                                    <td>{indexOfFirstUser + index + 1}</td>
                                    <td>{user.username}</td>
                                    <td>{user.fullName}</td>
                                    <td>{user.email}</td>
                                    <td>{user.analysisCount}</td>
                                    <td>{user.mostFrequentDisease}</td>
                                    <td>{user.diseaseCounts["ใบปกติ"]}</td>
                                    <td>{user.diseaseCounts["โรคแอนแทรคโนส"]}</td>
                                    <td>{user.diseaseCounts["โรคราดำ"]}</td>
                                    <td>{user.diseaseCounts["โรคใบจุดนูน"]}</td>
                                </tr>
                            ))
                        )}
                    </tbody>
                </table>
                <div className="pagination">
                        <button onClick={() => goToPage(currentPage - 1)} disabled={currentPage === 1}>⬅️ ก่อนหน้า</button>
                        <span>หน้า {currentPage} จาก {totalPages}</span>
                        <button onClick={() => goToPage(currentPage + 1)} disabled={currentPage === totalPages}>ถัดไป ➡️</button>
                </div>
            </div>
        </div>
        
    );
}

export default AdminReport;