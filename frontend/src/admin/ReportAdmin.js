import React, { useEffect, useState } from "react";
import {
    getFirestore, collection, query, where, getDocs, addDoc, updateDoc, arrayUnion, serverTimestamp
} from "firebase/firestore";
import { getAuth, onAuthStateChanged } from "firebase/auth";
import { useNavigate } from "react-router-dom";
import "../css/reportadmin.css";

import pdfMake from "pdfmake/build/pdfmake";
import pdfFonts from "../PDF/vfs_fonts";

pdfMake.vfs = pdfFonts.pdfMake.vfs;

pdfMake.fonts = {
    Sarabun: {
        normal: "Sarabun-Bold.ttf",
        bold: "Sarabun-Bold.ttf",
        italics: "Sarabun-Bold.ttf",
        bolditalics: "Sarabun-Bold.ttf"
    }
};

function ReportAdmin() {
    const [usersReport, setUsersReport] = useState([]);
    const [loading, setLoading] = useState(true);
    const [currentPage, setCurrentPage] = useState(1);
    const [saving, setSaving] = useState(false);
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

                const analysisSnapshot = await getDocs(collection(db, "AnalysisHistory"));

                // map ตาม userId
                const userReportsMap = {};
                analysisSnapshot.docs.forEach(doc => {
                    const data = doc.data();
                    const userId = data.userId;
                    if (!userReportsMap[userId]) userReportsMap[userId] = [];
                    userReportsMap[userId].push({
                        analysisId: doc.id,
                        diseaseId: data.diseaseName || data.predictedClass || null,
                        timestamp: data.timestamp || null
                    });
                });

                const diseasesList = ["ใบปกติ", "โรคแอนแทรคโนส", "โรคราดำ", "โรคใบจุดนูน"];

                const reportData = users.map(user => {
                    const analyses = userReportsMap[user.id] || [];

                    const diseaseCountMap = {};
                    analyses.forEach(a => {
                        const disease = a.diseaseId || "ไม่ระบุโรค";
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
                        analysisCount: analyses.length,
                        mostFrequentDisease: mostFrequentDiseaseEntry
                            ? `${mostFrequentDiseaseEntry[0]} (${mostFrequentDiseaseEntry[1]} ครั้ง)`
                            : "-",
                        diseaseCounts,
                        analysisHistory: analyses
                    };
                });

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

    // ✅ ฟังก์ชันบันทึกลง Firestore (ReportDataAdmin)
    // ✅ ฟังก์ชันบันทึกลง Firestore (ReportDataAdmin)
    const saveReportToFirestore = async (usersReport) => {
        if (!Array.isArray(usersReport) || usersReport.length === 0) {
            console.warn("ไม่มีข้อมูล usersReport ที่จะบันทึก:", usersReport);
            return;
        }

        const db = getFirestore();
        const colRef = collection(db, "ReportDataAdmin");

        for (const user of usersReport) {
            const q = query(colRef, where("UserID", "==", user.id));
            const querySnapshot = await getDocs(q);

            // ✅ แปลง analysis ของ user เป็น array object (safe check)
            const analysisArray = (user.analysisHistory || []).map(a => ({
                AnalysisID: a.analysisId || "-",
                DateReUser: a.timestamp?.seconds
                    ? new Date(a.timestamp.seconds * 1000).toISOString()
                    : null,
                DiseaseID: a.diseaseId || null
            }));

            if (querySnapshot.empty) {
                // 👉 ยังไม่มี UserID นี้ → สร้าง doc ใหม่
                await addDoc(colRef, {
                    DateReAdmin: serverTimestamp(),
                    UserID: user.id,
                    Analysis: analysisArray
                });
            } else {
                // 👉 มี UserID อยู่แล้ว → อัปเดต doc เดิม (เพิ่ม array)
                const docRef = querySnapshot.docs[0].ref;
                for (const analysis of analysisArray) {
                    await updateDoc(docRef, {
                        Analysis: arrayUnion(analysis),
                        DateReAdmin: serverTimestamp()
                    });
                }
            }
        }
    };

    // ✅ ฟังก์ชันสร้าง PDF
    const generatePDF = async () => {
        setSaving(true);
        try {
            const docDefinition = {
                pageOrientation: 'landscape',
                pageMargins: [40, 40, 40, 40],
                content: [
                    {
                        text: "รายงานผลการใช้งานของสมาชิก",
                        style: "header",
                        alignment: 'center',
                        margin: [0, 0, 0, 20]
                    },
                    {
                        table: {
                            headerRows: 1,
                            widths: ['auto', 60, 'auto', '*', 60, 'auto', 'auto', 'auto', 50, 60],
                            body: [
                                [
                                    { text: "ลำดับ", style: "tableHeader" },
                                    { text: "ชื่อบัญชี", style: "tableHeader" },
                                    { text: "ชื่อเต็ม", style: "tableHeader" },
                                    { text: "อีเมล", style: "tableHeader" },
                                    { text: "จำนวนภาพที่วิเคราะห์", style: "tableHeader" },
                                    { text: "โรคที่พบมากที่สุด", style: "tableHeader" },
                                    { text: "ใบปกติ", style: "tableHeader" },
                                    { text: "โรคแอนแทรคโนส", style: "tableHeader" },
                                    { text: "โรคราดำ", style: "tableHeader" },
                                    { text: "โรคใบจุดนูน", style: "tableHeader" }
                                ],
                                ...usersReport.map((user, index) => [
                                    { text: (index + 1).toString(), style: "tableCell" },
                                    { text: sanitize(user.username), style: "tableCell" },
                                    { text: sanitize(user.fullName), style: "tableCell" },
                                    { text: sanitize(user.email), style: "tableCell" },
                                    { text: toNumber(user.analysisCount).toString(), style: "tableCell" },
                                    { text: sanitize(user.mostFrequentDisease), style: "tableCell" },
                                    { text: toNumber(user.diseaseCounts?.["ใบปกติ"]).toString(), style: "tableCell" },
                                    { text: toNumber(user.diseaseCounts?.["โรคแอนแทรคโนส"]).toString(), style: "tableCell" },
                                    { text: toNumber(user.diseaseCounts?.["โรคราดำ"]).toString(), style: "tableCell" },
                                    { text: toNumber(user.diseaseCounts?.["โรคใบจุดนูน"]).toString(), style: "tableCell" },
                                ])
                            ]
                        },
                        layout: {
                            fillColor: (rowIndex) => (rowIndex === 0 ? "#CCCCCC" : null),
                            hLineWidth: () => 1,
                            vLineWidth: () => 1,
                            hLineColor: () => "#000000",
                            vLineColor: () => "#000000",
                            paddingLeft: () => 5,
                            paddingRight: () => 5,
                        },
                        alignment: "center"
                    }
                ],
                styles: {
                    header: { fontSize: 18, bold: true, font: "Sarabun" },
                    tableHeader: { fontSize: 10, bold: true, alignment: "left", font: "Sarabun", margin: [2, 2, 2, 2] },
                    tableCell: { fontSize: 9, alignment: "left", font: "Sarabun", margin: [2, 2, 2, 2] }
                },
                defaultStyle: { font: "Sarabun" }
            };
            alert('ดาวน์โหลดรายงานสำเร็จ!')

            // ✅ ดาวน์โหลด PDF
            pdfMake.createPdf(docDefinition).download("รายงานสมาชิก.pdf");

            // ✅ บันทึกลง Firestore ด้วย (ส่ง usersReport ไปด้วย)
            await saveReportToFirestore(usersReport);

        } catch (error) {
            alert("เกิดข้อผิดพลาด: " + error.message);
        } finally {
            setSaving(false);
        }
    };

    const totalPages = Math.ceil(usersReport.length / usersPerPage);
    const indexOfLastUser = currentPage * usersPerPage;
    const indexOfFirstUser = indexOfLastUser - usersPerPage;
    const currentUsers = usersReport.slice(indexOfFirstUser, indexOfLastUser);

    const goToPage = (pageNum) => {
        if (pageNum >= 1 && pageNum <= totalPages) setCurrentPage(pageNum);
    };

    if (loading) return <p>กำลังโหลดรายงาน...</p>;

    return (
        <div className="table-wrapper">
            <div className="admin-report-container">
                <h2>รายงานผลการใช้งานของสมาชิก</h2>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", marginBottom: "10px" }}>
                    <button
                        className="generate-report-btn"
                        onClick={generatePDF}
                        disabled={saving}
                        style={{ backgroundColor: saving ? "#ccc" : "#007bff", cursor: saving ? "not-allowed" : "pointer" }}
                    >
                        {saving ? "กำลังสร้างรายงาน..." : "สร้างรายงาน PDF"}
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

export default ReportAdmin;
