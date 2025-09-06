import { getFirestore, collection, addDoc, serverTimestamp, getDocs } from "firebase/firestore";

/**
 * บันทึกข้อมูลสถิติลง Firebase Firestore
 * @param {Object} diseaseStats - สถิติโรคทั้งหมด {disease: count}
 * @param {Array} filteredPredictions - ข้อมูลการวิเคราะห์ที่กรองแล้ว
 * @param {Object} filters - ตัวกรองข้อมูล (optional)
 */
export const saveStatisticsToFirestore = async (diseaseStats, filteredPredictions, filters = {}) => {
    const db = getFirestore();

    try {
        // ตรวจสอบ input
        if (!filteredPredictions || !Array.isArray(filteredPredictions)) {
            console.warn("filteredPredictions ต้องเป็น array");
            return null;
        }

        // คำนวณจำนวนการวิเคราะห์ทั้งหมด
        const totalAnalyses = filteredPredictions.length;

        // หาโรคที่พบบ่อยที่สุด
        const sortedDiseases = Object.entries(diseaseStats)
            .sort(([, a], [, b]) => b - a);

        const mostCommonDisease = sortedDiseases.length > 0 ? sortedDiseases[0] : null;

        if (!mostCommonDisease) {
            console.warn("ไม่มีข้อมูลโรคสำหรับบันทึก");
            return null;
        }

        const [mostCommonDiseaseName, mostCommonDiseaseCount] = mostCommonDisease;

        // เตรียมข้อมูลตัวกรองสำหรับบันทึก
        const filterInfo = {
            startDate: filters.startDate || null,
            endDate: filters.endDate || null,
            selectedProvince: filters.selectedProvince || null,
            selectedDistrict: filters.selectedDistrict || null
        };

        // 1. บันทึกลงตาราง StatisticAnaly (เก็บ TotalAnaly, วันที่, และ diseaseBreakdown)
        const statisticAnalyDoc = await addDoc(collection(db, "StatisticAnaly"), {
            DateStatAnaly: serverTimestamp(),
            TotalAnaly: totalAnalyses,
            diseaseBreakdown: diseaseStats, // เก็บรายละเอียดของโรคแต่ละชนิด
            filterInfo: filterInfo, // เก็บข้อมูลตัวกรอง
        });

        // 2. บันทึกลงตาราง StatisticMostComDisease
        const statisticMostComDiseaseDoc = await addDoc(collection(db, "StatisticMostComDisease"), {
            StatMostDisID: statisticAnalyDoc.id,
            DateStatMostDis: serverTimestamp(),
            CountMostDis: mostCommonDiseaseCount,
            MostComDisease: mostCommonDiseaseName,
            DiseaseID: generateDiseaseId(mostCommonDiseaseName),
        });

        console.log("บันทึกสถิติการวิเคราะห์สำเร็จ:", {
            statisticAnalyId: statisticAnalyDoc.id,
            totalAnalyses,
            mostCommonDisease: mostCommonDiseaseName
        });

        return {
            statisticAnalyId: statisticAnalyDoc.id,
            statisticMostComDiseaseId: statisticMostComDiseaseDoc.id,
            totalAnalyses,
            mostCommonDisease: {
                name: mostCommonDiseaseName,
                count: mostCommonDiseaseCount
            }
        };

    } catch (error) {
        console.error("เกิดข้อผิดพลาดในการบันทึกสถิติการวิเคราะห์:", error);
        throw error;
    }
};

/**
 * บันทึกข้อมูลสถิติผู้ใช้งานลง Firebase Firestore
 * @param {string} userCollectionName - ชื่อ collection ของผู้ใช้ (เช่น "users" หรือ "accounts")
 */
export const saveUserStatisticsToFirestore = async (userCollectionName = "users") => {
    const db = getFirestore();

    try {
        // นับจำนวนผู้ใช้งานทั้งหมดในระบบ
        const usersSnapshot = await getDocs(collection(db, userCollectionName));
        const totalUsers = usersSnapshot.size;

        // บันทึกลงตาราง StatisticUser
        const statisticUserDoc = await addDoc(collection(db, "StatisticUser"), {
            DateStatUser: serverTimestamp(),
            TotalUser: totalUsers,
        });

        console.log("บันทึกสถิติผู้ใช้งานสำเร็จ:", {
            StatUserID: statisticUserDoc.id,
            TotalUser: totalUsers
        });

        return {
            StatUserID: statisticUserDoc.id,
            TotalUser: totalUsers,
            DateStatUser: new Date().toISOString()
        };

    } catch (error) {
        console.error("เกิดข้อผิดพลาดในการบันทึกสถิติผู้ใช้งาน:", error);
        throw error;
    }
};

/**
 * บันทึกสถิติทั้งหมด (การวิเคราะห์และผู้ใช้งาน)
 * @param {Object} diseaseStats - สถิติโรคทั้งหมด {disease: count}
 * @param {Array} filteredPredictions - ข้อมูลการวิเคราะห์ที่กรองแล้ว
 * @param {Object} filters - ตัวกรองข้อมูล
 * @param {string} userCollectionName - ชื่อ collection ของผู้ใช้
 */
export const saveAllStatisticsToFirestore = async (diseaseStats, filteredPredictions, filters = {}, userCollectionName = "users") => {
    try {
        // บันทึกสถิติการวิเคราะห์และโรค
        const analysisStats = await saveStatisticsToFirestore(diseaseStats, filteredPredictions, filters);
        
        // บันทึกสถิติผู้ใช้งาน
        const userStats = await saveUserStatisticsToFirestore(userCollectionName);

        return {
            analysisStatistics: analysisStats,
            userStatistics: userStats
        };

    } catch (error) {
        console.error("เกิดข้อผิดพลาดในการบันทึกสถิติทั้งหมด:", error);
        throw error;
    }
};

/**
 * สร้างรหัสโรคจากชื่อโรค
 */
const generateDiseaseId = (diseaseName) => {
    const diseaseIdMap = {
        "ใบปกติ": "NORMAL_001",
        "ราดำ": "BLACK_SPOT_002",
        "จุดนูน": "ANTHRACNOSE_003",
        "แอนแทรคโนส": "ANTHRACNOSE_004"
    };

    return diseaseIdMap[diseaseName] || `UNKNOWN_${Date.now()}`;
};