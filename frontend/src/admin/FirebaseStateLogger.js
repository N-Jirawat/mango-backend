import { getFirestore, collection, addDoc, serverTimestamp } from "firebase/firestore";

/**
 * บันทึกข้อมูลสถิติลง Firebase Firestore
 * @param {Object} diseaseStats - สถิติโรคทั้งหมด {disease: count}
 * @param {Array} filteredPredictions - ข้อมูลการวิเคราะห์ที่กรองแล้ว
 */
export const saveStatisticsToFirestore = async (diseaseStats, filteredPredictions) => {
    const db = getFirestore();

    try {
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

        // 1. บันทึกลงตาราง StatisticAnaly (เก็บ TotalAnaly, วันที่, และ diseaseBreakdown)
        const statisticAnalyDoc = await addDoc(collection(db, "StatisticAnaly"), {
            DateStatAnaly: serverTimestamp(),
            TotalAnaly: totalAnalyses,
            diseaseBreakdown: diseaseStats, // เก็บรายละเอียดของโรคแต่ละชนิด
        });

        // 2. บันทึกลงตาราง StatisticMostComDisease
        // ก่อนหน้านี้มี AnalysisID
        const statisticMostComDiseaseDoc = await addDoc(collection(db, "StatisticMostComDisease"), {
            StatMostDisID: statisticAnalyDoc.id,
            DateStatMostDis: serverTimestamp(),
            CountMostDis: mostCommonDiseaseCount,
            MostComDisease: mostCommonDiseaseName,
            DiseaseID: generateDiseaseId(mostCommonDiseaseName),
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
