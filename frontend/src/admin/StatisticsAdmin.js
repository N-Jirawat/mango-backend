import React, { useEffect, useState, useMemo, useCallback } from "react";
import { getFirestore, collection, getDocs } from "firebase/firestore";
import "../css/StatisticsAdmin.css";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
  PieChart, Pie, Cell, LineChart, Line, ResponsiveContainer
} from "recharts";

// เพิ่ม pdfMake imports
import pdfMake from "pdfmake/build/pdfmake";
import pdfFonts from "../PDF/vfs_fonts"; // ไฟล์ฟอนต์ Sarabun ต้องมีในโฟลเดอร์นี้

// กำหนดฟอนต์ให้ pdfMake
pdfMake.vfs = pdfFonts.pdfMake.vfs;

pdfMake.fonts = {
  Sarabun: {
    normal: "Sarabun-Bold.ttf",
    bold: "Sarabun-Bold.ttf",
    italics: "Sarabun-Bold.ttf",
    bolditalics: "Sarabun-Bold.ttf"
  }
};

function StatisticsAdmin() {
  const [diseaseStats, setDiseaseStats] = useState({});
  const [districtDiseaseMap, setDistrictDiseaseMap] = useState({});
  const [loading, setLoading] = useState(true);
  const [usersMap, setUsersMap] = useState({});
  const [allPredictions, setAllPredictions] = useState([]);
  const [timelineData, setTimelineData] = useState([]);

  // เปลี่ยนจาก startDate, endDate เป็น object ที่รวมตัวกรองทั้งหมด
  const [filters, setFilters] = useState({
    startDate: '',
    endDate: '',
    selectedProvince: '',
    selectedDistrict: ''
  });

  // ✅ ฟังก์ชันแปลงเดือน - wrapped in useCallback
  const formatMonthLabel = useCallback((monthKey) => {
    const [year, month] = monthKey.split('-');
    const monthNames = [
      'ม.ค.', 'ก.พ.', 'มี.ค.', 'เม.ย.', 'พ.ค.', 'มิ.ย.',
      'ก.ค.', 'ส.ค.', 'ก.ย.', 'ต.ค.', 'พ.ย.', 'ธ.ค.'
    ];
    return `${monthNames[parseInt(month) - 1]} ${parseInt(year) + 543}`;
  }, []);

  // ✅ ดึงรายชื่อจังหวัดและอำเภอที่ไม่ซ้ำ
  const availableProvinces = useMemo(() => {
    const provinces = [...new Set(Object.values(usersMap).map(user => user.province))];
    return provinces.filter(province => province && province !== "ไม่ระบุจังหวัด").sort();
  }, [usersMap]);

  const availableDistricts = useMemo(() => {
    if (!filters.selectedProvince) {
      const districts = [...new Set(Object.values(usersMap).map(user => user.district))];
      return districts.filter(district => district && district !== "ไม่ระบุอำเภอ").sort();
    }

    // กรองอำเภอตามจังหวัดที่เลือก
    const districtsInProvince = Object.values(usersMap)
      .filter(user => user.province === filters.selectedProvince)
      .map(user => user.district);

    return [...new Set(districtsInProvince)].filter(district => district && district !== "ไม่ระบุอำเภอ").sort();
  }, [usersMap, filters.selectedProvince]);

  // ✅ ฟังก์ชันประมวลผลสถิติ
  const processStatistics = useCallback((predictions, usersMapTemp) => {
    const diseaseMap = {};
    const districtMap = {};
    const monthlyData = {};

    predictions.forEach((prediction) => {
      const { disease, userId, timestamp } = prediction;
      const userInfo = usersMapTemp[userId];
      const district = userInfo?.district || "ไม่ระบุอำเภอ";

      diseaseMap[disease] = (diseaseMap[disease] || 0) + 1;

      if (!districtMap[district]) {
        districtMap[district] = {};
      }
      districtMap[district][disease] = (districtMap[district][disease] || 0) + 1;

      if (timestamp && timestamp instanceof Date && !isNaN(timestamp.getTime())) {
        const monthKey = `${timestamp.getFullYear()}-${String(timestamp.getMonth() + 1).padStart(2, '0')}`;
        if (!monthlyData[monthKey]) {
          monthlyData[monthKey] = { month: monthKey, count: 0, diseases: {} };
        }
        monthlyData[monthKey].count++;
        monthlyData[monthKey].diseases[disease] = (monthlyData[monthKey].diseases[disease] || 0) + 1;
      }
    });

    setDiseaseStats(diseaseMap);
    setDistrictDiseaseMap(districtMap);

    const timelineArray = Object.values(monthlyData)
      .sort((a, b) => a.month.localeCompare(b.month))
      .map(item => ({
        month: formatMonthLabel(item.month),
        count: item.count,
        ...item.diseases
      }));

    setTimelineData(timelineArray);
  }, [formatMonthLabel]);

  // ✅ แยก variables เพื่อหลีกเลี่ยง ESLint warning
  const { startDate, endDate, selectedProvince, selectedDistrict } = filters;

  // ✅ กรองข้อมูล predictions ด้วย useMemo (เพิ่มการกรองจังหวัดและอำเภอ)
  const filteredPredictions = useMemo(() => {
    console.log("=== Filtering Data ===");
    console.log("All predictions:", allPredictions.length);
    console.log("Filters:", { startDate, endDate, selectedProvince, selectedDistrict });

    const filtered = allPredictions.filter(prediction => {
      // กรองตามวันที่
      if (prediction.timestamp && prediction.timestamp instanceof Date) {
        const predictionDate = prediction.timestamp;

        if (startDate) {
          const predictionDateString = predictionDate.toISOString().split('T')[0];
          if (predictionDateString < startDate) {
            return false;
          }
        }

        if (endDate) {
          const predictionDateString = predictionDate.toISOString().split('T')[0];
          if (predictionDateString > endDate) {
            return false;
          }
        }
      } else if (startDate || endDate) {
        // ถ้ามีการตั้งค่าวันที่แต่ timestamp ไม่ถูกต้อง ให้ข้าม
        return false;
      }

      // กรองตามจังหวัดและอำเภอ
      const userInfo = usersMap[prediction.userId];
      if (!userInfo) {
        // ถ้าไม่มีข้อมูลผู้ใช้ และมีการกรองตามพื้นที่ ให้ข้าม
        if (selectedProvince || selectedDistrict) {
          return false;
        }
        return true;
      }

      // กรองตามจังหวัด
      if (selectedProvince && userInfo.province !== selectedProvince) {
        return false;
      }

      // กรองตามอำเภอ
      if (selectedDistrict && userInfo.district !== selectedDistrict) {
        return false;
      }

      return true;
    });

    console.log("Filtered predictions:", filtered.length);
    console.log("=== End Filtering ===");
    return filtered;
  }, [allPredictions, startDate, endDate, selectedProvince, selectedDistrict, usersMap]);

  // ✅ useEffect สำหรับดึงข้อมูลครั้งแรก
  useEffect(() => {
    const fetchStatistics = async () => {
      const db = getFirestore();
      setLoading(true);

      try {
        console.log("เริ่มดึงข้อมูล...");

        // ดึงข้อมูล users
        const usersSnapshot = await getDocs(collection(db, "users"));
        const usersMapTemp = {};

        usersSnapshot.forEach(doc => {
          const user = doc.data();
          const userId = doc.id;

          usersMapTemp[userId] = {
            district: user.district || "ไม่ระบุอำเภอ",
            province: user.province || "ไม่ระบุจังหวัด",
            role: user.role || "user",
            name: user.displayName || user.name || "ไม่ระบุชื่อ"
          };
        });

        console.log("ดึงข้อมูล users แล้ว:", Object.keys(usersMapTemp).length, "คน");
        setUsersMap(usersMapTemp);

        // ดึงข้อมูล predictions ทั้งหมด (ไม่กรองตอนนี้)
        const predictionSnapshot = await getDocs(collection(db, "prediction_results"));

        if (predictionSnapshot.empty) {
          console.log("ไม่มีข้อมูลใน prediction_results");
          setAllPredictions([]);
          setLoading(false);
          return;
        }

        const predictionsData = [];

        predictionSnapshot.forEach(doc => {
          const data = doc.data();

          // แปลง timestamp - เพิ่ม console.log เพื่อ debug
          let createdAt = null;
          if (data.timestamp?.seconds) {
            createdAt = new Date(data.timestamp.seconds * 1000);
          } else if (data.timestamp?.toDate) {
            createdAt = data.timestamp.toDate();
          } else if (data.createdAt?.seconds) {
            createdAt = new Date(data.createdAt.seconds * 1000);
          } else if (data.createdAt?.toDate) {
            createdAt = data.createdAt.toDate();
          } else if (data.timestamp instanceof Date) {
            createdAt = data.timestamp;
          } else if (data.createdAt instanceof Date) {
            createdAt = data.createdAt;
          }

          const predictionItem = {
            id: doc.id,
            disease: data.diseaseName || data.predictedClass || "ไม่ระบุโรค",
            userId: data.userId || "ไม่ทราบผู้ใช้",
            timestamp: createdAt,
            confidence: data.confidence || 0,
            rawData: data // เก็บข้อมูลดิบไว้สำหรับ debug
          };

          // เพิ่มข้อมูลเฉพาะที่มี timestamp ที่ถูกต้อง
          if (createdAt && createdAt instanceof Date && !isNaN(createdAt.getTime())) {
            predictionsData.push(predictionItem);
            console.log("✅ Added prediction:", doc.id, "Date:", createdAt.toISOString().split('T')[0]);
          } else {
            console.warn("❌ Skipping prediction with invalid timestamp:", doc.id, {
              timestamp: data.timestamp,
              createdAt: data.createdAt,
              parsedDate: createdAt
            });
          }
        });

        console.log("ดึงข้อมูล predictions แล้ว:", predictionsData.length, "รายการ");
        setAllPredictions(predictionsData);

        // ประมวลผลสถิติครั้งแรก (ข้อมูลทั้งหมด)
        console.log("กำลังประมวลผลสถิติ...");
        processStatistics(predictionsData, usersMapTemp);

      } catch (error) {
        console.error("เกิดข้อผิดพลาดในการดึงข้อมูลสถิติ:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchStatistics();
  }, [processStatistics]); // ดึงข้อมูลครั้งเดียว

  // ✅ useEffect สำหรับประมวลผลใหม่เมื่อกรองข้อมูล
  useEffect(() => {
    if (filteredPredictions.length >= 0 && Object.keys(usersMap).length > 0) {
      console.log("ประมวลผลใหม่เมื่อกรองข้อมูล:", filteredPredictions.length, "รายการ");
      processStatistics(filteredPredictions, usersMap);
    }
  }, [filteredPredictions, usersMap, processStatistics]);

  // ✅ ฟังก์ชันอัพเดทตัวกรอง
  const updateFilter = (key, value) => {
    setFilters(prev => {
      const newFilters = { ...prev, [key]: value };

      // ถ้าเปลี่ยนจังหวัด ให้รีเซ็ตอำเภอ
      if (key === 'selectedProvince') {
        newFilters.selectedDistrict = '';
      }

      return newFilters;
    });
  };

  // ✅ รีเซ็ตตัวกรอง
  const resetFilter = () => {
    setFilters({
      startDate: '',
      endDate: '',
      selectedProvince: '',
      selectedDistrict: ''
    });
  };

  // ✅ ตรวจสอบว่ามีการใช้ตัวกรองหรือไม่
  const hasActiveFilters = startDate || endDate || selectedProvince || selectedDistrict;

  // ✅ ฟังก์ชันสร้าง PDF
  // เพิ่ม function นี้ก่อน handleDownloadPDF
  const generateChartImage = (chartType, data, options = {}) => {
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = options.width || 800;
      canvas.height = options.height || 400;

      // สร้างพื้นหลังสีขาว
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      if (chartType === 'pie') {
        drawPieChart(ctx, data, canvas.width, canvas.height);
      } else if (chartType === 'bar') {
        drawBarChart(ctx, data, canvas.width, canvas.height, options);
      }

      // แปลงเป็น base64
      const imageData = canvas.toDataURL('image/png');
      resolve(imageData);
    });
  };

  const drawPieChart = (ctx, data, width, height) => {
    const centerX = width / 2 - 50; // เลื่อนวงกลมไปทางซ้ายเล็กน้อย
    const centerY = height / 2;
    const radius = Math.min(width, height) / 4; // ลดขนาดวงกลมลง

    const colors = [
      "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
      "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"
    ];

    const total = data.reduce((sum, item) => sum + item.value, 0);
    let currentAngle = -Math.PI / 2; // เริ่มจากด้านบน

    // วาดชิ้นส่วนของวงกลม
    data.forEach((item, index) => {
      const sliceAngle = (item.value / total) * 2 * Math.PI;

      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.arc(centerX, centerY, radius, currentAngle, currentAngle + sliceAngle);
      ctx.closePath();
      ctx.fillStyle = colors[index % colors.length];
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 3;
      ctx.stroke();

      // วาดเปอร์เซ็นต์และเส้นชี้
      if (item.percentage > 3) { // แสดงเฉพาะที่มีพื้นที่เพียงพอ
        const labelAngle = currentAngle + sliceAngle / 2;

        // จุดบนวงกลม
        const innerX = centerX + Math.cos(labelAngle) * (radius * 0.8);
        const innerY = centerY + Math.sin(labelAngle) * (radius * 0.8);

        // จุดปลายเส้นชี้ (ห่างจากวงกลม)
        const outerX = centerX + Math.cos(labelAngle) * (radius * 1.3);
        const outerY = centerY + Math.sin(labelAngle) * (radius * 1.3);

        // จุดสำหรับข้อความ (แนวนอน)
        const textX = outerX + (labelAngle > Math.PI / 2 || labelAngle < -Math.PI / 2 ? -30 : 30);
        const textY = outerY;

        // วาดเส้นชี้
        ctx.strokeStyle = '#666666';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(innerX, innerY);
        ctx.lineTo(outerX, outerY);
        ctx.lineTo(textX, textY);
        ctx.stroke();

        // วาดจุดที่ปลายเส้น
        ctx.fillStyle = '#666666';
        ctx.beginPath();
        ctx.arc(outerX, outerY, 2, 0, 2 * Math.PI);
        ctx.fill();

        // วาดข้อความเปอร์เซ็นต์
        ctx.fillStyle = '#000000';
        ctx.font = 'bold 12px Arial';
        ctx.textAlign = labelAngle > Math.PI / 2 || labelAngle < -Math.PI / 2 ? 'right' : 'left';
        ctx.textBaseline = 'middle';

        // สร้างพื้นหลังสีขาวให้ข้อความ
        const textWidth = ctx.measureText(`${item.percentage}%`).width;
        const textHeight = 16;
        const bgX = labelAngle > Math.PI / 2 || labelAngle < -Math.PI / 2 ? textX - textWidth - 4 : textX - 4;
        const bgY = textY - textHeight / 2;

        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.fillRect(bgX, bgY, textWidth + 8, textHeight);
        ctx.strokeStyle = '#cccccc';
        ctx.lineWidth = 1;
        ctx.strokeRect(bgX, bgY, textWidth + 8, textHeight);

        ctx.fillStyle = '#000000';
        ctx.fillText(`${item.percentage}%`, textX, textY);
      }

      currentAngle += sliceAngle;
    });

    // วาด Legend ทางขวา
    const legendX = centerX + radius + 80;
    const legendY = centerY - (data.length * 25) / 2;

    ctx.font = 'bold 16px Arial';
    ctx.fillStyle = '#2E7D32';
    ctx.textAlign = 'left';
    ctx.fillText('สัดส่วนโรค', legendX, legendY - 20);

    data.forEach((item, index) => {
      const y = legendY + (index * 30);

      // วาดสี่เหลี่ยมสี (ใหญ่ขึ้น)
      ctx.fillStyle = colors[index % colors.length];
      ctx.fillRect(legendX, y - 10, 18, 18);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.strokeRect(legendX, y - 10, 18, 18);

      // วาดข้อความ
      ctx.fillStyle = '#333333';
      ctx.font = 'bold 11px Arial';
      ctx.fillText(item.name, legendX + 25, y - 5);

      ctx.fillStyle = '#666666';
      ctx.font = '10px Arial';
      ctx.fillText(`${item.value} ครั้ง (${item.percentage}%)`, legendX + 25, y + 8);
    });

    // วาดหัวเรื่องกราฟ
    ctx.fillStyle = '#2E7D32';
    ctx.font = 'bold 18px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('สัดส่วนโรคที่พบ', width / 2, 20);
  };

  const drawBarChart = (ctx, data, width, height, options) => {
    const margin = { top: 50, right: 50, bottom: 150, left: 80 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    if (data.length === 0) return;

    // หาค่าสูงสุด
    const diseases = Object.keys(data[0]).filter(key =>
      key !== 'district' && key !== 'locationLabel'
    );

    const maxValue = Math.max(...data.map(item =>
      diseases.reduce((sum, disease) => sum + (item[disease] || 0), 0)
    ));

    const colors = [
      "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
      "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"
    ];

    const barWidth = chartWidth / data.length * 0.8;
    const barSpacing = chartWidth / data.length * 0.2;

    // วาดแกน Y
    ctx.strokeStyle = '#cccccc';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = margin.top + (chartHeight / 5) * i;
      const value = maxValue - (maxValue / 5) * i;

      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(margin.left + chartWidth, y);
      ctx.stroke();

      // ป้ายแกน Y
      ctx.fillStyle = '#666666';
      ctx.font = '10px Arial';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(Math.round(value).toString(), margin.left - 10, y);
    }

    // วาดแท่งกราฟ
    data.forEach((item, index) => {
      const x = margin.left + (index * (barWidth + barSpacing)) + barSpacing / 2;
      let stackY = margin.top + chartHeight;

      diseases.forEach((disease, diseaseIndex) => {
        const value = item[disease] || 0;
        if (value > 0) {
          const barHeight = (value / maxValue) * chartHeight;

          ctx.fillStyle = colors[diseaseIndex % colors.length];
          ctx.fillRect(x, stackY - barHeight, barWidth, barHeight);

          stackY -= barHeight;
        }
      });

      // ป้ายแกน X
      ctx.fillStyle = '#333333';
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';

      // แบ่งข้อความยาวๆ
      const label = item.locationLabel || item.district;
      const words = label.split(' ');
      const maxWordsPerLine = 2;

      for (let i = 0; i < words.length; i += maxWordsPerLine) {
        const line = words.slice(i, i + maxWordsPerLine).join(' ');
        const lineY = margin.top + chartHeight + 15 + (Math.floor(i / maxWordsPerLine) * 12);
        ctx.fillText(line, x + barWidth / 2, lineY);
      }
    });

    // วาดเส้นแกน
    ctx.strokeStyle = '#333333';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top);
    ctx.lineTo(margin.left, margin.top + chartHeight);
    ctx.lineTo(margin.left + chartWidth, margin.top + chartHeight);
    ctx.stroke();

    // วาดหัวเรื่อง
    ctx.fillStyle = '#333333';
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(options.title || 'การกระจายโรคตามพื้นที่', width / 2, 10);

    // วาด Legend
    const legendX = width - 200;
    const legendY = margin.top;

    ctx.font = 'bold 12px Arial';
    ctx.fillStyle = '#000000';
    ctx.textAlign = 'left';
    ctx.fillText('โรค:', legendX, legendY);

    diseases.forEach((disease, index) => {
      const y = legendY + 20 + (index * 20);

      // วาดสี่เหลี่ยมสี
      ctx.fillStyle = colors[index % colors.length];
      ctx.fillRect(legendX, y - 6, 12, 12);

      // วาดข้อความ
      ctx.fillStyle = '#000000';
      ctx.font = '10px Arial';
      ctx.fillText(disease, legendX + 20, y);
    });
  };

  // แก้ไข handleDownloadPDF function
  const handleDownloadPDF = async () => {
    try {
      const currentDate = new Date().toLocaleDateString("th-TH");

      // สร้างรูปกราฟ
      console.log("กำลังสร้างกราฟ...");

      let pieChartImage = null;
      let barChartImage = null;

      // สร้างกราฟวงกลม
      if (pieData.length > 0) {
        pieChartImage = await generateChartImage('pie', pieData, { width: 600, height: 400 });
      }

      // สร้างกราฟแท่ง
      if (chartData.length > 0) {
        barChartImage = await generateChartImage('bar', chartData, {
          width: 800,
          height: 500,
          title: 'การกระจายโรคตามพื้นที่'
        });
      }

      // สร้างข้อมูลสำหรับตัวกรอง
      const filterInfo = [];
      if (startDate) {
        filterInfo.push(`วันที่เริ่มต้น: ${new Date(startDate + 'T00:00:00').toLocaleDateString('th-TH')}`);
      }
      if (endDate) {
        filterInfo.push(`วันที่สิ้นสุด: ${new Date(endDate + 'T00:00:00').toLocaleDateString('th-TH')}`);
      }
      if (selectedProvince) {
        filterInfo.push(`จังหวัด: ${selectedProvince}`);
      }
      if (selectedDistrict) {
        filterInfo.push(`อำเภอ: ${selectedDistrict}`);
      }

      // สร้างตารางสถิติโรค
      const diseaseTableData = [
        [{ text: 'ชื่อโรค', style: 'tableHeader' }, { text: 'จำนวน (ครั้ง)', style: 'tableHeader' }, { text: 'เปอร์เซ็นต์', style: 'tableHeader' }]
      ];

      const totalDiseaseCount = Object.values(diseaseStats).reduce((sum, val) => sum + val, 0);
      Object.entries(diseaseStats)
        .sort(([, a], [, b]) => b - a)
        .forEach(([disease, count]) => {
          const percentage = totalDiseaseCount > 0 ? ((count / totalDiseaseCount) * 100).toFixed(1) : 0;
          diseaseTableData.push([
            disease,
            count.toString(),
            `${percentage}%`
          ]);
        });

      // สร้างตารางสถิติตามพื้นที่
      const areaTableData = [
        [{ text: 'พื้นที่', style: 'tableHeader' }, { text: 'จำนวนรวม (ครั้ง)', style: 'tableHeader' }, { text: 'โรคที่พบ', style: 'tableHeader' }]
      ];

      Object.entries(districtDiseaseMap).forEach(([district, diseases]) => {
        const matchingUser = Object.values(usersMap).find(u => u.district === district);
        const province = matchingUser?.province || "ไม่ระบุจังหวัด";
        const totalInDistrict = Object.values(diseases).reduce((sum, val) => sum + val, 0);
        const diseaseList = Object.entries(diseases)
          .sort(([, a], [, b]) => b - a)
          .map(([diseaseName, count]) => `${diseaseName} (${count})`)
          .join(', ');

        areaTableData.push([
          `${district}, ${province}`,
          totalInDistrict.toString(),
          diseaseList
        ]);
      });

      // สร้าง Timeline Table
      const timelineTableData = [
        [{ text: 'เดือน', style: 'tableHeader' }, { text: 'จำนวนรวม', style: 'tableHeader' }]
      ];

      // เพิ่มหัวข้อโรคแต่ละชนิด
      const diseaseNames = Object.keys(diseaseStats);
      diseaseNames.forEach(disease => {
        timelineTableData[0].push({ text: disease, style: 'tableHeader' });
      });

      // เพิ่มข้อมูลแต่ละเดือน
      timelineData.forEach(monthData => {
        const row = [monthData.month, monthData.count.toString()];
        diseaseNames.forEach(disease => {
          row.push((monthData[disease] || 0).toString());
        });
        timelineTableData.push(row);
      });

      // สร้างสรุปรายเดือน
      const monthlySummary = timelineData.map(monthData => {
        return `${monthData.month} จำนวนโรครวม = ${monthData.count} ครั้ง`;
      }).join('\n');

      // หาตำแหน่งของคอลัมน์ที่มี text === 'จำนวนรวม'
      const totalIndex = timelineTableData[0].findIndex(cell => cell.text === 'จำนวนรวม');

      // ถ้าเจอ "จำนวนรวม"
      if (totalIndex !== -1) {
        timelineTableData.forEach(row => {
          row.splice(totalIndex, 1);  // ลบเซลล์ในแต่ละแถวออก
        });
      }

      const docDefinition = {
        content: [
          { text: 'รายงานสถิติการวิเคราะห์โรคใบมะม่วง', style: 'header' },
          { text: `วันที่สร้างรายงาน: ${currentDate}`, style: 'subheader' },

          // ข้อมูลตัวกรอง
          ...(hasActiveFilters ? [
            { text: 'เงื่อนไขการกรองข้อมูล:', style: 'sectionHeader' },
            {
              ul: filterInfo,
              margin: [0, 0, 0, 15]
            }
          ] : []),

          // สรุปผล
          { text: 'สรุปผลการวิเคราะห์:', style: 'sectionHeader' },
          {
            columns: [
              { text: `ประเภทโรคที่พบ: ${Object.keys(diseaseStats).length} ชนิด`, width: '*' }
            ],
            margin: [0, 0, 0, 15]
          },
          {
            columns: [
              { text: `พื้นที่ที่มีการวิเคราะห์: ${Object.keys(districtDiseaseMap).length} อำเภอ`, width: '*' },
              { text: `ช่วงเวลา: ${timelineData.length} เดือน`, width: '*' }
            ],
            margin: [0, 0, 0, 20]
          },

          // กราฟวงกลม
          ...(pieChartImage ? [
            { text: 'สัดส่วนโรคที่พบ (กราฟวงกลม):', style: 'sectionHeader' },
            {
              image: pieChartImage,
              width: 500,
              alignment: 'center',
              margin: [0, 0, 0, 20]
            }
          ] : []),

          // ตารางสถิติโรค
          { text: 'สถิติการพบโรคแต่ละชนิด:', style: 'sectionHeader', pageBreak: 'before' },
          {
            table: {
              headerRows: 1,
              widths: ['*', 'auto', 'auto'],
              body: diseaseTableData
            },
            style: 'table',
            margin: [0, 0, 0, 20]
          },

          // กราฟแท่ง
          ...(barChartImage ? [
            { text: 'การกระจายโรคตามพื้นที่ (กราฟแท่ง):', style: 'sectionHeader' },
            {
              image: barChartImage,
              width: 500,
              alignment: 'center',
              margin: [0, 0, 0, 0]
            }
          ] : []),

          // สถิติตามพื้นที่
          ...(Object.keys(districtDiseaseMap).length > 0 ? [
            { text: 'สถิติการพบโรคตามพื้นที่:', style: 'sectionHeader', pageBreak: 'before' },
            {
              table: {
                headerRows: 1,
                widths: ['*', 'auto', '*'],
                body: areaTableData
              },
              style: 'table',
              margin: [0, 0, 0, 20]
            }
          ] : []),

          // Timeline
          ...(timelineData.length > 0 ? [
            { text: 'แนวโน้มการวิเคราะห์รายเดือน:', style: 'sectionHeader' },
            {
              table: {
                headerRows: 1,
                widths: Array(timelineTableData[0].length).fill('*'),
                body: timelineTableData,
              },
              style: 'table',
              margin: [0, 0, 0, 20]
            },
            { text: 'สรุปรายเดือน:', style: 'subsectionHeader' },
            {
              text: monthlySummary,
              style: 'monthlySummary',
              margin: [0, 0, 0, 20]
            }
          ] : [])

        ].filter(Boolean),
        defaultStyle: {
          font: "Sarabun",
          fontSize: 12,
        },
        styles: {
          header: {
            fontSize: 20,
            bold: true,
            margin: [0, 0, 0, 10],
            alignment: 'center',
            color: '#2E7D32'
          },
          subheader: {
            fontSize: 14,
            margin: [0, 0, 0, 20],
            alignment: 'center',
            color: '#666'
          },
          sectionHeader: {
            fontSize: 16,
            bold: true,
            margin: [0, 20, 0, 10],
            color: '#2E7D32'
          },
          subsectionHeader: {
            fontSize: 14,
            bold: true,
            margin: [0, 10, 0, 5],
            color: '#2E7D32'
          },
          tableHeader: {
            bold: true,
            fillColor: '#E8F5E8',
            color: '#2E7D32'
          },
          table: {
            margin: [0, 5, 0, 15]
          },
          monthlySummary: {
            fontSize: 11,
            color: '#666'
          }
        },
        pageMargins: [40, 60, 40, 60]
      };

      // สร้างชื่อไฟล์
      let fileName = `รายงานสถิติโรคใบมะม่วง_${currentDate}`;
      if (hasActiveFilters) {
        if (selectedProvince) fileName += `_${selectedProvince}`;
        if (selectedDistrict) fileName += `_${selectedDistrict}`;
        if (startDate) fileName += `_${startDate}`;
        if (endDate) fileName += `_${endDate}`;
      }
      fileName += '.pdf';

      console.log("กำลังสร้าง PDF...");
      pdfMake.createPdf(docDefinition).download(fileName);

    } catch (error) {
      console.error("เกิดข้อผิดพลาดในการสร้าง PDF:", error);
      alert("เกิดข้อผิดพลาดในการสร้างไฟล์ PDF");
    }
  };

  const chartData = useMemo(() => {
    return Object.entries(districtDiseaseMap).map(([district, diseases]) => {
      const matchingUser = Object.values(usersMap).find(u => u.district === district);
      const province = matchingUser?.province || "ไม่ระบุจังหวัด";
      return {
        district,
        locationLabel: `${district}, ${province}`,
        ...diseases,
      };
    });
  }, [districtDiseaseMap, usersMap]);

  const pieData = useMemo(() => {
    const totalCount = Object.values(diseaseStats).reduce((sum, val) => sum + val, 0);
    return Object.entries(diseaseStats).map(([disease, count]) => ({
      name: disease,
      value: count,
      percentage: totalCount > 0 ? ((count / totalCount) * 100).toFixed(1) : 0
    }));
  }, [diseaseStats]);

  const colors = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"
  ];

  const getColorFromIndex = (index) => {
    return colors[index % colors.length];
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
    <div className="statistics-admin-container">
      <div className="header-section">
        <h1>สถิติการวิเคราะห์โรคใบมะม่วง</h1>

        {/* เพิ่มปุ่ม Download PDF */}
        <div className="header-actions">
          <button
            onClick={handleDownloadPDF}
            className="download-pdf-btn"
            disabled={Object.keys(diseaseStats).length === 0}
            title={Object.keys(diseaseStats).length === 0 ? "ไม่มีข้อมูลสำหรับสร้าง PDF" : "ดาวน์โหลดรายงาน PDF"}
          >
            📄 ดาวน์โหลดรายงาน PDF
          </button>
        </div>
      </div>

      {/* Enhanced Filter Section */}
      <div className="filter-section">
        <div className="filter-container">
          <h3>🔍 กรองข้อมูล</h3>

          {/* Date Filters */}
          <div className="filter-row">
            <h4>📅 ช่วงเวลา</h4>
            <div className="date-inputs">
              <div className="input-group">
                <label>
                  วันที่เริ่มต้น:
                  <input
                    type="date"
                    value={filters.startDate}
                    onChange={(e) => updateFilter('startDate', e.target.value)}
                  />
                </label>
              </div>
              <div className="input-group">
                <label>
                  วันที่สิ้นสุด:
                  <input
                    type="date"
                    value={filters.endDate}
                    onChange={(e) => updateFilter('endDate', e.target.value)}
                  />
                </label>
              </div>
            </div>
          </div>

          {/* Location Filters */}
          <div className="filter-row">
            <h4>🏛️ พื้นที่</h4>
            <div className="location-inputs">
              <div className="input-group">
                <label>
                  จังหวัด:
                  <select
                    value={filters.selectedProvince}
                    onChange={(e) => updateFilter('selectedProvince', e.target.value)}
                  >
                    <option value="">ทุกจังหวัด</option>
                    {availableProvinces.map(province => (
                      <option key={province} value={province}>
                        {province}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="input-group">
                <label>
                  อำเภอ:
                  <select
                    value={filters.selectedDistrict}
                    onChange={(e) => updateFilter('selectedDistrict', e.target.value)}
                    disabled={!availableDistricts.length}
                  >
                    <option value="">ทุกอำเภอ</option>
                    {availableDistricts.map(district => (
                      <option key={district} value={district}>
                        {district}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            </div>
          </div>

          {/* Reset Button */}
          <div className="button-group">
            <button onClick={resetFilter} className="reset-btn" disabled={!hasActiveFilters}>
              🔄 รีเซ็ตตัวกรอง
            </button>
          </div>

          {/* Filter Status */}
          <div className="filter-status">
            {hasActiveFilters ? (
              <div className="filter-info">
                <span className="filter-badge active">
                  🔍 กำลังกรองข้อมูล: <strong>{filteredPredictions.length}</strong> จาก <strong>{allPredictions.length}</strong> รายการ
                </span>
                <div className="active-filters">
                  {startDate && (
                    <span className="filter-tag">
                      📅 เริ่ม: {new Date(startDate + 'T00:00:00').toLocaleDateString('th-TH')}
                    </span>
                  )}
                  {endDate && (
                    <span className="filter-tag">
                      📅 สิ้นสุด: {new Date(endDate + 'T00:00:00').toLocaleDateString('th-TH')}
                    </span>
                  )}
                  {selectedProvince && (
                    <span className="filter-tag">
                      🏛️ จังหวัด: {selectedProvince}
                    </span>
                  )}
                  {selectedDistrict && (
                    <span className="filter-tag">
                      🏢 อำเภอ: {selectedDistrict}
                    </span>
                  )}
                </div>
              </div>
            ) : (
              <div className="filter-info">
                <span className="filter-badge">
                  📊 แสดงข้อมูลทั้งหมด: <strong>{allPredictions.length}</strong> รายการ
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="summary-cards">
        <div className="summary-card">
          <div className="card-icon">🦠</div>
          <div className="card-content">
            <h3>ประเภทโรค</h3>
            <div className="card-number">{Object.keys(diseaseStats).length}</div>
            <p>ชนิด</p>
          </div>
        </div>

        <div className="summary-card">
          <div className="card-icon">📍</div>
          <div className="card-content">
            <h3>พื้นที่</h3>
            <div className="card-number">{Object.keys(districtDiseaseMap).length}</div>
            <p>อำเภอ</p>
          </div>
        </div>

        <div className="summary-card">
          <div className="card-icon">📈</div>
          <div className="card-content">
            <h3>ช่วงเวลา</h3>
            <div className="card-number">{timelineData.length}</div>
            <p>เดือน</p>
          </div>
        </div>
      </div>

      {/* Charts Section */}
      <div className="charts-section">
        {/* Disease Statistics */}
        <div className="chart-container">
          <div className="chart-header">
            <h3>📊 สัดส่วนโรคที่พบ</h3>
          </div>
          <div className="chart-content">
            <div className="chart-left">
              {Object.keys(diseaseStats).length > 0 ? (
                <div className="disease-list">
                  {Object.entries(diseaseStats)
                    .sort(([, a], [, b]) => b - a)
                    .map(([disease, count], index) => {
                      const totalCount = Object.values(diseaseStats).reduce((sum, val) => sum + val, 0);
                      const percentage = totalCount > 0 ? ((count / totalCount) * 100).toFixed(1) : 0;
                      return (
                        <div key={disease} className="disease-item">
                          <div className="disease-info">
                            <div className="disease-label">
                              <div
                                className="disease-color"
                                style={{ backgroundColor: getColorFromIndex(index) }}
                              ></div>
                              <span className="disease-name">{disease}</span>
                            </div>
                            <div className="disease-stats">
                              <span className="disease-count">{count} ครั้ง</span>
                              <span className="disease-percentage">({percentage}%)</span>
                            </div>
                          </div>
                          <div className="disease-bar">
                            <div
                              className="disease-fill"
                              style={{
                                width: `${percentage}%`,
                                backgroundColor: getColorFromIndex(index)
                              }}
                            ></div>
                          </div>
                        </div>
                      );
                    })}
                </div>
              ) : (
                <div className="no-data">
                  <div className="no-data-icon">📊</div>
                  <p>ยังไม่มีข้อมูلการวิเคราะห์</p>
                </div>
              )}
            </div>

            {pieData.length > 0 && (
              <div className="chart-right">
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      dataKey="value"
                      label={({ name, percentage }) => `${percentage}%`}
                    >
                      {pieData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={getColorFromIndex(index)} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value, name) => [`${value} ครั้ง`, name]} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        </div>

        {/* Timeline Chart */}
        {timelineData.length > 0 && (
          <div className="chart-container">
            <div className="chart-header">
              <h3>📈 แนวโน้มการวิเคราะห์รายเดือน</h3>
              <p style={{ fontSize: '14px', color: '#666', margin: '5px 0' }}>
                แสดงแนวโน้มการวิเคราะห์แต่ละโรคตามเดือน
              </p>
            </div>
            <div className="chart-content">
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={timelineData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip
                    formatter={(value, name) => [`${value} ครั้ง`, name]}
                  />

                  {/* เส้นแต่ละโรค */}
                  {Object.keys(diseaseStats).map((disease, idx) => (
                    <Line
                      key={disease}
                      type="monotoneX"
                      dataKey={disease}
                      stroke={getColorFromIndex(idx)}
                      strokeWidth={4}
                      dot={{
                        fill: getColorFromIndex(idx),
                        strokeWidth: 3,
                        r: 8,
                        stroke: '#fff'
                      }}
                      activeDot={{
                        r: 10,
                        stroke: getColorFromIndex(idx),
                        strokeWidth: 3,
                        fill: '#fff'
                      }}
                      name={disease}
                      connectNulls={false}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>

              {/* Legend สำหรับ Timeline */}
              <div className="timeline-legend" style={{ marginTop: '20px' }}>
                <h4>โรคที่แสดงในกราฟ</h4>
                <div className="legend-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px' }}>
                  {Object.keys(diseaseStats).map((disease, idx) => (
                    <div key={disease} className="legend-item" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <div
                        style={{
                          width: '20px',
                          height: '20px',
                          borderRadius: '50%',
                          backgroundColor: getColorFromIndex(idx),
                          border: '3px solid #fff',
                          boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                        }}
                      ></div>
                      <span style={{ fontWeight: '500' }}>{disease}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* District Chart */}
        {chartData.length > 0 && (
          <div className="chart-container full-width">
            <div className="chart-header">
              <h3>📍 การกระจายโรคตามพื้นที่</h3>
            </div>
            <div className="chart-content">
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={chartData} margin={{ top: 20, right: 30, bottom: 60, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="locationLabel"
                    angle={-45}
                    textAnchor="end"
                    interval={0}
                    height={100}
                  />
                  <YAxis />
                  <Tooltip />
                  {Object.keys(diseaseStats).map((disease, idx) => (
                    <Bar
                      key={disease}
                      dataKey={disease}
                      stackId="a"
                      fill={getColorFromIndex(idx)}
                    />
                  ))}
                </BarChart>
              </ResponsiveContainer>

              <div className="chart-legend">
                <h4>สีแทนโรค</h4>
                <div className="legend-grid">
                  {Object.keys(diseaseStats).map((disease, idx) => (
                    <div key={disease} className="legend-item">
                      <div
                        className="legend-color"
                        style={{ backgroundColor: getColorFromIndex(idx) }}
                      ></div>
                      <span className="legend-text">{disease}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {!Object.keys(diseaseStats).length && (
        <div className="empty-state">
          <div className="empty-icon">📊</div>
          <h3>ไม่มีข้อมูลสถิติ</h3>
          <p>ยังไม่มีการวิเคราะห์โรคในระบบ หรือไม่มีข้อมูลในช่วงเวลาที่เลือก</p>
        </div>
      )}
    </div>
  );
}

export default StatisticsAdmin;