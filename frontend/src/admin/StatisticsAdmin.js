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

  // เพิ่ม state สำหรับ fullscreen และ zoom
  const [fullscreenChart, setFullscreenChart] = useState(null);
  const [zoomLevel, setZoomLevel] = useState(100); // Zoom level เป็นเปอร์เซ็นต์

  // เปลี่ยนจาก startDate, endDate เป็น object ที่รวมตัวกรองทั้งหมด
  const [filters, setFilters] = useState({
    startDate: '',
    endDate: '',
    selectedProvince: '',
    selectedDistrict: ''
  });

  // กำหนดสีที่ใช้ในกราฆทั้งหน้าจอและ PDF
  const chartColors = useMemo(() => [
    "#757575", // เขียว - สำหรับใบปกติ
    "#E67E22", // เทา - สำหรับราดำ  
    "#3498DB", // น้ำเงิน - สำหรับจุดนูน
    "#2ECC71"  // ส้ม - สำหรับแอนแทรค 
  ], []);

  // ฟังก์ชันแปลงเดือน - wrapped in useCallback
  const formatMonthLabel = useCallback((monthKey) => {
    const [year, month] = monthKey.split('-');
    const monthNames = [
      'ม.ค.', 'ก.พ.', 'มี.ค.', 'เม.ย.', 'พ.ค.', 'มิ.ย.',
      'ก.ค.', 'ส.ค.', 'ก.ย.', 'ต.ค.', 'พ.ย.', 'ธ.ค.'
    ];
    return `${monthNames[parseInt(month) - 1]} ${parseInt(year) + 543}`;
  }, []);

  // ดึงรายชื่อจังหวัดและอำเภอที่ไม่ซ้ำ
  const availableProvinces = useMemo(() => {
    const provinces = [...new Set(Object.values(usersMap).map(user => user.province))];
    return provinces.filter(province => province && province !== "ไม่ระบุจังหวัด").sort();
  }, [usersMap]);

  const availableDistricts = useMemo(() => {
    if (!filters.selectedProvince) {
      const districts = [...new Set(Object.values(usersMap).map(user => user.district))];
      return districts.filter(district => district && district !== "ไม่ระบุอำเภอ").sort();
    }

    const districtsInProvince = Object.values(usersMap)
      .filter(user => user.province === filters.selectedProvince)
      .map(user => user.district);

    return [...new Set(districtsInProvince)].filter(district => district && district !== "ไม่ระบุอำเภอ").sort();
  }, [usersMap, filters.selectedProvince]);

  // ฟังก์ชันประมวลผลสถิติ
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

  const { startDate, endDate, selectedProvince, selectedDistrict } = filters;

  // กรองข้อมูล predictions ด้วย useMemo
  const filteredPredictions = useMemo(() => {
    console.log("=== Filtering Data ===");
    console.log("All predictions:", allPredictions.length);
    console.log("Filters:", { startDate, endDate, selectedProvince, selectedDistrict });

    const filtered = allPredictions.filter(prediction => {
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
        return false;
      }

      const userInfo = usersMap[prediction.userId];
      if (!userInfo) {
        if (selectedProvince || selectedDistrict) {
          return false;
        }
        return true;
      }

      if (selectedProvince && userInfo.province !== selectedProvince) {
        return false;
      }

      if (selectedDistrict && userInfo.district !== selectedDistrict) {
        return false;
      }

      return true;
    });

    console.log("Filtered predictions:", filtered.length);
    return filtered;
  }, [allPredictions, startDate, endDate, selectedProvince, selectedDistrict, usersMap]);

  // useEffect สำหรับดึงข้อมูลครั้งแรก
  useEffect(() => {
    const fetchStatistics = async () => {
      const db = getFirestore();
      setLoading(true);

      try {
        console.log("เริ่มดึงข้อมูล...");

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
            rawData: data
          };

          if (createdAt && createdAt instanceof Date && !isNaN(createdAt.getTime())) {
            predictionsData.push(predictionItem);
            console.log("✅ Added prediction:", doc.id, "Date:", createdAt.toISOString().split('T')[0]);
          } else {
            console.warn("❌ Skipping prediction with invalid timestamp:", doc.id);
          }
        });

        console.log("ดึงข้อมูล predictions แล้ว:", predictionsData.length, "รายการ");
        setAllPredictions(predictionsData);

        console.log("กำลังประมวลผลสถิติ...");
        processStatistics(predictionsData, usersMapTemp);

      } catch (error) {
        console.error("เกิดข้อผิดพลาดในการดึงข้อมูลสถิติ:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchStatistics();
  }, [processStatistics]);

  // useEffect สำหรับประมวลผลใหม่เมื่อกรองข้อมูล
  useEffect(() => {
    if (filteredPredictions.length >= 0 && Object.keys(usersMap).length > 0) {
      console.log("ประมวลผลใหม่เมื่อกรองข้อมูล:", filteredPredictions.length, "รายการ");
      processStatistics(filteredPredictions, usersMap);
    }
  }, [filteredPredictions, usersMap, processStatistics]);

  // ฟังก์ชันอัพเดทตัวกรอง
  const updateFilter = (key, value) => {
    setFilters(prev => {
      const newFilters = { ...prev, [key]: value };

      if (key === 'selectedProvince') {
        newFilters.selectedDistrict = '';
      }

      return newFilters;
    });
  };

  // รีเซ็ตตัวกรอง
  const resetFilter = () => {
    setFilters({
      startDate: '',
      endDate: '',
      selectedProvince: '',
      selectedDistrict: ''
    });
  };

  const hasActiveFilters = startDate || endDate || selectedProvince || selectedDistrict;

  // ฟังก์ชันเปิด/ปิด fullscreen
  const openFullscreen = (chartType) => {
    setFullscreenChart(chartType);
    setZoomLevel(100); // รีเซ็ต zoom เมื่อเปิด fullscreen
  };

  const closeFullscreen = () => {
    setFullscreenChart(null);
    setZoomLevel(100); // รีเซ็ต zoom เมื่อปิด fullscreen
  };

  // ฟังก์ชันสร้างกราฟ PDF ที่ใช้สีเดียวกัน
  const generateChartImage = (chartType, data, options = {}) => {
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = options.width || 800;
      canvas.height = options.height || 400;

      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      if (chartType === 'pie') {
        drawPieChart(ctx, data, canvas.width, canvas.height);
      } else if (chartType === 'bar') {
        drawBarChart(ctx, data, canvas.width, canvas.height, options);
      } else if (chartType === 'line') {
        drawLineChart(ctx, data, canvas.width, canvas.height, options);
      }

      const imageData = canvas.toDataURL('image/png');
      resolve(imageData);
    });
  };

  // แก้ไขฟังก์ชัน drawPieChart
  const drawPieChart = (ctx, data, width, height) => {
    const centerX = width / 2 - 50;
    const centerY = height / 2;
    const radius = Math.min(width, height) / 4;

    const total = data.reduce((sum, item) => sum + item.value, 0);
    let currentAngle = -Math.PI / 2;

    // สร้างรายชื่อโรคที่เรียงลำดับ
    const sortedDiseases = Object.keys(diseaseStats).sort();

    data.forEach((item, index) => {
      const sliceAngle = (item.value / total) * 2 * Math.PI;

      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.arc(centerX, centerY, radius, currentAngle, currentAngle + sliceAngle);
      ctx.closePath();

      // ใช้สีที่ตรงกับโรค ไม่ใช่ index ของ data
      const colorIndex = sortedDiseases.indexOf(item.name);
      ctx.fillStyle = chartColors[colorIndex % chartColors.length];
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 3;
      ctx.stroke();

      // วาด label
      if (item.percentage > 3) {
        const labelAngle = currentAngle + sliceAngle / 2;
        const innerX = centerX + Math.cos(labelAngle) * (radius * 0.8);
        const innerY = centerY + Math.sin(labelAngle) * (radius * 0.8);
        const outerX = centerX + Math.cos(labelAngle) * (radius * 1.3);
        const outerY = centerY + Math.sin(labelAngle) * (radius * 1.3);
        const textX = outerX + (labelAngle > Math.PI / 2 || labelAngle < -Math.PI / 2 ? -30 : 30);
        const textY = outerY;

        ctx.strokeStyle = '#666666';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(innerX, innerY);
        ctx.lineTo(outerX, outerY);
        ctx.lineTo(textX, textY);
        ctx.stroke();

        ctx.fillStyle = '#666666';
        ctx.beginPath();
        ctx.arc(outerX, outerY, 2, 0, 2 * Math.PI);
        ctx.fill();

        ctx.fillStyle = '#000000';
        ctx.font = 'bold 12px Arial';
        ctx.textAlign = labelAngle > Math.PI / 2 || labelAngle < -Math.PI / 2 ? 'right' : 'left';
        ctx.textBaseline = 'middle';

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

    // Legend - ใช้ลำดับเดียวกับกราฟ
    const legendX = centerX + radius + 80;
    const legendY = centerY - (data.length * 25) / 2;

    ctx.font = 'bold 16px Arial';
    ctx.fillStyle = '#2E7D32';
    ctx.textAlign = 'left';
    ctx.fillText('สัดส่วนโรค', legendX, legendY - 20);

    data.forEach((item, index) => {
      const y = legendY + (index * 30);

      // ใช้สีที่ตรงกับโรค
      const colorIndex = sortedDiseases.indexOf(item.name);
      ctx.fillStyle = chartColors[colorIndex % chartColors.length];
      ctx.fillRect(legendX, y - 10, 18, 18);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.strokeRect(legendX, y - 10, 18, 18);

      ctx.fillStyle = '#333333';
      ctx.font = 'bold 11px Arial';
      ctx.fillText(item.name, legendX + 25, y - 5);

      ctx.fillStyle = '#666666';
      ctx.font = '10px Arial';
      ctx.fillText(`${item.value} ครั้ง (${item.percentage}%)`, legendX + 25, y + 8);
    });

    ctx.fillStyle = '#2E7D32';
    ctx.font = 'bold 18px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('สัดส่วนโรคที่พบ', width / 2, 20);
  };

  // แก้ไขฟังก์ชัน drawBarChart
  const drawBarChart = (ctx, data, width, height, options) => {
    const margin = { top: 50, right: 50, bottom: 150, left: 80 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    if (data.length === 0) return;

    // เรียงลำดับโรคให้สอดคล้องกัน
    const sortedDiseases = Object.keys(diseaseStats).sort();
    const diseases = sortedDiseases.filter(disease =>
      data.some(item => (item[disease] || 0) > 0)
    );

    const maxValue = Math.max(...data.map(item =>
      diseases.reduce((sum, disease) => sum + (item[disease] || 0), 0)
    ));

    const barWidth = chartWidth / data.length * 0.8;
    const barSpacing = chartWidth / data.length * 0.2;

    // แกน Y
    ctx.strokeStyle = '#cccccc';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = margin.top + (chartHeight / 5) * i;
      const value = maxValue - (maxValue / 5) * i;

      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(margin.left + chartWidth, y);
      ctx.stroke();

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

      diseases.forEach((disease) => {
        const value = item[disease] || 0;
        if (value > 0) {
          const barHeight = (value / maxValue) * chartHeight;

          // ใช้ index ที่ถูกต้องตามการเรียงลำดับโรค
          const colorIndex = sortedDiseases.indexOf(disease);
          ctx.fillStyle = chartColors[colorIndex % chartColors.length];
          ctx.fillRect(x, stackY - barHeight, barWidth, barHeight);

          stackY -= barHeight;
        }
      });

      // ป้ายแกน X
      ctx.fillStyle = '#333333';
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';

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

    // หัวเรื่อง
    ctx.fillStyle = '#333333';
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(options.title || 'แนวโน้มการวิเคราะห์รายเดือน', width / 2, 10);

    // Legend
    const legendX = width - 200;
    const legendY = margin.top;

    ctx.font = 'bold 12px Arial';
    ctx.fillStyle = '#000000';
    ctx.textAlign = 'left';
    ctx.fillText('โรค:', legendX, legendY);

    diseases.forEach((disease, index) => {
      const y = legendY + 20 + (index * 20);

      // ใช้ index ที่ถูกต้องตามการเรียงลำดับโรค
      const colorIndex = sortedDiseases.indexOf(disease);
      const color = chartColors[colorIndex % chartColors.length];

      ctx.fillStyle = color;
      ctx.fillRect(legendX, y - 6, 12, 12);

      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1;
      ctx.strokeRect(legendX, y - 6, 12, 12);

      ctx.fillStyle = '#000000';
      ctx.font = '10px Arial';
      ctx.fillText(disease, legendX + 20, y);
    });
  };

  // เพิ่มฟังก์ชัน drawLineChart
  const drawLineChart = (ctx, data, width, height, options) => {
    const margin = { top: 60, right: 100, bottom: 80, left: 80 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    if (data.length === 0) return;

    // เรียงลำดับโรคให้สอดคล้องกัน
    const sortedDiseases = Object.keys(diseaseStats).sort();
    const diseases = sortedDiseases.filter(disease =>
      data.some(item => (item[disease] || 0) > 0)
    );

    const maxValue = Math.max(...data.map(item =>
      diseases.reduce((max, disease) => Math.max(max, item[disease] || 0), 0)
    ));

    // วาดกริด Y
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = margin.top + (chartHeight / 5) * i;
      const value = maxValue - (maxValue / 5) * i;

      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(margin.left + chartWidth, y);
      ctx.stroke();

      // ป้าย Y
      ctx.fillStyle = '#666666';
      ctx.font = '11px Arial';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(Math.round(value).toString(), margin.left - 10, y);
    }

    // วาดเส้นกราฟสำหรับแต่ละโรค
    diseases.forEach((disease, diseaseIndex) => {
      const colorIndex = sortedDiseases.indexOf(disease);
      const color = chartColors[colorIndex % chartColors.length];

      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 3;

      let firstPoint = true;
      ctx.beginPath();

      data.forEach((item, dataIndex) => {
        const value = item[disease] || 0;
        if (value > 0) {
          const x = margin.left + (dataIndex / (data.length - 1)) * chartWidth;
          const y = margin.top + chartHeight - (value / maxValue) * chartHeight;

          if (firstPoint) {
            ctx.moveTo(x, y);
            firstPoint = false;
          } else {
            ctx.lineTo(x, y);
          }

          // วาดจุด
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, 2 * Math.PI);
          ctx.fill();
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2;
          ctx.stroke();
          ctx.strokeStyle = color;
          ctx.lineWidth = 3;
        }
      });

      if (!firstPoint) {
        ctx.stroke();
      }
    });

    // ป้ายแกน X
    ctx.fillStyle = '#333333';
    ctx.font = '10px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    data.forEach((item, index) => {
      const x = margin.left + (index / (data.length - 1)) * chartWidth;
      const label = item.month;

      // แบ่งป้ายเป็นหลายบรรทัดถ้าจำเป็น
      const words = label.split(' ');
      words.forEach((word, wordIndex) => {
        ctx.fillText(word, x, margin.top + chartHeight + 15 + (wordIndex * 12));
      });
    });

    // วาดเส้นแกน
    ctx.strokeStyle = '#333333';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top);
    ctx.lineTo(margin.left, margin.top + chartHeight);
    ctx.lineTo(margin.left + chartWidth, margin.top + chartHeight);
    ctx.stroke();

    // หัวเรื่อง
    ctx.fillStyle = '#2E7D32';
    ctx.font = 'bold 18px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(options.title || 'แนวโน้มการวิเคราะห์รายเดือน', width / 2, 20);

    // Legend
    const legendX = width - 180;
    const legendY = margin.top;

    ctx.font = 'bold 12px Arial';
    ctx.fillStyle = '#2E7D32';
    ctx.textAlign = 'left';
    ctx.fillText('โรค:', legendX, legendY);

    diseases.forEach((disease, index) => {
      const y = legendY + 25 + (index * 22);
      const colorIndex = sortedDiseases.indexOf(disease);
      const color = chartColors[colorIndex % chartColors.length];

      // วาดเส้นตัวอย่าง
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(legendX, y - 2);
      ctx.lineTo(legendX + 20, y - 2);
      ctx.stroke();

      // วาดจุด
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(legendX + 10, y - 2, 3, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1;
      ctx.stroke();

      // ป้ายชื่อโรค
      ctx.fillStyle = '#000000';
      ctx.font = '11px Arial';
      ctx.fillText(disease, legendX + 28, y + 2);
    });
  };

  const handleDownloadPDF = async () => {
    try {
      const currentDate = new Date().toLocaleDateString("th-TH");

      console.log("กำลังสร้างกราฟ...");

      let pieChartImage = null;
      let barChartImage = null;
      let lineChartImage = null;

      if (pieData.length > 0) {
        pieChartImage = await generateChartImage('pie', pieData, { width: 600, height: 400 });
      }

      if (chartData.length > 0) {
        barChartImage = await generateChartImage('bar', chartData, {
          width: 800,
          height: 500,
          title: 'การกระจายโรคตามพื้นที่'
        });
      }

      if (timelineData.length > 0) {
        lineChartImage = await generateChartImage('line', timelineData, {
          width: 800,
          height: 400,
          title: 'แนวโน้มการวิเคราะห์รายเดือน'
        });
      }

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

      const timelineTableData = [
        [{ text: 'เดือน', style: 'tableHeader' }, { text: 'จำนวนรวม', style: 'tableHeader' }]
      ];

      const diseaseNames = Object.keys(diseaseStats);
      diseaseNames.forEach(disease => {
        timelineTableData[0].push({ text: disease, style: 'tableHeader' });
      });

      timelineData.forEach(monthData => {
        const row = [monthData.month, monthData.count.toString()];
        diseaseNames.forEach(disease => {
          row.push((monthData[disease] || 0).toString());
        });
        timelineTableData.push(row);
      });

      const monthlySummary = timelineData.map(monthData => {
        return `${monthData.month} จำนวนโรครวม = ${monthData.count} ครั้ง`;
      }).join('\n');

      const totalIndex = timelineTableData[0].findIndex(cell => cell.text === 'จำนวนรวม');

      if (totalIndex !== -1) {
        timelineTableData.forEach(row => {
          row.splice(totalIndex, 1);
        });
      }

      const docDefinition = {
        content: [
          { text: 'รายงานสถิติการวิเคราะห์โรคใบมะม่วง', style: 'header' },
          { text: `วันที่สร้างรายงาน: ${currentDate}`, style: 'subheader' },

          ...(hasActiveFilters ? [
            { text: 'เงื่อนไขการกรองข้อมูล:', style: 'sectionHeader' },
            {
              ul: filterInfo,
              margin: [0, 0, 0, 15]
            }
          ] : []),

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

          ...(pieChartImage ? [
            { text: 'สัดส่วนโรคที่พบ (กราฟวงกลม):', style: 'sectionHeader' },
            {
              image: pieChartImage,
              width: 500,
              alignment: 'center',
              margin: [0, 0, 0, 20]
            }
          ] : []),

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

          ...(lineChartImage ? [
            { text: 'แนวโน้มการวิเคราะห์รายเดือน (กราฟเส้น):', style: 'sectionHeader' },
            {
              image: lineChartImage,
              width: 500,
              alignment: 'center',
              margin: [0, 0, 0, 20]
            }
          ] : []),

          ...(barChartImage ? [
            { text: 'การกระจายโรคตามพื้นที่ (กราฟแท่ง):', style: 'sectionHeader', pageBreak: 'before' },
            {
              image: barChartImage,
              width: 500,
              alignment: 'center',
              margin: [0, 0, 0, 0]
            }
          ] : []),

          ...(Object.keys(districtDiseaseMap).length > 0 ? [
            { text: 'สถิติการพบโรคตามพื้นที่:', style: 'sectionHeader' },
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

          ...(timelineData.length > 0 ? [
            { text: 'แนวโน้มการวิเคราะห์รายเดือน:', style: 'sectionHeader', pageBreak: 'before' },
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

  // ส่วนของ FullscreenChart Component ที่แก้ไขแล้ว พร้อมฟีเจอร์ zoom
  const FullscreenChart = ({ type, onClose }) => {
    const chartDataToShow = useMemo(() => {
      if (type === 'line') return timelineData;
      if (type === 'bar') return chartData;
      if (type === 'pie') return pieData;
      return [];
    }, [type]);

    const diseaseNames = useMemo(() => {
      return Object.keys(diseaseStats).sort();
    }, []);

    // คำนวณขนาดกราฟตาม zoom level
    // คำนวณขนาดกราฟตาม zoom level โดยไม่ใช้ useMemo
    const baseWidth = type === 'bar'
      ? Math.max(1000, chartDataToShow.length * 150)
      : Math.max(1000, chartDataToShow.length * 80);

    const chartSize = {
      width: `${Math.floor((baseWidth * zoomLevel) / 100)}px`,
      height: `${Math.floor((450 * zoomLevel) / 100)}px`
    };

    const handleZoomIn = useCallback(() => {
      setZoomLevel(prev => Math.min(prev + 25, 300));
    }, []);

    const handleZoomOut = useCallback(() => {
      setZoomLevel(prev => Math.max(prev - 25, 50));
    }, []);

    return (
      <div className="fullscreen-overlay" onClick={onClose}>
        <div className="fullscreen-container" onClick={(e) => e.stopPropagation()}>
          <div className="fullscreen-header">
            <h2>
              {type === 'line' && 'แนวโน้มการวิเคราะห์รายเดือน'}
              {type === 'bar' && 'การกระจายโรคตามพื้นที่'}
              {type === 'pie' && 'สัดส่วนโรคที่พบ'}
            </h2>

            {/* Zoom Controls */}
            <div className="zoom-controls">
              <button
                className="zoom-btn zoom-out"
                onClick={handleZoomOut}
                disabled={zoomLevel <= 50}
                title="ซูมออก"
              >
                -
              </button>
              <button
                className="zoom-btn zoom-in"
                onClick={handleZoomIn}
                disabled={zoomLevel >= 300}
                title="ซูมเข้า"
              >
                +
              </button>
            </div>

            <button className="close-btn" onClick={onClose}>✕</button>
          </div>
          <div className="fullscreen-chart">
            {(type === 'bar' || type === 'line') ? (
              <div className="fullscreen-scrollable-chart">
                <div
                  className="fullscreen-chart-wrapper"
                  style={chartSize}
                >
                  <ResponsiveContainer width="100%" height="100%">
                    {type === 'bar' && (
                      <BarChart data={chartDataToShow} margin={{ top: 20, right: 30, bottom: 80, left: 40 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          dataKey="locationLabel"
                          angle={-45}
                          textAnchor="end"
                          height={120}
                          tick={{ fontSize: Math.max(10, 14 * zoomLevel / 100) }}
                          interval={0}
                        />
                        <YAxis tick={{ fontSize: Math.max(8, 12 * zoomLevel / 100) }} />
                        <Tooltip formatter={(value, name) => [`${value} ครั้ง`, name]} />
                        {diseaseNames.map((disease, idx) => (
                          <Bar
                            key={disease}
                            dataKey={disease}
                            stackId="a"
                            fill={chartColors[idx % chartColors.length]}
                          />
                        ))}
                      </BarChart>
                    )}
                    {type === 'line' && (
                      <LineChart data={chartDataToShow} margin={{ top: 40, right: 50, left: 40, bottom: 80 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          dataKey="month"
                          angle={-30}
                          textAnchor="end"
                          height={70}
                          tick={{ fontSize: Math.max(8, 12 * zoomLevel / 100) }}
                          interval={0}
                        />
                        <YAxis
                          tick={{ fontSize: Math.max(8, 12 * zoomLevel / 100) }}
                          domain={[0, 'dataMax + 2']}
                        />
                        <Tooltip formatter={(value) => [`${value} ครั้ง`]} />
                        {diseaseNames.map((disease, idx) => (
                          <Line
                            key={disease}
                            type="monotone"
                            dataKey={disease}
                            stroke={chartColors[idx % chartColors.length]}
                            strokeWidth={Math.max(2, 3 * zoomLevel / 100)}
                            dot={{ r: Math.max(3, 6 * zoomLevel / 100) }}
                          />
                        ))}
                      </LineChart>
                    )}
                  </ResponsiveContainer>
                </div>
              </div>
            ) : (
              <div style={{
                transform: `scale(${zoomLevel / 100})`,
                transformOrigin: 'center center',
                transition: 'transform 0.3s ease'
              }}>
                <ResponsiveContainer width="100%" height={450}>
                  <PieChart>
                    <Pie
                      data={chartDataToShow}
                      cx="50%"
                      cy="50%"
                      outerRadius={120 * Math.min(zoomLevel / 100, 2)}
                      dataKey="value"
                      label={({ name, percentage }) => `${name}: ${percentage}%`}
                      labelLine={false}
                    >
                      {chartDataToShow.map((entry, index) => {
                        const diseaseNames = Object.keys(diseaseStats).sort();
                        const colorIndex = diseaseNames.indexOf(entry.name);
                        return (
                          <Cell key={`cell-${index}`} fill={chartColors[colorIndex % chartColors.length]} />
                        );
                      })}
                    </Pie>
                    <Tooltip formatter={(value, name) => [`${value} ครั้ง`, name]} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        </div>
      </div>
    );
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
    <>
      <div className="statistics-admin-container">
        <div className="header-section">
          <h2>สถิติการวิเคราะห์โรคใบมะม่วง</h2>
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
          {/* Disease Statistics - มีปุ่ม fullscreen */}
          <div className="chart-container">
            <div className="chart-header">
              <h3>📊 สัดส่วนโรคที่พบ</h3>
              {pieData.length > 0 && (
                <button
                  className="fullscreen-btn"
                  onClick={() => openFullscreen('pie')}
                  title="ดูแบบเต็มหน้าจอ"
                >
                  ⛶
                </button>
              )}
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
                        const diseaseNames = Object.keys(diseaseStats).sort();
                        const colorIndex = diseaseNames.indexOf(disease);
                        return (
                          <div key={disease} className="disease-item">
                            <div className="disease-info">
                              <div className="disease-label">
                                <div
                                  className="disease-color"
                                  style={{ backgroundColor: chartColors[colorIndex % chartColors.length] }}
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
                                  backgroundColor: chartColors[colorIndex % chartColors.length]
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
                    <p>ยังไม่มีข้อมูลการวิเคราะห์</p>
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
                        {pieData.map((entry, index) => {
                          const diseaseNames = Object.keys(diseaseStats).sort();
                          const colorIndex = diseaseNames.indexOf(entry.name);
                          return (
                            <Cell key={`cell-${index}`} fill={chartColors[colorIndex % chartColors.length]} />
                          );
                        })}
                      </Pie>
                      <Tooltip formatter={(value, name) => [`${value} ครั้ง`, name]} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          </div>

          {/* Timeline Chart - มีปุ่ม fullscreen และแสดงกราฟในคอนเทนเนอร์ที่เลื่อนได้ */}
          {timelineData.length > 0 && (
            <div className="chart-container">
              <div className="chart-header">
                <h3>📈 แนวโน้มการวิเคราะห์รายเดือน</h3>
                <button
                  className="fullscreen-btn"
                  onClick={() => openFullscreen('line')}
                  title="ดูแบบเต็มหน้าจอ"
                >
                  ⛶
                </button>
                <p style={{ fontSize: '14px', color: '#666', margin: '5px 0' }}>
                  แสดงแนวโน้มการวิเคราะห์แต่ละโรคตามเดือน
                </p>
              </div>
              <div className="chart-content">
                <div className="scrollable-chart-container">
                  <div className="chart-wrapper" style={{ minWidth: `${Math.max(800, timelineData.length * 60)}px` }}>
                    <ResponsiveContainer width="100%" height={400}>
                      <LineChart data={timelineData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          dataKey="month"
                          interval={0}
                          angle={-45}
                          textAnchor="end"
                          height={100}
                        />
                        <YAxis />
                        <Tooltip formatter={(value, name) => [`${value} ครั้ง`, name]} />

                        {Object.keys(diseaseStats).map((disease, idx) => {
                          const diseaseNames = Object.keys(diseaseStats).sort();
                          const colorIndex = diseaseNames.indexOf(disease);
                          return (
                            <Line
                              key={disease}
                              type="monotoneX"
                              dataKey={disease}
                              stroke={chartColors[colorIndex % chartColors.length]}
                              strokeWidth={4}
                              dot={{
                                fill: chartColors[colorIndex % chartColors.length],
                                strokeWidth: 3,
                                r: 8,
                                stroke: '#fff'
                              }}
                              activeDot={{
                                r: 10,
                                stroke: chartColors[colorIndex % chartColors.length],
                                strokeWidth: 3,
                                fill: '#fff'
                              }}
                              name={disease}
                              connectNulls={false}
                            />
                          );
                        })}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="timeline-legend" style={{ marginTop: '20px' }}>
                  <h4>โรคที่แสดงในกราฟ</h4>
                  <div className="legend-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px' }}>
                    {Object.keys(diseaseStats).map((disease) => {
                      const diseaseNames = Object.keys(diseaseStats).sort();
                      const colorIndex = diseaseNames.indexOf(disease);
                      return (
                        <div key={disease} className="legend-item" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                          <div
                            style={{
                              width: '20px',
                              height: '20px',
                              borderRadius: '50%',
                              backgroundColor: chartColors[colorIndex % chartColors.length],
                              border: '3px solid #fff',
                              boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                            }}
                          ></div>
                          <span style={{ fontWeight: '500' }}>{disease}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* District Chart - มีปุ่ม fullscreen และแสดงกราฟในคอนเทนเนอร์ที่เลื่อนได้ */}
          {chartData.length > 0 && (
            <div className="chart-container full-width">
              <div className="chart-header">
                <h3>📍 การกระจายโรคตามพื้นที่</h3>
                <button
                  className="fullscreen-btn"
                  onClick={() => openFullscreen('bar')}
                  title="ดูแบบเต็มหน้าจอ"
                >
                  ⛶
                </button>
                <p style={{ fontSize: '14px', color: '#666', margin: '5px 0' }}>
                  การกระจายโรคแต่ละชนิดตามพื้นที่
                </p>
              </div>
              <div className="chart-content">
                <div className="scrollable-chart-container">
                  <div className="chart-wrapper" style={{ minWidth: `${Math.max(800, chartData.length * 120)}px` }}>
                    <ResponsiveContainer width="100%" height={400}>
                      <BarChart data={chartData} margin={{ top: 20, right: 30, bottom: 120, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          dataKey="locationLabel"
                          angle={-45}
                          textAnchor="end"
                          interval={0}
                          height={120}
                        />
                        <YAxis />
                        <Tooltip formatter={(value, name) => [`${value} ครั้ง`, name]} />
                        {Object.keys(diseaseStats).map((disease, idx) => {
                          const diseaseNames = Object.keys(diseaseStats).sort();
                          const colorIndex = diseaseNames.indexOf(disease);
                          return (
                            <Bar
                              key={disease}
                              dataKey={disease}
                              stackId="a"
                              fill={chartColors[colorIndex % chartColors.length]}
                            />
                          );
                        })}
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="chart-legend">
                  <h4>สีแทนโรค</h4>
                  <div className="legend-grid">
                    {Object.keys(diseaseStats).map((disease) => {
                      const diseaseNames = Object.keys(diseaseStats).sort();
                      const colorIndex = diseaseNames.indexOf(disease);
                      return (
                        <div key={disease} className="legend-item">
                          <div
                            className="legend-color"
                            style={{ backgroundColor: chartColors[colorIndex % chartColors.length] }}
                          ></div>
                          <span className="legend-text">{disease}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Download PDF Button */}
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

        {!Object.keys(diseaseStats).length && (
          <div className="empty-state">
            <div className="empty-icon">📊</div>
            <h3>ไม่มีข้อมูลสถิติ</h3>
            <p>ยังไม่มีการวิเคราะห์โรคในระบบ หรือไม่มีข้อมูลในช่วงเวลาที่เลือก</p>
          </div>
        )}
      </div>

      {/* Fullscreen Chart Modal */}
      {fullscreenChart && (
        <FullscreenChart
          type={fullscreenChart}
          onClose={closeFullscreen}
        />
      )}
    </>
  );
}

export default StatisticsAdmin;