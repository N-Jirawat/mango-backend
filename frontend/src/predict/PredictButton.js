import React, { useState } from 'react';
import ImageUpload from './ImageUpLoad'; // import ImageUpload component
import { useNavigate } from 'react-router-dom';
import '../css/predictbutton.css';

function PredictPage() {
  const [predictionResult, setPredictionResult] = useState(null);
  const navigate = useNavigate();

  const handleGoHome = () => {
    navigate('/');
  };

  return (
    <div className="predict-page-container">
      <div className="predict-header">
        <button onClick={handleGoHome} className="back-button">
          ⬅️ หน้าหลัก
        </button>
      </div>

      <ImageUpload setPredictionResult={setPredictionResult} />
      {predictionResult && (
        <div style={{ marginTop: '20px' }}>
          <h3>ผลการทำนาย:</h3>
          <p>โรค: {predictionResult.prediction}</p>
          <p>ความมั่นใจ: {(predictionResult.confidence * 100).toFixed(4)}%</p>
          <p>ความแม่นยำ: {(predictionResult.accuracy * 100).toFixed(4)}%</p>
        </div>
      )}
    </div>
  );
}

export default PredictPage;