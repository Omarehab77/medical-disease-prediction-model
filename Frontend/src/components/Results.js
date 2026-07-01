import React from 'react';
import './Results.css';
import { motion } from 'framer-motion';
import { FaCheckCircle, FaTimesCircle, FaRedo, FaDownload } from 'react-icons/fa';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import { toast } from 'react-toastify'; // Add this import

const Results = ({ data, onReset }) => {
  const { prediction, probability, recommendations, details } = data;
  const isPositive = prediction === 1 || prediction === 'Positive' || prediction === 'Yes';

  const chartData = [
    { name: 'Risk', value: probability * 100 },
    { name: 'No Risk', value: (1 - probability) * 100 }
  ];

  const COLORS = ['#ef4444', '#22c55e'];

  const handleDownloadReport = () => {
    // Generate report download
    toast.info('Report download feature coming soon!');
  };

  return (
    <motion.div 
      className="results-container"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className={`result-header ${isPositive ? 'positive' : 'negative'}`}>
        <div className="result-icon">
          {isPositive ? <FaTimesCircle /> : <FaCheckCircle />}
        </div>
        <div className="result-text">
          <h2>{isPositive ? 'High Risk Detected' : 'Low Risk Detected'}</h2>
          <p>
            {isPositive 
              ? 'Based on the analysis, there is a significant risk. Please consult a healthcare professional.'
              : 'Based on the analysis, the risk appears to be low. Continue maintaining a healthy lifestyle.'}
          </p>
        </div>
      </div>

      <div className="result-details">
        <div className="confidence-section">
          <h3>Confidence Analysis</h3>
          <div className="confidence-display">
            <div className="confidence-number">
              <span className="value">{(probability * 100).toFixed(1)}%</span>
              <span className="label">Confidence Level</span>
            </div>
            <div className="confidence-chart">
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={chartData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="recommendations-section">
          <h3>Recommendations</h3>
          <ul>
            {recommendations.map((rec, idx) => (
              <motion.li 
                key={idx}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.1 }}
              >
                {rec}
              </motion.li>
            ))}
          </ul>
        </div>

        {details && (
          <div className="details-section">
            <h3>Detailed Analysis</h3>
            <div className="details-grid">
              {Object.entries(details).map(([key, value]) => (
                <div key={key} className="detail-item">
                  <span className="detail-label">{key.replace(/_/g, ' ')}:</span>
                  <span className="detail-value">{value}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="result-actions">
        <button className="action-btn secondary" onClick={onReset}>
          <FaRedo />
          New Prediction
        </button>
        <button className="action-btn primary" onClick={handleDownloadReport}>
          <FaDownload />
          Download Report
        </button>
      </div>
    </motion.div>
  );
};

export default Results;