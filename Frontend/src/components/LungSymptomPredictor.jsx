// src/components/LungSymptomPredictor.jsx
import React, { useState } from 'react';
import './LungSymptomPredictor.css';
import api from '../api';

const LUNG_ENDPOINT = '/api/lung/symptom_predict';

const LungSymptomPredictor = () => {
  const [symptomText, setSymptomText] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handlePredict = async () => {
    if (!symptomText.trim()) {
      setError('Please describe your symptoms.');
      return;
    }
    setLoading(true);
    setResults(null);
    setError(null);

    try {
      const response = await api.post(LUNG_ENDPOINT, {
        symptoms: symptomText,
        top_k: 3,
      });

      if (response.data.success) {
        setResults({
          predictions: response.data.predictions || [],
          input: response.data.input,
        });
      } else {
        throw new Error('Prediction failed');
      }
    } catch (err) {
      console.error('Lung prediction error:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to get prediction.');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handlePredict();
    }
  };

  return (
    <div className="page-container lung-symptom-predictor">
      <h2>Lung Symptom Check</h2>
      <p className="page-subtitle">
        Describe your respiratory symptoms in detail, and our AI will suggest possible conditions.
      </p>

      <textarea
        className="symptom-textarea"
        rows="4"
        placeholder="e.g., I have a persistent cough, shortness of breath, and chest pain..."
        value={symptomText}
        onChange={(e) => setSymptomText(e.target.value)}
        onKeyPress={handleKeyPress}
        disabled={loading}
      />

      <button className="predict-btn" onClick={handlePredict} disabled={loading}>
        {loading ? 'Analyzing...' : 'Analyze Symptoms'}
      </button>

      {error && (
        <div className="error-message">
          <span>⚠️ {error}</span>
        </div>
      )}

      {results && results.predictions.length > 0 && (
        <div className="results">
          <h3>🔬 Possible Conditions</h3>
          <ul className="prediction-list">
            {results.predictions.map((pred, index) => {
              const displayName = pred.disease.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
              const confidence = pred.confidence || 0;
              const barColor = confidence > 0.7 ? '#4CAF50' : confidence > 0.4 ? '#FFB800' : '#FF6584';
              return (
                <li key={index}>
                  <span className="disease-name">{displayName}</span>
                  <span className="confidence">{(confidence * 100).toFixed(0)}%</span>
                  <div className="confidence-bar">
                    <div
                      className="bar-fill"
                      style={{ width: `${Math.min(Math.max(confidence * 100, 0), 100)}%`, background: barColor }}
                    />
                  </div>
                </li>
              );
            })}
          </ul>
          <p className="disclaimer">
            ⚠️ This is an AI‑assisted screening tool. Always consult a healthcare professional for a definitive diagnosis.
          </p>
        </div>
      )}
    </div>
  );
};

export default LungSymptomPredictor;