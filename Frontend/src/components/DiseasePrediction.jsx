// src/components/DiseasePrediction.jsx
import React, { useState } from 'react';
import './DiseasePrediction.css';
import api from '../api';

// 👇 Map disease → backend endpoint (with /predict)
const ENDPOINTS = {
  diabetes: '/api/diabetes/predict',
  heart: '/api/heart/predict',
  breast: '/api/breast/predict',
};

// 👇 Field definitions for the UI (labels match backend feature names)
const diseaseFields = {
  diabetes: [
    { name: 'Pregnancies', label: 'Pregnancies', type: 'number', step: 1 },
    { name: 'Glucose', label: 'Glucose Level', type: 'number', step: 0.1 },
    { name: 'BloodPressure', label: 'Blood Pressure (mm Hg)', type: 'number', step: 1 },
    { name: 'SkinThickness', label: 'Skin Thickness (mm)', type: 'number', step: 1 },
    { name: 'Insulin', label: 'Insulin Level', type: 'number', step: 0.1 },
    { name: 'BMI', label: 'BMI', type: 'number', step: 0.1 },
    { name: 'DiabetesPedigreeFunction', label: 'Diabetes Pedigree Function', type: 'number', step: 0.01 },
    { name: 'Age', label: 'Age', type: 'number', step: 1 },
  ],
  heart: [
    { name: 'age', label: 'Age', type: 'number', step: 1 },
    { name: 'sex', label: 'Sex', type: 'select', options: [{ value: 0, label: 'Female' }, { value: 1, label: 'Male' }] },
    { name: 'cp', label: 'Chest Pain Type', type: 'select', options: [{ value: 0, label: 'Typical Angina' }, { value: 1, label: 'Atypical Angina' }, { value: 2, label: 'Non-anginal Pain' }, { value: 3, label: 'Asymptomatic' }] },
    { name: 'trestbps', label: 'Resting Blood Pressure (mm Hg)', type: 'number', step: 1 },
    { name: 'chol', label: 'Cholesterol (mg/dl)', type: 'number', step: 1 },
    { name: 'fbs', label: 'Fasting Blood Sugar > 120', type: 'select', options: [{ value: 0, label: 'False' }, { value: 1, label: 'True' }] },
    { name: 'restecg', label: 'Resting ECG Results', type: 'select', options: [{ value: 0, label: 'Normal' }, { value: 1, label: 'ST-T Wave Abnormality' }, { value: 2, label: 'Left Ventricular Hypertrophy' }] },
    { name: 'thalach', label: 'Max Heart Rate Achieved', type: 'number', step: 1 },
    { name: 'exang', label: 'Exercise Induced Angina', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
    { name: 'oldpeak', label: 'ST Depression Induced by Exercise', type: 'number', step: 0.1 },
    { name: 'slope', label: 'Slope of Peak Exercise ST Segment', type: 'select', options: [{ value: 0, label: 'Upsloping' }, { value: 1, label: 'Flat' }, { value: 2, label: 'Downsloping' }] },
    { name: 'ca', label: 'Number of Major Vessels (0-3)', type: 'select', options: [{ value: 0, label: '0' }, { value: 1, label: '1' }, { value: 2, label: '2' }, { value: 3, label: '3' }] },
    { name: 'thal', label: 'Thalassemia', type: 'select', options: [{ value: 0, label: 'Normal' }, { value: 1, label: 'Fixed Defect' }, { value: 2, label: 'Reversible Defect' }, { value: 3, label: 'Not Described' }] },
  ],
  breast: [
    // ---- Mean values (10) ----
    { name: 'radius_mean', label: 'Radius (mean)', type: 'number', step: 0.01 },
    { name: 'texture_mean', label: 'Texture (mean)', type: 'number', step: 0.01 },
    { name: 'perimeter_mean', label: 'Perimeter (mean)', type: 'number', step: 0.01 },
    { name: 'area_mean', label: 'Area (mean)', type: 'number', step: 0.1 },
    { name: 'smoothness_mean', label: 'Smoothness (mean)', type: 'number', step: 0.001 },
    { name: 'compactness_mean', label: 'Compactness (mean)', type: 'number', step: 0.001 },
    { name: 'concavity_mean', label: 'Concavity (mean)', type: 'number', step: 0.001 },
    { name: 'concave_points_mean', label: 'Concave Points (mean)', type: 'number', step: 0.001 },
    { name: 'symmetry_mean', label: 'Symmetry (mean)', type: 'number', step: 0.001 },
    { name: 'fractal_dimension_mean', label: 'Fractal Dimension (mean)', type: 'number', step: 0.001 },
    // ---- Standard Error values (10) ----
    { name: 'radius_se', label: 'Radius (se)', type: 'number', step: 0.01 },
    { name: 'texture_se', label: 'Texture (se)', type: 'number', step: 0.01 },
    { name: 'perimeter_se', label: 'Perimeter (se)', type: 'number', step: 0.01 },
    { name: 'area_se', label: 'Area (se)', type: 'number', step: 0.1 },
    { name: 'smoothness_se', label: 'Smoothness (se)', type: 'number', step: 0.001 },
    { name: 'compactness_se', label: 'Compactness (se)', type: 'number', step: 0.001 },
    { name: 'concavity_se', label: 'Concavity (se)', type: 'number', step: 0.001 },
    { name: 'concave_points_se', label: 'Concave Points (se)', type: 'number', step: 0.001 },
    { name: 'symmetry_se', label: 'Symmetry (se)', type: 'number', step: 0.001 },
    { name: 'fractal_dimension_se', label: 'Fractal Dimension (se)', type: 'number', step: 0.001 },
    // ---- Worst values (10) ----
    { name: 'radius_worst', label: 'Radius (worst)', type: 'number', step: 0.01 },
    { name: 'texture_worst', label: 'Texture (worst)', type: 'number', step: 0.01 },
    { name: 'perimeter_worst', label: 'Perimeter (worst)', type: 'number', step: 0.01 },
    { name: 'area_worst', label: 'Area (worst)', type: 'number', step: 0.1 },
    { name: 'smoothness_worst', label: 'Smoothness (worst)', type: 'number', step: 0.001 },
    { name: 'compactness_worst', label: 'Compactness (worst)', type: 'number', step: 0.001 },
    { name: 'concavity_worst', label: 'Concavity (worst)', type: 'number', step: 0.001 },
    { name: 'concave_points_worst', label: 'Concave Points (worst)', type: 'number', step: 0.001 },
    { name: 'symmetry_worst', label: 'Symmetry (worst)', type: 'number', step: 0.001 },
    { name: 'fractal_dimension_worst', label: 'Fractal Dimension (worst)', type: 'number', step: 0.001 },
  ],
};

const DiseasePrediction = ({ disease }) => {
  const [formData, setFormData] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fields = diseaseFields[disease] || [];

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setError(null);

    // Convert all values to numbers (backend expects floats)
    const payload = {};
    Object.keys(formData).forEach((key) => {
      const val = formData[key];
      payload[key] = isNaN(parseFloat(val)) ? val : parseFloat(val);
    });

    try {
      const endpoint = ENDPOINTS[disease];
      if (!endpoint) throw new Error('Unknown disease type');

      console.log(`📡 Calling: ${api.defaults.baseURL}${endpoint}`);

      const response = await api.post(endpoint, payload);

      // Response format from backend: { prediction, confidence, probability, risk_level, recommendation, details }
      setResult({
        prediction: response.data.prediction === 1 ? 'Positive' : 'Negative',
        confidence: response.data.confidence || 0,
        probability: response.data.probability || 0,
        risk_level: response.data.risk_level || 'LOW',
        recommendation: response.data.recommendation || '',
        details: response.data.details || {},
      });
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to get prediction.');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const title = disease.charAt(0).toUpperCase() + disease.slice(1);

  return (
    <div className="page-container disease-prediction">
      <h2 className="page-title">{title} Disease Prediction</h2>
      <p className="page-subtitle">
        Enter the required parameters below to get a prediction.
      </p>

      <form onSubmit={handleSubmit} className="prediction-form">
        {fields.map((field) => (
          <div className="input-group" key={field.name}>
            <label htmlFor={field.name}>{field.label}</label>
            {field.type === 'select' ? (
              <select
                id={field.name}
                name={field.name}
                value={formData[field.name] || ''}
                onChange={handleChange}
                required
              >
                <option value="">Select...</option>
                {field.options.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="number"
                id={field.name}
                name={field.name}
                step={field.step || 'any'}
                value={formData[field.name] || ''}
                onChange={handleChange}
                required
                placeholder={`Enter ${field.label.toLowerCase()}`}
              />
            )}
          </div>
        ))}
        <button type="submit" className="predict-btn" disabled={loading}>
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </form>

      {error && (
        <div className="error-message">
          <span>⚠️ {error}</span>
        </div>
      )}

      {result && (
        <div className="result-card">
          <h3>Prediction Result</h3>
          <div className="result-row">
            <span className="result-label">Diagnosis:</span>
            <span className={`result-value ${result.prediction.toLowerCase()}`}>
              {result.prediction}
            </span>
          </div>
          <div className="result-row">
            <span className="result-label">Confidence:</span>
            <span className="result-value">{(result.confidence * 100).toFixed(1)}%</span>
          </div>
          <div className="confidence-bar-container">
            <div
              className="confidence-bar-fill"
              style={{ width: `${Math.min(Math.max(result.confidence * 100, 0), 100)}%` }}
            />
          </div>
          <div className="result-row">
            <span className="result-label">Risk Level:</span>
            <span className={`result-value risk-${result.risk_level.toLowerCase()}`}>
              {result.risk_level}
            </span>
          </div>
          <p className="result-details">{result.recommendation}</p>
          <p className="disclaimer">
            ⚠️ This is an AI‑generated prediction. Always consult a healthcare professional.
          </p>
        </div>
      )}
    </div>
  );
};

export default DiseasePrediction;