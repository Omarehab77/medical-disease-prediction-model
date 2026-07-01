import React, { useState } from 'react';
import './PredictionForm.css';
import { motion } from 'framer-motion';
import { FaArrowRight } from 'react-icons/fa';

const PredictionForm = ({ fields, onSubmit, isLoading }) => {
  const [formData, setFormData] = useState({});
  const [errors, setErrors] = useState({});

  const handleChange = (name, value) => {
    setFormData(prev => ({ ...prev, [name]: value }));
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: '' }));
    }
  };

  const validateForm = () => {
    const newErrors = {};
    fields.forEach(field => {
      if (!formData[field.name] && formData[field.name] !== 0) {
        newErrors[field.name] = 'This field is required';
      }
    });
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (validateForm()) {
      onSubmit(formData);
    }
  };

  const renderField = (field) => {
    switch (field.type) {
      case 'select':
        return (
          <select
            value={formData[field.name] || ''}
            onChange={(e) => handleChange(field.name, e.target.value)}
            className={errors[field.name] ? 'error' : ''}
          >
            <option value="">Select an option</option>
            {field.options.map((option, idx) => (
              <option key={idx} value={option}>{option}</option>
            ))}
          </select>
        );
      case 'number':
        return (
          <input
            type="number"
            step="any"
            value={formData[field.name] || ''}
            onChange={(e) => handleChange(field.name, parseFloat(e.target.value) || '')}
            className={errors[field.name] ? 'error' : ''}
            placeholder={`Enter ${field.label}`}
          />
        );
      default:
        return (
          <input
            type="text"
            value={formData[field.name] || ''}
            onChange={(e) => handleChange(field.name, e.target.value)}
            className={errors[field.name] ? 'error' : ''}
            placeholder={`Enter ${field.label}`}
          />
        );
    }
  };

  return (
    <motion.form 
      className="prediction-form"
      onSubmit={handleSubmit}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <h3>Patient Information</h3>
      <div className="form-grid">
        {fields.map((field, index) => (
          <motion.div 
            key={field.name}
            className="form-field"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
          >
            <label htmlFor={field.name}>
              {field.label}
              <span className="required">*</span>
            </label>
            {renderField(field)}
            {errors[field.name] && (
              <span className="field-error">{errors[field.name]}</span>
            )}
          </motion.div>
        ))}
      </div>

      <div className="form-actions">
        <button 
          type="submit" 
          className="submit-btn"
          disabled={isLoading}
        >
          {isLoading ? (
            <>
              <span className="spinner"></span>
              Predicting...
            </>
          ) : (
            <>
              Predict Disease
              <FaArrowRight className="btn-icon" />
            </>
          )}
        </button>
      </div>
    </motion.form>
  );
};

export default PredictionForm;