import React, { useState } from 'react';
import './Home.css';

const Home = () => {
  const [formData, setFormData] = useState({
    patient_id: '',
    gender: 'Male',
    tenure_months: 12,
    visits_last_year: 0,
    diseases: '',
    insurance_type: 'Government',
    satisfaction_score: 5,
    total_billing: 1000,
    missed_appointments: 0,
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setPrediction(null);

    try {
      // Replace with your actual ML model API endpoint
      const response = await fetch('YOUR_MODEL_API_ENDPOINT', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      const result = await response.json();
      setPrediction(result);
    } catch (err) {
      setError('Error predicting churn. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Insurance Churn Predictor</h1>
      
      <form onSubmit={handleSubmit} className="form">
        <div className="form-group">
          <label>Patient ID:</label>
          <input
            type="text"
            name="patient_id"
            value={formData.patient_id}
            onChange={handleChange}
            required
          />
        </div>

        <div className="form-group">
          <label>Gender:</label>
          <select name="gender" value={formData.gender} onChange={handleChange}>
            <option value="Male">Male</option>
            <option value="Female">Female</option>
          </select>
        </div>

        <div className="form-group">
          <label>Tenure (months):</label>
          <input
            type="number"
            name="tenure_months"
            value={formData.tenure_months}
            onChange={handleChange}
            min="0"
          />
        </div>

        <div className="form-group">
          <label>Visits Last Year:</label>
          <input
            type="number"
            name="visits_last_year"
            value={formData.visits_last_year}
            onChange={handleChange}
            min="0"
          />
        </div>

        <div className="form-group">
          <label>Diseases (comma separated):</label>
          <input
            type="text"
            name="diseases"
            value={formData.diseases}
            onChange={handleChange}
            placeholder="e.g., Diabetes, Hypertension"
          />
        </div>

        <div className="form-group">
          <label>Insurance Type:</label>
          <select name="insurance_type" value={formData.insurance_type} onChange={handleChange}>
            <option value="Government">Government</option>
            <option value="Private">Private</option>
            <option value="None">None</option>
          </select>
        </div>

        <div className="form-group">
          <label>Satisfaction Score (1-10):</label>
          <input
            type="range"
            name="satisfaction_score"
            value={formData.satisfaction_score}
            onChange={handleChange}
            min="1"
            max="10"
          />
          <span>{formData.satisfaction_score}</span>
        </div>

        <div className="form-group">
          <label>Total Billing ($):</label>
          <input
            type="number"
            name="total_billing"
            value={formData.total_billing}
            onChange={handleChange}
            min="0"
          />
        </div>

        <div className="form-group">
          <label>Missed Appointments:</label>
          <input
            type="number"
            name="missed_appointments"
            value={formData.missed_appointments}
            onChange={handleChange}
            min="0"
          />
        </div>

        <button type="submit" disabled={loading} className="submit-btn">
          {loading ? 'Predicting...' : 'Predict Churn'}
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      {prediction && (
        <div className="result">
          <h2>Prediction Result</h2>
          <div className={`prediction ${prediction.churn === 'Yes' ? 'churn-yes' : 'churn-no'}`}>
            <h3>Will Churn: {prediction.churn}</h3>
            {prediction.probability && (
              <p>Probability: {(prediction.probability * 100).toFixed(2)}%</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default Home;