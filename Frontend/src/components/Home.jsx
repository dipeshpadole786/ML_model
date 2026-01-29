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
  const [batchFile, setBatchFile] = useState(null);
  const [batchResults, setBatchResults] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);

  // API URL - Change this to your backend URL
  const API_URL = 'http://localhost:5000'; // For development
  // const API_URL = 'https://your-api-domain.com'; // For production

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleFileChange = (e) => {
    setBatchFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setPrediction(null);
    setBatchResults(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      const result = await response.json();
      
      if (!response.ok) {
        throw new Error(result.error || 'Prediction failed');
      }

      setPrediction(result);
    } catch (err) {
      setError(err.message || 'Error predicting churn. Please try again.');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleBatchSubmit = async (e) => {
    e.preventDefault();
    if (!batchFile) {
      setError('Please select a CSV file');
      return;
    }

    setBatchLoading(true);
    setError('');
    setPrediction(null);
    setBatchResults(null);

    try {
      const formData = new FormData();
      formData.append('file', batchFile);

      const response = await fetch(`${API_URL}/predict-batch`, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      
      if (!response.ok) {
        throw new Error(result.error || 'Batch prediction failed');
      }

      setBatchResults(result);
    } catch (err) {
      setError(err.message || 'Error processing batch prediction.');
      console.error('Batch prediction error:', err);
    } finally {
      setBatchLoading(false);
    }
  };

  const handleRetrain = async () => {
    try {
      const response = await fetch(`${API_URL}/reload-model`, {
        method: 'POST',
      });
      const result = await response.json();
      alert(result.message || 'Model reloaded successfully');
    } catch (err) {
      alert('Error reloading model: ' + err.message);
    }
  };

  // Generate sample CSV
  const downloadSampleCSV = () => {
    const sampleData = `patient_id,gender,tenure_months,visits_last_year,diseases,insurance_type,satisfaction_score,total_billing,missed_appointments
1001,Male,24,8,Diabetes,Private,8,18500,2
1002,Female,6,3,Hypertension,Government,3,32500,7
1003,Male,36,12,None,None,7,12500,1
1004,Female,18,5,Asthma,Private,6,21500,4
1005,Male,48,15,Diabetes;Hypertension,Government,4,42500,8`;

    const blob = new Blob([sampleData], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sample_patients.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="container">
      <h1>🏥 Patient Churn Predictor</h1>
      <p className="subtitle">Predict which patients are likely to churn from your healthcare service</p>
      
      <div className="tabs">
        <div className="tab-content">
          {/* Single Prediction Tab */}
          <div className="tab-pane active">
            <h2>Single Patient Prediction</h2>
            <form onSubmit={handleSubmit} className="form">
              <div className="form-row">
                <div className="form-group">
                  <label>Patient ID:</label>
                  <input
                    type="text"
                    name="patient_id"
                    value={formData.patient_id}
                    onChange={handleChange}
                    required
                    placeholder="Enter patient ID"
                  />
                </div>

                <div className="form-group">
                  <label>Gender:</label>
                  <select name="gender" value={formData.gender} onChange={handleChange}>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                  </select>
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label>Tenure (months):</label>
                  <input
                    type="number"
                    name="tenure_months"
                    value={formData.tenure_months}
                    onChange={handleChange}
                    min="0"
                    max="120"
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
                    max="50"
                  />
                </div>
              </div>

              <div className="form-group">
                <label>Chronic Diseases:</label>
                <input
                  type="text"
                  name="diseases"
                  value={formData.diseases}
                  onChange={handleChange}
                  placeholder="e.g., Diabetes, Hypertension (or leave empty for none)"
                />
                <small className="hint">Leave empty if no chronic diseases</small>
              </div>

              <div className="form-row">
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
                  <div className="slider-container">
                    <input
                      type="range"
                      name="satisfaction_score"
                      value={formData.satisfaction_score}
                      onChange={handleChange}
                      min="1"
                      max="10"
                      className="slider"
                    />
                    <span className="slider-value">{formData.satisfaction_score}</span>
                  </div>
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label>Total Billing ($):</label>
                  <input
                    type="number"
                    name="total_billing"
                    value={formData.total_billing}
                    onChange={handleChange}
                    min="0"
                    step="100"
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
                    max="20"
                  />
                </div>
              </div>

              <button type="submit" disabled={loading} className="submit-btn">
                {loading ? '🔍 Predicting...' : '📊 Predict Churn'}
              </button>
            </form>
          </div>
          
          {/* Batch Prediction Tab */}
          <div className="tab-pane">
            <h2>Batch Prediction from CSV</h2>
            <div className="batch-section">
              <div className="batch-info">
                <p>Upload a CSV file with patient data to get predictions for multiple patients.</p>
                <button 
                  type="button" 
                  className="sample-btn"
                  onClick={downloadSampleCSV}
                >
                  📥 Download Sample CSV
                </button>
              </div>
              
              <form onSubmit={handleBatchSubmit} className="batch-form">
                <div className="form-group">
                  <label>Select CSV File:</label>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileChange}
                    required
                  />
                  {batchFile && (
                    <p className="file-info">Selected: {batchFile.name}</p>
                  )}
                </div>
                
                <button 
                  type="submit" 
                  disabled={batchLoading || !batchFile}
                  className="submit-btn"
                >
                  {batchLoading ? 'Processing...' : '📁 Predict Batch'}
                </button>
              </form>
            </div>
          </div>
        </div>
      </div>

      {error && <div className="error-message">⚠️ {error}</div>}

      {/* Single Prediction Results */}
      {prediction && (
        <div className="result-card">
          <h2>🎯 Prediction Result</h2>
          <div className={`prediction-box ${prediction.churn === 'Yes' ? 'high-risk' : 'low-risk'}`}>
            <div className="prediction-header">
              <h3>Patient: {prediction.patient_id || 'Unknown'}</h3>
              <span className={`risk-badge ${prediction.risk_level?.toLowerCase()}`}>
                {prediction.risk_level || 'Unknown'} Risk
              </span>
            </div>
            
            <div className="prediction-details">
              <div className="prediction-main">
                <div className="churn-status">
                  <span className="status-label">Will Churn:</span>
                  <span className={`status-value ${prediction.churn === 'Yes' ? 'churn-yes' : 'churn-no'}`}>
                    {prediction.churn}
                  </span>
                </div>
                
                <div className="probability-meter">
                  <div className="probability-label">
                    Churn Probability: {(prediction.churn_probability * 100).toFixed(1)}%
                  </div>
                  <div className="meter">
                    <div 
                      className="meter-fill"
                      style={{ width: `${prediction.churn_probability * 100}%` }}
                    ></div>
                  </div>
                </div>
              </div>
              
              <div className="prediction-info">
                <div className="info-item">
                  <span className="info-label">Confidence:</span>
                  <span className="info-value">{(prediction.confidence * 100).toFixed(1)}%</span>
                </div>
                
                <div className="info-item">
                  <span className="info-label">No Churn Probability:</span>
                  <span className="info-value">{(prediction.no_churn_probability * 100).toFixed(1)}%</span>
                </div>
              </div>
              
              <div className="recommendation">
                <h4>📋 Recommendation:</h4>
                <p>{prediction.recommendation}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Batch Prediction Results */}
      {batchResults && (
        <div className="result-card">
          <h2>📊 Batch Prediction Results</h2>
          <div className="batch-summary">
            <div className="summary-stats">
              <div className="stat">
                <span className="stat-label">Total Patients:</span>
                <span className="stat-value">{batchResults.total}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Predicted to Churn:</span>
                <span className="stat-value churn-count">
                  {batchResults.churn_count} ({((batchResults.churn_count / batchResults.total) * 100).toFixed(1)}%)
                </span>
              </div>
            </div>
            
            <div className="predictions-table">
              <h4>Individual Predictions:</h4>
              <div className="table-container">
                <table>
                  <thead>
                    <tr>
                      <th>Patient ID</th>
                      <th>Churn Prediction</th>
                      <th>Probability</th>
                      <th>Risk Level</th>
                    </tr>
                  </thead>
                  <tbody>
                    {batchResults.predictions.map((pred, index) => (
                      <tr key={index}>
                        <td>{pred.patient_id}</td>
                        <td>
                          <span className={`prediction-badge ${pred.churn === 'Yes' ? 'badge-churn' : 'badge-no-churn'}`}>
                            {pred.churn}
                          </span>
                        </td>
                        <td>{(pred.churn_probability * 100).toFixed(1)}%</td>
                        <td>
                          <span className={`risk-badge-small ${pred.risk_level.toLowerCase()}`}>
                            {pred.risk_level}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            
            <button 
              className="download-btn"
              onClick={() => {
                // Convert to CSV and download
                const csvContent = [
                  ['Patient ID', 'Churn Prediction', 'Churn Probability', 'Risk Level'],
                  ...batchResults.predictions.map(p => [
                    p.patient_id, 
                    p.churn, 
                    `${(p.churn_probability * 100).toFixed(2)}%`,
                    p.risk_level
                  ])
                ].map(row => row.join(',')).join('\n');
                
                const blob = new Blob([csvContent], { type: 'text/csv' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'batch_predictions.csv';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
              }}
            >
              📥 Download Results as CSV
            </button>
          </div>
        </div>
      )}

      <div className="api-actions">
        <button onClick={handleRetrain} className="action-btn">
          🔄 Reload ML Model
        </button>
        <div className="api-status">
          <span className="status-indicator">●</span>
          <span>API: {API_URL}</span>
        </div>
      </div>
    </div>
  );
};

export default Home;