import React, { useState, useEffect } from 'react';
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
    const [apiStatus, setApiStatus] = useState('checking');
    const [features, setFeatures] = useState([]);

    // API URL - Use localhost for development
    const API_URL = 'https://ml-model-12.onrender.com/';

    // Check API status on component mount
    useEffect(() => {
        checkApiStatus();
    }, []);

    const checkApiStatus = async () => {
        try {
            const response = await fetch(`${API_URL}/health`);
            const data = await response.json();

            if (data.status === 'healthy') {
                setApiStatus('connected');

                // Get feature information
                const featuresResponse = await fetch(`${API_URL}/features`);
                const featuresData = await featuresResponse.json();
                if (featuresData.features) {
                    setFeatures(featuresData.features);
                }
            } else {
                setApiStatus('error');
            }
        } catch (err) {
            setApiStatus('error');
            console.error('API connection error:', err);
        }
    };

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
            // Prepare data for API
            const requestData = {
                ...formData,
                satisfaction_score: parseInt(formData.satisfaction_score),
                tenure_months: parseInt(formData.tenure_months),
                visits_last_year: parseInt(formData.visits_last_year),
                total_billing: parseFloat(formData.total_billing),
                missed_appointments: parseInt(formData.missed_appointments),
                diseases: formData.diseases || '' // Ensure diseases is string
            };

            console.log('Sending request:', requestData);

            const response = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData),
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || `HTTP ${response.status}: Prediction failed`);
            }

            console.log('Received response:', result);
            setPrediction(result);
        } catch (err) {
            setError(err.message || 'Error predicting churn. Please try again.');
            console.error('Prediction error:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleRetrain = async () => {
        try {
            const response = await fetch(`${API_URL}/reload-model`, {
                method: 'POST',
            });
            const result = await response.json();
            alert(result.message || 'Model reloaded successfully');
            checkApiStatus(); // Refresh status
        } catch (err) {
            alert('Error reloading model: ' + err.message);
        }
    };

    return (
        <div className="container">
            <h1>🏥 Patient Churn Predictor</h1>
            <p className="subtitle">Predict which patients are likely to churn from your healthcare service</p>

            {/* API Status Indicator */}
            <div className={`api-status-indicator ${apiStatus}`}>
                <span className="status-dot"></span>
                <span className="status-text">
                    {apiStatus === 'connected' ? '✅ API Connected' :
                        apiStatus === 'checking' ? '🔄 Checking API...' :
                            '❌ API Disconnected'}
                </span>
                {apiStatus === 'connected' && features.length > 0 && (
                    <span className="features-info">({features.length} features loaded)</span>
                )}
            </div>

            {/* Debug Info (visible in development) */}
            {process.env.NODE_ENV === 'development' && features.length > 0 && (
                <div className="debug-info">
                    <details>
                        <summary>Model Features (Debug)</summary>
                        <div className="features-list">
                            {features.map((feature, index) => (
                                <div key={index} className="feature-item">
                                    {index + 1}. {feature}
                                </div>
                            ))}
                        </div>
                    </details>
                </div>
            )}

            <div className="form-container">
                <h2>Single Patient Prediction</h2>
                <form onSubmit={handleSubmit} className="form">
                    <div className="form-grid">
                        <div className="form-group">
                            <label>Patient ID:</label>
                            <input
                                type="text"
                                name="patient_id"
                                value={formData.patient_id}
                                onChange={handleChange}
                                required
                                placeholder="Enter patient ID"
                                disabled={apiStatus !== 'connected' || loading}
                            />
                        </div>

                        <div className="form-group">
                            <label>Gender:</label>
                            <select
                                name="gender"
                                value={formData.gender}
                                onChange={handleChange}
                                disabled={apiStatus !== 'connected' || loading}
                            >
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
                                max="120"
                                disabled={apiStatus !== 'connected' || loading}
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
                                disabled={apiStatus !== 'connected' || loading}
                            />
                        </div>

                        <div className="form-group">
                            <label>Chronic Diseases:</label>
                            <input
                                type="text"
                                name="diseases"
                                value={formData.diseases}
                                onChange={handleChange}
                                placeholder="e.g., Diabetes, Hypertension"
                                disabled={apiStatus !== 'connected' || loading}
                            />
                            <small className="hint">Leave empty if no chronic diseases</small>
                        </div>

                        <div className="form-group">
                            <label>Insurance Type:</label>
                            <select
                                name="insurance_type"
                                value={formData.insurance_type}
                                onChange={handleChange}
                                disabled={apiStatus !== 'connected' || loading}
                            >
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
                                    disabled={apiStatus !== 'connected' || loading}
                                />
                                <span className="slider-value">{formData.satisfaction_score}</span>
                            </div>
                        </div>

                        <div className="form-group">
                            <label>Total Billing ($):</label>
                            <input
                                type="number"
                                name="total_billing"
                                value={formData.total_billing}
                                onChange={handleChange}
                                min="0"
                                step="100"
                                disabled={apiStatus !== 'connected' || loading}
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
                                disabled={apiStatus !== 'connected' || loading}
                            />
                        </div>
                    </div>

                    <button
                        type="submit"
                        disabled={loading || apiStatus !== 'connected'}
                        className="submit-btn"
                    >
                        {loading ? '🔍 Predicting...' : '📊 Predict Churn'}
                    </button>
                </form>
            </div>

            {error && (
                <div className="error-message">
                    <h3>❌ Error</h3>
                    <p>{error}</p>
                    {apiStatus === 'error' && (
                        <p className="api-help">
                            Make sure the ML API is running. Open terminal in ML1 folder and run:
                            <code>python api.py</code>
                        </p>
                    )}
                </div>
            )}

            {prediction && (
                <div className="result-card">
                    <h2>🎯 Prediction Result</h2>

                    <div className="patient-info">
                        <h3>Patient: {prediction.patient_id}</h3>
                        <div className="model-info">
                            Model: {prediction.model_info?.type || 'Random Forest'} |
                            Features: {prediction.model_info?.features_used || 'Unknown'}
                        </div>
                    </div>

                    <div className={`prediction-main ${prediction.churn === 'Yes' ? 'churn-yes' : 'churn-no'}`}>
                        <div className="churn-status">
                            <span className="status-label">Will Churn:</span>
                            <span className="status-value">{prediction.churn}</span>
                        </div>

                        <div className="probability-display">
                            <div className="probability-label">
                                Churn Probability: {(prediction.churn_probability * 100).toFixed(1)}%
                            </div>
                            <div className="probability-bar">
                                <div
                                    className="probability-fill"
                                    style={{
                                        width: `${prediction.churn_probability * 100}%`,
                                        backgroundColor: prediction.risk_color
                                    }}
                                ></div>
                            </div>
                            <div className="probability-numbers">
                                <span>0%</span>
                                <span>50%</span>
                                <span>100%</span>
                            </div>
                        </div>

                        <div className="risk-indicator">
                            <span className="risk-label">Risk Level:</span>
                            <span
                                className="risk-badge"
                                style={{ backgroundColor: prediction.risk_color }}
                            >
                                {prediction.risk_level}
                            </span>
                        </div>
                    </div>

                    <div className="recommendation-section">
                        <h4>📋 Recommendation:</h4>
                        <p className="recommendation-text">{prediction.recommendation}</p>

                        {prediction.action_items && prediction.action_items.length > 0 && (
                            <div className="action-items">
                                <h5>Action Items:</h5>
                                <ul>
                                    {prediction.action_items.map((item, index) => (
                                        <li key={index}>{item}</li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>

                    <div className="confidence-section">
                        <div className="confidence-item">
                            <span className="confidence-label">Confidence:</span>
                            <span className="confidence-value">
                                {(prediction.confidence * 100).toFixed(1)}%
                            </span>
                        </div>
                        <div className="confidence-item">
                            <span className="confidence-label">No Churn Probability:</span>
                            <span className="confidence-value">
                                {(prediction.no_churn_probability * 100).toFixed(1)}%
                            </span>
                        </div>
                    </div>

                    {prediction.key_factors && (
                        <div className="key-factors">
                            <h5>Key Factors Influencing Prediction:</h5>
                            <ul>
                                {prediction.key_factors.map((factor, index) => (
                                    <li key={index}>{factor}</li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            )}

            <div className="api-info">
                <p>API Endpoint: <code>{API_URL}</code></p>
                <button
                    onClick={checkApiStatus}
                    className="refresh-btn"
                    disabled={loading}
                >
                    🔄 Refresh Status
                </button>
            </div>
        </div>
    );
};

export default Home;
