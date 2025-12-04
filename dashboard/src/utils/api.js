import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response.data;
  },
  (error) => {
    console.error('API Response Error:', error);
    
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;
      throw new Error(`API Error ${status}: ${data.message || 'Unknown error'}`);
    } else if (error.request) {
      // Request made but no response received
      throw new Error('Network error: Unable to reach the server');
    } else {
      // Something else happened
      throw new Error(`Request error: ${error.message}`);
    }
  }
);

// Fleet data endpoints
export const fetchFleetData = async () => {
  try {
    return await api.get('/fleet/overview');
  } catch (error) {
    console.warn('Fleet API unavailable, using mock data');
    throw error;
  }
};

export const fetchVehicleData = async (vehicleId) => {
  try {
    return await api.get(`/vehicle/${vehicleId}/data`);
  } catch (error) {
    console.warn('Vehicle data API unavailable, using mock data');
    throw error;
  }
};

export const fetchVehiclePrediction = async (vehicleId) => {
  try {
    return await api.get(`/vehicle/${vehicleId}/prediction`);
  } catch (error) {
    console.warn('Prediction API unavailable, using mock data');
    throw error;
  }
};

export const fetchWhatIfAnalysis = async (vehicleId, scenarios) => {
  try {
    return await api.post(`/vehicle/${vehicleId}/what-if`, { scenarios });
  } catch (error) {
    console.warn('What-if API unavailable, using mock data');
    throw error;
  }
};

// Health check endpoint
export const checkApiHealth = async () => {
  try {
    return await api.get('/health');
  } catch (error) {
    return { status: 'unavailable', error: error.message };
  }
};

export default api;