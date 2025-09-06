import axios from 'axios';

// Create API client with base URL pointing to our proxy
export const apiClient = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add any auth headers if needed in the future
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle common errors
    if (error.response?.status === 404) {
      console.warn('API endpoint not found:', error.config?.url);
    } else if (error.response?.status >= 500) {
      console.error('Server error:', error.response?.data);
    }
    return Promise.reject(error);
  }
);