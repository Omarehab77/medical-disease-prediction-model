// src/api.js
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000'; // change to your backend URL

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

export default api;