// api.js
import axios from "axios";

const api = axios.create({
  baseURL: "https://omar7987-one-health.hf.space",
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
});

export default api;