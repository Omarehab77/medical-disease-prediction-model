// src/App.js
import React, { useState, useEffect } from 'react';
import './App.css';
import Chatbot from './components/Chatbot';
import DiseasePrediction from './components/DiseasePrediction';
import LungSymptomPredictor from './components/LungSymptomPredictor';
import LandingPage from './components/LandingPage';
import { Routes, Route, useLocation } from 'react-router-dom';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import Sidebar from './components/Sidebar';
import { AnimatePresence } from 'framer-motion';

function App() {
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  const location = useLocation();

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Show sidebar only when not on landing page
  const showSidebar = location.pathname !== '/';

  return (
    <div className="app">
      {showSidebar && <Sidebar isMobile={isMobile} />}
      <div className={`main-content ${!showSidebar ? 'full-width' : ''}`}>
        <AnimatePresence mode="wait">
          <Routes location={location} key={location.pathname}>
            <Route path="/" element={<LandingPage />} />
            <Route path="/chat" element={<Chatbot />} />
            <Route path="/diabetes" element={<DiseasePrediction disease="diabetes" />} />
            <Route path="/heart" element={<DiseasePrediction disease="heart" />} />
            <Route path="/breast-cancer" element={<DiseasePrediction disease="breast" />} />
            <Route path="/lung-symptom" element={<LungSymptomPredictor />} />
          </Routes>
        </AnimatePresence>
      </div>
      <ToastContainer position="bottom-right" autoClose={3000} />
    </div>
  );
}

export default App;