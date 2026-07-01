// src/components/LandingPage.jsx
import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import './LandingPage.css';

const logo = '/images/Logo.jpeg';

const quotes = [
  "Health is not about the weight you lose, but about the life you gain.",
  "The greatest wealth is health.",
  "Your health is an investment, not an expense.",
  "Wellness is the complete integration of body, mind, and spirit.",
];

const images = [
  'https://images.unsplash.com/photo-1579684385127-1ef15d508118?w=400&h=300&fit=crop',
  'https://images.unsplash.com/photo-1582750433449-648ed127bb54?w=400&h=300&fit=crop',
  'https://images.unsplash.com/photo-1530023367847-a683933f4172?w=400&h=300&fit=crop',
];

const LandingPage = () => {
  const navigate = useNavigate();
  const containerRef = useRef(null);
  const [quoteIndex, setQuoteIndex] = useState(0);

  // Auto‑rotate quotes
  useEffect(() => {
    const timer = setInterval(() => {
      setQuoteIndex((prev) => (prev + 1) % quotes.length);
    }, 4000);
    return () => clearInterval(timer);
  }, []);

  // Mouse parallax for shapes
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!containerRef.current) return;
      const { clientX, clientY } = e;
      const x = (clientX / window.innerWidth - 0.5) * 30;
      const y = (clientY / window.innerHeight - 0.5) * 30;
      containerRef.current.style.setProperty('--mouse-x', `${x}px`);
      containerRef.current.style.setProperty('--mouse-y', `${y}px`);
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  // Particles
  const particles = Array.from({ length: 50 }, (_, i) => ({
    id: i,
    size: Math.random() * 4 + 2,
    x: Math.random() * 100,
    y: Math.random() * 100,
    duration: Math.random() * 20 + 12,
    delay: Math.random() * 10,
  }));

  return (
    <div className="landing-page" ref={containerRef}>
      {/* Animated grid background */}
      <div className="grid-bg" />

      {/* Gradient orbs */}
      <div className="bg-orbs">
        <div className="orb orb1" />
        <div className="orb orb2" />
        <div className="orb orb3" />
        <div className="orb orb4" />
      </div>

      {/* Floating particles */}
      <div className="particles">
        {particles.map((p) => (
          <div
            key={p.id}
            className="particle"
            style={{
              width: p.size,
              height: p.size,
              left: `${p.x}%`,
              top: `${p.y}%`,
              animationDuration: `${p.duration}s`,
              animationDelay: `${p.delay}s`,
            }}
          />
        ))}
      </div>

      {/* Geometric shapes with parallax */}
      <div className="shapes-layer" style={{ transform: 'translate(var(--mouse-x), var(--mouse-y))' }}>
        <div className="shape shape-triangle" />
        <div className="shape shape-hexagon" />
        <div className="shape shape-star" />
        <div className="shape shape-ring" />
        <div className="shape shape-diamond" />
      </div>

      {/* Main content – fit in viewport */}
      <div className="landing-content">
        <motion.img
          src={logo}
          alt="OneHealth AI"
          className="landing-logo"
          initial={{ scale: 0.8, opacity: 0, rotate: -10 }}
          animate={{ scale: 1, opacity: 1, rotate: 0 }}
          transition={{ duration: 0.8, delay: 0.2, type: 'spring' }}
        />

        <motion.h1
          className="landing-title"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          OneHealth AI
        </motion.h1>

        <motion.div
          className="subtitle-wrapper"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
        >
          <span className="subtitle-prefix">Your</span>
          <span className="subtitle-typewriter">intelligent health companion</span>
        </motion.div>

        {/* Quote carousel – only one at a time */}
        <div className="quote-carousel">
          <AnimatePresence mode="wait">
            <motion.p
              key={quoteIndex}
              className="quote"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.6 }}
            >
              “{quotes[quoteIndex]}”
            </motion.p>
          </AnimatePresence>
        </div>

        {/* Images – smaller, horizontal */}
        <div className="landing-images">
          {images.map((src, i) => (
            <motion.img
              key={i}
              src={src}
              alt="Health"
              className="landing-img"
              whileHover={{ scale: 1.08, rotate: 2, y: -4 }}
              transition={{ type: 'spring', stiffness: 300 }}
              initial={{ y: 30, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 1.2 + i * 0.1 }}
            />
          ))}
        </div>

        {/* Button – always visible */}
        <motion.button
          className="landing-btn"
          whileHover={{ scale: 1.05, boxShadow: '0 0 40px rgba(108, 99, 255, 0.6)' }}
          whileTap={{ scale: 0.95 }}
          onClick={() => navigate('/chat')}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.6 }}
        >
          <span className="btn-text">Get Started</span>
          <span className="btn-ripple" />
        </motion.button>

        <div className="landing-footer">
          <p>© 2026 OneHealth AI</p>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;