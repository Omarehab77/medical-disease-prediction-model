// src/components/Sidebar.jsx
import React, { useState } from 'react';
import './Sidebar.css';
import { 
  FaRobot, 
  FaHeart, 
  FaLungs, 
  FaBolt,
  FaBars,
  FaTimes,
  FaUserMd,
  FaAppleAlt,
  FaHeartbeat,
} from 'react-icons/fa';
import { NavLink } from 'react-router-dom';
import { motion } from 'framer-motion';

const Sidebar = ({ isMobile }) => {
  const [isOpen, setIsOpen] = useState(!isMobile);

  const toggleSidebar = () => {
    setIsOpen(!isOpen);
  };

  const menuItems = [
    { path: '/chat', icon: <FaRobot />, label: 'AI Doctor Chat' },
    { path: '/diabetes', icon: <FaAppleAlt />, label: 'Diabetes' },
    { path: '/heart', icon: <FaHeart />, label: 'Heart Disease' },
    { path: '/breast-cancer', icon: <FaHeartbeat />, label: 'Breast Cancer' },
    { path: '/lung-symptom', icon: <FaLungs />, label: 'Lung Symptom Check' },
  ];

  return (
    <>
      {isMobile && (
        <button className="menu-toggle" onClick={toggleSidebar}>
          {isOpen ? <FaTimes /> : <FaBars />}
        </button>
      )}
      
      <motion.div 
        className={`sidebar ${isOpen ? 'open' : 'closed'}`}
        initial={{ x: -250 }}
        animate={{ x: isOpen ? 0 : -250 }}
        transition={{ duration: 0.3 }}
      >
        <div className="sidebar-header">
          <img src="/images/Logo.jpeg" alt="OneHealth AI" className="sidebar-logo" />
          <h2>OneHealth AI</h2>
          <p className="subtitle">Smart Health Assistant</p>
        </div>

        <div className="sidebar-status">
          <div className="status-dot"></div>
          <span>AI Active</span>
        </div>

        <nav className="sidebar-nav">
          {menuItems.map((item, index) => (
            <NavLink
              key={index}
              to={item.path}
              className={({ isActive }) => 
                `nav-link ${isActive ? 'active' : ''}`
              }
              onClick={() => isMobile && setIsOpen(false)}
            >
              <span className="icon">{item.icon}</span>
              <span className="label">{item.label}</span>
              {item.path === '/chat' && <span className="badge">Live</span>}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="version">v4.0.0</div>
          <div className="powered">Powered by AI & ML</div>
        </div>
      </motion.div>
    </>
  );
};

export default Sidebar;