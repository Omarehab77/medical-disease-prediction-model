import React, { useState, useRef, useEffect } from 'react';
import './Chatbot.css';
import { FaPaperPlane, FaMicrophone, FaRobot, FaUser, FaStop, FaRedo } from 'react-icons/fa';
import { motion, AnimatePresence } from 'framer-motion';
import api from '../api';
import { toast } from 'react-toastify';
import { ThreeDots } from 'react-loader-spinner';

console.log("API URL:", process.env.REACT_APP_API_URL);

const Chatbot = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      text: "Hello! I'm OneHealth AI. 👋\n\nI can help you with:\n• Symptom analysis\n• Disease prediction\n• Health recommendations\n• Medical information\n\nFeel free to describe your symptoms or ask any health-related questions!",
      timestamp: new Date().toLocaleTimeString(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [patientId, setPatientId] = useState(() => {
    return localStorage.getItem('patientId') || `patient_${Date.now()}`;
  });
  const [sessionId, setSessionId] = useState(() => {
    return localStorage.getItem('sessionId') || null;
  });
  const [isConnected, setIsConnected] = useState(true);
  const [conversationState, setConversationState] = useState('initial');
  const [predictionComplete, setPredictionComplete] = useState(false);
  const [riskLevel, setRiskLevel] = useState('LOW');
  const [symptoms, setSymptoms] = useState([]);

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const [recognition, setRecognition] = useState(null);

  // Check backend connection on mount
  useEffect(() => {
    const checkConnection = async () => {
      try {
        await api.get('/health');
        setIsConnected(true);
        console.log('✅ Connected to backend');
      } catch (error) {
        setIsConnected(false);
        console.error('❌ Failed to connect to backend:', error);
        toast.error('Cannot connect to the server. Please make sure the backend is running on port 8000.');
      }
    };
    checkConnection();
  }, []);

  useEffect(() => {
    localStorage.setItem('patientId', patientId);
  }, [patientId]);

  useEffect(() => {
    if (sessionId) localStorage.setItem('sessionId', sessionId);
  }, [sessionId]);

  useEffect(() => {
    scrollToBottom();
    // Initialize speech recognition
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      const recognitionInstance = new SpeechRecognition();
      recognitionInstance.lang = 'en-US';
      recognitionInstance.continuous = false;
      recognitionInstance.interimResults = true;

      recognitionInstance.onresult = (event) => {
        let finalTranscript = '';
        let interimTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) finalTranscript += transcript;
          else interimTranscript += transcript;
        }
        const transcript = finalTranscript || interimTranscript;
        if (transcript) setInput(transcript);
      };

      recognitionInstance.onend = () => {
        setIsRecording(false);
        setTimeout(() => {
          if (input.trim()) handleSendMessage();
        }, 500);
      };

      recognitionInstance.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setIsRecording(false);
        if (event.error !== 'no-speech') toast.error('Voice recognition error. Please try again.');
      };

      setRecognition(recognitionInstance);
    }

    return () => {
      if (recognition) recognition.abort();
    };
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;
    if (!isConnected) {
      toast.error('Cannot send message: Not connected to server');
      return;
    }

    const userMessage = {
      id: Date.now(),
      type: 'user',
      text: input,
      timestamp: new Date().toLocaleTimeString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const messageToSend = input;
    setInput('');
    setIsLoading(true);

    try {
      const response = await api.post('/api/chat', {
        message: messageToSend,
        patient_id: patientId,
        session_id: sessionId,
      });

      if (response.status === 200 && response.data) {
        if (response.data.session_id) setSessionId(response.data.session_id);
        if (response.data.state) setConversationState(response.data.state);
        if (response.data.prediction_complete) setPredictionComplete(true);
        if (response.data.risk_level) setRiskLevel(response.data.risk_level);
        if (response.data.symptoms) setSymptoms(response.data.symptoms);

        const botMessage = {
          id: Date.now() + 1,
          type: 'bot',
          text: response.data.response || "I received your message but couldn't generate a proper response.",
          timestamp: new Date().toLocaleTimeString(),
          risk_level: response.data.risk_level || 'LOW',
          symptoms: response.data.symptoms || [],
          confidence: response.data.confidence || {},
          diseases: response.data.diseases || {},
          predictions: response.data.predictions || [],
          is_emergency: response.data.is_emergency || false,
          state: response.data.state || 'initial',
          prediction_complete: response.data.prediction_complete || false,
        };

        setMessages((prev) => [...prev, botMessage]);

        if (response.data.is_emergency) toast.error('🚨 EMERGENCY: Please seek immediate medical attention!');
        else if (response.data.risk_level === 'HIGH') toast.warning('⚠️ High Risk: Please consult a doctor within 24 hours');
        else if (response.data.symptoms?.length > 0 && response.data.state !== 'result') {
          toast.info(`Detected: ${response.data.symptoms.join(', ')}`);
        }
        if (response.data.prediction_complete) toast.success('✅ Analysis complete! Check the prediction results.');
      } else {
        throw new Error('Invalid response from server');
      }
    } catch (error) {
      console.error('Error sending message:', error);
      let errorMessage = "I'm having trouble connecting to the server. ";
      if (error.code === 'ECONNABORTED') errorMessage += 'The request timed out. Please try again.';
      else if (error.response) errorMessage += `Server error: ${error.response.status}`;
      else if (error.request) errorMessage += 'Please check if the backend is running on port 8000.';
      else errorMessage += 'Please try again later.';

      const errorBotMessage = {
        id: Date.now() + 1,
        type: 'bot',
        text: errorMessage,
        timestamp: new Date().toLocaleTimeString(),
      };
      setMessages((prev) => [...prev, errorBotMessage]);
      toast.error('Failed to connect to the server');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const toggleVoiceRecognition = () => {
    if (!recognition) {
      toast.warning('Voice recognition is not supported in your browser');
      return;
    }
    if (isRecording) {
      recognition.stop();
      setIsRecording(false);
    } else {
      try {
        recognition.start();
        setIsRecording(true);
        toast.info('Listening... Speak now');
      } catch (error) {
        console.error('Error starting voice recognition:', error);
        toast.error('Failed to start voice recognition');
      }
    }
  };

  // NEW: Handle image upload with preview
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Text file handling (legacy)
    if (file.type.startsWith('text/') || file.name.match(/\.(txt|csv|json)$/)) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target.result;
        let preview = content.substring(0, 100);
        if (content.length > 100) preview += '...';
        setInput(`I've uploaded a file: ${file.name}\nContent preview: ${preview}`);
      };
      reader.readAsText(file);
      event.target.value = '';
      return;
    }

    // Image upload – show preview in user message, then send to API
    if (!isConnected) {
      toast.error('Not connected to server');
      event.target.value = '';
      return;
    }

    // Read image as data URL for preview
    const reader = new FileReader();
    reader.onload = (e) => {
      const imageDataUrl = e.target.result;

      // Create user message with image preview
      const userMsg = {
        id: Date.now(),
        type: 'user',
        text: `📤 Uploaded image: ${file.name}`,
        image: imageDataUrl,  // store the image data for rendering
        timestamp: new Date().toLocaleTimeString(),
      };
      setMessages((prev) => [...prev, userMsg]);

      // Now send to backend
      sendImageToBackend(file);
    };
    reader.readAsDataURL(file);
    event.target.value = '';
  };

  // Separate function to send image to backend
  const sendImageToBackend = async (file) => {
    const loadingMsgId = Date.now() + 1;
    const loadingMsg = {
      id: loadingMsgId,
      type: 'bot',
      text: '🧠 Analyzing brain MRI image...',
      timestamp: new Date().toLocaleTimeString(),
    };
    setMessages((prev) => [...prev, loadingMsg]);
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await api.post('/api/brain/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 30000,
      });

      if (response.status === 200 && response.data.success) {
        const { prediction, confidence, risk_level, probabilities } = response.data;
        const botMsg = {
          id: Date.now() + 2,
          type: 'bot',
          text: `**Brain MRI Analysis Complete**\n\n` +
                `**Prediction:** ${prediction}\n` +
                `**Confidence:** ${(confidence * 100).toFixed(1)}%\n` +
                `**Risk Level:** ${risk_level}`,
          timestamp: new Date().toLocaleTimeString(),
          is_emergency: risk_level === 'HIGH',
          risk_level: risk_level,
          predictions: [{ disease: prediction, confidence: confidence }],
          prediction_complete: true,
          probabilities: probabilities,
        };
        setMessages((prev) => prev.map((msg) => (msg.id === loadingMsgId ? botMsg : msg)));
        if (risk_level === 'HIGH') toast.warning('⚠️ Tumor detected. Please consult a specialist.');
        else toast.success('✅ No tumor detected.');
      } else {
        throw new Error('Invalid response');
      }
    } catch (error) {
      console.error('Brain prediction error:', error);
      const errorMsg = {
        id: Date.now() + 2,
        type: 'bot',
        text: `❌ Failed to analyze image: ${error.message || 'Unknown error'}`,
        timestamp: new Date().toLocaleTimeString(),
      };
      setMessages((prev) => prev.map((msg) => (msg.id === loadingMsgId ? errorMsg : msg)));
      toast.error('Brain analysis failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewChat = () => {
    setSessionId(null);
    localStorage.removeItem('sessionId');
    setMessages([messages[0]]);
    setConversationState('initial');
    setPredictionComplete(false);
    setRiskLevel('LOW');
    setSymptoms([]);
    toast.info('Started a new conversation');
  };

  const formatMessageText = (text) => {
    const lines = text.split('\n');
    return lines.map((line, i) => {
      let formattedLine = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
      formattedLine = formattedLine.replace(/•/g, '• ');
      return (
        <React.Fragment key={i}>
          <span dangerouslySetInnerHTML={{ __html: formattedLine }} />
          <br />
        </React.Fragment>
      );
    });
  };

  const renderPredictionCard = (message) => {
    if (!message.predictions || message.predictions.length === 0) {
      if (!message.diseases || typeof message.diseases !== 'object' || Object.keys(message.diseases).length === 0)
        return null;
      const entries = Object.entries(message.diseases)
        .filter(([key, value]) => typeof key === 'string' && typeof value === 'number' && value >= 0.1);
      if (entries.length === 0) return null;
      return (
        <div className="prediction-card">
          <h4>🔬 Possible Conditions</h4>
          {entries.map(([condition, score]) => {
            let displayName = condition.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
            return (
              <div key={condition} className="condition-item">
                <span>{displayName}</span>
                <div className="condition-bar">
                  <div className="condition-fill" style={{ width: `${Math.min(Math.max(score * 100, 0), 100)}%` }} />
                  <span>{(score * 100).toFixed(0)}%</span>
                </div>
              </div>
            );
          })}
          {message.risk_level === 'EMERGENCY' && <div className="emergency-badge">🚨 EMERGENCY</div>}
          {message.risk_level === 'HIGH' && <div className="high-risk-badge">⚠️ High Risk</div>}
          {message.prediction_complete && <div className="prediction-complete-badge">✅ Analysis Complete</div>}
        </div>
      );
    }

    return (
      <div className="prediction-card">
        <h4>🔬 Prediction Results</h4>
        {message.predictions.map((pred, index) => {
          let displayName = pred.disease.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
          return (
            <div key={index} className="condition-item">
              <span>{displayName}</span>
              <div className="condition-bar">
                <div
                  className="condition-fill"
                  style={{
                    width: `${Math.min(Math.max((pred.confidence || 0) * 100, 0), 100)}%`,
                    background:
                      pred.risk_level === 'EMERGENCY' ? '#ff4444' : pred.risk_level === 'HIGH' ? '#ff9800' : '#4CAF50',
                  }}
                />
                <span>{((pred.confidence || 0) * 100).toFixed(0)}%</span>
              </div>
            </div>
          );
        })}
        {message.risk_level === 'EMERGENCY' && <div className="emergency-badge">🚨 EMERGENCY - Call 911</div>}
        {message.risk_level === 'HIGH' && <div className="high-risk-badge">⚠️ High Risk - Consult Doctor</div>}
        {message.prediction_complete && <div className="prediction-complete-badge">✅ Analysis Complete</div>}
      </div>
    );
  };

  const ConnectionStatus = () => (
    <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
      <span className="status-dot"></span>
      {isConnected ? 'Connected' : 'Disconnected'}
    </div>
  );

  const StateIndicator = () => {
    const stateLabels = {
      initial: '🟢 Initial',
      collecting: '🟡 Collecting',
      follow_up: '🟠 Follow-up',
      ready_for_prediction: '🔵 Ready',
      predicting: '🔄 Predicting',
      result: '✅ Result',
      emergency: '🔴 Emergency',
      disease_info: '📚 Information',
    };
    return <div className="state-indicator">{stateLabels[conversationState] || '🟢 Active'}</div>;
  };

  // Status bar
  const StatusBar = () => (
    <div className="status-bar">
      <div className="status-item">
        <span className="status-label">Risk Level:</span>
        <span className={`status-value risk-${riskLevel.toLowerCase()}`}>{riskLevel}</span>
      </div>
      <div className="status-item">
        <span className="status-label">Symptoms:</span>
        <span className="status-value">{symptoms.length > 0 ? symptoms.join(', ') : 'None'}</span>
      </div>
      <div className="status-item">
        <span className="status-label">Session:</span>
        <span className="status-value">{sessionId ? sessionId.slice(0, 10) : 'New'}</span>
      </div>
      <button className="status-new-chat-btn" onClick={handleNewChat} title="Start new session">
        <FaRedo /> New
      </button>
    </div>
  );

  return (
    <div className="chatbot-container fade-in">
      <div className="chat-header">
        <div className="chat-header-info">
          <img src="/images/Logo.jpeg" alt="OneHealth AI" className="chat-logo" />
          <div>
            <h2>OneHealth AI</h2>
            <p>Patient: {patientId.slice(0, 10)}...</p>
          </div>
        </div>
        <div className="header-actions">
          <StateIndicator />
          <ConnectionStatus />
        </div>
      </div>

      <StatusBar />

      <div className="chat-messages">
        <AnimatePresence>
          {messages.map((message) => (
            <motion.div
              key={message.id}
              className={`message ${message.type}`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              <div className="message-avatar">{message.type === 'bot' ? <FaRobot /> : <FaUser />}</div>
              <div className="message-content">
                <div className="message-text">
                  {formatMessageText(message.text)}
                  {/* Display image if present */}
                  {message.image && (
                    <div className="message-image">
                      <img src={message.image} alt="Uploaded" />
                    </div>
                  )}
                </div>
                {message.type === 'bot' && renderPredictionCard(message)}
                <div className="message-time">{message.timestamp}</div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {isLoading && (
          <div className="message bot">
            <div className="message-avatar">
              <FaRobot />
            </div>
            <div className="message-content">
              <div className="typing-indicator">
                <ThreeDots color="#667eea" height={10} width={40} />
                <span style={{ marginLeft: '10px' }}>Analyzing your symptoms...</span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-container">
        <div className="input-actions">
          <button
            className={`action-btn ${isRecording ? 'recording' : ''}`}
            onClick={toggleVoiceRecognition}
            title={isRecording ? 'Stop recording' : 'Start voice input'}
          >
            {isRecording ? <FaStop /> : <FaMicrophone />}
          </button>
          <button className="action-btn" onClick={() => fileInputRef.current?.click()} title="Upload file">
            📎
          </button>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            style={{ display: 'none' }}
            accept="image/*,.txt,.csv,.json"
          />
        </div>

        <div className="input-wrapper">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={isRecording ? 'Listening...' : 'Describe your symptoms or ask a question...'}
            rows={1}
            className="chat-input"
            disabled={!isConnected}
          />
          <button
            className="send-btn"
            onClick={handleSendMessage}
            disabled={!input.trim() || isLoading || !isConnected}
          >
            <FaPaperPlane />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;