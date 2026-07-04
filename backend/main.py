"""
COMPLETE MEDICAL AI ASSISTANT - ALL BLOCKS (FIXED VERSION)
===========================================================
"""
# ============================================================================
# BLOCK 1: Production Medical AI Assistant - Complete Setup Layer
# ============================================================================

# Fix Windows encoding
import sys
import io
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (AttributeError, ValueError, OSError):
        pass

# Imports
import os
import re
import json
import warnings
import logging
import hashlib
import random
import time
import platform
import gc
from pathlib import Path
from typing import (
    List, Dict, Optional, Any, Union,
    Tuple, Set, Callable, TypeVar, Generic,
    OrderedDict
)
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, Counter, deque, OrderedDict
from functools import lru_cache, wraps

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings

try:
    import uvicorn
    UVICORN_AVAILABLE = True
except ImportError:
    UVICORN_AVAILABLE = False

try:
    import nest_asyncio
    NEST_ASYNCIO_AVAILABLE = True
except ImportError:
    NEST_ASYNCIO_AVAILABLE = False

warnings.filterwarnings('ignore')
print("All imports loaded successfully!")

# ----------------------------------------------------------------------------
# Safe logging with emoji removal for Windows
# ----------------------------------------------------------------------------
def safe_emoji_remove(text: str) -> str:
    if not text or sys.platform != 'win32':
        return text
    try:
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F700-\U0001F77F"
            "\U0001F780-\U0001F7FF"
            "\U0001F800-\U0001F8FF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FA6F"
            "\U0001FA70-\U0001FAFF"
            "\u2600-\u26FF"
            "\u2700-\u27BF"
            "]",
            flags=re.UNICODE
        )
        return emoji_pattern.sub('', text)
    except:
        return text

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[41m',
        'RESET': '\033[0m'
    }
    def format(self, record):
        log_message = super().format(record)
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        return f"{color}{log_message}{self.COLORS['RESET']}"

def setup_logging(log_dir: Path = Path("logs"), log_level: str = "INFO", console_output: bool = True) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("medical_ai")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()

    class SafeLogFilter(logging.Filter):
        def filter(self, record):
            if sys.platform == 'win32':
                record.msg = safe_emoji_remove(record.msg)
            return True

    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_dir / "medical_ai.log", encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(SafeLogFilter())
    logger.addHandler(file_handler)

    error_handler = logging.FileHandler(log_dir / "errors.log", encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    error_handler.addFilter(SafeLogFilter())
    logger.addHandler(error_handler)

    warning_handler = logging.FileHandler(log_dir / "warnings.log", encoding='utf-8')
    warning_handler.setLevel(logging.WARNING)
    warning_handler.setFormatter(file_formatter)
    warning_handler.addFilter(SafeLogFilter())
    logger.addHandler(warning_handler)

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(SafeLogFilter())
        logger.addHandler(console_handler)

    return logger

logger = setup_logging()
logger.info("=" * 80)
logger.info("Medical AI Assistant v4.0 - Initializing")
logger.info("=" * 80)
logger.info(f"   Python Version: {platform.python_version()}")
logger.info(f"   Working Directory: {Path.cwd()}")
logger.info(f"   Operating System: {platform.system()} {platform.release()}")
logger.info(f"   CPU Cores: {os.cpu_count()}")
logger.info(f"   Start Time: {datetime.now().isoformat()}")
logger.info("=" * 80)

# ----------------------------------------------------------------------------
# Random seed
# ----------------------------------------------------------------------------
def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    logger.info(f"Random seed set to: {seed}")

set_random_seed(42)

# ----------------------------------------------------------------------------
# Settings class with extra_fields support
# ----------------------------------------------------------------------------
class Settings(BaseSettings):
    APP_NAME: str = "Medical AI Assistant"
    APP_DESCRIPTION: str = "AI-powered medical triage assistant with ML-based disease prediction"
    VERSION: str = "4.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_PREFIX: str = "/api"

    MODELS_DIR: Path = Path("models")
    DATABASE_DIR: Path = Path("database")
    LOG_DIR: Path = Path("logs")
    CACHE_DIR: Path = Path("cache")
    TEMP_DIR: Path = Path("temp")

    ALLOWED_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    SUPPORTED_DISEASES: List[str] = ["diabetes", "heart", "breast"]

    MODEL_NAMES: Dict[str, str] = {
        "diabetes": "diabetes_model",
        "heart": "heart_model",
        "breast": "breast_model"
    }
    PREPROCESSOR_NAMES: Dict[str, str] = {
        "diabetes": "diabetes_preprocessor",
        "heart": "heart_preprocessor",
        "breast": "breast_preprocessor"
    }

    DATABASE_FILES: List[str] = [
        "Healthcare.csv",
        "disease_description.csv",
        "disease_precaution.csv",
        "symptom_severity.csv",
        "heart.csv",
        "diabetes.csv",
        "breast-cancer.csv"
    ]

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    DEVICE: str = "cpu"
    MAX_SENTENCE_LENGTH: int = 128

    SYMPTOM_MATCH_THRESHOLD: int = 85
    SIMILARITY_THRESHOLD: float = 0.7
    CONFIDENCE_THRESHOLD: float = 0.3
    EMERGENCY_THRESHOLD: float = 0.7
    MIN_SIMILARITY: float = 0.3
    MAX_SIMILARITY: float = 1.0

    MAX_FUZZY_CANDIDATES: int = 10
    MAX_RETRIEVED_SYMPTOMS: int = 10
    MAX_RETRIEVED_DISEASES: int = 5
    MAX_DIAGNOSIS_CANDIDATES: int = 5
    MAX_SEARCH_RESULTS: int = 5

    MAX_HISTORY: int = 100
    MAX_SESSIONS: int = 1000
    MAX_QUESTIONS_PER_SYMPTOM: int = 5
    MAX_FOLLOWUP_ROUNDS: int = 10
    MAX_RETRIES: int = 3
    SESSION_TIMEOUT_MINUTES: int = 30

    CACHE_TTL_SECONDS: int = 3600
    MAX_CACHE_SIZE: int = 1000

    MAX_MEMORY_MB: int = 2048

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"

    def __init__(self, **kwargs):
        valid_keys = set(self.__class__.__annotations__.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        super().__init__(**filtered_kwargs)

    def create_directories(self) -> None:
        for directory in [self.MODELS_DIR, self.DATABASE_DIR, self.LOG_DIR,
                          self.CACHE_DIR, self.TEMP_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory ready: {directory}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "APP_NAME": self.APP_NAME,
            "VERSION": self.VERSION,
            "DEBUG": self.DEBUG,
            "HOST": self.HOST,
            "PORT": self.PORT,
            "MODELS_DIR": str(self.MODELS_DIR),
            "DATABASE_DIR": str(self.DATABASE_DIR),
            "SUPPORTED_DISEASES": self.SUPPORTED_DISEASES,
            "EMBEDDING_MODEL": self.EMBEDDING_MODEL,
            "DEVICE": self.DEVICE,
        }

settings = Settings()
settings.create_directories()
logger.info("Settings loaded successfully")
logger.debug(f"   Settings: {settings.to_dict()}")

# ----------------------------------------------------------------------------
# Disease configuration
# ----------------------------------------------------------------------------
DISEASE_CONFIG: Dict[str, Dict[str, Any]] = {
    "heart": {
        "display_name": "Heart Disease",
        "aliases": [
            "heart", "heart disease", "cardiac disease", "cardiac problem",
            "heart issue", "cardiovascular disease", "coronary artery disease",
            "heart condition", "heart problem", "cardiac condition"
        ],
        "model_path": settings.MODELS_DIR / "heart_model.pkl",
        "preprocessor_path": settings.MODELS_DIR / "heart_preprocessor.pkl",
        "csv_path": settings.DATABASE_DIR / "heart.csv",
        "target_column": "target",
        "prediction_type": "binary",
        "supported_features": [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ],
        "risk_factors": [
            "high blood pressure", "high cholesterol", "smoking",
            "diabetes", "obesity", "family history", "age", "stress"
        ],
        "recommended_specialist": "Cardiologist",
        "recommended_tests": [
            "ECG", "Stress Test", "Echocardiogram", "Cardiac Catheterization"
        ]
    },
    "diabetes": {
        "display_name": "Diabetes",
        "aliases": [
            "diabetes", "blood sugar", "high sugar", "diabetic",
            "type 2 diabetes", "type 2", "sugar disease", "hyperglycemia",
            "insulin resistance", "high blood sugar", "sugar problem"
        ],
        "model_path": settings.MODELS_DIR / "diabetes_model.pkl",
        "preprocessor_path": settings.MODELS_DIR / "diabetes_preprocessor.pkl",
        "csv_path": settings.DATABASE_DIR / "diabetes.csv",
        "target_column": "Outcome",
        "prediction_type": "binary",
        "supported_features": [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ],
        "risk_factors": [
            "family history", "obesity", "physical inactivity", "age",
            "gestational diabetes", "high blood pressure", "high cholesterol"
        ],
        "recommended_specialist": "Endocrinologist",
        "recommended_tests": [
            "Fasting Blood Sugar", "HbA1c", "Oral Glucose Tolerance Test"
        ]
    },
    "breast": {
        "display_name": "Breast Cancer",
        "aliases": [
            "breast", "breast cancer", "tumor", "breast tumor",
            "breast mass", "breast lump", "breast carcinoma",
            "breast malignancy", "breast neoplasm"
        ],
        "model_path": settings.MODELS_DIR / "breast_model.pkl",
        "preprocessor_path": settings.MODELS_DIR / "breast_preprocessor.pkl",
        "csv_path": settings.DATABASE_DIR / "breast-cancer.csv",
        "target_column": "diagnosis",
        "prediction_type": "binary",
        "supported_features": [
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
            "smoothness_mean", "compactness_mean", "concavity_mean",
            "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
            "radius_se", "texture_se", "perimeter_se", "area_se",
            "smoothness_se", "compactness_se", "concavity_se",
            "concave_points_se", "symmetry_se", "fractal_dimension_se",
            "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
            "smoothness_worst", "compactness_worst", "concavity_worst",
            "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
        ],
        "risk_factors": [
            "family history", "age", "genetic mutations (BRCA1/2)",
            "personal history", "dense breast tissue", "hormone exposure"
        ],
        "recommended_specialist": "Oncologist",
        "recommended_tests": [
            "Mammogram", "Ultrasound", "MRI", "Biopsy"
        ]
    }
}
logger.info(f"Disease configuration loaded: {len(DISEASE_CONFIG)} diseases")

# ----------------------------------------------------------------------------
# Database configuration
# ----------------------------------------------------------------------------
DATABASE_CONFIG: Dict[str, Dict[str, Any]] = {
    "healthcare": {
        "file_name": "Healthcare.csv",
        "path": settings.DATABASE_DIR / "Healthcare.csv",
        "description": "Patient symptom-disease mappings",
        "required": True
    },
    "disease_description": {
        "file_name": "disease_description.csv",
        "path": settings.DATABASE_DIR / "disease_description.csv",
        "description": "Disease descriptions and information",
        "required": True
    },
    "disease_precaution": {
        "file_name": "disease_precaution.csv",
        "path": settings.DATABASE_DIR / "disease_precaution.csv",
        "description": "Disease precautions and recommendations",
        "required": True
    },
    "symptom_severity": {
        "file_name": "symptom_severity.csv",
        "path": settings.DATABASE_DIR / "symptom_severity.csv",
        "description": "Symptom severity scores",
        "required": True
    },
    "heart": {
        "file_name": "heart.csv",
        "path": settings.DATABASE_DIR / "heart.csv",
        "description": "Heart disease training data",
        "required": True
    },
    "diabetes": {
        "file_name": "diabetes.csv",
        "path": settings.DATABASE_DIR / "diabetes.csv",
        "description": "Diabetes training data",
        "required": True
    },
    "breast_cancer": {
        "file_name": "breast-cancer.csv",
        "path": settings.DATABASE_DIR / "breast-cancer.csv",
        "description": "Breast cancer training data",
        "required": True
    }
}
logger.info(f"Database configuration loaded: {len(DATABASE_CONFIG)} files")

# ----------------------------------------------------------------------------
# Global paths
# ----------------------------------------------------------------------------
MODEL_PATHS: Dict[str, Path] = {
    disease: DISEASE_CONFIG[disease]["model_path"]
    for disease in settings.SUPPORTED_DISEASES
}
PREPROCESSOR_PATHS: Dict[str, Path] = {
    disease: DISEASE_CONFIG[disease]["preprocessor_path"]
    for disease in settings.SUPPORTED_DISEASES
}
CSV_PATHS: Dict[str, Path] = {
    config["file_name"]: config["path"]
    for config in DATABASE_CONFIG.values()
}
TRAINING_SUMMARY_PATH: Path = settings.MODELS_DIR / "training_summary.json"
logger.info("Global paths configured")
logger.debug(f"   Model paths: {len(MODEL_PATHS)}")
logger.debug(f"   Preprocessor paths: {len(PREPROCESSOR_PATHS)}")
logger.debug(f"   CSV paths: {len(CSV_PATHS)}")

# ----------------------------------------------------------------------------
# Global containers
# ----------------------------------------------------------------------------
MODELS: Dict[str, Any] = {}
PREPROCESSORS: Dict[str, Any] = {}
SCALERS: Dict[str, StandardScaler] = {}  # new
PREPROCESSOR_FEATURES: Dict[str, List[str]] = {}  # store feature names per disease
DATASETS: Dict[str, pd.DataFrame] = {}
EMBEDDINGS: Dict[str, Any] = {}
CHAT_SESSIONS: Dict[str, Any] = {}
DISEASE_DESCRIPTIONS: Dict[str, str] = {}
DISEASE_PRECAUTIONS: Dict[str, List[str]] = {}
SYMPTOM_SEVERITY: Dict[str, int] = {}
SYMPTOM_DISEASE_MAP: Dict[str, List[str]] = {}
HEALTHCARE_DATA: Optional[pd.DataFrame] = None
SYMPTOM_INDEX: Optional[Dict] = None
QUESTION_ENGINE: Optional[Any] = None
logger.info("Global containers initialized (empty)")

# ----------------------------------------------------------------------------
# Global constants
# ----------------------------------------------------------------------------
class Constants:
    CONVERSATION_STATES = {
        "GREETING": "greeting",
        "SYMPTOM_COLLECTION": "symptom_collection",
        "QUESTIONING": "questioning",
        "ANALYZING": "analyzing",
        "DIAGNOSING": "diagnosing",
        "EMERGENCY": "emergency",
        "COMPLETE": "complete",
        "EXIT": "exit"
    }
    GREETING_KEYWORDS = [
        "hi", "hello", "hey", "greetings", "good morning",
        "good afternoon", "good evening", "howdy", "hey there",
        "hi there", "hello there", "good day"
    ]
    EXIT_KEYWORDS = [
        "bye", "goodbye", "exit", "quit", "stop", "end",
        "see you", "later", "good night", "farewell"
    ]
    CONFIRMATION_WORDS = [
        "yes", "yeah", "yep", "sure", "okay", "ok",
        "correct", "right", "true", "absolutely", "definitely",
        "of course", "certainly", "indeed"
    ]
    NEGATIVE_WORDS = [
        "no", "not", "never", "none", "nope", "nah",
        "no way", "not at all", "negative", "false", "incorrect"
    ]
    THANK_YOU_WORDS = [
        "thank", "thanks", "thank you", "thanks a lot",
        "appreciate", "grateful", "thanks for your help"
    ]
    HELP_KEYWORDS = [
        "help", "support", "assist", "guide", "explain",
        "what should I do", "how do I", "can you help"
    ]
    EMERGENCY_KEYWORDS = [
        "heart attack", "stroke", "call 911", "emergency",
        "severe bleeding", "unconscious", "can't breathe",
        "chest pain", "difficulty breathing", "severe pain",
        "severe headache", "numbness", "speech difficulty"
    ]
    UNKNOWN_RESPONSE = "I'm not sure what you're describing. Could you please rephrase?"
    ERROR_RESPONSE = "I apologize, but I encountered an error. Please try again."
    EMERGENCY_RESPONSE = "Please call 911 immediately! This is a medical emergency."
    EXIT_RESPONSE = "Take care! Remember to consult a healthcare professional for an accurate diagnosis."
    GREETING_RESPONSE = "Hello! I'm your AI Medical Assistant. How can I help you today?"
    HELP_RESPONSE = "I can help you understand your symptoms and identify potential health conditions. Please describe what you're feeling."
    THANK_YOU_RESPONSE = "You're welcome! Is there anything else I can help you with?"
    FALLBACK_RESPONSE = "I'm not sure how to respond. Could you please rephrase or ask something else?"
    RISK_LABELS = {
        "LOW": "Low Risk",
        "MEDIUM": "Medium Risk",
        "HIGH": "High Risk",
        "EMERGENCY": "Emergency"
    }
    RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "EMERGENCY"]
    DEFAULT_RISK_LEVEL = "LOW"
    SUPPORTED_LANGUAGES = ["en", "ar"]
    DEFAULT_LANGUAGE = "en"
    EMERGENCY_COMBINATIONS = {
        "heart_attack": {
            "symptoms": ["chest pain", "arm pain", "shortness of breath", "sweating", "nausea"],
            "required": ["chest pain"],
            "min_match": 2,
            "severity": 9,
            "message": "URGENT: Possible Heart Attack"
        },
        "stroke": {
            "symptoms": ["confusion", "speech problems", "numbness", "vision changes", "severe headache"],
            "required": ["confusion", "speech problems"],
            "min_match": 2,
            "severity": 9,
            "message": "URGENT: Possible Stroke"
        }
    }
    NON_EMERGENCY_SYMPTOMS = [
        "thirst", "fatigue", "headache", "cough", "runny nose",
        "sore throat", "back pain", "joint pain", "muscle pain"
    ]
    MAX_HISTORY = 100
    MAX_RESULTS = 5
    MAX_QUESTIONS_PER_SYMPTOM = 5
    MAX_FOLLOWUP_ROUNDS = 10
    MAX_DIAGNOSIS_CANDIDATES = 5
    MAX_RETRIEVED_SYMPTOMS = 10
    MAX_RETRIEVED_DISEASES = 5
    SIMILARITY_THRESHOLD = 0.7
    CONFIDENCE_THRESHOLD = 0.3
    EMERGENCY_THRESHOLD = 0.7
    SYMPTOM_MATCH_THRESHOLD = 85
    MIN_CONFIDENCE = 0.1
    MAX_CONFIDENCE = 1.0
    CACHE_TTL_SECONDS = 3600
    MAX_CACHE_SIZE = 1000

# ----------------------------------------------------------------------------
# NLP configuration
# ----------------------------------------------------------------------------
NLP_CONFIG: Dict[str, Any] = {
    "embedding_model": settings.EMBEDDING_MODEL,
    "embedding_dimension": settings.EMBEDDING_DIMENSION,
    "device": settings.DEVICE,
    "max_sentence_length": settings.MAX_SENTENCE_LENGTH,
    "similarity_threshold": settings.SIMILARITY_THRESHOLD,
    "min_similarity": settings.MIN_SIMILARITY,
    "max_similarity": settings.MAX_SIMILARITY,
    "fuzzy_match_threshold": settings.SYMPTOM_MATCH_THRESHOLD,
    "max_fuzzy_candidates": settings.MAX_FUZZY_CANDIDATES,
    "max_retrieved_symptoms": settings.MAX_RETRIEVED_SYMPTOMS,
    "max_retrieved_diseases": settings.MAX_RETRIEVED_DISEASES,
    "max_search_results": settings.MAX_SEARCH_RESULTS,
    "use_fuzzy_matching": True,
    "use_embedding_similarity": True,
    "use_synonym_matching": True,
    "use_stemming": True,
    "use_lemmatization": True,
}
logger.info(f"NLP configuration loaded: {len(NLP_CONFIG)} parameters")

# ----------------------------------------------------------------------------
# Session configuration
# ----------------------------------------------------------------------------
SESSION_CONFIG: Dict[str, Any] = {
    "session_timeout_minutes": settings.SESSION_TIMEOUT_MINUTES,
    "max_sessions": settings.MAX_SESSIONS,
    "max_history": settings.MAX_HISTORY,
    "max_questions": settings.MAX_QUESTIONS_PER_SYMPTOM,
    "max_followup_questions": settings.MAX_FOLLOWUP_ROUNDS,
    "max_retries": settings.MAX_RETRIES,
    "conversation_expiration_hours": 24,
    "session_expiration_minutes": 30,
    "persist_sessions": False,
    "session_storage_dir": settings.CACHE_DIR / "sessions",
}
logger.info(f"Session configuration loaded: {len(SESSION_CONFIG)} parameters")

# ----------------------------------------------------------------------------
# FastAPI configuration
# ----------------------------------------------------------------------------
FASTAPI_CONFIG: Dict[str, Any] = {
    "title": settings.APP_NAME,
    "description": settings.APP_DESCRIPTION,
    "version": settings.VERSION,
    "debug": settings.DEBUG,
    "host": settings.HOST,
    "port": settings.PORT,
    "api_prefix": settings.API_PREFIX,
    "allowed_origins": settings.ALLOWED_ORIGINS,
    "allow_credentials": settings.CORS_ALLOW_CREDENTIALS,
    "allow_methods": settings.CORS_ALLOW_METHODS,
    "allow_headers": settings.CORS_ALLOW_HEADERS,
    "docs_url": "/docs",
    "redoc_url": "/redoc",
    "openapi_url": "/openapi.json",
}
logger.info(f"FastAPI configuration loaded: {len(FASTAPI_CONFIG)} parameters")

# ----------------------------------------------------------------------------
# Project validation
# ----------------------------------------------------------------------------
def validate_project() -> Dict[str, Any]:
    results = {
        "valid": True,
        "timestamp": datetime.now().isoformat(),
        "database": {"files": [], "missing": [], "invalid": []},
        "models": {"files": [], "missing": [], "invalid": []},
        "warnings": [],
        "errors": [],
        "summary": {
            "total_required": 0,
            "total_found": 0,
            "total_missing": 0
        }
    }
    for config in DATABASE_CONFIG.values():
        file_path = config["path"]
        file_name = config["file_name"]
        if file_path.exists():
            results["database"]["files"].append(file_name)
            try:
                pd.read_csv(file_path, nrows=1)
            except Exception as e:
                results["database"]["invalid"].append(file_name)
                results["warnings"].append(f"Invalid CSV file: {file_name} - {str(e)}")
        else:
            results["database"]["missing"].append(file_name)
            if config["required"]:
                results["warnings"].append(f"Required database file missing: {file_name}")
    required_model_files = []
    for disease in settings.SUPPORTED_DISEASES:
        required_model_files.append(f"{settings.MODEL_NAMES[disease]}.pkl")
        required_model_files.append(f"{settings.PREPROCESSOR_NAMES[disease]}.pkl")
    required_model_files.append("training_summary.json")
    for file_name in required_model_files:
        file_path = settings.MODELS_DIR / file_name
        if file_path.exists():
            results["models"]["files"].append(file_name)
            if file_name.endswith('.pkl'):
                try:
                    joblib.load(file_path)
                except Exception as e:
                    results["models"]["invalid"].append(file_name)
                    results["warnings"].append(f"Invalid model file: {file_name} - {str(e)}")
        else:
            results["models"]["missing"].append(file_name)
            results["warnings"].append(f"Required model file missing: {file_name}")
    total_required = len(DATABASE_CONFIG) + len(required_model_files)
    total_found = len(results["database"]["files"]) + len(results["models"]["files"])
    total_missing = len(results["database"]["missing"]) + len(results["models"]["missing"])
    results["summary"] = {
        "total_required": total_required,
        "total_found": total_found,
        "total_missing": total_missing
    }
    if results["errors"] or results["database"]["missing"] or results["models"]["missing"]:
        results["valid"] = False
    if results["valid"]:
        logger.info("Project validation PASSED")
    else:
        logger.warning("Project validation has WARNINGS")
    logger.info(f"   Database: {len(results['database']['files'])} found, {len(results['database']['missing'])} missing")
    logger.info(f"   Models: {len(results['models']['files'])} found, {len(results['models']['missing'])} missing")
    logger.info(f"   Summary: {results['summary']['total_found']}/{results['summary']['total_required']} files found")
    if results["warnings"]:
        logger.warning(f"   Warnings: {len(results['warnings'])}")
        for warning in results["warnings"][:5]:
            logger.warning(f"      - {warning}")
        if len(results["warnings"]) > 5:
            logger.warning(f"      ... and {len(results['warnings']) - 5} more")
    return results

validation_results = validate_project()

# ----------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------
def get_model_path(disease: str) -> Optional[Path]:
    if disease in MODEL_PATHS:
        return MODEL_PATHS[disease]
    return None

def get_preprocessor_path(disease: str) -> Optional[Path]:
    if disease in PREPROCESSOR_PATHS:
        return PREPROCESSOR_PATHS[disease]
    return None

def get_dataset_path(dataset_name: str) -> Optional[Path]:
    if dataset_name in DATABASE_CONFIG:
        return DATABASE_CONFIG[dataset_name]["path"]
    return None

def validate_disease(disease: str) -> bool:
    return disease in DISEASE_CONFIG

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = normalize_text(text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def safe_read_json(file_path: Path, default: Any = None) -> Any:
    try:
        if not file_path.exists():
            return default
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading JSON {file_path}: {e}")
        return default

def safe_write_json(data: Any, file_path: Path, indent: int = 2) -> bool:
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, default=str)
        return True
    except Exception as e:
        logger.error(f"Error writing JSON {file_path}: {e}")
        return False

def safe_load_csv(file_path: Path, **kwargs) -> Optional[pd.DataFrame]:
    try:
        if not file_path.exists():
            return None
        return pd.read_csv(file_path, **kwargs)
    except Exception as e:
        logger.error(f"Error reading CSV {file_path}: {e}")
        return None

def safe_load_model(file_path: Path) -> Optional[Any]:
    try:
        if not file_path.exists():
            return None
        return joblib.load(file_path)
    except Exception as e:
        logger.error(f"Error loading model {file_path}: {e}")
        return None

def is_valid_disease(disease: str) -> bool:
    if not disease:
        return False
    disease = disease.strip().lower()
    invalid_terms = ['unknown', 'other', 'none', 'null', 'nan', '']
    if disease in invalid_terms:
        return False
    if len(disease) < 2:
        return False
    return True

def format_percentage(value: float) -> float:
    return max(0.0, min(100.0, round(value, 1)))

def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]

def get_disease_by_alias(alias: str) -> Optional[str]:
    alias = normalize_text(alias)
    for disease, config in DISEASE_CONFIG.items():
        for disease_alias in config["aliases"]:
            if disease_alias in alias or alias in disease_alias:
                return disease
    return None

def check_dependencies() -> Dict[str, bool]:
    return {
        "sentence_transformers": SENTENCE_TRANSFORMERS_AVAILABLE,
        "rapidfuzz": RAPIDFUZZ_AVAILABLE,
        "nltk": NLTK_AVAILABLE,
        "uvicorn": UVICORN_AVAILABLE,
        "nest_asyncio": NEST_ASYNCIO_AVAILABLE,
    }

def get_memory_usage_mb() -> float:
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0

dependencies = check_dependencies()
logger.info(f"Dependencies: {dependencies}")

memory_usage = get_memory_usage_mb()
if memory_usage > 0:
    logger.info(f"Memory Usage: {memory_usage:.1f} MB")

logger.info("=" * 80)
logger.info("BLOCK 1: Project Setup Complete")
logger.info("=" * 80)
logger.info(f"   Base Directory: {Path.cwd()}")
logger.info(f"   Models Directory: {settings.MODELS_DIR}")
logger.info(f"   Database Directory: {settings.DATABASE_DIR}")
logger.info(f"   Log Directory: {settings.LOG_DIR}")
logger.info(f"   Cache Directory: {settings.CACHE_DIR}")
logger.info(f"   Supported Diseases: {settings.SUPPORTED_DISEASES}")
logger.info(f"   Database Files: {len(DATABASE_CONFIG)}")
logger.info(f"   Disease Configs: {len(DISEASE_CONFIG)}")
logger.info("=" * 80)
logger.info("Block 2 will load all data and models")
logger.info("Block 3 will implement the chat engine")
logger.info("Block 4 will create FastAPI endpoints")
logger.info("=" * 80)

# ============================================================================
# BLOCK 2: Complete Production-Grade Initialization - DATA-DRIVEN VERSION
# ============================================================================

import gc
import time
import re
import math
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

logger = logging.getLogger("medical_ai")

# Ensure global variables exist
if 'MODELS' not in globals():
    MODELS = {}
if 'PREPROCESSORS' not in globals():
    PREPROCESSORS = {}
if 'SCALERS' not in globals():
    SCALERS = {}
if 'PREPROCESSOR_FEATURES' not in globals():
    PREPROCESSOR_FEATURES = {}
if 'DATASETS' not in globals():
    DATASETS = {}
if 'DATASET_METADATA' not in globals():
    DATASET_METADATA = {}
if 'EMBEDDINGS' not in globals():
    EMBEDDINGS = {}
if 'CHAT_SESSIONS' not in globals():
    CHAT_SESSIONS = {}
if 'DISEASE_DESCRIPTIONS' not in globals():
    DISEASE_DESCRIPTIONS = {}
if 'DISEASE_PRECAUTIONS' not in globals():
    DISEASE_PRECAUTIONS = {}
if 'SYMPTOM_SEVERITY' not in globals():
    SYMPTOM_SEVERITY = {}
if 'SYMPTOM_DISEASE_MAP' not in globals():
    SYMPTOM_DISEASE_MAP = {}
if 'SYMPTOM_INDEX' not in globals():
    SYMPTOM_INDEX = {}
if 'HEALTHCARE_DATA' not in globals():
    HEALTHCARE_DATA = None
if 'SYMPTOM_SYNONYMS' not in globals():
    SYMPTOM_SYNONYMS = {}
if 'DISEASE_SYMPTOMS' not in globals():
    DISEASE_SYMPTOMS = {}
if 'DISEASE_STATS' not in globals():
    DISEASE_STATS = {}
if 'FEATURE_METADATA' not in globals():
    FEATURE_METADATA = {}
if 'FEATURE_RANGES' not in globals():
    FEATURE_RANGES = {}
if 'FUZZY_CACHE' not in globals():
    FUZZY_CACHE = {}
if 'QUESTION_ENGINE' not in globals():
    QUESTION_ENGINE = None
if 'INITIALIZATION_STATUS' not in globals():
    INITIALIZATION_STATUS = {}

# ===== store feature defaults for missing values =====
FEATURE_DEFAULTS: Dict[str, Dict[str, float]] = {}

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_and_inspect_dataset(file_path: Path, name: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    logger.info(f"Loading dataset: {name}")
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return None, {}
    try:
        df = pd.read_csv(file_path, low_memory=False)
        logger.info(f"   Loaded {df.shape[0]} rows, {df.shape[1]} columns")
        if df.empty:
            logger.error(f"Dataset {name} is empty")
            return None, {}
        metadata = {
            "name": name,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": df.columns.tolist()
        }
        target_aliases = ['target', 'outcome', 'diagnosis', 'class', 'label', 'result']
        target_column = None
        for col in df.columns:
            col_lower = col.lower().strip()
            for alias in target_aliases:
                if alias in col_lower or col_lower in alias:
                    target_column = col
                    break
            if target_column:
                break
        metadata["detected_target"] = target_column
        if target_column:
            metadata["feature_columns"] = [col for col in df.columns if col != target_column]
        else:
            metadata["feature_columns"] = df.columns.tolist()
        logger.info(f"   Target: {metadata['detected_target']}")
        return df, metadata
    except Exception as e:
        logger.error(f"Error loading dataset {name}: {e}")
        return None, {}

# ---------------------------------------------------------------------------
# Build data-driven symptom index
# ---------------------------------------------------------------------------
def build_symptom_index_from_healthcare(healthcare_df: pd.DataFrame) -> Dict[str, Any]:
    index = {
        "symptom_to_diseases": defaultdict(list),
        "disease_to_symptoms": defaultdict(list),
        "disease_counts": Counter(),
        "total_records": 0,
        "all_symptoms": set(),
        "all_diseases": set(),
        "symptom_list": [],
        "disease_list": [],
        "cond_prob": {}
    }
    if healthcare_df is None or healthcare_df.empty:
        logger.warning("Healthcare.csv not loaded; using fallback common diseases.")
        return index
    symptom_col = None
    disease_col = None
    for col in healthcare_df.columns:
        col_lower = col.lower().strip()
        if 'symptom' in col_lower:
            symptom_col = col
        if 'disease' in col_lower:
            disease_col = col
    if symptom_col is None or disease_col is None:
        logger.warning("Could not find symptom/disease columns in Healthcare.csv.")
        return index
    for _, row in healthcare_df.iterrows():
        symptoms_str = str(row[symptom_col]) if pd.notna(row[symptom_col]) else ""
        disease = str(row[disease_col]).strip() if pd.notna(row[disease_col]) else ""
        if not symptoms_str or not disease:
            continue
        symptoms_str = symptoms_str.replace('|', ',').replace(';', ',').replace('_', ' ')
        for raw_symptom in symptoms_str.split(','):
            symptom = raw_symptom.strip().lower()
            if not symptom or len(symptom) < 2:
                continue
            index["all_symptoms"].add(symptom)
            index["all_diseases"].add(disease)
            if symptom not in index["symptom_to_diseases"]:
                index["symptom_to_diseases"][symptom] = []
            if disease not in index["symptom_to_diseases"][symptom]:
                index["symptom_to_diseases"][symptom].append(disease)
            if disease not in index["disease_to_symptoms"]:
                index["disease_to_symptoms"][disease] = []
            if symptom not in index["disease_to_symptoms"][disease]:
                index["disease_to_symptoms"][disease].append(symptom)
            index["disease_counts"][disease] += 1
            index["total_records"] += 1
    index["symptom_list"] = sorted(index["all_symptoms"])
    index["disease_list"] = sorted(index["all_diseases"])
    alpha = 1.0
    num_symptoms = len(index["all_symptoms"])
    index["cond_prob"] = {}
    for disease in index["disease_list"]:
        total = index["disease_counts"][disease]
        if total == 0:
            continue
        index["cond_prob"][disease] = {}
        for symptom in index["symptom_list"]:
            count = sum(1 for d in index["symptom_to_diseases"].get(symptom, []) if d == disease)
            prob = (count + alpha) / (total + alpha * num_symptoms) if num_symptoms > 0 else 0.5
            index["cond_prob"][disease][symptom] = prob
    logger.info(f"Built symptom index: {len(index['symptom_list'])} symptoms, {len(index['disease_list'])} diseases")
    return index

# ---------------------------------------------------------------------------
# Merge common diseases (fallback)
# ---------------------------------------------------------------------------
def add_common_diseases(index: Dict[str, Any]) -> Dict[str, Any]:
    common_diseases = {
        "Influenza (Flu)": {
            "symptoms": ["fever", "cough", "fatigue", "headache", "muscle pain", "chills", "sore throat"],
            "description": "Influenza is a contagious respiratory illness caused by influenza viruses.",
            "precautions": ["get vaccinated", "wash hands frequently", "avoid close contact", "stay home when sick"]
        },
        "Common Cold": {
            "symptoms": ["cough", "runny nose", "sore throat", "fatigue", "fever", "sneezing"],
            "description": "The common cold is a viral infection of your nose and throat.",
            "precautions": ["rest", "drink fluids", "use saline nasal spray", "avoid cold foods"]
        },
        "Pneumonia": {
            "symptoms": ["fever", "cough", "shortness of breath", "chest pain", "fatigue", "phlegm", "chills"],
            "description": "Pneumonia is an infection that inflames the air sacs in one or both lungs.",
            "precautions": ["consult doctor", "medication", "rest", "follow up"]
        },
        "Gastroenteritis": {
            "symptoms": ["nausea", "vomiting", "diarrhea", "abdominal pain", "fever", "dehydration"],
            "description": "Gastroenteritis is inflammation of the digestive tract.",
            "precautions": ["stay hydrated", "rest", "eat bland foods", "avoid dairy and fatty foods"]
        },
        "Urinary Tract Infection": {
            "symptoms": ["burning micturition", "frequent urination", "fever", "abdominal pain", "foul smell of urine"],
            "description": "A urinary tract infection (UTI) is an infection in any part of the urinary system.",
            "precautions": ["drink plenty of water", "cranberry juice", "avoid irritants", "consult doctor"]
        },
        "Migraine": {
            "symptoms": ["headache", "nausea", "visual disturbances", "sensitivity to light", "sensitivity to sound"],
            "description": "A migraine is a headache that can cause severe throbbing pain.",
            "precautions": ["rest in a quiet dark room", "apply cold or warm compresses", "stay hydrated", "avoid triggers"]
        },
        "Tuberculosis": {
            "symptoms": ["cough", "fever", "night sweats", "weight loss", "fatigue", "chest pain", "phlegm", "blood in sputum"],
            "description": "Tuberculosis (TB) is an infectious disease caused by Mycobacterium tuberculosis.",
            "precautions": ["cover mouth when coughing", "consult doctor", "medication", "rest"]
        },
        "COVID-19": {
            "symptoms": ["fever", "cough", "fatigue", "loss of taste", "loss of smell", "shortness of breath", "headache", "muscle pain"],
            "description": "COVID-19 is a disease caused by the SARS-CoV-2 virus.",
            "precautions": ["wear mask", "social distancing", "wash hands", "consult doctor"]
        },
        "Bronchitis": {
            "symptoms": ["cough", "phlegm", "fatigue", "chest discomfort", "shortness of breath", "fever"],
            "description": "Bronchitis is an inflammation of the lining of your bronchial tubes.",
            "precautions": ["rest", "drink fluids", "use a humidifier", "avoid irritants"]
        }
    }
    for disease, data in common_diseases.items():
        if disease not in index["disease_to_symptoms"]:
            index["disease_to_symptoms"][disease] = []
        for symptom in data["symptoms"]:
            symptom = symptom.lower().strip()
            if symptom not in index["symptom_to_diseases"]:
                index["symptom_to_diseases"][symptom] = []
            if disease not in index["symptom_to_diseases"][symptom]:
                index["symptom_to_diseases"][symptom].append(disease)
            if symptom not in index["disease_to_symptoms"][disease]:
                index["disease_to_symptoms"][disease].append(symptom)
            index["all_symptoms"].add(symptom)
        index["all_diseases"].add(disease)
        if disease not in index["disease_counts"]:
            index["disease_counts"][disease] = 1
        else:
            index["disease_counts"][disease] += 1
        if disease not in DISEASE_DESCRIPTIONS:
            DISEASE_DESCRIPTIONS[disease] = data["description"]
        if disease not in DISEASE_PRECAUTIONS:
            DISEASE_PRECAUTIONS[disease] = data["precautions"]
    index["total_records"] += len(common_diseases)
    index["symptom_list"] = sorted(index["all_symptoms"])
    index["disease_list"] = sorted(index["all_diseases"])
    alpha = 1.0
    num_symptoms = len(index["all_symptoms"])
    for disease in index["disease_list"]:
        total = index["disease_counts"][disease]
        if total == 0:
            continue
        index["cond_prob"][disease] = {}
        for symptom in index["symptom_list"]:
            count = sum(1 for d in index["symptom_to_diseases"].get(symptom, []) if d == disease)
            prob = (count + alpha) / (total + alpha * num_symptoms) if num_symptoms > 0 else 0.5
            index["cond_prob"][disease][symptom] = prob
    logger.info(f"   Added {len(common_diseases)} common diseases with symptoms.")
    return index

# ---------------------------------------------------------------------------
# Generate synonyms
# ---------------------------------------------------------------------------
def generate_synonyms(symptom_list: List[str]) -> Dict[str, List[str]]:
    logger.info("Generating symptom synonyms...")
    synonyms = {}
    synonym_map = {
        'chest pain': ['chest discomfort', 'chest tightness', 'pain in chest'],
        'shortness of breath': ['difficulty breathing', 'breathlessness', 'hard to breathe'],
        'fatigue': ['tiredness', 'exhaustion', 'low energy'],
        'headache': ['head pain', 'migraine'],
        'nausea': ['queasiness', 'upset stomach'],
        'dizziness': ['lightheadedness', 'vertigo'],
        'fever': ['high temperature', 'chills'],
        'cough': ['coughing', 'hacking'],
        'sweating': ['perspiration', 'clamminess'],
        'thirst': ['thirsty', 'dry mouth'],
        'weight loss': ['losing weight', 'lost weight'],
        'blurred vision': ['blurry vision', 'vision problems'],
        'numbness': ['tingling', 'loss of sensation'],
        'back pain': ['backache', 'lower back pain'],
        'abdominal pain': ['stomach pain', 'belly pain'],
        'vomiting': ['throwing up', 'puking'],
        'diarrhea': ['loose stools', 'watery stool'],
        'rash': ['skin rash', 'red spots'],
        'muscle pain': ['muscle ache', 'myalgia'],
        'chills': ['shivering', 'cold flashes'],
        'appetite loss': ['loss of appetite', 'not hungry'],
        'malaise': ['general discomfort', 'feeling unwell'],
        'phlegm': ['mucus', 'sputum'],
        'runny nose': ['runny nose', 'nasal congestion'],
        'sore throat': ['throat pain', 'scratchy throat'],
        'night sweats': ['night sweating', 'night perspiration'],
        'loss of taste': ['taste loss', 'no taste'],
        'loss of smell': ['smell loss', 'no smell'],
        'burning micturition': ['burning urination', 'painful urination'],
        'frequent urination': ['peeing often', 'urinary frequency'],
        'visual disturbances': ['vision problems', 'blurry vision'],
        'sensitivity to light': ['light sensitivity', 'photophobia'],
        'sensitivity to sound': ['sound sensitivity', 'phonophobia'],
        'insomnia': ['sleep problems', "can't sleep", 'trouble sleeping'],
        'anxiety': ['anxious', 'nervous', 'worry'],
        'joint pain': ['joint ache', 'arthralgia'],
        'swelling': ['swollen', 'edema'],
        'depression': ['sadness', 'low mood', 'hopelessness']
    }
    for symptom in symptom_list:
        if symptom in synonym_map:
            synonyms[symptom] = synonym_map[symptom]
        else:
            synonyms[symptom] = [symptom]
    logger.info(f"   Synonyms generated for {len(synonyms)} symptoms")
    return synonyms

# ---------------------------------------------------------------------------
# Main initialization
# ---------------------------------------------------------------------------
def initialize_medical_ai() -> Dict[str, Any]:
    global MODELS, PREPROCESSORS, SCALERS, PREPROCESSOR_FEATURES, DATASETS, DATASET_METADATA
    global EMBEDDINGS, HEALTHCARE_DATA
    global DISEASE_DESCRIPTIONS, DISEASE_PRECAUTIONS, SYMPTOM_SEVERITY
    global SYMPTOM_DISEASE_MAP, SYMPTOM_INDEX, SYMPTOM_SYNONYMS
    global DISEASE_SYMPTOMS, DISEASE_STATS, FEATURE_METADATA, FEATURE_RANGES
    global FUZZY_CACHE, QUESTION_ENGINE, INITIALIZATION_STATUS
    global FEATURE_DEFAULTS

    logger.info("=" * 80)
    logger.info("BLOCK 2: Initializing Medical AI Assistant (DATA-DRIVEN VERSION)")
    logger.info("=" * 80)

    start_time = time.time()
    status = {
        "success": True,
        "datasets": {},
        "models": {},
        "knowledge": {},
        "warnings": [],
        "errors": [],
        "timing": {}
    }

    # 1. Load datasets
    logger.info("\nLoading Datasets...")
    t0 = time.time()
    datasets = {}
    dataset_metadata = {}
    for name, config in DATABASE_CONFIG.items():
        df, metadata = load_and_inspect_dataset(config["path"], name)
        if df is not None:
            datasets[name] = df
            dataset_metadata[name] = metadata
            status["datasets"][name] = {"loaded": True, "rows": len(df), "columns": len(df.columns), "target": metadata.get("detected_target")}
        else:
            status["datasets"][name] = {"loaded": False}
            status["errors"].append(f"Failed to load dataset: {name}")
    status["timing"]["datasets"] = time.time() - t0

    DATASETS = datasets
    DATASET_METADATA = dataset_metadata
    HEALTHCARE_DATA = datasets.get("healthcare")

    # 2. Build knowledge base
    logger.info("\nBuilding Knowledge Base...")
    t0 = time.time()

    # Load disease descriptions
    desc_df = datasets.get("disease_description")
    if desc_df is not None:
        disease_col = None
        desc_col = None
        for col in desc_df.columns:
            col_lower = col.lower().strip()
            if 'disease' in col_lower or 'prognosis' in col_lower or 'diagnosis' in col_lower:
                disease_col = col
            if 'description' in col_lower:
                desc_col = col
        if disease_col and desc_col:
            for _, row in desc_df.iterrows():
                disease = str(row[disease_col]).strip() if pd.notna(row[disease_col]) else ""
                description = str(row[desc_col]).strip() if pd.notna(row[desc_col]) else ""
                if disease and description:
                    DISEASE_DESCRIPTIONS[disease] = description

    # Load precautions
    prec_df = datasets.get("disease_precaution")
    if prec_df is not None:
        disease_col = None
        for col in prec_df.columns:
            if 'disease' in col.lower():
                disease_col = col
                break
        if disease_col:
            prec_cols = [col for col in prec_df.columns if 'precaution' in col.lower()]
            for _, row in prec_df.iterrows():
                disease = str(row[disease_col]).strip() if pd.notna(row[disease_col]) else ""
                if disease:
                    prec_list = [str(row[col]).strip() for col in prec_cols if pd.notna(row[col]) and str(row[col]).strip()]
                    if prec_list:
                        DISEASE_PRECAUTIONS[disease] = prec_list

    # Load symptom severity
    sev_df = datasets.get("symptom_severity")
    if sev_df is not None:
        symptom_col = None
        severity_col = None
        for col in sev_df.columns:
            if 'symptom' in col.lower():
                symptom_col = col
            if 'severity' in col.lower() or 'score' in col.lower():
                severity_col = col
        if symptom_col and severity_col:
            for _, row in sev_df.iterrows():
                symptom = str(row[symptom_col]).lower().strip().replace('_', ' ') if pd.notna(row[symptom_col]) else ""
                score = row[severity_col] if pd.notna(row[severity_col]) else 3
                if symptom:
                    try:
                        SYMPTOM_SEVERITY[symptom] = int(score)
                    except:
                        SYMPTOM_SEVERITY[symptom] = 3

    # Build symptom index from Healthcare.csv
    symptom_index = build_symptom_index_from_healthcare(HEALTHCARE_DATA)
    symptom_index = add_common_diseases(symptom_index)

    # Add fallback symptoms
    fallback_symptoms = [
        "chest pain", "shortness of breath", "fatigue", "headache",
        "nausea", "fever", "cough", "sweating", "thirst", "weight loss",
        "blurred vision", "numbness", "dizziness", "joint pain", "back pain",
        "abdominal pain", "vomiting", "diarrhea", "constipation", "rash",
        "muscle pain", "chills", "appetite loss", "malaise", "phlegm",
        "runny nose", "sore throat", "night sweats", "loss of taste", "loss of smell",
        "burning micturition", "frequent urination", "visual disturbances",
        "sensitivity to light", "sensitivity to sound", "chest discomfort",
        "insomnia", "anxiety", "swelling", "depression"
    ]
    for sym in fallback_symptoms:
        if sym not in symptom_index["all_symptoms"]:
            symptom_index["all_symptoms"].add(sym)

    symptom_index["symptom_list"] = sorted(list(symptom_index["all_symptoms"]))
    symptom_index["disease_list"] = sorted(list(symptom_index["all_diseases"]))

    if symptom_index.get("total_records", 0) == 0:
        symptom_index["total_records"] = len(symptom_index["disease_list"])

    SYMPTOM_INDEX = symptom_index
    SYMPTOM_DISEASE_MAP = symptom_index.get("symptom_to_diseases", {})
    DISEASE_SYMPTOMS = symptom_index.get("disease_to_symptoms", {})

    # Generate synonyms
    symptom_list = symptom_index.get("symptom_list", [])
    SYMPTOM_SYNONYMS = generate_synonyms(symptom_list)

    status["symptom_index"] = {
        "symptoms": len(symptom_list),
        "diseases": len(symptom_index.get("disease_list", [])),
        "mappings": len(SYMPTOM_DISEASE_MAP),
        "synonyms": len(SYMPTOM_SYNONYMS)
    }
    status["timing"]["knowledge"] = time.time() - t0

    # 3. Load base models, preprocessors and scalers
    logger.info("\nLoading Models...")
    t0 = time.time()
    models = {}
    preprocessors = {}
    scalers = {}
    preprocessor_features = {}

    for disease in settings.SUPPORTED_DISEASES:
        dataset = datasets.get(disease)
        if dataset is None:
            status["errors"].append(f"Dataset for {disease} not loaded")
            continue
        metadata = dataset_metadata.get(disease, {})
        feature_cols = metadata.get("feature_columns", [])
        if not feature_cols:
            status["errors"].append(f"No feature columns for {disease}")
            continue
        model_path = DISEASE_CONFIG[disease]["model_path"]
        preprocessor_path = DISEASE_CONFIG[disease]["preprocessor_path"]
        scaler_path = settings.MODELS_DIR / f"{disease}_scaler.pkl"
        try:
            if model_path.exists():
                model = joblib.load(model_path)
                models[disease] = model
                logger.info(f"   Model loaded: {disease}")
        except Exception as e:
            logger.error(f"Failed to load model {disease}: {e}")
        try:
            if preprocessor_path.exists():
                preprocessor = joblib.load(preprocessor_path)
                preprocessors[disease] = preprocessor
                # Extract feature names from the preprocessor
                if hasattr(preprocessor, 'feature_names_'):
                    # custom attribute set in training
                    preprocessor_features[disease] = preprocessor.feature_names_
                elif hasattr(preprocessor, 'feature_names_in_'):
                    preprocessor_features[disease] = list(preprocessor.feature_names_in_)
                else:
                    # fallback: try to get from column names in the first transformer
                    try:
                        if hasattr(preprocessor, 'transformers_'):
                            all_cols = []
                            for _, trans, cols in preprocessor.transformers_:
                                if cols is not None:
                                    all_cols.extend(cols)
                            preprocessor_features[disease] = all_cols
                    except:
                        preprocessor_features[disease] = feature_cols
                logger.info(f"   Preprocessor loaded: {disease}")
        except Exception as e:
            logger.error(f"Failed to load preprocessor {disease}: {e}")
        try:
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                scalers[disease] = scaler
                logger.info(f"   Scaler loaded: {disease}")
        except Exception as e:
            logger.warning(f"Scaler not found for {disease}")

    # Scan for additional models (any *_model.pkl)
    logger.info("   Scanning for additional models...")
    model_files = list(settings.MODELS_DIR.glob("*_model.pkl"))
    for model_file in model_files:
        disease_name = model_file.stem.replace("_model", "")
        if disease_name not in models:
            try:
                model = joblib.load(model_file)
                models[disease_name] = model
                logger.info(f"   Model loaded from file: {disease_name}")
            except Exception as e:
                logger.error(f"Failed to load model from {model_file}: {e}")

    preprocessor_files = list(settings.MODELS_DIR.glob("*_preprocessor.pkl"))
    for preprocessor_file in preprocessor_files:
        disease_name = preprocessor_file.stem.replace("_preprocessor", "")
        if disease_name not in preprocessors:
            try:
                preprocessor = joblib.load(preprocessor_file)
                preprocessors[disease_name] = preprocessor
                # extract feature names as above
                if hasattr(preprocessor, 'feature_names_'):
                    preprocessor_features[disease_name] = preprocessor.feature_names_
                elif hasattr(preprocessor, 'feature_names_in_'):
                    preprocessor_features[disease_name] = list(preprocessor.feature_names_in_)
                else:
                    try:
                        if hasattr(preprocessor, 'transformers_'):
                            all_cols = []
                            for _, trans, cols in preprocessor.transformers_:
                                if cols is not None:
                                    all_cols.extend(cols)
                            preprocessor_features[disease_name] = all_cols
                    except:
                        preprocessor_features[disease_name] = []
                logger.info(f"   Preprocessor loaded from file: {disease_name}")
            except Exception as e:
                logger.error(f"Failed to load preprocessor from {preprocessor_file}: {e}")

    # Also load scalers for additional models if they exist
    scaler_files = list(settings.MODELS_DIR.glob("*_scaler.pkl"))
    for scaler_file in scaler_files:
        disease_name = scaler_file.stem.replace("_scaler", "")
        if disease_name not in scalers:
            try:
                scaler = joblib.load(scaler_file)
                scalers[disease_name] = scaler
                logger.info(f"   Scaler loaded from file: {disease_name}")
            except Exception as e:
                logger.error(f"Failed to load scaler from {scaler_file}: {e}")

    MODELS = models
    PREPROCESSORS = preprocessors
    SCALERS = scalers
    PREPROCESSOR_FEATURES = preprocessor_features
    status["models_loaded"] = len(MODELS)
    status["timing"]["models"] = time.time() - t0

    # 4. Embeddings (optional)
    logger.info("\nGenerating Embeddings...")
    t0 = time.time()
    EMBEDDINGS = {}
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL, device=settings.DEVICE)
            EMBEDDINGS["model"] = embedding_model
            logger.info(f"   Embedding model loaded")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
    status["timing"]["embeddings"] = time.time() - t0

    # 5. Compute feature defaults (medians) for missing values
    logger.info("\nComputing feature defaults for missing values...")
    for disease in settings.SUPPORTED_DISEASES:
        df = datasets.get(disease)
        if df is None:
            continue
        # Use the actual feature names from the preprocessor if available, else from config
        feature_names = PREPROCESSOR_FEATURES.get(disease, DISEASE_CONFIG[disease]["supported_features"])
        medians = {}
        for f in feature_names:
            if f in df.columns:
                medians[f] = float(df[f].median())
            else:
                logger.warning(f"Feature '{f}' not found in dataset {disease}, using 0")
                medians[f] = 0.0
        FEATURE_DEFAULTS[disease] = medians
        logger.info(f"   Defaults for {disease}: {len(medians)} features")

    elapsed_time = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("BLOCK 2: Initialization Complete (DATA-DRIVEN VERSION)")
    logger.info("=" * 80)
    logger.info(f"   DATASETS LOADED: {len(DATASETS)}")
    logger.info(f"   MODELS LOADED: {len(MODELS)}")
    logger.info(f"   SCALERS LOADED: {len(SCALERS)}")
    logger.info(f"   SYMPTOMS INDEXED: {len(SYMPTOM_INDEX.get('symptom_list', []))}")
    logger.info(f"   DISEASES INDEXED: {len(SYMPTOM_INDEX.get('disease_list', []))}")
    logger.info(f"   SYNONYMS: {len(SYMPTOM_SYNONYMS)}")
    logger.info(f"   FEATURE DEFAULTS: {len(FEATURE_DEFAULTS)} diseases")
    logger.info(f"   INITIALIZATION TIME: {elapsed_time:.2f}s")
    logger.info("=" * 80)

    status["success"] = len(status["errors"]) == 0
    status["total_time"] = elapsed_time
    INITIALIZATION_STATUS = status
    return status

# Run initialization
initialization_status = initialize_medical_ai()

# Safety fallback – ensure SYMPTOM_INDEX is valid
if not isinstance(SYMPTOM_INDEX, dict) or not SYMPTOM_INDEX.get('symptom_list'):
    logger.error("SYMPTOM_INDEX is empty or invalid! Using fallback symptom list (no mappings).")
    fallback_symptoms = [
        "chest pain", "shortness of breath", "fatigue", "headache",
        "nausea", "fever", "cough", "sweating", "thirst", "weight loss",
        "blurred vision", "numbness", "dizziness", "joint pain", "back pain",
        "abdominal pain", "vomiting", "diarrhea", "constipation", "rash",
        "muscle pain", "chills", "appetite loss", "malaise", "phlegm",
        "insomnia", "anxiety", "swelling"
    ]
    SYMPTOM_INDEX = {
        'symptom_list': fallback_symptoms,
        'disease_list': [],
        'symptom_to_diseases': {},
        'disease_to_symptoms': {},
        'cond_prob': {},
        'disease_counts': {},
        'total_records': 1
    }
    SYMPTOM_DISEASE_MAP = {}
    DISEASE_SYMPTOMS = {}

if not isinstance(SYMPTOM_SYNONYMS, dict) or not SYMPTOM_SYNONYMS:
    logger.error("SYMPTOM_SYNONYMS missing. Generating from fallback list.")
    SYMPTOM_SYNONYMS = generate_synonyms(SYMPTOM_INDEX.get('symptom_list', fallback_symptoms))

logger.info(f"Final SYMPTOM_INDEX symptom count: {len(SYMPTOM_INDEX.get('symptom_list', []))}")
logger.info(f"Final SYMPTOM_SYNONYMS keys: {len(SYMPTOM_SYNONYMS)}")

# ============================================================================
# 6. BRAIN TUMOR DETECTION MODEL (TensorFlow)
# ============================================================================

import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

BRAIN_MODEL_PATH = settings.MODELS_DIR / "Brain_Tumor_Detection_model.h5"
BRAIN_MODEL = None

if BRAIN_MODEL_PATH.exists():
    try:
        BRAIN_MODEL = load_model(
            BRAIN_MODEL_PATH,
            compile=False,
            custom_objects={"softmax_v2": tf.nn.softmax}
        )
        logger.info(f"✅ Brain tumor model loaded from {BRAIN_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load brain model: {e}")
else:
    logger.warning(f"Brain model not found at {BRAIN_MODEL_PATH}")

BRAIN_CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def preprocess_brain_image(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image")
    img = cv2.resize(img, (200, 200))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=-1)   # (200,200,1)
    img = np.expand_dims(img, axis=0)    # (1,200,200,1)
    return img

def predict_brain_tumor(image_bytes: bytes) -> dict:
    if BRAIN_MODEL is None:
        raise RuntimeError("Brain model not loaded")
    img = preprocess_brain_image(image_bytes)
    probs = BRAIN_MODEL.predict(img, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    class_name = BRAIN_CLASSES[pred_idx]
    risk = "HIGH" if class_name != "notumor" else "LOW"
    return {
        "class": class_name,
        "confidence": confidence,
        "risk_level": risk,
        "probabilities": {cls: float(prob) for cls, prob in zip(BRAIN_CLASSES, probs)}
    }

# ============================================================================
# 7. LUNG NLP MODEL (BERT)
# ============================================================================

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LUNG_MODEL_PATH = settings.MODELS_DIR / "Lung_Model"
LUNG_LABEL_ENCODER_PATH = settings.MODELS_DIR / "label_encoder.pkl"
LUNG_TOKENIZER = None
LUNG_MODEL = None
LUNG_ENCODER = None

if LUNG_MODEL_PATH.exists() and LUNG_LABEL_ENCODER_PATH.exists():
    try:
        LUNG_TOKENIZER = AutoTokenizer.from_pretrained(str(LUNG_MODEL_PATH))
        LUNG_MODEL = AutoModelForSequenceClassification.from_pretrained(str(LUNG_MODEL_PATH))
        LUNG_ENCODER = joblib.load(LUNG_LABEL_ENCODER_PATH)
        if torch.cuda.is_available():
            LUNG_MODEL.to("cuda")
        logger.info(f"✅ Lung NLP model loaded from {LUNG_MODEL_PATH}")
        logger.info(f"   Classes: {len(LUNG_ENCODER.classes_)}")
    except Exception as e:
        logger.error(f"Failed to load lung NLP model: {e}")
else:
    logger.warning(f"Lung model files not found at {LUNG_MODEL_PATH} or label_encoder.pkl missing")

def predict_lung_disease(text: str, top_k: int = 3) -> list:
    if LUNG_MODEL is None or LUNG_TOKENIZER is None or LUNG_ENCODER is None:
        raise RuntimeError("Lung NLP model not loaded")
    device = next(LUNG_MODEL.parameters()).device
    LUNG_MODEL.eval()
    inputs = LUNG_TOKENIZER(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = LUNG_MODEL(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    top_indices = probs.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            "disease": LUNG_ENCODER.inverse_transform([idx])[0],
            "confidence": float(probs[idx])
        })
    return results

logger.info("✅ All models loaded (Brain TF + Lung NLP)")

# ============================================================================
# BLOCK 3: Medical Chat Engine - FINAL WITH FORCED RESET (UPDATED)
# ============================================================================

import re
import time
import uuid
from typing import List, Dict, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

logger = logging.getLogger("medical_ai.chat_engine")

# ----------------------------------------------------------------------------
# Data classes (unchanged)
# ----------------------------------------------------------------------------
@dataclass
class ConversationMessage:
    role: str
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DetectedSymptom:
    symptom: str
    confidence: float
    source: str
    original_text: str
    severity: int = 3
    is_confirmed: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DiseasePrediction:
    disease: str
    confidence: float
    matched_symptoms: List[str]
    missing_symptoms: List[str]
    severity_score: float
    risk_level: str
    description: str
    precautions: List[str]
    recommendation: str

@dataclass
class ChatSession:
    session_id: str
    created_at: datetime
    updated_at: datetime
    messages: List[ConversationMessage]
    detected_symptoms: List[DetectedSymptom]
    confirmed_symptoms: List[str]
    denied_symptoms: List[str]
    predicted_diseases: List[DiseasePrediction]
    answered_questions: List[str]
    missing_information: List[str]
    risk_level: str
    risk_score: float
    confidence_score: float
    is_complete: bool
    state: str
    follow_up_count: int
    max_follow_ups: int
    last_question_symptom: Optional[str]
    asked_symptoms: List[str]
    diagnosis_complete: bool
    question_history: List[str]
    user_profile: Dict[str, Any] = field(default_factory=dict)
    medical_history: List[str] = field(default_factory=list)
    diagnosis_report: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "messages": len(self.messages),
            "confirmed_symptoms": self.confirmed_symptoms,
            "denied_symptoms": self.denied_symptoms,
            "risk_level": self.risk_level,
            "state": self.state,
            "follow_up_count": self.follow_up_count,
            "diagnosis_complete": self.diagnosis_complete
        }

# ----------------------------------------------------------------------------
# Conversation states (unchanged)
# ----------------------------------------------------------------------------
class ConversationState:
    INITIAL = "initial"
    COLLECTING = "collecting"
    FOLLOW_UP = "follow_up"
    READY_FOR_PREDICTION = "ready_for_prediction"
    PREDICTING = "predicting"
    RESULT = "result"
    EMERGENCY = "emergency"
    DISEASE_INFO = "disease_info"

# ----------------------------------------------------------------------------
# NLP Pipeline (unchanged)
# ----------------------------------------------------------------------------
class NLPPipeline:
    def __init__(self):
        self.stopwords = set()
        try:
            from nltk.corpus import stopwords
            self.stopwords = set(stopwords.words('english'))
        except:
            pass
        self.yes_patterns = [
            r'^yes$', r'^yeah$', r'^yep$', r'^sure$', r'^ok$',
            r'^okay$', r'^correct$', r'^right$', r'^true$',
            r'^absolutely$', r'^definitely$', r'^of course$',
            r'^i do$', r'^i have$', r'^i am$'
        ]
        self.no_patterns = [
            r'^no$', r'^not$', r'^never$', r'^none$', r'^nope$',
            r'^nah$', r'^no way$', r'^not at all$', r'^negative$',
            r'^false$', r'^incorrect$', r'^i don\'t$', r'^i dont$',
            r'^i do not$', r'^i am not$'
        ]
        self.greeting_patterns = [
            r'^(hi|hello|hey|howdy|greetings|good morning|good afternoon|good evening|hi there|hello there|hey there)\s*$',
            r'^(how are you|what\'s up|sup|yo)\s*$'
        ]

    def normalize_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def is_yes_response(self, text: str) -> bool:
        text = text.lower().strip()
        for pattern in self.yes_patterns:
            if re.match(pattern, text):
                return True
        if RAPIDFUZZ_AVAILABLE and len(text) > 0:
            yes_words = ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'correct', 'right', 'true', 'absolutely', 'definitely']
            for word in yes_words:
                if fuzz.ratio(text, word) >= 85:
                    return True
        if text in ['y', 'ye']:
            return True
        return False

    def is_no_response(self, text: str) -> bool:
        text = text.lower().strip()
        for pattern in self.no_patterns:
            if re.match(pattern, text):
                return True
        if RAPIDFUZZ_AVAILABLE and len(text) > 0:
            no_words = ['no', 'not', 'never', 'none', 'nope', 'nah', 'no way', 'not at all', 'negative', 'false', 'incorrect']
            for word in no_words:
                if fuzz.ratio(text, word) >= 85:
                    return True
        if text in ['n']:
            return True
        return False

    def is_greeting(self, text: str) -> bool:
        text = text.lower().strip()
        for pattern in self.greeting_patterns:
            if re.match(pattern, text):
                return True
        return False

    def process(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}
        return {
            'original': text,
            'normalized': self.normalize_text(text),
            'is_yes': self.is_yes_response(text),
            'is_no': self.is_no_response(text),
            'is_greeting': self.is_greeting(text)
        }

# ----------------------------------------------------------------------------
# Symptom Extractor (enhanced with more fuzzy + keyword detection)
# ----------------------------------------------------------------------------
class SymptomExtractor:
    def __init__(self):
        self.nlp = NLPPipeline()
        self.symptom_list = SYMPTOM_INDEX.get('symptom_list', [])
        self.symptom_synonyms = SYMPTOM_SYNONYMS
        self.symptom_severity = SYMPTOM_SEVERITY

    def extract_symptoms(self, text: str) -> List[DetectedSymptom]:
        if not text or len(text.strip()) < 2:
            return []

        if self.nlp.is_greeting(text):
            logger.info("   Message is a greeting; skipping symptom extraction.")
            return []

        if len(text.strip()) <= 3:
            logger.info("   Very short message; skipping symptom extraction.")
            return []

        logger.info(f"Extracting symptoms from: '{text[:50]}...'")
        normalized_text = self.nlp.normalize_text(text)
        detected = []
        found_symptoms = set()

        # 1. EXACT MATCHING
        for symptom in self.symptom_list:
            if symptom in normalized_text:
                symptom_words = symptom.lower().split()
                if all(word in normalized_text for word in symptom_words):
                    severity = self.symptom_severity.get(symptom, 3)
                    detected.append(DetectedSymptom(
                        symptom=symptom,
                        confidence=1.0,
                        source='exact',
                        original_text=text,
                        severity=severity
                    ))
                    found_symptoms.add(symptom)
                    logger.info(f"   Exact match: '{symptom}'")

        # 2. SYNONYM MATCHING
        if len(text) > 5:
            for symptom, synonyms in self.symptom_synonyms.items():
                if symptom in found_symptoms:
                    continue
                for synonym in synonyms:
                    if synonym in normalized_text:
                        severity = self.symptom_severity.get(symptom, 3)
                        detected.append(DetectedSymptom(
                            symptom=symptom,
                            confidence=0.85,
                            source='synonym',
                            original_text=text,
                            severity=severity
                        ))
                        found_symptoms.add(symptom)
                        logger.info(f"   Synonym match: '{symptom}' from '{synonym}'")
                        break

        # 3. FUZZY MATCHING (token-based) – more aggressive
        if RAPIDFUZZ_AVAILABLE and len(detected) < 10:
            tokens = normalized_text.split()
            for token in tokens:
                if len(token) < 3:
                    continue
                # try matching each token against symptom list
                for symptom in self.symptom_list:
                    if symptom in found_symptoms:
                        continue
                    # use partial_ratio for phrases
                    score = fuzz.partial_ratio(token, symptom)
                    if score >= 75:  # lowered threshold
                        severity = self.symptom_severity.get(symptom, 3)
                        detected.append(DetectedSymptom(
                            symptom=symptom,
                            confidence=0.65,
                            source='fuzzy',
                            original_text=text,
                            severity=severity
                        ))
                        found_symptoms.add(symptom)
                        logger.info(f"   Fuzzy match: '{symptom}' (token='{token}', score={score})")
                        break

        # 4. If still no symptoms, check for common symptom-indicating phrases
        if not detected:
            indicator_patterns = [
                r'\b(have|feel|suffer|experience|am having|got)\b.*\b(pain|ache|cough|fever|headache|nausea|fatigue|dizziness|rash|swelling|numbness|shortness of breath|chest pain|back pain|joint pain|sore throat|runny nose|chills|sweating|vomiting|diarrhea|constipation|insomnia|anxiety|depression)\b'
            ]
            for pattern in indicator_patterns:
                matches = re.findall(pattern, text.lower())
                if matches:
                    # Treat as a symptom description even if we didn't match exact name
                    logger.info("   Symptom-indicating keywords found, but no exact match; will trigger reset.")
                    # We'll add a dummy symptom to force reset
                    detected.append(DetectedSymptom(
                        symptom="symptom_description",
                        confidence=0.5,
                        source='keyword',
                        original_text=text,
                        severity=3
                    ))
                    break

        logger.info(f"   Extracted {len(detected)} symptoms")
        return detected

# ----------------------------------------------------------------------------
# Disease Ranker (unchanged)
# ----------------------------------------------------------------------------
class DiseaseRanker:
    def __init__(self):
        self.symptom_disease_map = SYMPTOM_DISEASE_MAP
        self.disease_symptoms = DISEASE_SYMPTOMS
        self.symptom_severity = SYMPTOM_SEVERITY
        self.disease_descriptions = DISEASE_DESCRIPTIONS
        self.disease_precautions = DISEASE_PRECAUTIONS

    def rank_diseases(self, symptoms: List[str], top_k: int = 5) -> List[DiseasePrediction]:
        if not symptoms:
            return []

        normalized_symptoms = [s.lower().strip() for s in symptoms]
        logger.info(f"Ranking diseases for symptoms: {normalized_symptoms}")

        candidate_diseases = set()
        for symptom in normalized_symptoms:
            if symptom in self.symptom_disease_map:
                candidate_diseases.update(self.symptom_disease_map[symptom])

        if not candidate_diseases:
            logger.info("   No candidate diseases found")
            return []

        predictions = []
        for disease in candidate_diseases:
            disease_symptom_list = self.disease_symptoms.get(disease, [])
            if not disease_symptom_list:
                continue
            matched = []
            missing = []
            for ds in disease_symptom_list:
                if ds in normalized_symptoms:
                    matched.append(ds)
                else:
                    missing.append(ds)
            match_ratio = len(matched) / len(disease_symptom_list) if disease_symptom_list else 0
            severity_score = 0
            if matched:
                severity_sum = sum(self.symptom_severity.get(s, 3) for s in matched)
                severity_score = severity_sum / len(matched) / 10
            confidence = (match_ratio * 0.6) + (severity_score * 0.3) + 0.1
            confidence = min(confidence, 1.0)
            risk_level = self._calculate_risk_level(matched, severity_score)
            description = self.disease_descriptions.get(disease, "No description available")
            precautions = self.disease_precautions.get(disease, [])
            recommendation = self._generate_recommendation(disease, matched, missing, risk_level)
            predictions.append(DiseasePrediction(
                disease=disease,
                confidence=confidence,
                matched_symptoms=matched,
                missing_symptoms=missing,
                severity_score=severity_score,
                risk_level=risk_level,
                description=description,
                precautions=precautions,
                recommendation=recommendation
            ))
            logger.info(f"   {disease}: confidence={confidence:.2f}, matched={matched}")

        predictions.sort(key=lambda x: x.confidence, reverse=True)
        return predictions[:top_k]

    def _calculate_risk_level(self, symptoms: List[str], severity_score: float) -> str:
        if not symptoms:
            return "LOW"
        emergency_symptoms = {
            'chest pain', 'difficulty breathing', 'severe headache',
            'confusion', 'speech problems', 'numbness', 'unconscious'
        }
        for symptom in symptoms:
            if symptom in emergency_symptoms:
                return "EMERGENCY"
        avg_severity = severity_score * 10
        if avg_severity >= 7:
            return "HIGH"
        elif avg_severity >= 4:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_recommendation(self, disease: str, matched: List[str], missing: List[str], risk_level: str) -> str:
        if risk_level == "EMERGENCY":
            return "URGENT: Please seek immediate medical attention or call 911."
        elif risk_level == "HIGH":
            return f"Please consult a healthcare professional as soon as possible."
        elif risk_level == "MEDIUM":
            return f"Consider consulting a doctor. Your symptoms warrant professional evaluation."
        else:
            return f"Continue monitoring your condition. Consult a doctor if symptoms worsen."

# ----------------------------------------------------------------------------
# Question Engine (unchanged)
# ----------------------------------------------------------------------------
class QuestionEngine:
    def __init__(self):
        self.symptom_disease_map = SYMPTOM_DISEASE_MAP
        self.disease_symptoms = DISEASE_SYMPTOMS
        self.asked_questions = set()

    def get_next_question(self, confirmed_symptoms: List[str],
                         predicted_diseases: List[DiseasePrediction],
                         asked_symptoms: List[str]) -> Optional[str]:
        if not confirmed_symptoms:
            return "Could you please describe your symptoms in more detail?"

        if predicted_diseases and predicted_diseases[0].confidence > 0.3:
            top = predicted_diseases[0]
            missing = [s for s in top.missing_symptoms if s not in asked_symptoms]
            if missing:
                symptom = missing[0]
                return f"Do you also experience **{symptom}**?"

        common_symptoms = ['fever', 'fatigue', 'headache', 'cough', 'nausea']
        for symptom in common_symptoms:
            if symptom not in confirmed_symptoms and symptom not in asked_symptoms:
                return f"Do you also experience **{symptom}**?"
        return None

# ----------------------------------------------------------------------------
# Chat Session Manager (with forced reset, enhanced post-diagnosis handling)
# ----------------------------------------------------------------------------
class ChatSessionManager:
    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
        self.max_sessions = 1000
        if 'CHAT_SESSIONS' in globals() and CHAT_SESSIONS:
            self.sessions = CHAT_SESSIONS
        self.symptom_extractor = SymptomExtractor()
        self.disease_ranker = DiseaseRanker()
        self.question_engine = QuestionEngine()
        self._prediction_cache = {}
        self._user_sessions: Dict[str, str] = {}
        logger.info("ChatSessionManager initialized (FINAL WITH FORCED RESET)")

    def get_or_create_session(self, user_id: Optional[str] = None,
                              session_id: Optional[str] = None) -> ChatSession:
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            logger.info(f"Reusing existing session: {session_id}")
            return session
        if user_id and user_id in self._user_sessions:
            existing_session_id = self._user_sessions[user_id]
            if existing_session_id in self.sessions:
                logger.info(f"Reusing session for user: {existing_session_id}")
                return self.sessions[existing_session_id]
        session = ChatSession(
            session_id=f"SESS-{uuid.uuid4().hex[:8].upper()}",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            messages=[],
            detected_symptoms=[],
            confirmed_symptoms=[],
            denied_symptoms=[],
            predicted_diseases=[],
            answered_questions=[],
            missing_information=[],
            risk_level="LOW",
            risk_score=0.0,
            confidence_score=0.0,
            is_complete=False,
            state=ConversationState.INITIAL,
            follow_up_count=0,
            max_follow_ups=5,
            last_question_symptom=None,
            asked_symptoms=[],
            diagnosis_complete=False,
            question_history=[]
        )
        self.sessions[session.session_id] = session
        if user_id:
            self._user_sessions[user_id] = session.session_id
        if 'CHAT_SESSIONS' in globals():
            CHAT_SESSIONS[session.session_id] = session
        self._cleanup_old_sessions()
        logger.info(f"Created new session: {session.session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        return self.sessions.get(session_id)

    def _cleanup_old_sessions(self):
        if len(self.sessions) <= self.max_sessions:
            return
        sorted_sessions = sorted(
            self.sessions.items(),
            key=lambda x: x[1].updated_at,
            reverse=True
        )
        self.sessions = dict(sorted_sessions[:self.max_sessions])

    def _add_message(self, session: ChatSession, role: str, content: str):
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now()
        )
        session.messages.append(message)
        if len(session.messages) > 100:
            session.messages = session.messages[-100:]
        session.updated_at = datetime.now()

    def _detect_intent(self, text: str, session: ChatSession) -> Dict[str, Any]:
        text_lower = text.lower().strip()
        if session.state == ConversationState.FOLLOW_UP:
            processed = self.symptom_extractor.nlp.process(text)
            if processed.get('is_yes', False):
                return {'intent': 'confirmation_yes'}
            if processed.get('is_no', False):
                return {'intent': 'confirmation_no'}
            if len(text_lower) <= 2:
                if text_lower.startswith('y'):
                    return {'intent': 'confirmation_yes'}
                elif text_lower.startswith('n'):
                    return {'intent': 'confirmation_no'}
            symptoms = self.symptom_extractor.extract_symptoms(text)
            if symptoms:
                return {'intent': 'symptom_diagnosis', 'symptoms': symptoms}
            return {'intent': 'general_chat'}

        patterns = [
            r'what is (?:a|an)?\s*[\w\s]+',
            r'tell me about (?:a|an)?\s*[\w\s]+',
            r'explain (?:a|an)?\s*[\w\s]+',
            r'information about (?:a|an)?\s*[\w\s]+',
            r'what are the symptoms of',
            r'how to prevent'
        ]
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return {'intent': 'disease_info'}
        symptoms = self.symptom_extractor.extract_symptoms(text)
        if symptoms:
            return {'intent': 'symptom_diagnosis', 'symptoms': symptoms}
        if text_lower in ['reset', 'start over', 'new session']:
            return {'intent': 'reset'}
        return {'intent': 'general_chat'}

    def process_user_message(self, message: str, user_id: Optional[str] = None,
                            session_id: Optional[str] = None) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info(f"Processing message: '{message[:50]}...'")
        logger.info(f"Session ID: {session_id}")
        if not message or len(message.strip()) < 1:
            return {
                'text': "Could you please describe your symptoms or ask a question?",
                'error': 'Empty message'
            }
        session = self.get_or_create_session(user_id, session_id)
        logger.info(f"Session state: {session.state}, Follow-ups: {session.follow_up_count}")
        self._add_message(session, 'user', message)
        try:
            intent_result = self._detect_intent(message, session)
            intent = intent_result.get('intent')
            logger.info(f"Detected intent: {intent}")
            if intent == 'reset':
                self.reset_conversation(session.session_id)
                session = self.get_session(session.session_id)
                return self._handle_general_chat(message, session)
            elif intent == 'disease_info':
                return self._handle_disease_info(message, session)
            elif intent == 'confirmation_yes':
                return self._handle_confirmation_yes(message, session)
            elif intent == 'confirmation_no':
                return self._handle_confirmation_no(message, session)
            elif intent == 'symptom_diagnosis':
                symptoms = intent_result.get('symptoms', [])
                return self._handle_symptom_diagnosis(message, session, symptoms)
            elif session.state == ConversationState.RESULT:
                return self._handle_post_diagnosis(message, session)
            else:
                return self._handle_general_chat(message, session)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            import traceback
            traceback.print_exc()
            return {
                'text': "I apologize, but I encountered an error. Please try again.",
                'error': str(e)
            }

    # ----- Handlers -----
    def _handle_disease_info(self, message: str, session: ChatSession) -> Dict[str, Any]:
        disease_name = self._extract_disease_name(message)
        if disease_name and disease_name in DISEASE_DESCRIPTIONS:
            response = f"**{disease_name.title()}**\n\n"
            response += f"{DISEASE_DESCRIPTIONS.get(disease_name, '')}\n\n"
            symptoms = DISEASE_SYMPTOMS.get(disease_name, [])
            if symptoms:
                response += "**Common Symptoms:**\n"
                for s in symptoms[:5]:
                    response += f"• {s}\n"
                response += "\n"
            precautions = DISEASE_PRECAUTIONS.get(disease_name, [])
            if precautions:
                response += "**Precautions:**\n"
                for p in precautions[:3]:
                    response += f"• {p}\n"
                response += "\n"
            response += "---\n*This information is for educational purposes only.*"
        else:
            response = "I don't have detailed information about that condition. Would you like to describe your symptoms instead?"
        self._add_message(session, 'assistant', response)
        session.state = ConversationState.DISEASE_INFO
        return {
            'text': response,
            'intent': 'disease_info',
            'is_emergency': False,
            'session_id': session.session_id,
            'state': session.state
        }

    def _extract_disease_name(self, text: str) -> Optional[str]:
        text_lower = text.lower().strip()
        for disease in DISEASE_DESCRIPTIONS.keys():
            if disease.lower() in text_lower:
                return disease
        return None

    def _handle_confirmation_yes(self, message: str, session: ChatSession) -> Dict[str, Any]:
        if session.last_question_symptom:
            symptom = session.last_question_symptom
            if symptom not in session.confirmed_symptoms:
                session.confirmed_symptoms.append(symptom)
                if symptom in session.denied_symptoms:
                    session.denied_symptoms.remove(symptom)
            logger.info(f"Confirmed symptom via yes: {symptom}")
        session.last_question_symptom = None
        return self._continue_diagnosis(session)

    def _handle_confirmation_no(self, message: str, session: ChatSession) -> Dict[str, Any]:
        if session.last_question_symptom:
            symptom = session.last_question_symptom
            if symptom not in session.denied_symptoms:
                session.denied_symptoms.append(symptom)
            if symptom in session.confirmed_symptoms:
                session.confirmed_symptoms.remove(symptom)
            logger.info(f"Denied symptom via no: {symptom}")
        session.last_question_symptom = None
        return self._continue_diagnosis(session)

    def _handle_symptom_diagnosis(self, message: str, session: ChatSession,
                                 detected_symptoms: List[DetectedSymptom]) -> Dict[str, Any]:
        for symptom in detected_symptoms:
            if symptom.symptom not in session.confirmed_symptoms:
                session.confirmed_symptoms.append(symptom.symptom)
                session.detected_symptoms.append(symptom)
                logger.info(f"Added symptom: {symptom.symptom}")
        if session.state == ConversationState.INITIAL:
            session.state = ConversationState.COLLECTING
        elif session.state == ConversationState.FOLLOW_UP:
            session.state = ConversationState.COLLECTING
        return self._continue_diagnosis(session)

    def _handle_general_chat(self, message: str, session: ChatSession) -> Dict[str, Any]:
        response = "I'm here to help with health-related questions. You can:\n\n"
        response += "• Ask about a specific condition (e.g., 'Tell me about diabetes')\n"
        response += "• Describe your symptoms (e.g., 'I have chest pain')\n"
        response += "• Get health information\n\n"
        response += "How can I assist you today?"
        self._add_message(session, 'assistant', response)
        return {
            'text': response,
            'is_emergency': False,
            'session_id': session.session_id,
            'state': session.state
        }

    # -------------------------------------------------------------------------
    #  POST-DIAGNOSIS HANDLER (FORCED RESET) – enhanced
    # -------------------------------------------------------------------------
    def _handle_post_diagnosis(self, message: str, session: ChatSession) -> Dict[str, Any]:
        logger.info(f"Post-diagnosis handling: '{message}'")

        # 1. Check if it's a question (answer without reset)
        question_keywords = ['what', 'how', 'why', 'should', 'can', 'tell', 'explain', 'describe', 'about']
        is_question = any(kw in message.lower() for kw in question_keywords)
        if is_question:
            logger.info("Message is a question – answering based on current diagnosis.")
            response = self._generate_post_diagnosis_answer(message, session)
            self._add_message(session, 'assistant', response)
            return {
                'text': response,
                'is_emergency': False,
                'session_id': session.session_id,
                'state': session.state
            }

        # 2. Extract symptoms (will use enhanced extractor)
        detected = self.symptom_extractor.extract_symptoms(message)
        logger.info(f"Extracted symptoms: {[s.symptom for s in detected]}")

        # 3. If ANY symptom is detected, reset and start fresh diagnosis
        if detected:
            logger.info("Symptoms detected – resetting session for new analysis.")
            self._reset_session_for_new_diagnosis(session)
            # Add the detected symptoms (even the dummy one)
            for symptom in detected:
                if symptom.symptom not in session.confirmed_symptoms:
                    session.confirmed_symptoms.append(symptom.symptom)
                    session.detected_symptoms.append(symptom)
                    logger.info(f"Added symptom after reset: {symptom.symptom}")
            return self._continue_diagnosis(session)

        # 4. If no symptoms, check for symptom-describing keywords (more aggressive)
        symptom_keywords = ['have', 'feel', 'my', 'pain', 'ache', 'symptom', 'hurt', 'suffering', 'experiencing', 'swell', 'swelling', 'ache', 'cough', 'fever', 'headache', 'nausea', 'dizziness', 'rash', 'itching', 'burning', 'numbness', 'tingling', 'weakness', 'fatigue', 'tired', 'shortness of breath', 'chest tightness', 'palpitations', 'indigestion', 'vomiting', 'diarrhea', 'constipation', 'insomnia']
        if any(word in message.lower() for word in symptom_keywords):
            logger.info("Symptom description detected but no specific symptoms extracted – resetting anyway.")
            self._reset_session_for_new_diagnosis(session)
            # Add a dummy symptom to avoid empty diagnosis
            dummy = DetectedSymptom(
                symptom="symptom_description",
                confidence=0.4,
                source='keyword',
                original_text=message,
                severity=3
            )
            session.confirmed_symptoms.append("symptom_description")
            session.detected_symptoms.append(dummy)
            return self._continue_diagnosis(session)

        # 5. Otherwise, treat as general chat
        logger.info("No symptoms or keywords – treating as general chat.")
        return self._handle_general_chat(message, session)

    def _reset_session_for_new_diagnosis(self, session: ChatSession):
        """Reset all diagnosis-related fields, keeping the session active."""
        session.confirmed_symptoms = []
        session.detected_symptoms = []
        session.predicted_diseases = []
        session.asked_symptoms = []
        session.follow_up_count = 0
        session.diagnosis_complete = False
        session.risk_level = "LOW"
        session.risk_score = 0.0
        session.state = ConversationState.COLLECTING
        logger.info(f"Session reset for new diagnosis. Session ID: {session.session_id}")

    # -------------------------------------------------------------------------
    #  POST-DIAGNOSIS ANSWER GENERATOR (unchanged)
    # -------------------------------------------------------------------------
    def _generate_post_diagnosis_answer(self, message: str, session: ChatSession) -> str:
        text_lower = message.lower().strip()
        if 'what should i do' in text_lower or 'what to do' in text_lower:
            if session.predicted_diseases:
                top = session.predicted_diseases[0]
                response = f"Based on your diagnosis of **{top.disease}**, here's what you should do:\n\n"
                response += f"{top.recommendation}\n\n"
                if top.precautions:
                    response += "**Precautions:**\n"
                    for p in top.precautions[:3]:
                        response += f"• {p}\n"
                return response
            else:
                return "Please describe your symptoms first so I can provide appropriate recommendations."
        if 'risk' in text_lower or 'emergency' in text_lower:
            if session.risk_level == "EMERGENCY":
                return "⚠️ **EMERGENCY**: Please call 911 immediately. Your symptoms require urgent medical attention."
            elif session.risk_level == "HIGH":
                return "⚠️ **High Risk**: Please consult a healthcare professional within 24 hours."
            else:
                return f"Your current risk level is **{session.risk_level}**. Monitor your symptoms and consult a doctor if they worsen."
        if 'symptom' in text_lower or 'condition' in text_lower:
            if session.confirmed_symptoms:
                return f"Your confirmed symptoms are: **{', '.join(session.confirmed_symptoms)}**"
            else:
                return "No symptoms have been confirmed yet. Please describe what you're feeling."
        return "I can help you with health-related questions. Based on our conversation, you have " + \
               f"{', '.join(session.confirmed_symptoms) if session.confirmed_symptoms else 'no confirmed symptoms'}. " + \
               "You can ask about symptoms, risk levels, or what to do next."

    # -------------------------------------------------------------------------
    #  CONTINUE DIAGNOSIS (unchanged)
    # -------------------------------------------------------------------------
    def _continue_diagnosis(self, session: ChatSession) -> Dict[str, Any]:
        symptoms = session.confirmed_symptoms
        logger.info(f"Continue diagnosis. Symptoms: {symptoms}")
        if len(symptoms) < 2:
            response = f"Thank you for describing your symptoms.\n\n"
            response += f"**Current symptoms:** {', '.join(symptoms)}\n\n"
            response += "Could you describe any other symptoms you're experiencing?"
            self._add_message(session, 'assistant', response)
            session.state = ConversationState.COLLECTING
            return {
                'text': response,
                'symptoms': symptoms,
                'state': session.state,
                'follow_up_count': session.follow_up_count,
                'is_emergency': False,
                'session_id': session.session_id
            }
        predictions = self.disease_ranker.rank_diseases(symptoms)
        session.predicted_diseases = predictions
        if predictions and predictions[0].confidence > 0.5:
            return self._generate_diagnosis_response(session)
        if session.follow_up_count >= session.max_follow_ups:
            return self._generate_diagnosis_response(session)
        next_question = self.question_engine.get_next_question(
            symptoms, predictions, session.asked_symptoms
        )
        if next_question:
            session.follow_up_count += 1
            session.state = ConversationState.FOLLOW_UP
            match = re.search(r'\*\*(.*?)\*\*', next_question)
            if match:
                session.last_question_symptom = match.group(1).lower().strip()
            else:
                session.last_question_symptom = None
            if session.last_question_symptom:
                session.asked_symptoms.append(session.last_question_symptom)
            response = f"Thank you for describing your symptoms.\n\n"
            if symptoms:
                response += f"**Current symptoms:** {', '.join(symptoms)}\n\n"
            response += next_question
            self._add_message(session, 'assistant', response)
            return {
                'text': response,
                'symptoms': symptoms,
                'state': session.state,
                'follow_up_count': session.follow_up_count,
                'is_emergency': False,
                'session_id': session.session_id
            }
        else:
            return self._generate_diagnosis_response(session)

    def _generate_diagnosis_response(self, session: ChatSession) -> Dict[str, Any]:
        symptoms = session.confirmed_symptoms
        predictions = session.predicted_diseases
        session.diagnosis_complete = True
        session.state = ConversationState.RESULT
        if predictions:
            top = predictions[0]
            response = f"**🏥 Diagnosis Complete**\n\n"
            response += f"**Primary Diagnosis: {top.disease}**\n"
            response += f"**Confidence:** {top.confidence:.1%}\n\n"
            if top.matched_symptoms:
                response += f"**Matched Symptoms:** {', '.join(top.matched_symptoms)}\n"
            if top.missing_symptoms:
                response += f"**Missing Symptoms:** {', '.join(top.missing_symptoms[:3])}\n"
            response += f"\n**Recommendation:** {top.recommendation}\n"
            if top.precautions:
                response += f"\n**Precautions:**\n"
                for p in top.precautions[:3]:
                    response += f"• {p}\n"
            if top.risk_level == "EMERGENCY":
                response += "\n⚠️ **URGENT: Please seek immediate medical attention!**"
            elif top.risk_level == "HIGH":
                response += "\n⚠️ **High Risk: Please consult a doctor soon.**"
            response += "\n\n---\n*This is an AI-powered analysis and should not replace professional medical advice.*"
            session.risk_level = top.risk_level
            session.risk_score = top.confidence
            self._add_message(session, 'assistant', response)
            return {
                'text': response,
                'predictions': [{'disease': p.disease, 'confidence': p.confidence} for p in predictions[:3]],
                'symptoms': symptoms,
                'risk_level': session.risk_level,
                'is_emergency': session.risk_level == 'EMERGENCY',
                'state': session.state,
                'prediction_complete': True,
                'session_id': session.session_id
            }
        else:
            response = "I couldn't identify any specific diseases based on your symptoms.\n\n"
            response += "Please provide more details or consult a healthcare professional."
            self._add_message(session, 'assistant', response)
            return {
                'text': response,
                'symptoms': symptoms,
                'state': session.state,
                'prediction_complete': True,
                'session_id': session.session_id
            }

    def reset_conversation(self, session_id: str) -> bool:
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.confirmed_symptoms = []
            session.denied_symptoms = []
            session.detected_symptoms = []
            session.predicted_diseases = []
            session.asked_symptoms = []
            session.question_history = []
            session.follow_up_count = 0
            session.risk_level = "LOW"
            session.risk_score = 0.0
            session.diagnosis_complete = False
            session.state = ConversationState.INITIAL
            session.messages = []
            session.updated_at = datetime.now()
            logger.info(f"Reset conversation for session: {session_id}")
            return True
        return False

# ----------------------------------------------------------------------------
# Medical Chat Engine
# ----------------------------------------------------------------------------
class MedicalChatEngine:
    def __init__(self):
        self.session_manager = ChatSessionManager()
        self.nlp = NLPPipeline()
        self.initialized = True
        self.model_loaded = False
        logger.info("Medical Chat Engine initialized (FINAL WITH FORCED RESET)")

    def process_message(self, message: str, session_id: Optional[str] = None,
                       user_id: Optional[str] = None) -> Dict[str, Any]:
        if not message or len(message.strip()) < 1:
            return {
                'text': "I didn't receive any message. Please describe your symptoms or ask a question.",
                'error': 'Empty message'
            }
        result = self.session_manager.process_user_message(
            message=message,
            user_id=user_id,
            session_id=session_id
        )
        if 'session_id' not in result:
            session = self.session_manager.get_or_create_session(user_id, session_id)
            result['session_id'] = session.session_id
        session = self.session_manager.get_session(result['session_id'])
        if session:
            result['state'] = session.state
            result['prediction_complete'] = session.diagnosis_complete
            result['symptoms'] = session.confirmed_symptoms
            result['follow_up_count'] = session.follow_up_count
            result['session'] = session.to_dict()
        logger.info(f"Response: state={result.get('state')}, symptoms={result.get('symptoms')}")
        return result

    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        session = self.session_manager.get_session(session_id)
        if not session:
            return None
        return session.to_dict()

    def reset_conversation(self, session_id: str) -> bool:
        return self.session_manager.reset_conversation(session_id)

    def validate_message(self, message: str) -> Dict[str, Any]:
        if not message or len(message.strip()) < 1:
            return {
                'valid': False,
                'suggestion': 'Please describe your symptoms or ask a question.'
            }
        return {'valid': True}

    def get_available_diseases(self) -> List[str]:
        return list(DISEASE_DESCRIPTIONS.keys())

# ----------------------------------------------------------------------------
# Global initialization
# ----------------------------------------------------------------------------
CHAT_ENGINE = MedicalChatEngine()
logger.info("Medical Chat Engine is ready (FINAL WITH FORCED RESET)")

# ============================================================================
# BLOCK 4: FastAPI Server - COMPLETE WITH ON-DEMAND MODEL LOADING & SCALER FIX
# ============================================================================

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import logging
import os
import json

logger = logging.getLogger("medical_ai")

# ----------------------------------------------------------------------------
# Pydantic models
# ----------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    patient_id: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    risk_level: str = "LOW"
    symptoms: List[str] = []
    diseases: Dict[str, float] = {}
    is_emergency: bool = False
    confidence: Dict[str, float] = {}
    suggestions: List[str] = []

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    probability: float
    risk_level: str
    recommendation: str
    details: Optional[Dict[str, Any]] = None

class LungSymptomRequest(BaseModel):
    symptoms: str
    top_k: int = 3

# ----------------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------------
app = FastAPI(
    title="Medical AI Assistant API",
    description="AI-powered medical triage with ML disease prediction",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------------
# Feature mapping (exact order from training) – now used only as fallback
# ----------------------------------------------------------------------------
DISEASE_FEATURES = {
    "diabetes": [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ],
    "heart": [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ],
    "parkinson": [
        "MDVP_Fo_Hz", "MDVP_Fhi_Hz", "MDVP_Flo_Hz",
        "MDVP_Jitter_percent", "MDVP_Jitter_Abs", "MDVP_RAP",
        "MDVP_PPQ", "Jitter_DDP", "MDVP_Shimmer",
        "MDVP_Shimmer_dB", "Shimmer_APQ3", "Shimmer_APQ5",
        "MDVP_APQ", "Shimmer_DDA", "NHR", "HNR",
        "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ],
    "liver": [
        "Sex", "age", "Total_Bilirubin", "Direct_Bilirubin",
        "Alkaline_Phosphotase", "Alamine_Aminotransferase",
        "Aspartate_Aminotransferase", "Total_Protiens",
        "Albumin", "Albumin_and_Globulin_Ratio"
    ],
    "hepatitis": [
        "Age", "Sex", "ALB", "ALP", "ALT", "AST", "BIL",
        "CHE", "CHOL", "CREA", "GGT", "PROT"
    ],
    "lung": [
        "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS",
        "ANXIETY", "PEER_PRESSURE", "CHRONIC_DISEASE",
        "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL_CONSUMING",
        "COUGHING", "SHORTNESS_OF_BREATH", "SWALLOWING_DIFFICULTY",
        "CHEST_PAIN"
    ],
    "kidney": [
        "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc",
        "ba", "bgr", "bu", "sc", "sod", "pot", "hemo",
        "pcv", "wc", "rc", "htn", "dm", "cad", "appet",
        "pe", "ane"
    ],
    "breast": [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean",
        "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se",
        "smoothness_se", "compactness_se", "concavity_se",
        "concave_points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
        "smoothness_worst", "compactness_worst", "concavity_worst",
        "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]
}

SELECT_MAPPING = {
    "sex": {"Female": 0, "Male": 1},
    "Sex": {"Female": 0, "Male": 1},
    "GENDER": {"Female": 0, "Male": 1},
    "fbs": {"No": 0, "Yes": 1},
    "exang": {"No": 0, "Yes": 1},
    "cp": {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3},
    "restecg": {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2},
    "slope": {"Upsloping": 0, "Flat": 1, "Downsloping": 2},
    "thal": {"Normal": 0, "Fixed Defect": 1, "Reversible Defect": 2},
    "SMOKING": {"No": 0, "Yes": 1},
    "YELLOW_FINGERS": {"No": 0, "Yes": 1},
    "ANXIETY": {"No": 0, "Yes": 1},
    "PEER_PRESSURE": {"No": 0, "Yes": 1},
    "CHRONIC_DISEASE": {"No": 0, "Yes": 1},
    "FATIGUE": {"No": 0, "Yes": 1},
    "ALLERGY": {"No": 0, "Yes": 1},
    "WHEEZING": {"No": 0, "Yes": 1},
    "ALCOHOL_CONSUMING": {"No": 0, "Yes": 1},
    "COUGHING": {"No": 0, "Yes": 1},
    "SHORTNESS_OF_BREATH": {"No": 0, "Yes": 1},
    "SWALLOWING_DIFFICULTY": {"No": 0, "Yes": 1},
    "CHEST_PAIN": {"No": 0, "Yes": 1},
    "rbc": {"Normal": 0, "Abnormal": 1},
    "pc": {"Normal": 0, "Abnormal": 1},
    "pcc": {"Present": 1, "Not Present": 0},
    "ba": {"Present": 1, "Not Present": 0},
    "htn": {"No": 0, "Yes": 1},
    "dm": {"No": 0, "Yes": 1},
    "cad": {"No": 0, "Yes": 1},
    "appet": {"Good": 0, "Poor": 1},
    "pe": {"No": 0, "Yes": 1},
    "ane": {"No": 0, "Yes": 1},
}

# ----------------------------------------------------------------------------
# Feature preparation with default imputation (using preprocessor's actual column names)
# ----------------------------------------------------------------------------
def prepare_features_for_prediction(disease: str, data: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a DataFrame with columns exactly matching the preprocessor's expectations.
    Handles missing features by using training-set medians.
    """
    # Get the feature names from the preprocessor (loaded earlier)
    feature_names = PREPROCESSOR_FEATURES.get(disease)
    if not feature_names:
        # Fallback: use the hardcoded list from DISEASE_CONFIG
        feature_names = DISEASE_CONFIG.get(disease, {}).get("supported_features", [])
        logger.warning(f"No preprocessor feature names for {disease}, using config fallback.")

    defaults = FEATURE_DEFAULTS.get(disease, {})
    row = {}
    missing = []

    for name in feature_names:
        # Check if the input data contains this feature (case-insensitive, normalize spaces)
        # Try to find the key in data by normalizing both names (remove underscores/spaces, lower)
        val = None
        for key, value in data.items():
            # Normalize both to compare: remove spaces and underscores, lower
            key_norm = re.sub(r'[_\s]+', '', key.lower())
            name_norm = re.sub(r'[_\s]+', '', name.lower())
            if key_norm == name_norm:
                val = value
                break

        if val is not None:
            # Convert value to float/int
            if isinstance(val, str):
                # Try mapping for categorical features (if any)
                mapping = SELECT_MAPPING.get(name)
                if mapping and val in mapping:
                    val = mapping[val]
                else:
                    try:
                        val = float(val)
                    except:
                        val = 0.0
            else:
                try:
                    val = float(val) if val is not None else 0.0
                except:
                    val = 0.0
        else:
            # Feature missing: use default (median from training)
            val = defaults.get(name, 0.0)
            missing.append(name)

        row[name] = val

    if missing:
        logger.warning(f"Missing features for {disease}: {missing} – using defaults (medians).")

    # Build DataFrame with one row
    df = pd.DataFrame([row])
    # Ensure columns are in the exact order expected by the preprocessor (if needed)
    # The preprocessor uses column names, so order doesn't matter, but we ensure all columns present.
    return df

# ----------------------------------------------------------------------------
# Get model and transformer (preprocessor or scaler)
# ----------------------------------------------------------------------------
def get_model_and_transformer(disease: str):
    model = MODELS.get(disease)
    if model is None:
        model_path = settings.MODELS_DIR / f"{disease}_model.pkl"
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                MODELS[disease] = model
                logger.info(f"✅ Loaded {disease} model on-demand from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise HTTPException(status_code=503, detail=f"Model for {disease} not found")
        else:
            raise HTTPException(status_code=503, detail=f"Model file for {disease} not found at {model_path}")

    # Prefer the full preprocessor
    preprocessor = PREPROCESSORS.get(disease)
    if preprocessor is None:
        preprocessor_path = settings.MODELS_DIR / f"{disease}_preprocessor.pkl"
        if preprocessor_path.exists():
            try:
                preprocessor = joblib.load(preprocessor_path)
                PREPROCESSORS[disease] = preprocessor
                # Also store feature names if not already
                if disease not in PREPROCESSOR_FEATURES:
                    if hasattr(preprocessor, 'feature_names_'):
                        PREPROCESSOR_FEATURES[disease] = preprocessor.feature_names_
                    elif hasattr(preprocessor, 'feature_names_in_'):
                        PREPROCESSOR_FEATURES[disease] = list(preprocessor.feature_names_in_)
                logger.info(f"✅ Loaded preprocessor for {disease} on-demand from {preprocessor_path}")
            except Exception as e:
                logger.warning(f"Failed to load preprocessor: {e}")
                preprocessor = None

    # Fallback to scaler if preprocessor not available
    scaler = None
    if preprocessor is None:
        scaler = SCALERS.get(disease)
        if scaler is None:
            scaler_path = settings.MODELS_DIR / f"{disease}_scaler.pkl"
            if scaler_path.exists():
                try:
                    scaler = joblib.load(scaler_path)
                    SCALERS[disease] = scaler
                    logger.info(f"✅ Loaded scaler for {disease} on-demand from {scaler_path}")
                except Exception as e:
                    logger.warning(f"Failed to load scaler: {e}")
                    scaler = None
        if scaler is None:
            logger.warning(f"No preprocessor or scaler found for {disease} – using raw features.")

    return model, preprocessor, scaler

# ----------------------------------------------------------------------------
# Prediction endpoint (generic)
# ----------------------------------------------------------------------------
async def _predict_generic(disease: str, data: Dict[str, Any]) -> PredictionResponse:
    try:
        model, preprocessor, scaler = get_model_and_transformer(disease)
        input_df = prepare_features_for_prediction(disease, data)

        # Apply preprocessor if available
        if preprocessor is not None:
            try:
                features = preprocessor.transform(input_df)
                logger.info(f"✅ Preprocessed features (first 5): {features[0][:5]}")
            except Exception as e:
                logger.error(f"❌ Preprocessor transform failed: {e}. Falling back to raw features.")
                features = input_df.values
        elif scaler is not None:
            try:
                features = scaler.transform(input_df)
                logger.info(f"✅ Scaled features (first 5): {features[0][:5]}")
            except Exception as e:
                logger.warning(f"Scaler transform failed: {e}. Using raw features.")
                features = input_df.values
        else:
            logger.warning(f"No preprocessor or scaler – using raw features.")
            features = input_df.values

        # Predict
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(features)[0]
                logger.info(f"Prediction probabilities: {proba}")
            except Exception as e:
                logger.warning(f"predict_proba failed: {e}. Using default.")
                proba = None
        else:
            proba = None

        prediction = model.predict(features)[0]
        logger.info(f"Prediction result: {prediction}")

        if proba is not None:
            if len(proba) == 2:
                probability = float(proba[1])
            else:
                probability = float(proba[0] if prediction == 0 else proba[1])
        else:
            probability = 0.5

        confidence = max(probability, 1 - probability) if proba is not None else 0.5
        risk_level = "HIGH" if prediction == 1 else "LOW"
        recommendation = "Consult a healthcare professional immediately." if prediction == 1 else "Your results are normal. Continue monitoring."

        return PredictionResponse(
            prediction=int(prediction),
            confidence=float(confidence),
            probability=float(probability),
            risk_level=risk_level,
            recommendation=recommendation,
            details={"features_used": PREPROCESSOR_FEATURES.get(disease, [])}
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Prediction error for {disease}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------
@app.post("/api/diabetes/predict", response_model=PredictionResponse)
async def predict_diabetes(data: Dict[str, Any]):
    return await _predict_generic("diabetes", data)

@app.post("/api/heart/predict", response_model=PredictionResponse)
async def predict_heart(data: Dict[str, Any]):
    return await _predict_generic("heart", data)

@app.post("/api/parkinson/predict", response_model=PredictionResponse)
async def predict_parkinson(data: Dict[str, Any]):
    return await _predict_generic("parkinson", data)

@app.post("/api/liver/predict", response_model=PredictionResponse)
async def predict_liver(data: Dict[str, Any]):
    return await _predict_generic("liver", data)

@app.post("/api/hepatitis/predict", response_model=PredictionResponse)
async def predict_hepatitis(data: Dict[str, Any]):
    return await _predict_generic("hepatitis", data)

@app.post("/api/lung/predict", response_model=PredictionResponse)
async def predict_lung(data: Dict[str, Any]):
    return await _predict_generic("lung", data)

@app.post("/api/kidney/predict", response_model=PredictionResponse)
async def predict_kidney(data: Dict[str, Any]):
    return await _predict_generic("kidney", data)

@app.post("/api/breast/predict", response_model=PredictionResponse)
async def predict_breast(data: Dict[str, Any]):
    return await _predict_generic("breast", data)

# ----------------------------------------------------------------------------
# Lung symptom prediction endpoint (NLP)
# ----------------------------------------------------------------------------
@app.post("/api/lung/symptom_predict")
async def predict_lung_symptoms(request: LungSymptomRequest):
    try:
        if not request.symptoms or len(request.symptoms.strip()) < 2:
            raise HTTPException(status_code=400, detail="Please describe your symptoms.")
        predictions = predict_lung_disease(request.symptoms, request.top_k)
        return {
            "success": True,
            "predictions": predictions,
            "input": request.symptoms
        }
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Lung symptom prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# ----------------------------------------------------------------------------
# Brain tumor prediction endpoint
# ----------------------------------------------------------------------------
@app.post("/api/brain/predict")
async def predict_brain(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        image_bytes = await file.read()
        result = predict_brain_tumor(image_bytes)
        return {
            "success": True,
            "prediction": result["class"],
            "confidence": result["confidence"],
            "risk_level": result["risk_level"],
            "probabilities": result["probabilities"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Brain prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# ----------------------------------------------------------------------------
# Health & Chat endpoints
# ----------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Medical AI Assistant API", "version": "4.0.0", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "4.0.0",
        "models_loaded": len(MODELS),
        "sessions_active": len(CHAT_ENGINE.session_manager.sessions) if CHAT_ENGINE else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"Chat request: session_id={request.session_id}, patient_id={request.patient_id}")
        result = CHAT_ENGINE.process_message(
            message=request.message,
            session_id=request.session_id,
            user_id=request.patient_id
        )
        response = ChatResponse(
            response=result.get('text', ''),
            session_id=result.get('session_id', ''),
            risk_level=result.get('risk_level', 'LOW'),
            symptoms=result.get('symptoms', []),
            diseases={p['disease']: p['confidence'] for p in result.get('predictions', [])[:3]},
            is_emergency=result.get('is_emergency', False),
            confidence={},
            suggestions=[]
        )
        return response
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset")
async def reset_conversation(request: Dict[str, str]):
    session_id = request.get('session_id')
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    success = CHAT_ENGINE.reset_conversation(session_id)
    return {"success": success, "session_id": session_id}

# ============================================================================
# Run server
# ============================================================================
if __name__ == "__main__":
    import os
    import uvicorn

    # Use environment variable PORT (for Render, Hugging Face, etc.) or default to 7860
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )