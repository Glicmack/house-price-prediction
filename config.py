# config.py
"""
Configuration file for Flask applications.
Usage: from config import config
"""

import os
from datetime import timedelta

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    MODEL_PATH = os.environ.get('MODEL_PATH', 'model.pkl')
    META_PATH = os.environ.get('META_PATH', 'metadata.json')
    
    # Session configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=1)
    
    # File upload (if needed later)
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    # Note: In real production, set SECRET_KEY environment variable!

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}