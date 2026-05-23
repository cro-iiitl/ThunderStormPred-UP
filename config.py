import os
from datetime import datetime

# App configuration
DEBUG = True
SECRET_KEY = 'your-secret-key'

# Data directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

# Uttar Pradesh region boundaries (approximate)
UP_BOUNDS = {
    'min_lat': 23.5,
    'max_lat': 30.5,
    'min_lon': 77.0,
    'max_lon': 84.5
}

# Data source API configurations
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY', 'bd5e378503939ddaee76f12ad7a97608')
WEATHER_API_URL = 'https://api.openweathermap.org/data/2.5/weather'

# IMD API configuration (replace with actual credentials if available)
IMD_API_KEY = os.getenv('IMD_API_KEY', '')
IMD_API_URL = os.getenv('IMD_API_URL', 'https://example.imd.gov.in/api/weather')

# Earth Networks Lightning Network API (replace with actual credentials if available)
ENTLN_API_KEY = os.getenv('ENTLN_API_KEY', '')
ENTLN_API_URL = os.getenv('ENTLN_API_URL', 'https://api.earthnetworks.com/v1/lightning')

# Data collection parameters
COLLECTION_INTERVAL_HOURS = 1

# Model parameters
SEQUENCE_LENGTH = 24  # 24 hours of data for prediction
FORECAST_HORIZON = 12  # Predict 12 hours ahead

# Standard weather parameters
FEATURES = [
    'temperature', 'humidity', 'pressure', 'wind_speed', 
    'wind_direction', 'precipitation', 'cloud_cover'
]

# Enhanced features for thunderstorm prediction
ENHANCED_FEATURES = FEATURES + [
    'cape',              # Convective Available Potential Energy
    'cin',               # Convective Inhibition
    'lifted_index',      # Lifted Index
    'k_index',           # K-Index (humidity & temperature)
    'precipitable_water', # Total column water
    'vertical_temp_diff', # Temperature differential
    'vertical_wind_shear', # Wind shear
    'dewpoint',          # Dewpoint temperature
    'relative_humidity_850mb', # Mid-level humidity
    'relative_humidity_500mb', # Upper-level humidity
]

# Target variable
TARGET = 'lightning_probability'

# Use mock data for development (no API calls)
USE_MOCK_DATA = True # Set to True for development