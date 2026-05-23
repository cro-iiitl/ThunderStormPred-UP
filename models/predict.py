import os
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
import sys
sys.path.append('..')
from config import MODEL_DIR, SEQUENCE_LENGTH, ENHANCED_FEATURES, UP_BOUNDS, FORECAST_HORIZON

def load_model():
    """Load the trained model"""
    model_path = os.path.join(MODEL_DIR, 'thunderstorm_prediction_model.h5')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    return model

def prepare_input_data(latest_data, sequence_length=SEQUENCE_LENGTH):
    """Prepare input data for prediction from the latest weather data"""
    # Group by location and create sequences
    locations = latest_data[['latitude', 'longitude']].drop_duplicates().values
    
    prediction_inputs = []
    location_info = []
    
    for lat, lon in locations:
        location_data = latest_data[(latest_data['latitude'] == lat) & 
                                    (latest_data['longitude'] == lon)]
        
        if len(location_data) >= sequence_length:
            # Sort by timestamp and take the latest sequence_length entries
            location_data = location_data.sort_values('timestamp').tail(sequence_length)
            
            # Check if we have all required features
            missing_features = [f for f in ENHANCED_FEATURES if f not in location_data.columns]
            if missing_features:
                # Try to impute missing features with reasonable defaults
                for feature in missing_features:
                    if feature == 'cape':
                        location_data[feature] = 500  # Moderate CAPE
                    elif feature == 'lifted_index':
                        location_data[feature] = 0    # Neutral stability
                    elif feature == 'k_index':
                        location_data[feature] = 25   # Moderate thunderstorm potential
                    elif feature == 'precipitable_water':
                        location_data[feature] = 30   # Moderate moisture
                    elif feature == 'dewpoint':
                        # Estimate dewpoint from temperature and humidity if available
                        if 'temperature' in location_data.columns and 'humidity' in location_data.columns:
                            t = location_data['temperature'].values
                            rh = location_data['humidity'].values
                            # Magnus approximation
                            a = 17.27
                            b = 237.7
                            alpha = ((a * t) / (b + t)) + np.log(rh/100.0)
                            location_data[feature] = (b * alpha) / (a - alpha)
                        else:
                            location_data[feature] = 15  # Default value
                    elif feature == 'vertical_temp_diff':
                        location_data[feature] = 15  # Default lapse rate
                    elif feature == 'vertical_wind_shear':
                        location_data[feature] = 20  # Moderate shear
                    elif feature == 'relative_humidity_850mb':
                        location_data[feature] = 70  # Moderately humid mid-levels
                    elif feature == 'relative_humidity_500mb':
                        location_data[feature] = 50  # Moderate upper-level humidity
                    else:
                        # For any other missing features, use a reasonable default
                        location_data[feature] = 0
            
            # Extract features in the correct order
            feature_data = location_data[ENHANCED_FEATURES].values
            
            prediction_inputs.append(feature_data)
            location_info.append({'latitude': lat, 'longitude': lon})
    
    return np.array(prediction_inputs), location_info

def generate_predictions(latest_data):
    """Generate thunderstorm predictions for Uttar Pradesh"""
    # Load the model
    try:
        model = load_model()
    except FileNotFoundError:
        print("Model not found. Using statistical prediction method instead.")
        return generate_statistical_predictions(latest_data)
    
    # Prepare the input data
    X_pred, locations = prepare_input_data(latest_data)
    
    if len(X_pred) == 0:
        print("No valid input sequences could be created.")
        return pd.DataFrame()
    
    # Get predictions
    predictions = model.predict(X_pred)
    
    # Create prediction DataFrame
    latest_timestamp = latest_data['timestamp'].max()
    forecast_times = [pd.Timestamp(latest_timestamp) + pd.Timedelta(hours=i+1) 
                      for i in range(FORECAST_HORIZON)]
    
    forecast_results = []
    
    for i, loc in enumerate(locations):
        pred_prob = float(predictions[i][0])
        
        # Add prediction for each forecast hour (with decaying certainty)
        for hour, forecast_time in enumerate(forecast_times):
            # Apply temporal decay to prediction confidence
            # Further in future = less certain
            decay_factor = 1.0 - (hour / FORECAST_HORIZON) * 0.5
            hourly_prob = pred_prob * decay_factor
            
            # Also apply seasonal adjustments
            current_month = datetime.now().month
            if 6 <= current_month <= 9:  # Monsoon season
                season_factor = 1.2  # Increase probability in monsoon season
            elif current_month in [4, 5, 10]:  # Pre/post monsoon
                season_factor = 1.0  # Normal probability
            else:  # Dry season
                season_factor = 0.7  # Decrease probability in dry season
                
            # Apply time-of-day adjustments (thunderstorms more common in afternoon/evening)
            hour_of_day = (forecast_time.hour + 5) % 24  # Convert UTC to IST (approx)
            if 12 <= hour_of_day <= 18:  # Afternoon/evening in IST
                time_factor = 1.3  # Increase probability
            elif 19 <= hour_of_day <= 23:  # Evening
                time_factor = 1.1  # Slight increase
            else:  # Night/morning
                time_factor = 0.8  # Decrease probability
            
            # Calculate final probability with all factors
            final_prob = hourly_prob * season_factor * time_factor
            
            # Cap between 0 and 1
            final_prob = max(0.0, min(1.0, final_prob))
            
            forecast_results.append({
                'latitude': loc['latitude'],
                'longitude': loc['longitude'],
                'forecast_time': forecast_time,
                'hours_ahead': hour + 1,
                'lightning_probability': final_prob,
                'prediction_generated': latest_timestamp,
                'risk_level': get_risk_level(final_prob)
            })
    
    return pd.DataFrame(forecast_results)

def generate_statistical_predictions(latest_data):
    """Generate predictions using statistical methods when model is not available"""
    print("Using statistical method for predictions")
    
    # Get unique locations
    locations = latest_data[['latitude', 'longitude']].drop_duplicates().values
    
    # Current timestamp
    latest_timestamp = latest_data['timestamp'].max()
    forecast_times = [pd.Timestamp(latest_timestamp) + pd.Timedelta(hours=i+1) 
                     for i in range(FORECAST_HORIZON)]
    
    forecast_results = []
    
    # For each location
    for lat, lon in locations:
        location_data = latest_data[(latest_data['latitude'] == lat) & 
                                   (latest_data['longitude'] == lon)]
        
        # Default base probability
        base_prob = 0.1
        
        # If we have observed lightning, increase probability
        if 'lightning_observed' in location_data.columns and location_data['lightning_observed'].max() > 0:
            base_prob = 0.8
        
        # Check for high CAPE
        if 'cape' in location_data.columns:
            max_cape = location_data['cape'].max()
            if max_cape > 2000:
                base_prob += 0.3
            elif max_cape > 1000:
                base_prob += 0.2
            elif max_cape > 500:
                base_prob += 0.1
        
        # Check for negative lifted index
        if 'lifted_index' in location_data.columns:
            min_li = location_data['lifted_index'].min()
            if min_li < -5:
                base_prob += 0.2
            elif min_li < 0:
                base_prob += 0.1
        
        # Spatial factor (eastern UP gets more storms)
        east_factor = (lon - UP_BOUNDS['min_lon']) / (UP_BOUNDS['max_lon'] - UP_BOUNDS['min_lon'])
        base_prob += 0.2 * east_factor
        
        # Add prediction for each forecast hour
        for hour, forecast_time in enumerate(forecast_times):
            # Decay probability with time
            decay_factor = 1.0 - (hour / FORECAST_HORIZON) * 0.5
            hourly_prob = base_prob * decay_factor
            
            # Time of day adjustment (more storms in afternoon/evening IST)
            hour_of_day = (forecast_time.hour + 5) % 24  # UTC to IST
            if 12 <= hour_of_day <= 18:  # Afternoon/evening
                time_factor = 1.3
            elif 19 <= hour_of_day <= 23:  # Evening
                time_factor = 1.1
            else:  # Night/morning
                time_factor = 0.7
                
            hourly_prob *= time_factor
            
            # Cap between 0 and 1
            hourly_prob = max(0.0, min(1.0, hourly_prob))
            
            forecast_results.append({
                'latitude': lat,
                'longitude': lon,
                'forecast_time': forecast_time,
                'hours_ahead': hour + 1,
                'lightning_probability': hourly_prob,
                'prediction_generated': latest_timestamp,
                'risk_level': get_risk_level(hourly_prob)
            })
    
    return pd.DataFrame(forecast_results)

def get_risk_level(probability):
    """Convert probability to risk level category"""
    if probability < 0.2:
        return 'Low'
    elif probability < 0.4:
        return 'Moderate'
    elif probability < 0.6:
        return 'Elevated'
    elif probability < 0.8:
        return 'High'
    else:
        return 'Severe'

if __name__ == "__main__":
    # This is just for testing - in production this would use live data
    from preprocessing.data_fusion import DataFusionProcessor
    
    processor = DataFusionProcessor()
    latest_data = processor.process_and_save()[0]
    
    if latest_data is not None:
        predictions = generate_predictions(latest_data)
        print(f"Generated {len(predictions)} predictions")
        print(predictions.head())
    else:
        print("No data available for prediction")