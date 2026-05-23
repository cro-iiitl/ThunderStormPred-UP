# app.py
import os
import time
import json
import requests
import pandas as pd
import numpy as np
import math
import random
from datetime import datetime, timedelta
import folium
from folium.plugins import HeatMap, MarkerCluster
import threading

# Import from project modules
from config import DEBUG, SECRET_KEY, PROCESSED_DATA_DIR, UP_BOUNDS, FEATURES

# Try to import the new modules, fall back to existing ones if not available
try:
    from preprocessing.data_fusion import DataFusionProcessor
    NEW_STRUCTURE = True
    print("Using enhanced data processing system")
except ImportError:
    from preprocessing.data_processor import WeatherDataProcessor
    NEW_STRUCTURE = False
    print("Using original data processing system")

try:
    from preprocessing.data_collector import WeatherDataCollector
    from models.predict import generate_predictions
except ImportError:
    print("ERROR: Required modules not found. Make sure data_collector.py and predict.py exist.")
    exit(1)

from flask import Flask, render_template, jsonify, request

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['DEBUG'] = DEBUG

# Global variables to store latest data
latest_weather_data = None
latest_predictions = None
data_last_updated = None

# Flag to use mock data for development
USE_MOCK_DATA = False  # Set to True to use mock data instead of API calls
# In app.py, add this at the top
USE_TRAINED_MODEL = False

# And in the predict function, ensure it's loading your trained model
def generate_predictions(latest_data):
    # Load the model
    try:
        model = load_model(os.path.join(MODEL_DIR, 'thunderstorm_prediction_model.h5'))
        print("Loaded trained model for prediction")
    except:
        print("No trained model found, using statistical prediction")
        return generate_statistical_predictions(latest_data)
    
def generate_mock_data():
    """Generate mock data for development when API access is limited"""
    global latest_weather_data, latest_predictions, data_last_updated
    
    print("Generating mock data for development...")
    
    # Create mock weather data
    mock_data = []
    
    # Create a grid for Uttar Pradesh
    lats = np.arange(UP_BOUNDS['min_lat'], UP_BOUNDS['max_lat'], 0.5)
    lons = np.arange(UP_BOUNDS['min_lon'], UP_BOUNDS['max_lon'], 0.5)
    
    # Generate random weather data for each point
    now = datetime.now()
    for lat in lats:
        for lon in lons:
            # Generate data for the last 24 hours
            for hour in range(24):
                timestamp = now - timedelta(hours=23-hour)
                
                # Eastern UP tends to have more thunderstorms
                east_factor = (lon - UP_BOUNDS['min_lon']) / (UP_BOUNDS['max_lon'] - UP_BOUNDS['min_lon'])
                
                # Current season affects thunderstorm likelihood
                current_month = now.month
                if 6 <= current_month <= 9:  # Monsoon season
                    season_factor = 1.5
                elif current_month in [4, 5, 10]:  # Pre/post monsoon
                    season_factor = 1.0
                else:  # Dry season
                    season_factor = 0.5
                
                # Time of day affects thunderstorm likelihood (afternoon/evening peak)
                hour_of_day = timestamp.hour
                if 12 <= hour_of_day <= 18:  # Afternoon/evening
                    time_factor = 1.5
                else:  # Night/morning
                    time_factor = 0.8
                
                # Random factor for variation
                random_factor = random.uniform(0.7, 1.3)
                
                # Calculate thunderstorm probability
                ts_prob = min(0.95, east_factor * season_factor * time_factor * random_factor * 0.8)
                
                # Generate data point
                data_point = {
                    'timestamp': timestamp,
                    'latitude': lat,
                    'longitude': lon,
                    'temperature': random.uniform(20, 35),
                    'humidity': random.uniform(40, 90),
                    'pressure': random.uniform(995, 1015),
                    'wind_speed': random.uniform(0, 25),
                    'wind_direction': random.uniform(0, 360),
                    'precipitation': random.uniform(0, 10) if random.random() < 0.3 else 0,
                    'cloud_cover': random.uniform(0, 100),
                    'lightning_observed': 1 if random.random() < ts_prob else 0
                }
                
                mock_data.append(data_point)
    
    # Convert to DataFrame
    latest_weather_data = pd.DataFrame(mock_data)
    
    # Generate mock predictions
    # Create prediction points at regular intervals
    forecast_times = [now + timedelta(hours=i+1) for i in range(12)]
    
    mock_predictions = []
    for lat in lats:
        for lon in lons:
            # Base probability influenced by location (more in eastern UP)
            east_factor = (lon - UP_BOUNDS['min_lon']) / (UP_BOUNDS['max_lon'] - UP_BOUNDS['min_lon'])
            season_factor = 1.0 if 6 <= current_month <= 9 else 0.5  # More in monsoon
            random_factor = random.random()
            base_prob = min(0.95, east_factor * season_factor * random_factor * 0.8)
            
            for hour, forecast_time in enumerate(forecast_times):
                # Decay factor for future hours
                decay_factor = 1.0 - (hour / 12) * 0.5
                
                # Time of day factor
                hour_of_day = forecast_time.hour
                if 12 <= hour_of_day <= 18:
                    time_factor = 1.3
                else:
                    time_factor = 0.7
                
                # Calculate probability
                probability = min(0.95, base_prob * decay_factor * time_factor)
                
                # Add some spatial clusters of high risk
                if hour == 0 and random.random() < 0.05:  # 5% chance of a storm cell on first hour
                    # Create a storm cell
                    probability = random.uniform(0.7, 0.95)
                
                mock_predictions.append({
                    'latitude': lat,
                    'longitude': lon,
                    'forecast_time': forecast_time,
                    'hours_ahead': hour + 1,
                    'lightning_probability': probability,
                    'prediction_generated': now,
                    'risk_level': get_risk_level(probability)
                })
    
    latest_predictions = pd.DataFrame(mock_predictions)
    data_last_updated = now
    
    print(f"Generated mock data with {len(latest_weather_data)} weather points and {len(latest_predictions)} predictions")

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

def start_data_collection_thread():
    """Start a background thread for data collection"""
    # If using mock data, generate it once and return
    if USE_MOCK_DATA:
        generate_mock_data()
        return
    
    # Initialize data collectors
    collector = WeatherDataCollector()
    
    if NEW_STRUCTURE:
        processor = DataFusionProcessor()
    else:
        processor = WeatherDataProcessor()
    
    def collection_loop():
        global latest_weather_data, latest_predictions, data_last_updated
        
        while True:
            try:
                print("Starting data collection cycle...")
                
                # Collect data
                weather_data = collector.collect_and_save()
                
                if weather_data:
                    print(f"Collected data from {len(weather_data)} locations")
                    
                    # Process the data
                    if NEW_STRUCTURE:
                        # For new structure with DataFusionProcessor
                        collected_data = {'openweather': [{'file': 'latest.json', 'data': weather_data}]}
                        df, _, _ = processor.process_and_save(collected_data)
                    else:
                        # For old structure with WeatherDataProcessor
                        df, _, _ = processor.process_and_save()
                    
                    if df is not None and not df.empty:
                        # Generate new predictions
                        print("Generating predictions...")
                        predictions = generate_predictions(df)
                        
                        # Update global variables
                        latest_weather_data = df
                        latest_predictions = predictions
                        data_last_updated = datetime.now()
                        
                        print(f"Data updated at {data_last_updated}")
                    else:
                        print("No data available after processing")
                else:
                    print("No data collected, using mock data as fallback")
                    # Generate mock data if API failed
                    generate_mock_data()
                
                # Sleep for one hour before next update
                print("Waiting for next data collection cycle...")
                time.sleep(3600)
                
            except Exception as e:
                print(f"Error in data collection thread: {e}")
                # Generate mock data if there was an error
                if latest_weather_data is None:
                    print("Generating mock data due to collection error")
                    generate_mock_data()
                time.sleep(300)  # Sleep for 5 minutes before retrying
    
    # Start the thread
    thread = threading.Thread(target=collection_loop, daemon=True)
    thread.start()
    print("Data collection thread started")

# Routes
@app.route('/')
def index():
    """Main dashboard page"""
    # If we don't have data yet, provide a message
    if latest_predictions is None:
        return render_template('index.html', 
                                has_data=False,
                                message="Data collection in progress. Please check back later.")
    
    # Get the current hour's predictions
    current_hour = latest_predictions[latest_predictions['hours_ahead'] == 1]
    
    # Create the map
    up_map = create_prediction_map(current_hour)
    
    # Get summary statistics
    stats = {
        'high_risk_areas': len(current_hour[current_hour['risk_level'].isin(['High', 'Severe'])]),
        'average_probability': current_hour['lightning_probability'].mean() * 100,
        'data_updated': data_last_updated.strftime('%Y-%m-%d %H:%M'),
        'forecast_time': current_hour['forecast_time'].iloc[0].strftime('%Y-%m-%d %H:%M') if not current_hour.empty else "Unknown"
    }
    
    # Get risk level distribution
    risk_counts = current_hour['risk_level'].value_counts().to_dict()
    
    return render_template('index.html',
                          has_data=True,
                          map=up_map._repr_html_(),
                          stats=stats,
                          risk_counts=risk_counts)

@app.route('/forecast')
def forecast():
    """Detailed forecast page"""
    if latest_predictions is None:
        return render_template('forecast.html', 
                               has_data=False,
                               message="Data collection in progress. Please check back later.")
    
    # Get hourly predictions
    hours = sorted(latest_predictions['hours_ahead'].unique())
    hourly_maps = {}
    
    for hour in hours:
        hour_data = latest_predictions[latest_predictions['hours_ahead'] == hour]
        hour_map = create_prediction_map(hour_data)
        hourly_maps[hour] = {
            'map': hour_map._repr_html_(),
            'forecast_time': hour_data['forecast_time'].iloc[0].strftime('%Y-%m-%d %H:%M') if not hour_data.empty else "Unknown",
            'high_risk_areas': len(hour_data[hour_data['risk_level'].isin(['High', 'Severe'])])
        }
    
    return render_template('forecast.html',
                          has_data=True,
                          hourly_maps=hourly_maps,
                          hours=hours)

@app.route('/historical')
def historical():
    """Historical data visualization page"""
    if latest_weather_data is None:
        return render_template('historical.html', 
                               has_data=False,
                               message="Data collection in progress. Please check back later.")
    
    # Get data for the last 7 days
    last_week = latest_weather_data[
        latest_weather_data['timestamp'] >= (datetime.now() - timedelta(days=7))
    ]
    
    # Aggregate data by day and count thunderstorm occurrences
    daily_counts = last_week.groupby(last_week['timestamp'].dt.date)['lightning_observed'].sum()
    dates = [str(date) for date in daily_counts.index]
    counts = daily_counts.values.tolist()
    
    # Get historical accuracy if available
    # This would require comparing past predictions with actual observations
    # For simplicity, we'll just return dummy data here
    accuracy_data = {
        'dates': dates,
        'accuracy': np.random.uniform(0.7, 0.95, size=len(dates)).tolist()
    }
    
    return render_template('historical.html',
                           has_data=True,
                           dates=json.dumps(dates),
                           counts=json.dumps(counts),
                           accuracy_data=json.dumps(accuracy_data))

@app.route('/lightning')
def lightning():
    """Real-time lightning visualization page"""
    # If we don't have data yet, provide a message
    if latest_predictions is None:
        return render_template('lightning.html', 
                               has_data=False,
                               message="Data collection in progress. Please check back later.")
    
    # Create lightning data for the past 24 hours
    lightning_data = generate_lightning_data()
    
    # Get summary statistics
    stats = {
        'strike_count': len(lightning_data),
        'average_intensity': round(np.mean([strike['intensity'] for strike in lightning_data])),
        'max_intensity': max([strike['intensity'] for strike in lightning_data]),
        'data_updated': data_last_updated.strftime('%Y-%m-%d %H:%M') if data_last_updated else "Unknown",
    }
    
    # Create the map
    lightning_map = create_lightning_map(lightning_data)
    
    return render_template('lightning.html',
                          has_data=True,
                          map=lightning_map._repr_html_(),
                          stats=stats,
                          lightning_data=json.dumps(lightning_data[:20]))  # Send first 20 for table display

@app.route('/ildn')
def ildn_lightning_tracker():
    """ILDN-style lightning tracker page"""
    if latest_predictions is None:
        return render_template('ildn.html', 
                              has_data=False,
                              message="Data collection in progress. Please check back later.")
    
    # Get current date/time for default range
    now = datetime.now()
    three_hours_ago = now - timedelta(hours=3)
    
    # Create default date range
    date_range = {
        'start': three_hours_ago.strftime('%Y-%m-%d'),
        'end': now.strftime('%Y-%m-%d')
    }
    
    # Create default time range
    time_range = {
        'start': three_hours_ago.strftime('%H:%M'),
        'end': now.strftime('%H:%M')
    }
    
    # Generate lightning data with more detailed attributes
    lightning_data = generate_enhanced_lightning_data()
    
    # Calculate statistics
    stats = {
        'total_strikes': len(lightning_data),
        'cg_strikes': len([s for s in lightning_data if s['type'] == 'cloud-to-ground']),
        'ic_strikes': len([s for s in lightning_data if s['type'] == 'intra-cloud']),
        'high_intensity': len([s for s in lightning_data if s['intensity'] >= 80]),
        'data_updated': now.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Get alert regions based on strike density
    alert_regions = generate_alert_regions(lightning_data)
    
    return render_template('ildn.html',
                          has_data=True,
                          stats=stats,
                          date_range=date_range,
                          time_range=time_range,
                          regions=get_available_regions(),
                          lightning_data=json.dumps(lightning_data[:100]),  # Limit to 100 for performance
                          alert_regions=json.dumps(alert_regions))

@app.route('/api/current_predictions')
def api_current_predictions():
    """API endpoint for current predictions"""
    if latest_predictions is None:
        return jsonify({'error': 'No prediction data available yet'})
    
    current_hour = latest_predictions[latest_predictions['hours_ahead'] == 1]
    
    # Convert to GeoJSON
    features = []
    for _, row in current_hour.iterrows():
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [row['longitude'], row['latitude']]
            },
            'properties': {
                'probability': float(row['lightning_probability']),
                'risk_level': row['risk_level'],
                'forecast_time': row['forecast_time'].strftime('%Y-%m-%d %H:%M')
            }
        }
        features.append(feature)
    
    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }
    
    return jsonify(geojson)

@app.route('/api/lightning_data')
def api_lightning_data():
    """API endpoint for lightning strike data"""
    if latest_predictions is None:
        return jsonify({'error': 'No prediction data available yet'})
    
    # Get time range filter
    time_range = request.args.get('timeRange', 'day')  # 'hour', 'day', 'week'
    
    # Generate lightning data
    all_lightning_data = generate_lightning_data()
    
    # Filter by time range
    now = datetime.now()
    if time_range == 'hour':
        cutoff = now - timedelta(hours=1)
    elif time_range == 'week':
        cutoff = now - timedelta(days=7)
    else:  # 'day' is default
        cutoff = now - timedelta(days=1)
    
    filtered_data = [strike for strike in all_lightning_data 
                     if datetime.fromisoformat(strike['timestamp']) >= cutoff]
    
    return jsonify({
        'count': len(filtered_data),
        'strikes': filtered_data
    })

def generate_lightning_data():
    """Generate lightning strike data based on prediction model"""
    # Use the existing prediction data to create realistic lightning distribution
    # Areas with higher thunderstorm probability should have more lightning strikes
    
    lightning_strikes = []
    
    # Only proceed if we have prediction data
    if latest_predictions is None:
        return lightning_strikes
    
    # Filter for locations with higher probabilities
    high_risk_areas = latest_predictions[latest_predictions['lightning_probability'] > 0.4]
    
    # Current time
    now = datetime.now()
    
    # Create a strike ID counter
    strike_id = 1
    
    # For each high risk area, generate some strikes
    for _, area in high_risk_areas.iterrows():
        # Number of strikes based on probability
        num_strikes = int(area['lightning_probability'] * 20)  # Up to 20 strikes per location
        
        for _ in range(num_strikes):
            # Randomize location slightly around the prediction point
            lat_jitter = random.uniform(-0.05, 0.05)
            lon_jitter = random.uniform(-0.05, 0.05)
            lat = area['latitude'] + lat_jitter
            lon = area['longitude'] + lon_jitter
            
            # Random time in the last 24 hours, weighted toward recent times for active areas
            if area['lightning_probability'] > 0.7:
                # More recent strikes for high probability areas
                hours_ago = random.uniform(0, 6)
            else:
                hours_ago = random.uniform(0, 24)
                
            strike_time = now - timedelta(hours=hours_ago)
            
            # Intensity related to probability
            base_intensity = area['lightning_probability'] * 50
            intensity = int(base_intensity + random.uniform(0, 50))
            
            # Determine strike type (cloud-to-ground vs intra-cloud)
            strike_type = 'cloud-to-ground' if random.random() < 0.8 else 'intra-cloud'
            
            # Polarity (mostly negative)
            polarity = 'negative' if random.random() < 0.9 else 'positive'
            
            lightning_strikes.append({
                'id': strike_id,
                'latitude': lat,
                'longitude': lon,
                'timestamp': strike_time.isoformat(),
                'intensity': intensity,
                'type': strike_type,
                'polarity': polarity
            })
            
            strike_id += 1
    
    # Sort by time (newest first)
    lightning_strikes.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return lightning_strikes

def generate_enhanced_lightning_data():
    """Generate enhanced lightning strike data with ILDN-compatible attributes"""
    # Filter for high risk areas
    high_risk_areas = latest_predictions[latest_predictions['lightning_probability'] > 0.4]
    
    lightning_strikes = []
    strike_id = 1
    now = datetime.now()
    
    # Clustering parameters to create realistic patterns
    cluster_centers = []
    if len(high_risk_areas) > 5:
        # Create 3-5 cluster centers from high risk areas
        sample_size = min(5, len(high_risk_areas))
        cluster_samples = high_risk_areas.sample(sample_size)
        
        for _, area in cluster_samples.iterrows():
            cluster_centers.append({
                'latitude': area['latitude'],
                'longitude': area['longitude'],
                'strength': area['lightning_probability']
            })
    else:
        # Use all high risk areas as cluster centers
        for _, area in high_risk_areas.iterrows():
            cluster_centers.append({
                'latitude': area['latitude'],
                'longitude': area['longitude'],
                'strength': area['lightning_probability']
            })
    
    # Generate strikes around cluster centers
    for center in cluster_centers:
        # Number of strikes proportional to storm strength
        num_strikes = int(center['strength'] * 50) + 5
        
        for _ in range(num_strikes):
            # Distance from center (most strikes close to center)
            distance = random.gammavariate(1, 0.02)  # Concentrated near center, with tail
            
            # Random direction
            angle = random.uniform(0, 2 * math.pi)
            
            # Calculate new position
            lat_offset = distance * math.cos(angle)
            lon_offset = distance * math.sin(angle)
            
            latitude = center['latitude'] + lat_offset
            longitude = center['longitude'] + lon_offset
            
            # Skip if outside Uttar Pradesh bounds
            if (latitude < UP_BOUNDS['min_lat'] or latitude > UP_BOUNDS['max_lat'] or
                longitude < UP_BOUNDS['min_lon'] or longitude > UP_BOUNDS['max_lon']):
                continue
            
            # Random time in last 3 hours, with more recent times more likely
            time_offset = random.triangular(0, 3*60*60, 3*60*60)  # seconds, weighted toward recent
            strike_time = now - timedelta(seconds=time_offset)
            
            # Determine strike type (CG vs IC)
            # Higher probability of CG for stronger storms
            is_cg = random.random() < (0.6 + center['strength'] * 0.3)
            
            # Intensity varies by type and cluster strength
            if is_cg:
                # CG strikes tend to be stronger
                intensity_base = 40 + center['strength'] * 60
                intensity = random.normalvariate(intensity_base, 15)
                intensity = max(10, min(200, intensity))  # Clamp values
            else:
                # IC strikes tend to be weaker
                intensity_base = 20 + center['strength'] * 40
                intensity = random.normalvariate(intensity_base, 10)
                intensity = max(5, min(100, intensity))  # Clamp values
            
            # Determine polarity (mostly negative)
            polarity = 'negative' if random.random() < 0.9 else 'positive'
            
            # Enhanced attributes for ILDN compatibility
            peak_current = intensity  # kA
            multiplicity = random.choices([1, 2, 3, 4, 5], weights=[60, 20, 10, 7, 3])[0]
            
            lightning_strikes.append({
                'id': f"strike-{strike_id}",
                'latitude': latitude,
                'longitude': longitude,
                'timestamp': strike_time.isoformat(),
                'intensity': round(intensity, 1),
                'type': 'cloud-to-ground' if is_cg else 'intra-cloud',
                'polarity': polarity,
                'peak_current': round(peak_current, 1),
                'multiplicity': multiplicity,
                'cluster_id': cluster_centers.index(center)
            })
            
            strike_id += 1
    
    # Sort by time (newest first)
    lightning_strikes.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return lightning_strikes

def generate_alert_regions(lightning_data):
    """Generate alert regions based on lightning strike density"""
    # Skip if no data
    if not lightning_data:
        return []
    
    # Grid the region
    grid_size = 0.5  # degrees
    lat_bins = np.arange(UP_BOUNDS['min_lat'], UP_BOUNDS['max_lat'] + grid_size, grid_size)
    lon_bins = np.arange(UP_BOUNDS['min_lon'], UP_BOUNDS['max_lon'] + grid_size, grid_size)
    
    # Count strikes in each grid cell
    grid_counts = {}
    for strike in lightning_data:
        # Find grid cell
        lat_bin = math.floor((strike['latitude'] - UP_BOUNDS['min_lat']) / grid_size) * grid_size + UP_BOUNDS['min_lat']
        lon_bin = math.floor((strike['longitude'] - UP_BOUNDS['min_lon']) / grid_size) * grid_size + UP_BOUNDS['min_lon']
        
        grid_key = f"{lat_bin:.2f}_{lon_bin:.2f}"
        
        if grid_key in grid_counts:
            grid_counts[grid_key]['count'] += 1
            
            # Track CG vs IC counts
            if strike['type'] == 'cloud-to-ground':
                grid_counts[grid_key]['cg_count'] += 1
            else:
                grid_counts[grid_key]['ic_count'] += 1
                
            # Track average intensity
            grid_counts[grid_key]['total_intensity'] += strike['intensity']
            
            # Track most recent strike
            strike_time = datetime.fromisoformat(strike['timestamp'])
            if strike_time > grid_counts[grid_key]['latest_time']:
                grid_counts[grid_key]['latest_time'] = strike_time
        else:
            strike_time = datetime.fromisoformat(strike['timestamp'])
            grid_counts[grid_key] = {
                'count': 1,
                'cg_count': 1 if strike['type'] == 'cloud-to-ground' else 0,
                'ic_count': 0 if strike['type'] == 'cloud-to-ground' else 1,
                'total_intensity': strike['intensity'],
                'latest_time': strike_time,
                'latitude': lat_bin + grid_size/2,  # Center of cell
                'longitude': lon_bin + grid_size/2  # Center of cell
            }
    
    # Create alert regions for cells with significant activity
    alert_regions = []
    
    for grid_key, data in grid_counts.items():
        # Skip if fewer than 5 strikes
        if data['count'] < 5:
            continue
        
        # Calculate average intensity
        avg_intensity = data['total_intensity'] / data['count']
        
        # Determine alert level based on count and intensity
        if data['count'] >= 20 or (data['count'] >= 10 and avg_intensity >= 70):
            alert_level = 'high'
        elif data['count'] >= 10 or (data['count'] >= 5 and avg_intensity >= 60):
            alert_level = 'medium'
        else:
            alert_level = 'low'
        
        # Add minutes ago for display
        minutes_ago = int((datetime.now() - data['latest_time']).total_seconds() / 60)
        
        # Create region name based on geographical area
        # (In a real implementation, this would use a reverse geocoder)
        lat, lon = data['latitude'], data['longitude']
        region_name = get_region_name(lat, lon)
        
        alert_regions.append({
            'latitude': data['latitude'],
            'longitude': data['longitude'],
            'count': data['count'],
            'cg_count': data['cg_count'],
            'ic_count': data['ic_count'],
            'avg_intensity': round(avg_intensity, 1),
            'minutes_ago': minutes_ago,
            'alert_level': alert_level,
            'region_name': region_name
        })
    
    # Sort by alert level (high to low) and then by count (high to low)
    alert_regions.sort(key=lambda x: (-1 if x['alert_level'] == 'high' else 
                                      -0.5 if x['alert_level'] == 'medium' else 0, 
                                     -x['count']))
    
    return alert_regions

def get_region_name(lat, lon):
    """Get region name based on latitude and longitude"""
    # Dictionary of major cities in UP with their coordinates
    cities = {
        'Lucknow': (26.8467, 80.9462),
        'Kanpur': (26.4499, 80.3319),
        'Varanasi': (25.3176, 82.9739),
        'Agra': (27.1767, 78.0081),
        'Prayagraj': (25.4358, 81.8463),
        'Ghaziabad': (28.6692, 77.4538),
        'Meerut': (28.9845, 77.7064),
        'Bareilly': (28.3670, 79.4304),
        'Aligarh': (27.8974, 78.0880),
        'Moradabad': (28.8386, 78.7733)
    }
    
    # Find the closest city
    closest_city = None
    min_distance = float('inf')
    
    for city, coords in cities.items():
        city_lat, city_lon = coords
        distance = ((lat - city_lat) ** 2 + (lon - city_lon) ** 2) ** 0.5
        
        if distance < min_distance:
            min_distance = distance
            closest_city = city
    
    # If within 50km (approx 0.5 degrees), use "near [city]"
    if min_distance < 0.5:
        return f"near {closest_city}"
    else:
        # Use cardinal directions from closest city
        direction = get_cardinal_direction(lat, lon, *cities[closest_city])
        return f"{direction} of {closest_city}"

def get_cardinal_direction(lat1, lon1, lat2, lon2):
    """Get cardinal direction from point 2 to point 1"""
    lat_diff = lat1 - lat2
    lon_diff = lon1 - lon2
    
    # Calculate angle
    angle = math.degrees(math.atan2(lat_diff, lon_diff))
    
    # Convert angle to cardinal direction
    if -22.5 <= angle < 22.5:
        return "East"
    elif 22.5 <= angle < 67.5:
        return "Northeast"
    elif 67.5 <= angle < 112.5:
        return "North"
    elif 112.5 <= angle < 157.5:
        return "Northwest"
    elif 157.5 <= angle or angle < -157.5:
        return "West"
    elif -157.5 <= angle < -112.5:
        return "Southwest"
    elif -112.5 <= angle < -67.5:
        return "South"
    else:  # -67.5 <= angle < -22.5
        return "Southeast"

def get_available_regions():
    """Get available regions for the selector"""
    return [
        {'id': 'UP', 'name': 'Uttar Pradesh', 'default': True},
        {'id': 'IN', 'name': 'All India', 'default': False},
        {'id': 'BR', 'name': 'Bihar', 'default': False},
        {'id': 'MP', 'name': 'Madhya Pradesh', 'default': False},
        {'id': 'HR', 'name': 'Haryana', 'default': False}
    ]

def create_lightning_map(lightning_data):
    """Create a Folium map with lightning strike visualization"""
    # Create the base map centered on Uttar Pradesh
    center_lat = (UP_BOUNDS['min_lat'] + UP_BOUNDS['max_lat']) / 2
    center_lon = (UP_BOUNDS['min_lon'] + UP_BOUNDS['max_lon']) / 2
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles='CartoDB dark_matter')
    
    # Add Uttar Pradesh boundary
    folium.Rectangle(
        bounds=[[UP_BOUNDS['min_lat'], UP_BOUNDS['min_lon']], 
                [UP_BOUNDS['max_lat'], UP_BOUNDS['max_lon']]],
        color='white',
        fill=False,
        weight=2
    ).add_to(m)
    
    # Add major cities in Uttar Pradesh
    cities = [
        {'name': 'Lucknow', 'lat': 26.8467, 'lon': 80.9462},
        {'name': 'Kanpur', 'lat': 26.4499, 'lon': 80.3319},
        {'name': 'Varanasi', 'lat': 25.3176, 'lon': 82.9739},
        {'name': 'Agra', 'lat': 27.1767, 'lon': 78.0081},
        {'name': 'Prayagraj', 'lat': 25.4358, 'lon': 81.8463}
    ]
    
    for city in cities:
        folium.CircleMarker(
            location=[city['lat'], city['lon']],
            radius=4,
            color='white',
            fill=True,
            fill_color='white',
            fill_opacity=0.7,
            popup=city['name']
        ).add_to(m)
    
    # Create strike markers with custom icons
    for strike in lightning_data:
        # Determine color based on intensity
        if strike['intensity'] >= 80:
            color = 'red'
        elif strike['intensity'] >= 60:
            color = 'orange'
        elif strike['intensity'] >= 40:
            color = 'yellow'
        else:
            color = 'blue'
        
        # Format time
        strike_time = datetime.fromisoformat(strike['timestamp'])
        formatted_time = strike_time.strftime('%H:%M:%S')
        
        # Create popup content
        popup_text = f"""
        <strong>Lightning Strike</strong><br>
        Time: {formatted_time}<br>
        Intensity: {strike['intensity']} kA<br>
        Type: {strike['type']}<br>
        Polarity: {strike['polarity']}
        """
        
        # Use different icons for different strike types
        if strike['type'] == 'cloud-to-ground':
            icon = folium.Icon(color=color, icon='bolt', prefix='fa')
        else:
            icon = folium.Icon(color=color, icon='cloud', prefix='fa')
        
        folium.Marker(
            location=[strike['latitude'], strike['longitude']],
            popup=folium.Popup(popup_text, max_width=200),
            icon=icon
        ).add_to(m)
    
    # Add a heatmap layer for strike density
    heat_data = [[float(strike['latitude']), float(strike['longitude']), min(strike['intensity']/20, 1.0)] 
                for strike in lightning_data]
    
    if heat_data:
        HeatMap(heat_data, radius=15, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1: 'red'},
                min_opacity=0.5, blur=10).add_to(m)
    
    # Add a legend
    legend_html = '''
    <div style="position: fixed; 
        bottom: 50px; right: 50px; width: 150px; height: 160px; 
        background-color: rgba(0, 0, 0, 0.7);
        border-radius: 5px; padding: 10px; z-index: 9999;">
        <p style="color: white; margin-bottom: 5px; font-weight: bold;">Lightning Intensity</p>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 15px; height: 15px; background-color: red; margin-right: 5px;"></div>
            <span style="color: white;">80+ kA (Severe)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 15px; height: 15px; background-color: orange; margin-right: 5px;"></div>
            <span style="color: white;">60-80 kA (Strong)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 15px; height: 15px; background-color: yellow; margin-right: 5px;"></div>
            <span style="color: white;">40-60 kA (Moderate)</span>
        </div>
        <div style="display: flex; align-items: center;">
            <div style="width: 15px; height: 15px; background-color: blue; margin-right: 5px;"></div>
            <span style="color: white;">0-40 kA (Light)</span>
        </div>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def create_prediction_map(prediction_data):
    """Create a Folium map with prediction data"""
    # Create the base map centered on Uttar Pradesh
    center_lat = (UP_BOUNDS['min_lat'] + UP_BOUNDS['max_lat']) / 2
    center_lon = (UP_BOUNDS['min_lon'] + UP_BOUNDS['max_lon']) / 2
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7)
    
    # Add a heatmap layer for lightning probability
    heat_data = [[row['latitude'], row['longitude'], row['lightning_probability']] 
                for _, row in prediction_data.iterrows()]
    
    HeatMap(heat_data, radius=15, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}).add_to(m)
    
    # Add markers for high-risk areas
    high_risk = prediction_data[prediction_data['risk_level'].isin(['High', 'Severe'])]
    
    if not high_risk.empty:
        marker_cluster = MarkerCluster().add_to(m)
        
        for _, row in high_risk.iterrows():
            popup_text = f"""
            <strong>Lightning Risk: {row['risk_level']}</strong><br>
            Probability: {row['lightning_probability']:.2f}<br>
            Forecast Time: {row['forecast_time'].strftime('%Y-%m-%d %H:%M')}
            """
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_text, max_width=200),
                icon=folium.Icon(color='red', icon='bolt', prefix='fa')
            ).add_to(marker_cluster)
    
    # Add Uttar Pradesh boundary
    folium.Rectangle(
        bounds=[[UP_BOUNDS['min_lat'], UP_BOUNDS['min_lon']], 
                [UP_BOUNDS['max_lat'], UP_BOUNDS['max_lon']]],
        color='blue',
        fill=False,
        weight=2
    ).add_to(m)
    
    # Add major cities in Uttar Pradesh
    cities = [
        {'name': 'Lucknow', 'lat': 26.8467, 'lon': 80.9462},
        {'name': 'Kanpur', 'lat': 26.4499, 'lon': 80.3319},
        {'name': 'Varanasi', 'lat': 25.3176, 'lon': 82.9739},
        {'name': 'Agra', 'lat': 27.1767, 'lon': 78.0081},
        {'name': 'Prayagraj', 'lat': 25.4358, 'lon': 81.8463}
    ]
    
    for city in cities:
        folium.CircleMarker(
            location=[city['lat'], city['lon']],
            radius=4,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.7,
            popup=city['name']
        ).add_to(m)
    
    return m

if __name__ == '__main__':
    # Set this to True to use mock data instead of API calls
    USE_MOCK_DATA = True  # Change to False for production with real API calls
    
    start_data_collection_thread()
    app.run(host='0.0.0.0', port=5000)