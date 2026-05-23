import os
import time
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
import sys
sys.path.append('..')
from config import IMD_API_KEY, IMD_API_URL, RAW_DATA_DIR, UP_BOUNDS

class IMDDataCollector:
    def __init__(self):
        self.api_key = IMD_API_KEY
        self.api_url = IMD_API_URL
        self.data_dir = RAW_DATA_DIR
        self.up_bounds = UP_BOUNDS
        
    def fetch_imd_data(self):
        """Fetch IMD weather data for Uttar Pradesh"""
        # In a real implementation, you would use the actual IMD API
        # For now, we'll create a simulated response based on typical IMD data format
        
        # API endpoints for different data types
        data_types = ['current', 'forecast', 'lightning', 'radar']
        
        all_data = {}
        for data_type in data_types:
            try:
                # In a production system, you would make actual API calls
                # For now, simulating a response
                
                # Simulated API call with authentication
                # headers = {'Authorization': f'Bearer {self.api_key}'}
                # params = {
                #     'state': 'UTTAR PRADESH',
                #     'format': 'json'
                # }
                # response = requests.get(f"{self.api_url}/{data_type}", headers=headers, params=params)
                # response.raise_for_status()
                # all_data[data_type] = response.json()
                
                # For simulation, creating mock IMD data
                all_data[data_type] = self._generate_mock_imd_data(data_type)
                
                time.sleep(1)  # Polite delay between requests
            except Exception as e:
                print(f"Error fetching IMD {data_type} data: {e}")
        
        return all_data
    
    def _generate_mock_imd_data(self, data_type):
        """Generate mock IMD data for simulation purposes"""
        # This would be replaced with actual API responses in production
        
        # Create a grid for Uttar Pradesh
        grid_points = []
        for lat in range(int(self.up_bounds['min_lat']*10), int(self.up_bounds['max_lat']*10)+1, 5):
            for lon in range(int(self.up_bounds['min_lon']*10), int(self.up_bounds['max_lon']*10)+1, 5):
                grid_points.append({
                    'latitude': lat/10.0,
                    'longitude': lon/10.0
                })
        
        # Current timestamp
        now = datetime.now()
        
        if data_type == 'current':
            # Current weather observations
            return {
                'timestamp': now.isoformat(),
                'stations': [
                    {
                        'station_id': f"UP{i:03d}",
                        'latitude': point['latitude'],
                        'longitude': point['longitude'],
                        'temperature': round(25 + 10 * ((point['latitude'] - self.up_bounds['min_lat']) / 
                                                 (self.up_bounds['max_lat'] - self.up_bounds['min_lat'])), 1),
                        'humidity': round(50 + 40 * ((point['longitude'] - self.up_bounds['min_lon']) / 
                                               (self.up_bounds['max_lon'] - self.up_bounds['min_lon'])), 1),
                        'pressure': round(1000 + 20 * ((point['latitude'] - self.up_bounds['min_lat']) / 
                                                 (self.up_bounds['max_lat'] - self.up_bounds['min_lat'])), 1),
                        'wind_speed': round(5 + 15 * ((point['longitude'] - self.up_bounds['min_lon']) / 
                                                (self.up_bounds['max_lon'] - self.up_bounds['min_lon'])), 1),
                        'wind_direction': round(360 * ((point['latitude'] + point['longitude']) % 1)),
                        'rainfall_1hr': round(5 * ((point['latitude'] + point['longitude']) % 1), 1),
                        'weather_condition': 'Thunderstorm' if ((point['latitude'] + point['longitude']) % 1) > 0.8 else 
                                             'Rain' if ((point['latitude'] + point['longitude']) % 1) > 0.6 else
                                             'Cloudy' if ((point['latitude'] + point['longitude']) % 1) > 0.4 else 'Clear'
                    } for i, point in enumerate(grid_points)
                ]
            }
        
        elif data_type == 'forecast':
            # Weather forecasts
            forecasts = []
            for hour in range(1, 25):  # 24-hour forecast
                hour_data = []
                for point in grid_points:
                    # Create some spatial and temporal variation
                    variation = (point['latitude'] + point['longitude'] + hour/24) % 1
                    
                    hour_data.append({
                        'latitude': point['latitude'],
                        'longitude': point['longitude'],
                        'forecast_hour': hour,
                        'temperature': round(25 + 10 * variation, 1),
                        'humidity': round(50 + 40 * variation, 1),
                        'pressure': round(1000 + 20 * variation, 1),
                        'wind_speed': round(5 + 15 * variation, 1),
                        'wind_direction': round(360 * variation),
                        'precipitation_probability': round(100 * variation, 1),
                        'weather_condition': 'Thunderstorm' if variation > 0.8 else 
                                             'Rain' if variation > 0.6 else
                                             'Cloudy' if variation > 0.4 else 'Clear'
                    })
                
                forecasts.append({
                    'forecast_time': (now + timedelta(hours=hour)).isoformat(),
                    'forecast_data': hour_data
                })
            
            return {
                'generated_at': now.isoformat(),
                'forecasts': forecasts
            }
        
        elif data_type == 'lightning':
            # Lightning strike data
            lightning_strikes = []
            
            # Generate some random lightning strikes in the region
            for _ in range(50):  # 50 random strikes
                lat = self.up_bounds['min_lat'] + (self.up_bounds['max_lat'] - self.up_bounds['min_lat']) * random.random()
                lon = self.up_bounds['min_lon'] + (self.up_bounds['max_lon'] - self.up_bounds['min_lon']) * random.random()
                
                # More strikes in northeast UP (typical during monsoon)
                if lat > (self.up_bounds['min_lat'] + self.up_bounds['max_lat'])/2 and lon > (self.up_bounds['min_lon'] + self.up_bounds['max_lon'])/2:
                    if random.random() < 0.7:  # 70% chance of additional strike
                        strike_time = now - timedelta(minutes=int(60 * random.random()))
                        lightning_strikes.append({
                            'strike_time': strike_time.isoformat(),
                            'latitude': lat,
                            'longitude': lon,
                            'amplitude': round(-10 - 90 * random.random(), 1),  # kA
                            'type': 'cloud_to_ground' if random.random() < 0.8 else 'intra_cloud'
                        })
            
            return {
                'period_start': (now - timedelta(hours=24)).isoformat(),
                'period_end': now.isoformat(),
                'total_strikes': len(lightning_strikes),
                'strikes': lightning_strikes
            }
        
        elif data_type == 'radar':
            # Radar reflectivity data (simplified)
            reflectivity_grid = []
            
            for lat in range(int(self.up_bounds['min_lat'] * 10), int(self.up_bounds['max_lat'] * 10) + 1, 1):
                for lon in range(int(self.up_bounds['min_lon'] * 10), int(self.up_bounds['max_lon'] * 10) + 1, 1):
                    lat_val = lat / 10.0
                    lon_val = lon / 10.0
                    
                    # Create a pattern with higher reflectivity in certain areas
                    # This would represent storm cells in a real radar image
                    dist_from_center = ((lat_val - (self.up_bounds['min_lat'] + self.up_bounds['max_lat'])/2)**2 + 
                                        (lon_val - (self.up_bounds['min_lon'] + self.up_bounds['max_lon'])/2)**2)**0.5
                    
                    # Add some random variations for realistic storm cells
                    variation = 0.5 * random.random()
                    reflectivity = max(0, 60 - 10 * dist_from_center + 20 * variation)
                    
                    if reflectivity > 5:  # Only include significant reflectivity
                        reflectivity_grid.append({
                            'latitude': lat_val,
                            'longitude': lon_val,
                            'reflectivity_dbz': round(reflectivity, 1)
                        })
            
            return {
                'timestamp': now.isoformat(),
                'radar_site': 'LUCKNOW',
                'scan_type': 'composite',
                'data': reflectivity_grid
            }
        
        return {}  # Default empty response for unknown data types
    
    def collect_and_save(self):
        """Collect and save IMD data"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M')
        imd_data = self.fetch_imd_data()
        
        if not imd_data:
            print("No IMD data collected")
            return None
            
        # Save the data
        filename = os.path.join(self.data_dir, f'imd_data_{timestamp}.json')
        with open(filename, 'w') as f:
            json.dump(imd_data, f)
        print(f"Saved IMD data to {filename}")
        
        return imd_data