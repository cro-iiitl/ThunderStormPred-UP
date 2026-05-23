import os
import json
import requests
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import sys
sys.path.append('..')
from config import ENTLN_API_KEY, ENTLN_API_URL, RAW_DATA_DIR, UP_BOUNDS

class LightningDataCollector:
    def __init__(self):
        self.api_key = ENTLN_API_KEY
        self.api_url = ENTLN_API_URL
        self.data_dir = RAW_DATA_DIR
        self.up_bounds = UP_BOUNDS
    
    def fetch_lightning_data(self):
        """Fetch lightning data for Uttar Pradesh"""
        # In a production system, you would use a real lightning data API
        # For simulation, we'll generate synthetic lightning data
        
        # Current time and 24 hours ago
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)
        
        # Format times
        start_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        try:
            # In a production environment, you would make a real API call:
            # params = {
            #     'start': start_str,
            #     'end': end_str,
            #     'minLat': self.up_bounds['min_lat'],
            #     'maxLat': self.up_bounds['max_lat'],
            #     'minLon': self.up_bounds['min_lon'],
            #     'maxLon': self.up_bounds['max_lon'],
            #     'apiKey': self.api_key
            # }
            # response = requests.get(self.api_url, params=params)
            # response.raise_for_status()
            # return response.json()
            
            # For simulation, generate synthetic lightning data
            return self._generate_synthetic_lightning_data(start_time, end_time)
            
        except Exception as e:
            print(f"Error fetching lightning data: {e}")
            return None
    
    def _generate_synthetic_lightning_data(self, start_time, end_time):
        """Generate synthetic lightning data for simulation purposes"""
        # Number of lightning strikes to generate
        # In monsoon season (Jun-Sep), more strikes
        current_month = datetime.now().month
        if 6 <= current_month <= 9:
            num_strikes = random.randint(500, 2000)  # Monsoon - lots of lightning
        elif current_month in [4, 5, 10]:
            num_strikes = random.randint(100, 500)   # Pre/post monsoon - moderate lightning
        else:
            num_strikes = random.randint(10, 100)    # Dry season - less lightning
        
        strikes = []
        
        # Time difference in seconds
        time_diff = int((end_time - start_time).total_seconds())
        
        # Generate random strikes with realistic patterns
        # Eastern UP typically gets more strikes
        for _ in range(num_strikes):
            # Random time within the period
            seconds_offset = random.randint(0, time_diff)
            strike_time = start_time + timedelta(seconds=seconds_offset)
            
            # Random location with bias towards eastern UP
            # Eastern UP (higher longitude) gets more strikes
            lon_bias = random.random() * 0.7  # 0-0.7 bias factor
            lon = self.up_bounds['min_lon'] + (self.up_bounds['max_lon'] - self.up_bounds['min_lon']) * (random.random() * (1 + lon_bias))
            
            # Cap longitude to max bounds
            lon = min(lon, self.up_bounds['max_lon'])
            
            # Random latitude
            lat = self.up_bounds['min_lat'] + (self.up_bounds['max_lat'] - self.up_bounds['min_lat']) * random.random()
            
            # Create clusters of strikes
            # If we're in a cluster region, add more strikes nearby
            if random.random() < 0.3:  # 30% chance of a cluster
                cluster_size = random.randint(5, 20)
                for _ in range(cluster_size):
                    # Nearby location and time
                    cluster_lat = lat + random.uniform(-0.1, 0.1)
                    cluster_lon = lon + random.uniform(-0.1, 0.1)
                    cluster_time = strike_time + timedelta(seconds=random.randint(-300, 300))
                    
                    # Ensure within bounds
                    cluster_lat = max(self.up_bounds['min_lat'], min(cluster_lat, self.up_bounds['max_lat']))
                    cluster_lon = max(self.up_bounds['min_lon'], min(cluster_lon, self.up_bounds['max_lon']))
                    
                    # Add strike data
                    strikes.append({
                        'time': cluster_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                        'latitude': cluster_lat,
                        'longitude': cluster_lon,
                        'amplitude': round(random.uniform(-150, -5), 1),  # kA
                        'type': 'cloud_to_ground' if random.random() < 0.7 else 'cloud_to_cloud',
                        'polarity': 'negative' if random.random() < 0.9 else 'positive'
                    })
            else:
                # Add single strike
                strikes.append({
                    'time': strike_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                    'latitude': lat,
                    'longitude': lon,
                    'amplitude': round(random.uniform(-150, -5), 1),  # kA
                    'type': 'cloud_to_ground' if random.random() < 0.7 else 'cloud_to_cloud',
                    'polarity': 'negative' if random.random() < 0.9 else 'positive'
                })
        
        # Sort strikes by time
        strikes.sort(key=lambda x: x['time'])
        
        return {
            'total_count': len(strikes),
            'start_time': start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'end_time': end_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'region': {
                'min_lat': self.up_bounds['min_lat'],
                'max_lat': self.up_bounds['max_lat'],
                'min_lon': self.up_bounds['min_lon'],
                'max_lon': self.up_bounds['max_lon'],
                'name': 'Uttar Pradesh'
            },
            'strikes': strikes
        }
    
    def collect_and_save(self):
        """Collect and save lightning data"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M')
        data = self.fetch_lightning_data()
        
        if not data:
            print("No lightning data collected")
            return None
        
        filename = os.path.join(self.data_dir, f'lightning_data_{timestamp}.json')
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"Saved lightning data to {filename} with {data['total_count']} strikes")
        
        return data