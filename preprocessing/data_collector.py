import os
import time
import json
import requests
import pandas as pd
import random
from datetime import datetime, timedelta
import sys
sys.path.append('..')
from config import WEATHER_API_KEY, WEATHER_API_URL, RAW_DATA_DIR, UP_BOUNDS, COLLECTION_INTERVAL_HOURS

class WeatherDataCollector:
    def __init__(self):
        self.api_key = WEATHER_API_KEY
        self.api_url = WEATHER_API_URL
        self.data_dir = RAW_DATA_DIR
        self.up_bounds = UP_BOUNDS
        
    def create_grid(self, resolution=1.0):
        """Create a grid of coordinates covering Uttar Pradesh"""
        lats = []
        lons = []
        lat = self.up_bounds['min_lat']
        while lat <= self.up_bounds['max_lat']:
            lon = self.up_bounds['min_lon']
            while lon <= self.up_bounds['max_lon']:
                lats.append(lat)
                lons.append(lon)
                lon += resolution
            lat += resolution
        return pd.DataFrame({'latitude': lats, 'longitude': lons})
    
    def fetch_weather_data(self, lat, lon):
        """Fetch current weather data for a specific location"""
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.api_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for coordinates ({lat}, {lon}): {e}")
            return None
    
    def collect_and_save(self):
        """Collect weather data for the entire grid and save it"""
        grid = self.create_grid(resolution=1.0)  # Increased resolution to reduce API calls
        timestamp = datetime.now().strftime('%Y%m%d%H%M')
        
        all_data = []
        for _, row in grid.iterrows():
            try:
                weather_data = self.fetch_weather_data(row['latitude'], row['longitude'])
                if weather_data:
                    weather_data['latitude'] = row['latitude']
                    weather_data['longitude'] = row['longitude']
                    weather_data['collection_time'] = timestamp
                    all_data.append(weather_data)
                # Add randomized delay to avoid hitting rate limits
                time.sleep(random.uniform(2.0, 3.0))
            except Exception as e:
                print(f"Error processing location ({row['latitude']}, {row['longitude']}): {e}")
                # Wait longer after an error to recover from rate limiting
                time.sleep(10)
                continue
        
        # Save the collected data
        if all_data:
            filename = os.path.join(self.data_dir, f'weather_data_{timestamp}.json')
            with open(filename, 'w') as f:
                json.dump(all_data, f)
            print(f"Saved weather data to {filename} with {len(all_data)} locations")
        else:
            print("No data collected from OpenWeatherMap")
        
        return all_data
    
    def run_collection_loop(self):
        """Run continuous data collection at specified intervals"""
        print("Starting OpenWeatherMap data collection loop...")
        while True:
            self.collect_and_save()
            print(f"Waiting for {COLLECTION_INTERVAL_HOURS} hours before next collection")
            time.sleep(COLLECTION_INTERVAL_HOURS * 3600)