import os
import json
import requests
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime, timedelta
import sys
sys.path.append('..')
from config import RAW_DATA_DIR, UP_BOUNDS

class NASAPowerDataCollector:
    def __init__(self):
        self.base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
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
    
    def fetch_power_data(self, lat, lon):
        """Fetch NASA POWER data for a specific location"""
        # In a production system, you would use the actual NASA POWER API
        # Here, we'll generate synthetic data for simulation
        
        try:
            # Real API call would look like this:
            # params = {
            #     'start': (datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),
            #     'end': datetime.now().strftime('%Y%m%d'),
            #     'latitude': lat,
            #     'longitude': lon,
            #     'parameters': 'T2M,RH2M,PS,WS10M,PRECTOT,KT,WS10M_MAX,T2MDEW,CAPE',
            #     'community': 'AG',
            #     'format': 'json'
            # }
            # response = requests.get(self.base_url, params=params)
            # response.raise_for_status()
            # return response.json()
            
            # Generate synthetic data
            return self._generate_synthetic_power_data(lat, lon)
            
        except Exception as e:
            print(f"Error fetching NASA POWER data for coordinates ({lat}, {lon}): {e}")
            return None
    
    def _generate_synthetic_power_data(self, lat, lon):
        """Generate synthetic NASA POWER data for simulation"""
        # Current date and 30 days ago
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Generate daily data for 30 days
        daily_data = {}
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')
            
            # Spatial variation factor (eastern UP tends to be more humid/stormy)
            east_factor = (lon - self.up_bounds['min_lon']) / (self.up_bounds['max_lon'] - self.up_bounds['min_lon'])
            
            # Seasonal variation (higher temps and more storms in summer monsoon)
            month = current_date.month
            if 6 <= month <= 9:  # Monsoon season
                season_factor = 1.5
            elif month in [4, 5, 10]:  # Pre/post monsoon
                season_factor = 1.2
            else:  # Dry season
                season_factor = 0.8
            
            # Random daily variations
            daily_factor = 0.8 + 0.4 * random.random()
            
            # Create realistic atmospheric parameters
            temp_celsius = 20 + 15 * season_factor * daily_factor
            humidity = 40 + 50 * east_factor * season_factor * daily_factor
            pressure = 990 + 30 * (1 - daily_factor)
            wind_speed = 2 + 8 * daily_factor
            precipitation = 0
            
            # Add precipitation on some days
            if random.random() < 0.3 * season_factor * east_factor:
                precipitation = random.uniform(0, 50) * season_factor * east_factor
            
            # Calculate dewpoint based on temp and humidity
            # Magnus approximation
            a = 17.27
            b = 237.7
            alpha = ((a * temp_celsius) / (b + temp_celsius)) + np.log(humidity/100.0)
            dewpoint = (b * alpha) / (a - alpha)
            
            # CAPE varies by region and season
            # Eastern UP and monsoon season get higher CAPE
            cape = random.uniform(0, 500) + 2000 * east_factor * season_factor * daily_factor
            
            # Record values
            daily_data[date_str] = {
                'T2M': round(temp_celsius, 2),
                'RH2M': round(humidity, 2),
                'PS': round(pressure, 2),
                'WS10M': round(wind_speed, 2),
                'PRECTOT': round(precipitation, 2),
                'T2MDEW': round(dewpoint, 2),
                'CAPE': round(cape, 2),
                'WS10M_MAX': round(wind_speed * (1.5 + random.random()), 2),
                'KT': round(0.4 + 0.4 * daily_factor, 2)  # Clearness index
            }
            
            current_date += timedelta(days=1)
        
        # Format data like NASA POWER API response
        return {
            'type': 'feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [lon, lat]
            },
            'properties': {
                'parameter': {
                    'T2M': daily_data,
                    'RH2M': daily_data,
                    'PS': daily_data,
                    'WS10M': daily_data,
                    'PRECTOT': daily_data,
                    'T2MDEW': daily_data,
                    'CAPE': daily_data,
                    'WS10M_MAX': daily_data,
                    'KT': daily_data
                }
            }
        }
    
    def collect_and_save(self):
        """Collect NASA POWER data for Uttar Pradesh grid and save it"""
        grid = self.create_grid(resolution=1.0)
        timestamp = datetime.now().strftime('%Y%m%d%H%M')
        
        all_data = []
        for _, row in grid.iterrows():
            try:
                power_data = self.fetch_power_data(row['latitude'], row['longitude'])
                if power_data:
                    all_data.append(power_data)
                # Delay to avoid API rate limits
                time.sleep(random.uniform(0.5, 1.5))
            except Exception as e:
                print(f"Error processing NASA POWER data for ({row['latitude']}, {row['longitude']}): {e}")
        
        if all_data:
            filename = os.path.join(self.data_dir, f'nasa_power_data_{timestamp}.json')
            with open(filename, 'w') as f:
                json.dump(all_data, f)
            print(f"Saved NASA POWER data to {filename} with {len(all_data)} grid points")
            return all_data
        else:
            print("No NASA POWER data collected")
            return None