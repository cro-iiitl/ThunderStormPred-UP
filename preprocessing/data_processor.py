import os
import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import sys
sys.path.append('..')
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES, TARGET, SEQUENCE_LENGTH

class WeatherDataProcessor:
    def __init__(self):
        self.raw_data_dir = RAW_DATA_DIR
        self.processed_data_dir = PROCESSED_DATA_DIR
        self.features = FEATURES
        self.target = TARGET
        self.sequence_length = SEQUENCE_LENGTH
        
    def load_raw_data(self):
        """Load all raw JSON weather data files"""
        all_data = []
        json_files = glob.glob(os.path.join(self.raw_data_dir, 'weather_data_*.json'))
        
        for file_path in sorted(json_files):
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_data.extend(data)
        
        return all_data
    
    def extract_features(self, raw_data):
        """Extract relevant features from raw weather data"""
        processed_data = []
        
        for entry in raw_data:
            # Extract and structure the data
            try:
                timestamp = datetime.strptime(entry['collection_time'], '%Y%m%d%H%M')
                
                processed_entry = {
                    'timestamp': timestamp,
                    'latitude': entry['latitude'],
                    'longitude': entry['longitude'],
                    'temperature': entry['main']['temp'],
                    'humidity': entry['main']['humidity'],
                    'pressure': entry['main']['pressure'],
                    'wind_speed': entry['wind']['speed'],
                    'wind_direction': entry['wind'].get('deg', 0),
                    'cloud_cover': entry['clouds']['all'],
                    'precipitation': entry.get('rain', {}).get('1h', 0),
                    'weather_main': entry['weather'][0]['main'],
                    'weather_description': entry['weather'][0]['description']
                }
                
                # Add derived features
                if 'thunderstorm' in processed_entry['weather_description'].lower():
                    processed_entry['lightning_observed'] = 1
                else:
                    processed_entry['lightning_observed'] = 0
                
                processed_data.append(processed_entry)
            except (KeyError, ValueError) as e:
                print(f"Error processing entry: {e}")
                continue
        
        return pd.DataFrame(processed_data)
    
    def create_time_series_dataset(self, df):
        """Create sequences for time series prediction"""
        df = df.sort_values(by='timestamp')
        
        # Group by location
        locations = df[['latitude', 'longitude']].drop_duplicates().values
        
        X_sequences = []
        y_labels = []
        
        for lat, lon in locations:
            location_data = df[(df['latitude'] == lat) & (df['longitude'] == lon)]
            
            feature_data = location_data[self.features].values
            target_data = location_data[self.target].values
            
            # Create sequences
            for i in range(len(location_data) - self.sequence_length):
                X_sequences.append(feature_data[i:i + self.sequence_length])
                y_labels.append(target_data[i + self.sequence_length])
        
        return np.array(X_sequences), np.array(y_labels)
    
    def process_and_save(self):
        """Process raw data and save it for training"""
        print("Loading raw weather data...")
        raw_data = self.load_raw_data()
        
        if not raw_data:
            print("No raw data found.")
            return
        
        print("Extracting features...")
        df = self.extract_features(raw_data)
        
        # Save the processed DataFrame
        processed_file = os.path.join(self.processed_data_dir, 'processed_weather_data.csv')
        df.to_csv(processed_file, index=False)
        print(f"Saved processed data to {processed_file}")
        
        # Create and save time series dataset
        print("Creating time series sequences...")
        X_sequences, y_labels = self.create_time_series_dataset(df)
        
        sequence_file = os.path.join(self.processed_data_dir, 'time_series_sequences.npz')
        np.savez(sequence_file, X=X_sequences, y=y_labels)
        print(f"Saved time series sequences to {sequence_file}")
        
        return df, X_sequences, y_labels

# For testing or manual processing
if __name__ == "__main__":
    processor = WeatherDataProcessor()
    processor.process_and_save()