import os
import glob
import json
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import sys
sys.path.append('..')
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, BASIC_FEATURES, ENHANCED_FEATURES, TARGET, SEQUENCE_LENGTH, UP_BOUNDS

class DataFusionProcessor:
    def __init__(self):
        self.raw_data_dir = RAW_DATA_DIR
        self.processed_data_dir = PROCESSED_DATA_DIR
        self.basic_features = BASIC_FEATURES
        self.enhanced_features = ENHANCED_FEATURES
        self.target = TARGET
        self.sequence_length = SEQUENCE_LENGTH
        self.up_bounds = UP_BOUNDS
        
    def load_raw_data(self, data_type=None, max_files=10):
        """Load raw data files of specified type or all if None"""
        all_data = {}
        
        # Define data types and their file patterns
        data_types = {
            'openweather': 'weather_data_*.json',
            'imd': 'imd_data_*.json',
            'gfs': 'gfs_data_*.nc',
            'lightning': 'lightning_data_*.json',
            'nasa': 'nasa_power_data_*.json'
        }
        
        # Filter by specific data type if requested
        if data_type and data_type in data_types:
            data_types = {data_type: data_types[data_type]}
        
        # Load each data type
        for dtype, pattern in data_types.items():
            all_data[dtype] = []
            
            # Find matching files
            file_pattern = os.path.join(self.raw_data_dir, pattern)
            files = sorted(glob.glob(file_pattern), reverse=True)[:max_files]  # Latest files first
            
            # Load each file
            for file_path in files:
                try:
                    if file_path.endswith('.json'):
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            all_data[dtype].append({
                                'file': os.path.basename(file_path),
                                'data': data
                            })
                    elif file_path.endswith('.nc'):
                        data = xr.open_dataset(file_path)
                        all_data[dtype].append({
                            'file': os.path.basename(file_path),
                            'data': data
                        })
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        return all_data
    
    def process_openweather_data(self, weather_files):
        """Process OpenWeatherMap data"""
        if not weather_files:
            return pd.DataFrame()
        
        processed_data = []
        
        for file_data in weather_files:
            data = file_data['data']
            
            # Extract and structure the data
            for entry in data:
                try:
                    timestamp = datetime.strptime(entry['collection_time'], '%Y%m%d%H%M')
                    
                    processed_entry = {
                        'timestamp': timestamp,
                        'data_source': 'openweather',
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
                    print(f"Error processing OpenWeatherMap entry: {e}")
                    continue
        
        return pd.DataFrame(processed_data)
    
    def process_imd_data(self, imd_files):
        """Process IMD data"""
        if not imd_files:
            return pd.DataFrame()
        
        processed_data = []
        
        for file_data in imd_files:
            data = file_data['data']
            
            # Process current observations
            if 'current' in data:
                current_data = data['current']
                for station in current_data.get('stations', []):
                    try:
                        processed_entry = {
                            'timestamp': datetime.fromisoformat(current_data['timestamp'].replace('Z', '+00:00')),
                            'data_source': 'imd_current',
                            'latitude': station['latitude'],
                            'longitude': station['longitude'],
                            'temperature': station['temperature'],
                            'humidity': station['humidity'],
                            'pressure': station['pressure'],
                            'wind_speed': station['wind_speed'],
                            'wind_direction': station['wind_direction'],
                            'precipitation': station.get('rainfall_1hr', 0),
                            'weather_description': station['weather_condition'],
                            'lightning_observed': 1 if 'thunderstorm' in station['weather_condition'].lower() else 0
                        }
                        processed_data.append(processed_entry)
                    except (KeyError, ValueError) as e:
                        print(f"Error processing IMD current entry: {e}")
            
            # Process forecast data
            if 'forecast' in data:
                forecast_data = data['forecast']
                for forecast in forecast_data.get('forecasts', []):
                    forecast_time = datetime.fromisoformat(forecast['forecast_time'].replace('Z', '+00:00'))
                    for point in forecast.get('forecast_data', []):
                        try:
                            processed_entry = {
                                'timestamp': forecast_time,
                                'data_source': 'imd_forecast',
                                'latitude': point['latitude'],
                                'longitude': point['longitude'],
                                'temperature': point['temperature'],
                                'humidity': point['humidity'],
                                'pressure': point['pressure'],
                                'wind_speed': point['wind_speed'],
                                'wind_direction': point['wind_direction'],
                                'precipitation': point.get('precipitation_probability', 0) / 100 * 5,  # Rough estimate
                                'weather_description': point['weather_condition'],
                                'lightning_observed': 1 if 'thunderstorm' in point['weather_condition'].lower() else 0
                            }
                            processed_data.append(processed_entry)
                        except (KeyError, ValueError) as e:
                            print(f"Error processing IMD forecast entry: {e}")
            
            # Process lightning data
            if 'lightning' in data:
                lightning_data = data['lightning']
                strike_counts = {}
                
                # Create a grid for aggregating strikes
                grid_res = 0.1  # ~10km grid resolution
                for lat in np.arange(self.up_bounds['min_lat'], self.up_bounds['max_lat'] + grid_res, grid_res):
                    for lon in np.arange(self.up_bounds['min_lon'], self.up_bounds['max_lon'] + grid_res, grid_res):
                        grid_key = (round(lat, 1), round(lon, 1))
                        strike_counts[grid_key] = 0
                
                # Count strikes in each grid cell
                for strike in lightning_data.get('strikes', []):
                    strike_lat = round(strike['latitude'] * 10) / 10
                    strike_lon = round(strike['longitude'] * 10) / 10
                    grid_key = (strike_lat, strike_lon)
                    
                    if grid_key in strike_counts:
                        strike_counts[grid_key] += 1
                
                # Create entries for each grid cell with lightning data
                period_end = datetime.fromisoformat(lightning_data['period_end'].replace('Z', '+00:00'))
                for grid_key, count in strike_counts.items():
                    if count > 0:  # Only include cells with strikes
                        processed_entry = {
                            'timestamp': period_end,
                            'data_source': 'imd_lightning',
                            'latitude': grid_key[0],
                            'longitude': grid_key[1],
                            'lightning_count': count,
                            'lightning_observed': 1 if count > 0 else 0
                        }
                        processed_data.append(processed_entry)
        
        return pd.DataFrame(processed_data)
    
    def process_gfs_data(self, gfs_files):
        """Process GFS forecast data"""
        if not gfs_files:
            return pd.DataFrame()
        
        processed_data = []
        
        for file_data in gfs_files:
            data = file_data['data']
            
            # Process the variables we care about
            # Convert 3D data (time, lat, lon) to a list of 2D records
            if isinstance(data, xr.Dataset):
                for t_idx, time in enumerate(data.time.values):
                    timestamp = pd.Timestamp(time)
                    
                    for lat_idx, lat in enumerate(data.latitude.values):
                        for lon_idx, lon in enumerate(data.longitude.values):
                            try:
                                # Extract variables at this point
                                entry = {
                                    'timestamp': timestamp,
                                    'data_source': 'gfs',
                                    'latitude': lat,
                                    'longitude': lon
                                }
                                
                                # Add all available variables from GFS
                                if 't2m' in data:
                                    entry['temperature'] = float(data.t2m.values[t_idx, lat_idx, lon_idx])
                                if 'rh2m' in data:
                                    entry['humidity'] = float(data.rh2m.values[t_idx, lat_idx, lon_idx])
                                if 'pres' in data:
                                    entry['pressure'] = float(data.pres.values[t_idx, lat_idx, lon_idx])
                                if 'u10' in data and 'v10' in data:
                                    u = float(data.u10.values[t_idx, lat_idx, lon_idx])
                                    v = float(data.v10.values[t_idx, lat_idx, lon_idx])
                                    entry['wind_speed'] = np.sqrt(u**2 + v**2)
                                    entry['wind_direction'] = np.degrees(np.arctan2(v, u)) % 360
                                if 'prcp' in data:
                                    entry['precipitation'] = float(data.prcp.values[t_idx, lat_idx, lon_idx])
                                
                                # Add extended weather variables
                                if 'cape' in data:
                                    entry['cape'] = float(data.cape.values[t_idx, lat_idx, lon_idx])
                                if 'cin' in data:
                                    entry['cin'] = float(data.cin.values[t_idx, lat_idx, lon_idx])
                                if 'li' in data:
                                    entry['lifted_index'] = float(data.li.values[t_idx, lat_idx, lon_idx])
                                if 'pwat' in data:
                                    entry['precipitable_water'] = float(data.pwat.values[t_idx, lat_idx, lon_idx])
                                
                                # Derive lightning probability from CAPE and other factors
                                if 'cape' in entry:
                                    cape = entry['cape']
                                    li = entry.get('lifted_index', 0)
                                    
                                    # Simple lightning probability model
                                    # High CAPE and negative LI = higher probability
                                    if cape > 2000 and li < -4:
                                        lightning_prob = 0.8
                                    elif cape > 1500 and li < -2:
                                        lightning_prob = 0.6
                                    elif cape > 1000 and li < 0:
                                        lightning_prob = 0.4
                                    elif cape > 500:
                                        lightning_prob = 0.2
                                    else:
                                        lightning_prob = 0.05
                                    
                                    entry['lightning_probability'] = lightning_prob
                                    entry['lightning_observed'] = 1 if lightning_prob > 0.6 else 0
                                
                                processed_data.append(entry)
                            except Exception as e:
                                print(f"Error processing GFS data point: {e}")
        
        return pd.DataFrame(processed_data)
    
    def process_lightning_data(self, lightning_files):
        """Process lightning strike data"""
        if not lightning_files:
            return pd.DataFrame()
        
        processed_data = []
        
        for file_data in lightning_files:
            data = file_data['data']
            
            # Aggregate lightning strikes into grid cells
            grid_res = 0.1  # ~10km grid resolution
            
            # Initialize grid with all relevant cells
            strike_counts = {}
            for lat in np.arange(self.up_bounds['min_lat'], self.up_bounds['max_lat'] + grid_res, grid_res):
                for lon in np.arange(self.up_bounds['min_lon'], self.up_bounds['max_lon'] + grid_res, grid_res):
                    lat_rounded = round(lat, 1)
                    lon_rounded = round(lon, 1)
                    grid_key = (lat_rounded, lon_rounded)
                    strike_counts[grid_key] = {
                        'count': 0,
                        'amplitude_sum': 0,
                        'cg_count': 0,  # cloud-to-ground
                        'ic_count': 0,  # intra-cloud
                        'timestamps': []
                    }
            
            # Count strikes in each grid cell
            for strike in data.get('strikes', []):
                strike_lat = round(strike['latitude'] * 10) / 10
                strike_lon = round(strike['longitude'] * 10) / 10
                grid_key = (strike_lat, strike_lon)
                
                if grid_key in strike_counts:
                    strike_counts[grid_key]['count'] += 1
                    strike_counts[grid_key]['amplitude_sum'] += abs(float(strike['amplitude']))
                    
                    if strike['type'] == 'cloud_to_ground':
                        strike_counts[grid_key]['cg_count'] += 1
                    else:
                        strike_counts[grid_key]['ic_count'] += 1
                    
                    # Store timestamp for temporal distribution
                    strike_time = datetime.fromisoformat(strike['time'].replace('Z', '+00:00'))
                    strike_counts[grid_key]['timestamps'].append(strike_time)
            
            # Calculate end time of period
            end_time = datetime.fromisoformat(data['end_time'].replace('Z', '+00:00'))
            
            # Create entries for each grid cell
            for grid_key, stats in strike_counts.items():
                if stats['count'] > 0:  # Only include cells with strikes
                    # Calculate average recent strike time
                    if stats['timestamps']:
                        latest_strikes = sorted(stats['timestamps'], reverse=True)[:min(10, len(stats['timestamps']))]
                        avg_strike_time = sum((t.timestamp() for t in latest_strikes)) / len(latest_strikes)
                        avg_strike_time = datetime.fromtimestamp(avg_strike_time)
                    else:
                        avg_strike_time = end_time
                    
                    processed_entry = {
                        'timestamp': avg_strike_time,
                        'data_source': 'lightning',
                        'latitude': grid_key[0],
                        'longitude': grid_key[1],
                        'lightning_count': stats['count'],
                        'lightning_amplitude_avg': stats['amplitude_sum'] / stats['count'] if stats['count'] > 0 else 0,
                        'lightning_cg_ratio': stats['cg_count'] / stats['count'] if stats['count'] > 0 else 0,
                        'lightning_observed': 1
                    }
                    processed_data.append(processed_entry)
        
        return pd.DataFrame(processed_data)
    
    def process_nasa_power_data(self, nasa_files):
        """Process NASA POWER data"""
        if not nasa_files:
            return pd.DataFrame()
        
        processed_data = []
        
        for file_data in nasa_files:
            data_array = file_data['data']
            
            # Process each point
            for point_data in data_array:
                try:
                    geometry = point_data['geometry']
                    lon = geometry['coordinates'][0]
                    lat = geometry['coordinates'][1]
                    
                    # Get parameters
                    params = point_data['properties']['parameter']
                    
                    # Process daily data
                    for date_str, values in params['T2M'].items():
                        try:
                            date = datetime.strptime(date_str, '%Y%m%d')
                            
                            # Extract values for this date across parameters
                            entry = {
                                'timestamp': date,
                                'data_source': 'nasa_power',
                                'latitude': lat,
                                'longitude': lon,
                                'temperature': params['T2M'][date_str],
                                'humidity': params['RH2M'][date_str],
                                'pressure': params['PS'][date_str],
                                'wind_speed': params['WS10M'][date_str],
                                'dewpoint': params['T2MDEW'][date_str],
                                'precipitation': params['PRECTOT'][date_str],
                                'max_wind_speed': params['WS10M_MAX'][date_str],
                                'cape': params['CAPE'][date_str] if 'CAPE' in params else 0
                            }
                            
                            # Compute derived values
                            # K-Index approximation (temperature, dewpoint, and lapse rate proxy)
                            t_diff = abs(entry['temperature'] - entry['dewpoint'])
                            if t_diff < 20 and entry['cape'] > 500:
                                k_index = 30  # Moderate thunderstorm potential
                            elif t_diff < 10 and entry['cape'] > 1000:
                                k_index = 35  # High thunderstorm potential
                            else:
                                k_index = 20  # Low thunderstorm potential
                            
                            entry['k_index'] = k_index
                            
                            # Lightning probability estimation
                            # Based on CAPE and K-Index
                            cape = entry['cape']
                            if cape > 2000 and k_index > 30:
                                lightning_prob = 0.8
                            elif cape > 1500 and k_index > 25:
                                lightning_prob = 0.6
                            elif cape > 1000 and k_index > 20:
                                lightning_prob = 0.4
                            elif cape > 500:
                                lightning_prob = 0.2
                            else:
                                lightning_prob = 0.05
                            
                            entry['lightning_probability'] = lightning_prob
                            entry['lightning_observed'] = 1 if lightning_prob > 0.7 else 0
                            
                            processed_data.append(entry)
                        except Exception as e:
                            print(f"Error processing NASA POWER date {date_str}: {e}")
                except Exception as e:
                    print(f"Error processing NASA POWER point data: {e}")
        
        return pd.DataFrame(processed_data)
    
    def fuse_data_sources(self, data_frames):
        """Combine data from multiple sources with prioritization and cross-validation"""
        if not data_frames or all(df.empty for df in data_frames):
            print("No data to fuse")
            return pd.DataFrame()
        
        # Concatenate all non-empty DataFrames
        non_empty_dfs = [df for df in data_frames if not df.empty]
        all_data = pd.concat(non_empty_dfs, ignore_index=True)
        
        if all_data.empty:
            return pd.DataFrame()
        
        # Convert timestamp to common format
        all_data['timestamp'] = pd.to_datetime(all_data['timestamp'])
        
        # Sort by timestamp
        all_data = all_data.sort_values('timestamp')
        
        # Create a common grid for the region
        grid_res = 0.1  # ~10km resolution
        grid_lats = np.arange(self.up_bounds['min_lat'], self.up_bounds['max_lat'] + grid_res, grid_res)
        grid_lons = np.arange(self.up_bounds['min_lon'], self.up_bounds['max_lon'] + grid_res, grid_res)
        
        # Group the data spatially and temporally
        # Round coordinates to grid resolution to create spatial bins
        all_data['grid_lat'] = np.round(all_data['latitude'] / grid_res) * grid_res
        all_data['grid_lon'] = np.round(all_data['longitude'] / grid_res) * grid_res
        
        # Bin timestamps to the nearest hour
        all_data['grid_time'] = all_data['timestamp'].dt.floor('H')
        
        # Group by grid cell and time
        grouped = all_data.groupby(['grid_time', 'grid_lat', 'grid_lon'])
        
        # Create fused dataset with priority order for each data source and field
        # Priority: 1. Lightning data (for lightning_observed)
        #           2. GFS forecast data (for CAPE and other derived values)
        #           3. IMD data
        #           4. NASA POWER data
        #           5. OpenWeatherMap data (fallback)
        
        fused_data = []
        for (time, lat, lon), group in grouped:
            try:
                fused_entry = {
                    'timestamp': time,
                    'latitude': lat,
                    'longitude': lon
                }
                
                # First, check if we have lightning data
                lightning_data = group[group['data_source'] == 'lightning']
                if not lightning_data.empty:
                    fused_entry['lightning_observed'] = 1
                    if 'lightning_count' in lightning_data.columns:
                        fused_entry['lightning_count'] = lightning_data['lightning_count'].max()
                    if 'lightning_amplitude_avg' in lightning_data.columns:
                        fused_entry['lightning_amplitude_avg'] = lightning_data['lightning_amplitude_avg'].mean()
                else:
                    # No direct lightning data, check other sources
                    if 'lightning_observed' in group.columns:
                        sources_with_lightning = group[group['lightning_observed'].notna()]
                        if not sources_with_lightning.empty:
                            # If any source observed lightning, mark as observed
                            if sources_with_lightning['lightning_observed'].max() > 0:
                                fused_entry['lightning_observed'] = 1
                            else:
                                fused_entry['lightning_observed'] = 0
                        else:
                            fused_entry['lightning_observed'] = 0
                    else:
                        fused_entry['lightning_observed'] = 0
                
                # Process standard weather variables with priority order
                for feature in self.basic_features:
                    if feature in group.columns:
                        # Priority order for different sources
                        for source in ['imd_current', 'gfs', 'nasa_power', 'openweather']:
                            source_data = group[group['data_source'] == source]
                            if not source_data.empty and feature in source_data.columns and not source_data[feature].isna().all():
                                fused_entry[feature] = source_data[feature].mean()
                                break
                
                # Process enhanced weather variables for thunderstorm prediction
                for feature in ['cape', 'cin', 'lifted_index', 'k_index', 'precipitable_water',
                                'dewpoint', 'lightning_probability']:
                    if feature in group.columns:
                        # Priority for severe weather parameters
                        for source in ['gfs', 'nasa_power']:
                            source_data = group[group['data_source'] == source]
                            if not source_data.empty and feature in source_data.columns and not source_data[feature].isna().all():
                                fused_entry[feature] = source_data[feature].mean()
                                break
                
                # Compute derived thunderstorm features if missing
                if 'cape' in fused_entry and 'lifted_index' not in fused_entry:
                    # Approximate lifted index from CAPE
                    cape = fused_entry['cape']
                    if cape > 2500:
                        fused_entry['lifted_index'] = -8
                    elif cape > 1500:
                        fused_entry['lifted_index'] = -5
                    elif cape > 1000:
                        fused_entry['lifted_index'] = -3
                    elif cape > 500:
                        fused_entry['lifted_index'] = -1
                    else:
                        fused_entry['lifted_index'] = 2
                
                # If we have temperature and dewpoint, compute derived parameters
                if 'temperature' in fused_entry and 'dewpoint' in fused_entry:
                    t = fused_entry['temperature']
                    td = fused_entry['dewpoint']
                    
                    # Temperature-dewpoint spread (lower = more moisture)
                    fused_entry['t_td_spread'] = t - td
                    
                    # Simplified K-Index if not already computed
                    if 'k_index' not in fused_entry:
                        if (t - td) < 10:  # Low spread = high moisture
                            fused_entry['k_index'] = 35  # Conducive for thunderstorms
                        elif (t - td) < 20:
                            fused_entry['k_index'] = 25  # Moderate potential
                        else:
                            fused_entry['k_index'] = 15  # Low potential
                
                # If we don't have a lightning probability yet, estimate it
                if 'lightning_probability' not in fused_entry:
                    # Start with a base probability
                    if fused_entry['lightning_observed'] == 1:
                        base_prob = 0.9  # Already observed lightning
                    else:
                        base_prob = 0.1  # No lightning observed yet
                    
                    # Adjust based on available parameters
                    if 'cape' in fused_entry:
                        cape = fused_entry['cape']
                        if cape > 2000:
                            base_prob += 0.4
                        elif cape > 1000:
                            base_prob += 0.2
                        elif cape > 500:
                            base_prob += 0.1
                    
                    if 'lifted_index' in fused_entry:
                        li = fused_entry['lifted_index']
                        if li < -5:
                            base_prob += 0.3
                        elif li < -2:
                            base_prob += 0.2
                        elif li < 0:
                            base_prob += 0.1
                    
                    if 'k_index' in fused_entry:
                        ki = fused_entry['k_index']
                        if ki > 35:
                            base_prob += 0.3
                        elif ki > 30:
                            base_prob += 0.2
                        elif ki > 25:
                            base_prob += 0.1
                    
                    # Cap probability between 0 and 1
                    fused_entry['lightning_probability'] = max(0, min(1, base_prob))
                
                fused_data.append(fused_entry)
            except Exception as e:
                print(f"Error fusing data for cell ({time}, {lat}, {lon}): {e}")
        
        # Convert to DataFrame
        fused_df = pd.DataFrame(fused_data)
        
        # Fill missing lightning_probability with reasonable defaults based on season
        if 'lightning_probability' in fused_df.columns and fused_df['lightning_probability'].isna().any():
            # Check current month for seasonal defaults
            current_month = datetime.now().month
            if 6 <= current_month <= 9:  # Monsoon season
                default_prob = 0.3
            elif current_month in [4, 5, 10]:  # Pre/post monsoon
                default_prob = 0.2
            else:  # Dry season
                default_prob = 0.05
                
            fused_df['lightning_probability'].fillna(default_prob, inplace=True)
        
        return fused_df
    
    def create_time_series_dataset(self, df):
        """Create sequences for time series prediction"""
        if df.empty:
            return np.array([]), np.array([])
            
        df = df.sort_values(by='timestamp')
        
        # Group by location
        locations = df[['latitude', 'longitude']].drop_duplicates().values
        
        X_sequences = []
        y_labels = []
        
        for lat, lon in locations:
            location_data = df[(df['latitude'] == lat) & (df['longitude'] == lon)]
            
            # Ensure we have all required features
            missing_features = [f for f in self.enhanced_features if f not in location_data.columns]
            if missing_features:
                print(f"Location ({lat}, {lon}) missing features: {missing_features}")
                continue
                
            feature_data = location_data[self.enhanced_features].values
            target_data = location_data[self.target].values
            
            # Create sequences
            for i in range(len(location_data) - self.sequence_length):
                X_sequences.append(feature_data[i:i + self.sequence_length])
                y_labels.append(target_data[i + self.sequence_length])
        
        return np.array(X_sequences), np.array(y_labels)
    
    def process_and_save(self, collected_data=None):
        """Process all available data sources and save combined dataset"""
        print("Starting data fusion and processing...")
        
        if collected_data is None:
            # Load all recent data files
            raw_data = self.load_raw_data()
        else:
            # Use provided data
            raw_data = collected_data
        
        # Process each data type
        data_frames = []
        
        if 'openweather' in raw_data and raw_data['openweather']:
            print("Processing OpenWeatherMap data...")
            openweather_df = self.process_openweather_data(raw_data['openweather'])
            data_frames.append(openweather_df)
            print(f"Processed {len(openweather_df)} OpenWeatherMap data points")
        
        if 'imd' in raw_data and raw_data['imd']:
            print("Processing IMD data...")
            imd_df = self.process_imd_data(raw_data['imd'])
            data_frames.append(imd_df)
            print(f"Processed {len(imd_df)} IMD data points")
        
        if 'gfs' in raw_data and raw_data['gfs']:
            print("Processing GFS data...")
            gfs_df = self.process_gfs_data(raw_data['gfs'])
            data_frames.append(gfs_df)
            print(f"Processed {len(gfs_df)} GFS data points")
        
        if 'lightning' in raw_data and raw_data['lightning']:
            print("Processing lightning data...")
            lightning_df = self.process_lightning_data(raw_data['lightning'])
            data_frames.append(lightning_df)
            print(f"Processed {len(lightning_df)} lightning data points")
        
        if 'nasa' in raw_data and raw_data['nasa']:
            print("Processing NASA POWER data...")
            nasa_df = self.process_nasa_power_data(raw_data['nasa'])
            data_frames.append(nasa_df)
            print(f"Processed {len(nasa_df)} NASA POWER data points")
        
        # Fuse all data sources
        print("Fusing data from multiple sources...")
        fused_df = self.fuse_data_sources(data_frames)
        
        if fused_df.empty:
            print("No data to save after fusion")
            return None, None, None
        
        print(f"Fused data contains {len(fused_df)} points")
        
        # Save the processed DataFrame
        timestamp = datetime.now().strftime('%Y%m%d%H%M')
        processed_file = os.path.join(self.processed_data_dir, f'fused_weather_data_{timestamp}.csv')
        fused_df.to_csv(processed_file, index=False)
        print(f"Saved fused data to {processed_file}")
        
        # Create and save time series dataset
        print("Creating time series sequences...")
        X_sequences, y_labels = self.create_time_series_dataset(fused_df)
        
        if len(X_sequences) > 0:
            sequence_file = os.path.join(self.processed_data_dir, f'time_series_sequences_{timestamp}.npz')
            np.savez(sequence_file, X=X_sequences, y=y_labels)
            print(f"Saved {len(X_sequences)} time series sequences to {sequence_file}")
        else:
            print("No time series sequences created")
        
        return fused_df, X_sequences, y_labels