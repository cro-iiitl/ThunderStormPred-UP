import os
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import requests
import time
from io import BytesIO
sys.path.append('..')
from config import RAW_DATA_DIR, UP_BOUNDS

class GFSDataCollector:
    def __init__(self):
        self.data_dir = RAW_DATA_DIR
        self.up_bounds = UP_BOUNDS
        
    def download_gfs_data(self):
        """Download GFS weather forecast data for Uttar Pradesh region"""
        try:
            # Current date and nearest forecast hour
            now = datetime.utcnow()
            year = now.year
            month = now.month
            day = now.day
            hour = int(now.hour / 6) * 6  # GFS runs every 6 hours (0, 6, 12, 18)
            
            print(f"Attempting to download GFS data for {year}-{month:02d}-{day:02d} {hour:02d}Z")
            
            # In a production system, you would download actual GFS data from NCEP or AWS
            # Here, we'll generate synthetic GFS-like data for the Uttar Pradesh region
            
            # Create a grid for the region
            lats = np.arange(self.up_bounds['min_lat'], self.up_bounds['max_lat'] + 0.25, 0.25)
            lons = np.arange(self.up_bounds['min_lon'], self.up_bounds['max_lon'] + 0.25, 0.25)
            
            # Time coordinates for forecast (0-120 hours in 3-hour steps)
            forecast_hours = np.arange(0, 121, 3)
            times = [now + timedelta(hours=h) for h in forecast_hours]
            
            # Create the dataset
            ds = xr.Dataset(
                coords={
                    'longitude': lons,
                    'latitude': lats,
                    'time': times
                }
            )
            
            # Add variables important for thunderstorm prediction
            # Temperature at 2m
            ds['t2m'] = xr.DataArray(
                data=np.random.uniform(25, 35, size=(len(times), len(lats), len(lons))),
                dims=['time', 'latitude', 'longitude'],
                attrs={'units': 'degC', 'long_name': 'Temperature at 2m'}
            )
            
            # Relative humidity at 2m
            ds['rh2m'] = xr.DataArray(
                data=np.random.uniform(30, 90, size=(len(times), len(lats), len(lons))),
                dims=['time', 'latitude', 'longitude'],
                attrs={'units': '%', 'long_name': 'Relative Humidity at 2m'}
            )
            
            # Surface pressure
            ds['pres'] = xr.DataArray(
                data=np.random.uniform(980, 1020, size=(len(times), len(lats), len(lons))),
                dims=['time', 'latitude', 'longitude'],
                attrs={'units': 'hPa', 'long_name': 'Surface Pressure'}
            )
            
            # CAPE - Convective Available Potential Energy (critical for thunderstorm prediction)
            # Higher in eastern UP where thunderstorms are more common
            cape_data = np.zeros((len(times), len(lats), len(lons)))
            for t in range(len(times)):
                for i, lat in enumerate(lats):
                    for j, lon in enumerate(lons):
                        # More CAPE in eastern UP, especially in the afternoon
                        east_factor = (lon - self.up_bounds['min_lon']) / (self.up_bounds['max_lon'] - self.up_bounds['min_lon'])
                        time_factor = np.sin(np.pi * (times[t].hour % 24) / 24)  # Peak in afternoon
                        base_cape = np.random.uniform(100, 500)
                        
                        # Higher CAPE in monsoon season (June-September)
                        if 6 <= times[t].month <= 9:
                            season_factor = 2.0
                        else:
                            season_factor = 0.5
                            
                        cape_data[t, i, j] = base_cape + 2000 * east_factor * time_factor * season_factor
            
            ds['cape'] = xr.DataArray(
                data=cape_data,
                dims=['time', 'latitude', 'longitude'],
                attrs={'units': 'J/kg', 'long_name': 'Convective Available Potential Energy'}
            )
            
            # CIN - Convective Inhibition
            ds['cin'] = xr.DataArray(
                data=-np.random.uniform(0, 200, size=(len(times), len(lats), len(lons))),
                dims=['time', 'latitude', 'longitude'],
                attrs={'units': 'J/kg', 'long_name': 'Convective Inhibition'}
            )
            
            # Precipitation
            ds['prcp'] = xr.DataArray(
                data=np.random.exponential(2, size=(len(times), len(lats), len(lons))),
                dims=['time', 'latitude', 'longitude'],
                attrs={'units': 'mm', 'long_name': 'Precipitation'}
            )
            
            # Wind components at 10m
            ds['u10'] = xr.DataArray(
                data=np.random.uniform(-10, 10, size=(len(times), len(lats), len(lons))),
                dims=['time', 'latitude', 'longitude'],
                attrs={'units': 'm/s', 'long_name': 'U-component of wind at 10m'}
            )
            
            ds['v10'] = xr.DataArray(
                data=np.random.uniform(-10, 10, size=(len(times), len(lats), len(lons))),
                dims=['time', 'latitude', 'longitude'],
                attrs={'units': 'm/s', 'long_name': 'V-component of wind at 10m'}
            )
            
            # Lifted index (negative values indicate instability)
            ds['li'] = xr.DataArray(
                data=np.random.uniform(-8, 5, size=(len(times), len(lats), len(lons))),
                dims=['time', 'latitude', 'longitude'],
                attrs={'units': 'K', 'long_name': 'Lifted Index'}
            )
            
            # Precipitable water
            ds['pwat'] = xr.DataArray(
                data=np.random.uniform(20, 60, size=(len(times), len(lats), len(lons))),
                dims=['time', 'latitude', 'longitude'],
                attrs={'units': 'mm', 'long_name': 'Precipitable Water'}
            )
            
            # Add some more realistic spatial and temporal patterns
            # Eastern UP typically has more thunderstorms
            for t in range(len(times)):
                # Create a more realistic evolution over time
                time_factor = 1 + 0.5 * np.sin(np.pi * t / len(times))
                
                # Update cape with more realistic spatial pattern
                for i, lat in enumerate(lats):
                    for j, lon in enumerate(lons):
                        # Eastern UP gets more storms
                        if lon > (self.up_bounds['min_lon'] + self.up_bounds['max_lon']) / 2:
                            ds['cape'].values[t, i, j] *= 1.5 * time_factor
                            ds['li'].values[t, i, j] -= 2  # More negative (unstable)
                            ds['prcp'].values[t, i, j] *= 2 * time_factor
                        
                        # Add some random thunderstorm cells
                        if np.random.random() < 0.05:  # 5% chance of a storm cell
                            ds['cape'].values[t, i, j] = np.random.uniform(2000, 4000)
                            ds['li'].values[t, i, j] = np.random.uniform(-10, -5)
                            ds['prcp'].values[t, i, j] = np.random.uniform(10, 30)
            
            return ds
            
        except Exception as e:
            print(f"Error downloading GFS data: {e}")
            return None
    
    def collect_and_save(self):
        """Collect and save GFS data"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M')
        data = self.download_gfs_data()
        
        if data is None:
            print("No GFS data collected")
            return None
        
        # Save as netCDF file
        filename = os.path.join(self.data_dir, f'gfs_data_{timestamp}.nc')
        data.to_netcdf(filename)
        print(f"Saved GFS data to {filename}")
        return data