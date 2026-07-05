import pandas as pd
import numpy as np
import os
import urllib.request

def download_and_preprocess_omni(num_samples=100000):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    raw_data_path = os.path.join(data_dir, 'omni2_all_years.dat')
    url = 'https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat'
    
    if not os.path.exists(raw_data_path):
        print(f"Downloading OMNI2 dataset from {url}...")
        urllib.request.urlretrieve(url, raw_data_path)
        print("Download complete.")
        
    print("Parsing dataset...")
    df = pd.read_csv(raw_data_path, sep=r'\s+', header=None, engine='python')
    
    df = df[[0, 1, 2, 16, 23, 22, 24]]
    df.columns = ['Year', 'Day', 'Hour', 'bz', 'density', 'temperature', 'speed']
    
    print("Cleaning missing values...")
    df['bz'] = df['bz'].replace(999.9, np.nan)
    df['density'] = df['density'].replace(999.9, np.nan)
    df['temperature'] = df['temperature'].replace(9999999., np.nan)
    df['speed'] = df['speed'].replace(9999., np.nan)
    df['speed'] = df['speed'].replace(999.9, np.nan)
    
    df['timestamp'] = pd.to_datetime(df['Year'].astype(str) + df['Day'].astype(str).str.zfill(3), format='%Y%j') + pd.to_timedelta(df['Hour'], unit='h')
    
    df.drop(columns=['Year', 'Day', 'Hour'], inplace=True)
    
    df.ffill(limit=3, inplace=True)
    df.dropna(inplace=True)
    
    print("Filtering to recent records...")
    df = df.tail(num_samples).copy()
    
    print("Feature Engineering...")
    df['speed_t-1'] = df['speed'].shift(1)
    df['speed_t-2'] = df['speed'].shift(2)
    df['density_t-1'] = df['density'].shift(1)
    df['bz_t-1'] = df['bz'].shift(1)
    
    df['target_speed'] = df['speed'].shift(-1)
    
    df.dropna(inplace=True)
    
    output_path = os.path.join(data_dir, 'space_weather_real.csv')
    df.to_csv(output_path, index=False)
    print(f"Real data processed and saved to {output_path}. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    download_and_preprocess_omni(100000)
