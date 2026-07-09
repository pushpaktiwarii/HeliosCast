import pandas as pd
import numpy as np
import os
import urllib.request
from datetime import timedelta

def download_and_preprocess_omni_5min(years=[2019, 2020, 2021, 2022, 2023]):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    dfs = []
    
    for year in years:
        file_name = f'omni_5min{year}.asc'
        raw_data_path = os.path.join(data_dir, file_name)
        url = f'https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/{file_name}'
        
        if not os.path.exists(raw_data_path):
            print(f"Downloading OMNI 5-min dataset for {year} from {url}...")
            try:
                urllib.request.urlretrieve(url, raw_data_path)
                print(f"Download for {year} complete.")
            except Exception as e:
                print(f"Failed to download {year}: {e}")
                continue
                
        print(f"Parsing dataset for {year}...")
        try:
            cols = [0, 1, 2, 3, 13, 14, 17, 18, 21, 25, 26, 27, 28, 29, 30]
            col_names = ['Year', 'Day', 'Hour', 'Minute', 'Bt', 'Bx', 'By', 'Bz', 
                         'Speed', 'Density', 'Temperature', 'Dynamic_Pressure', 'Electric_Field', 
                         'Plasma_Beta', 'Alfven_Mach']
                         
            df = pd.read_csv(raw_data_path, sep=r'\s+', header=None, engine='python', usecols=cols)
            df.columns = col_names
            dfs.append(df)
        except Exception as e:
            print(f"Error parsing {year}: {e}")
            
    if not dfs:
        print("No data was loaded!")
        return None
        
    print("Concatenating data...")
    df = pd.concat(dfs, ignore_index=True)
    
    print("Cleaning missing values (999.9, 9999.9, etc.)...")
    # Replace anything >= 900 in specific columns since NASA uses 999.9 etc.
    # Note: Speed could realistically be > 900 (up to ~2000 km/s in extreme events)
    # But NASA uses 9999.99 for missing speed in 5-min.
    
    df['Bx'] = np.where(df['Bx'] > 900, np.nan, df['Bx'])
    df['By'] = np.where(df['By'] > 900, np.nan, df['By'])
    df['Bz'] = np.where(df['Bz'] > 900, np.nan, df['Bz'])
    df['Bt'] = np.where(df['Bt'] > 900, np.nan, df['Bt'])
    df['Temperature'] = np.where(df['Temperature'] > 9000000, np.nan, df['Temperature'])
    df['Density'] = np.where(df['Density'] > 900, np.nan, df['Density'])
    df['Speed'] = np.where(df['Speed'] > 9000, np.nan, df['Speed'])
    df['Dynamic_Pressure'] = np.where(df['Dynamic_Pressure'] > 90, np.nan, df['Dynamic_Pressure'])
    df['Electric_Field'] = np.where(df['Electric_Field'] > 900, np.nan, df['Electric_Field'])
    df['Plasma_Beta'] = np.where(df['Plasma_Beta'] > 900, np.nan, df['Plasma_Beta'])
    df['Alfven_Mach'] = np.where(df['Alfven_Mach'] > 900, np.nan, df['Alfven_Mach'])
    
    print("Creating timestamp...")
    df['timestamp'] = pd.to_datetime(df['Year'].astype(str) + df['Day'].astype(str).str.zfill(3), format='%Y%j') + \
                      pd.to_timedelta(df['Hour'], unit='h') + pd.to_timedelta(df['Minute'], unit='m')
                      
    df['Month'] = df['timestamp'].dt.month
    df.drop(columns=['Year', 'Day'], inplace=True)
    
    print("Interpolating and Forward filling missing values...")
    df.interpolate(method='linear', limit_direction='both', inplace=True)
    df.fillna(df.median(), inplace=True)
    
    # Drop rows that still have NAs (e.g. large gaps)
    
    
    print("Feature Engineering...")
    for lag in [1, 3]: 
        periods = lag * 12 
        df[f'speed_lag_{lag}h'] = df['Speed'].shift(periods)
        df[f'density_lag_{lag}h'] = df['Density'].shift(periods)
        df[f'bz_lag_{lag}h'] = df['Bz'].shift(periods)
        
    df['speed_ma_1h'] = df['Speed'].rolling(window=12).mean()
    df['bz_ma_1h'] = df['Bz'].rolling(window=12).mean()
    df['speed_roc'] = df['Speed'].pct_change()
    df['bz_std_1h'] = df['Bz'].rolling(window=12).std()
    
    df['target_speed'] = df['Speed'].shift(-12)
    
    def calculate_risk(row):
        if row['Bz'] < -10 and row['Speed'] > 600: return 3
        elif row['Bz'] < -5 and row['Speed'] > 500: return 2
        elif row['Bz'] < 0 and row['Speed'] > 400: return 1
        else: return 0
        
    df['future_bz'] = df['Bz'].shift(-12)
    df['target_risk'] = df.apply(lambda row: calculate_risk({'Bz': row['future_bz'], 'Speed': row['target_speed']}) if pd.notna(row['future_bz']) else np.nan, axis=1)
    df.drop(columns=['future_bz'], inplace=True)
    
    # Drop rows that have NaN target variables due to the 1-hour shift
    df.dropna(subset=['target_speed', 'target_risk'], inplace=True)
    
    
    df.columns = [c.lower() for c in df.columns]
    
    output_path = os.path.join(data_dir, 'space_weather_real.csv')
    df.to_csv(output_path, index=False)
    print(f"Data processed and saved to {output_path}. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    download_and_preprocess_omni_5min([2019, 2020, 2021, 2022, 2023])
