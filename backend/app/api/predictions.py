from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import os
import pandas as pd
import requests
import asyncio
import time
from app.ml.predictor import predictor
from datetime import datetime, timedelta

router = APIRouter()

NOAA_CACHE = {"data": None, "timestamp": 0}

class PredictionRequest(BaseModel):
    speed: float
    density: float
    temperature: float
    bz: float
    speed_t1: float
    speed_t2: float
    density_t1: float
    bz_t1: float

@router.post("/predict")
async def predict_solar_wind(req: PredictionRequest):
    try:
        result = predictor.predict(
            req.speed, req.density, req.temperature, req.bz,
            req.speed_t1, req.speed_t2, req.density_t1, req.bz_t1
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def fetch_live_noaa_data():
    global NOAA_CACHE
    now = time.time()
    
    if now - NOAA_CACHE["timestamp"] < 30:
        if NOAA_CACHE["data"]:
            return NOAA_CACHE["data"]
        else:
            raise Exception("NOAA API is on a 30-second cooldown due to a previous failure.")
            
    NOAA_CACHE["timestamp"] = now

    plasma_url = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"
    mag_url = "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"
    
    plasma_data = requests.get(plasma_url, timeout=5).json()
    mag_data = requests.get(mag_url, timeout=5).json()
    
    plasma_df = pd.DataFrame(plasma_data)
    plasma_df['time_tag'] = pd.to_datetime(plasma_df['time_tag'])
    plasma_df.rename(columns={
        'proton_density': 'density',
        'proton_speed': 'speed',
        'proton_temperature': 'temperature'
    }, inplace=True)
    plasma_df[['density', 'speed', 'temperature']] = plasma_df[['density', 'speed', 'temperature']].apply(pd.to_numeric)
    
    mag_df = pd.DataFrame(mag_data)
    mag_df['time_tag'] = pd.to_datetime(mag_df['time_tag'])
    mag_df[['bz_gsm']] = mag_df[['bz_gsm']].apply(pd.to_numeric)
    
    merged = pd.merge_asof(plasma_df.sort_values('time_tag'), mag_df.sort_values('time_tag'), on='time_tag', direction='nearest')
    merged.dropna(subset=['speed', 'density', 'temperature', 'bz_gsm'], inplace=True)
    
    if len(merged) == 0:
        raise ValueError("No valid live data available")
        
    if len(merged) > 24:
        merged.set_index('time_tag', inplace=True)
        hourly = merged.resample('1h').mean(numeric_only=True).dropna()
        if len(hourly) >= 24:
            history_df = hourly.tail(24).reset_index()
        else:
            history_df = merged.tail(24).reset_index()
    else:
        history_df = merged.copy()
        
    history = []
    for _, row in history_df.iterrows():
        time_tag = row['time_tag'] if 'time_tag' in row else row.name
        timestamp_str = time_tag.isoformat()
        if not timestamp_str.endswith('Z') and '+' not in timestamp_str:
            timestamp_str += 'Z'
        history.append({
            "timestamp": timestamp_str,
            "speed": float(round(row['speed'], 2)),
            "density": float(round(row['density'], 2)),
            "temperature": float(round(row['temperature'], 2)),
            "bz": float(round(row['bz_gsm'], 2))
        })
        
    latest_raw = merged.iloc[-1]
    latest_time = latest_raw.name if 'time_tag' not in latest_raw else latest_raw['time_tag']
    latest_timestamp_str = latest_time.isoformat()
    if not latest_timestamp_str.endswith('Z') and '+' not in latest_timestamp_str:
        latest_timestamp_str += 'Z'
        
    current_data = {
        "timestamp": latest_timestamp_str,
        "speed": float(round(latest_raw['speed'], 2)),
        "density": float(round(latest_raw['density'], 2)),
        "temperature": float(round(latest_raw['temperature'], 2)),
        "bz": float(round(latest_raw['bz_gsm'], 2))
    }
        
    NOAA_CACHE["data"] = (history, current_data)
    
    import json
    import os
    try:
        cache_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'latest_cache.json')
        with open(cache_path, 'w') as f:
            json.dump({"history": history, "current": current_data}, f)
    except Exception as e:
        print(f"Failed to write cache file: {e}")
        
    return history, current_data

import json

@router.get("/current-conditions")
async def get_current_conditions_api():
    try:
        history, current = await asyncio.to_thread(fetch_live_noaa_data)
        return {
            "current": current,
            "history": history,
            "is_cached": False
        }
    except Exception as e:
        cache_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'latest_cache.json')
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                data = json.load(f)
            return {
                "current": data["current"],
                "history": data["history"],
                "is_cached": True
            }
        raise HTTPException(status_code=503, detail=f"NOAA API is down and no cache exists: {str(e)}")

@router.get("/model-info")
async def get_model_info():
    try:
        models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        info_path = os.path.join(models_dir, 'model_info.json')
        
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                return json.load(f)
        
        # Fallback default info if file doesn't exist yet
        return {
            "algorithm": "Random Forest Regressor",
            "dataset": "OMNI2 (NASA/NOAA)",
            "features": ['speed', 'density', 'temperature', 'bz', 'speed_t-1', 'speed_t-2', 'density_t-1', 'bz_t-1'],
            "rmse": 17.89,
            "mae": 10.44,
            "interpretability": "SHAP (TreeExplainer)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
