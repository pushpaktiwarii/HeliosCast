from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Dict
import random
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
    
    # Enforce 60-second cooldown on both success and failure
    if now - NOAA_CACHE["timestamp"] < 60:
        if NOAA_CACHE["data"]:
            return NOAA_CACHE["data"]
        else:
            raise Exception("NOAA API is on a 60-second cooldown due to a previous failure.")
            
    # Mark attempt time to trigger cooldown
    NOAA_CACHE["timestamp"] = now

    plasma_url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json"
    mag_url = "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json"
    
    plasma_data = requests.get(plasma_url, timeout=5).json()
    mag_data = requests.get(mag_url, timeout=5).json()
    
    plasma_df = pd.DataFrame(plasma_data[1:], columns=plasma_data[0])
    plasma_df['time_tag'] = pd.to_datetime(plasma_df['time_tag'])
    plasma_df[['density', 'speed', 'temperature']] = plasma_df[['density', 'speed', 'temperature']].apply(pd.to_numeric)
    
    mag_df = pd.DataFrame(mag_data[1:], columns=mag_data[0])
    mag_df['time_tag'] = pd.to_datetime(mag_df['time_tag'])
    mag_df[['bz_gsm']] = mag_df[['bz_gsm']].apply(pd.to_numeric)
    
    merged = pd.merge_asof(plasma_df.sort_values('time_tag'), mag_df.sort_values('time_tag'), on='time_tag', direction='nearest')
    merged.dropna(subset=['speed', 'density', 'temperature', 'bz_gsm'], inplace=True)
    
    if len(merged) == 0:
        raise ValueError("No valid live data available")
        
    if len(merged) > 24:
        merged.set_index('time_tag', inplace=True)
        hourly = merged.resample('1h').mean().dropna()
        if len(hourly) >= 24:
            history_df = hourly.tail(24).reset_index()
        else:
            history_df = merged.tail(24).reset_index()
    else:
        history_df = merged.copy()
        
    history = []
    for _, row in history_df.iterrows():
        # Handle index vs column if we reset index or not
        time_tag = row['time_tag'] if 'time_tag' in row else row.name
        timestamp_str = time_tag.isoformat()
        if not timestamp_str.endswith('Z') and '+' not in timestamp_str:
            timestamp_str += 'Z'
        history.append({
            "timestamp": timestamp_str,
            "speed": round(row['speed'], 2),
            "density": round(row['density'], 2),
            "temperature": round(row['temperature'], 2),
            "bz": round(row['bz_gsm'], 2)
        })
        
    # Get the absolute latest 1-minute record for "current" conditions
    latest_raw = merged.iloc[-1]
    latest_time = latest_raw.name if 'time_tag' not in latest_raw else latest_raw['time_tag']
    latest_timestamp_str = latest_time.isoformat()
    if not latest_timestamp_str.endswith('Z') and '+' not in latest_timestamp_str:
        latest_timestamp_str += 'Z'
        
    current_data = {
        "timestamp": latest_timestamp_str,
        "speed": round(latest_raw['speed'], 2),
        "density": round(latest_raw['density'], 2),
        "temperature": round(latest_raw['temperature'], 2),
        "bz": round(latest_raw['bz_gsm'], 2)
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
        history, current = fetch_live_noaa_data()
        return {
            "current": current,
            "history": history,
            "is_cached": False
        }
    except Exception as e:
        # Fallback to recorded 24h cache file
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

async def get_current_conditions():
    # Used by websocket
    try:
        history, current = fetch_live_noaa_data()
        return {
            "current": current,
            "history": history,
            "is_cached": False
        }
    except Exception as e:
        # Fallback to recorded 24h cache file
        cache_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'latest_cache.json')
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                data = json.load(f)
            return {
                "current": data["current"],
                "history": data["history"],
                "is_cached": True
            }
        return {
            "error": "NOAA API Offline",
            "current": {},
            "history": []
        }

@router.websocket("/ws/current-conditions")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await get_current_conditions()
            await websocket.send_json(data)
            await asyncio.sleep(2)  # Emit real-time data every 2 seconds
    except WebSocketDisconnect:
        pass
    except Exception as e:
        pass # Handle cases where socket closes forcefully (e.g. socket.send() exception)
