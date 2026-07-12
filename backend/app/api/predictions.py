from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import pandas as pd
import requests
import asyncio
import time
import json
import traceback
from datetime import datetime
from app.ml.predictor import predictor
from app.db import db

router = APIRouter()

NOAA_CACHE = {"data": None, "timestamp": 0, "history_df": None}

class PredictionRequest(BaseModel):
    speed: Optional[float] = None
    density: Optional[float] = None
    temperature: Optional[float] = None
    bz: Optional[float] = None

@router.post("/predict")
async def predict_solar_wind(req: PredictionRequest):
    try:
        # Use cached history if available for dynamic lags
        history_df = NOAA_CACHE.get("history_df", None)
        
        current_data = {
            "speed": req.speed,
            "density": req.density,
            "temperature": req.temperature,
            "bz": req.bz
        }
        
        result = predictor.predict(current_data, history_df=history_df)
        
        # Log to DB
        predicted_speed = result["prediction"]["predicted_speed"]
        risk_class = result["prediction"]["storm_risk"]
        db.log_prediction(predicted_speed, risk_class)
        
        # If we have actual current speed, we don't update actuals here because it's a manual POST
        # which lacks proper historical context. We only log the prediction.
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def fetch_noaa_alerts():
    try:
        url = "https://services.swpc.noaa.gov/products/alerts.json"
        # Use requests since this is simple JSON
        import requests
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            # Return top 5 most recent alerts
            return data[:5]
        return []
    except Exception as e:
        print(f"Error fetching alerts: {e}")
        return []

def fetch_live_noaa_data():
    global NOAA_CACHE
    now = time.time()
    
    if now - NOAA_CACHE["timestamp"] < 30:
        if NOAA_CACHE["data"]:
            return NOAA_CACHE["data"]
            
    NOAA_CACHE["timestamp"] = now

    plasma_url = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"
    mag_url = "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"
    
    plasma_data = requests.get(plasma_url, timeout=5).json()
    mag_data = requests.get(mag_url, timeout=5).json()
    
    plasma_df = pd.DataFrame(plasma_data)
    plasma_df['time_tag'] = pd.to_datetime(plasma_df['time_tag'])
    plasma_df.rename(columns={'proton_density': 'density', 'proton_speed': 'speed', 'proton_temperature': 'temperature'}, inplace=True)
    plasma_df[['density', 'speed', 'temperature']] = plasma_df[['density', 'speed', 'temperature']].apply(pd.to_numeric)
    
    mag_df = pd.DataFrame(mag_data)
    mag_df['time_tag'] = pd.to_datetime(mag_df['time_tag'])
    mag_df[['bx_gsm', 'by_gsm', 'bz_gsm', 'bt']] = mag_df[['bx_gsm', 'by_gsm', 'bz_gsm', 'bt']].apply(pd.to_numeric)
    
    merged = pd.merge_asof(plasma_df.sort_values('time_tag'), mag_df.sort_values('time_tag'), on='time_tag', direction='nearest')
    merged.dropna(subset=['speed', 'density', 'bz_gsm'], inplace=True)
    
    if len(merged) == 0:
        raise ValueError("No valid live data available")
        
    merged.set_index('time_tag', inplace=True)
    # 5-minute resampling to match training data
    resampled = merged.resample('5min').mean(numeric_only=True).dropna()
    if len(resampled) < 2:
        resampled = merged.resample('1min').mean(numeric_only=True).dropna()
        
    history_df = resampled.reset_index()
    history_df.rename(columns={'bz_gsm': 'bz'}, inplace=True)
    NOAA_CACHE["history_df"] = history_df
    
    # Generate 24h, 7d, 30d views (For demo, we just use the fetched recent data, 
    # normally this would require a DB of past month or pulling from an archive)
    history_24h = []
    for _, row in history_df.tail(288).iterrows():  # 288 5-min intervals = 24h
        time_tag = row['time_tag'] if 'time_tag' in row else row.name
        history_24h.append({
            "timestamp": time_tag.isoformat() + 'Z' if '+' not in time_tag.isoformat() else time_tag.isoformat(),
            "speed": float(round(row['speed'], 2)) if pd.notna(row['speed']) else 0,
            "density": float(round(row['density'], 2)) if pd.notna(row['density']) else 0,
            "bz": float(round(row['bz'], 2)) if pd.notna(row['bz']) else 0,
            "pressure": float(round(1.67e-6 * row['density'] * (row['speed']**2), 2)) if pd.notna(row['density']) else 0
        })
        
    latest_raw = merged.iloc[-1]
    time_tag_val = latest_raw.name if 'time_tag' not in latest_raw else latest_raw['time_tag']
    current_data = {
        "timestamp": time_tag_val.isoformat() + 'Z',
        "speed": float(round(latest_raw['speed'], 2)),
        "density": float(round(latest_raw['density'], 2)),
        "temperature": float(round(latest_raw.get('temperature', 100000), 2)),
        "bz": float(round(latest_raw['bz_gsm'], 2)),
        "bx": float(round(latest_raw.get('bx_gsm', 0), 2)),
        "by": float(round(latest_raw.get('by_gsm', 0), 2)),
        "bt": float(round(latest_raw.get('bt', abs(latest_raw['bz_gsm'])), 2)),
        "dynamic_pressure": float(round(1.67e-6 * latest_raw['density'] * (latest_raw['speed']**2), 2))
    }
        
    NOAA_CACHE["data"] = ({"24h": history_24h, "7d": [], "30d": []}, current_data)
    
    return NOAA_CACHE["data"]

LAST_PREDICTED_TIMESTAMP = None

@router.get("/current-conditions")
async def get_current_conditions_api():
    global LAST_PREDICTED_TIMESTAMP
    try:
        history, current = await asyncio.to_thread(fetch_live_noaa_data)
        
        # Auto-predict for the live dashboard
        pred_result = predictor.predict(current, history_df=NOAA_CACHE.get("history_df"))
        
        if current["timestamp"] != LAST_PREDICTED_TIMESTAMP:
            db.log_prediction(pred_result["prediction"]["predicted_speed"], pred_result["prediction"]["storm_risk"])
            
            # Combine history and current to check for any matches
            full_history = history["24h"] + [current]
            db.update_actuals(full_history)

            LAST_PREDICTED_TIMESTAMP = current["timestamp"]
            
        return {
            "current": current,
            "history": history,
            "prediction": pred_result,
            "is_cached": False
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"NOAA API error: {str(e)}")

@router.get("/history")
async def get_history():
    history, _ = await asyncio.to_thread(fetch_live_noaa_data)
    return history

@router.get("/prediction-history")
async def get_prediction_history():
    return db.get_prediction_history(limit=50)

@router.get("/model-training-stats")
async def get_model_info():
    try:
        models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        info_path = os.path.join(models_dir, 'model_info.json')
        
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                return json.load(f)
        
        return {"error": "Model info not found. Train models first."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_alerts():
    try:
        alerts = await asyncio.to_thread(fetch_noaa_alerts)
        return alerts
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"NOAA API error: {str(e)}")

async def background_prediction_task():
    """Continuously fetches data and logs predictions every 5 minutes in the background."""
    print("Started background prediction task...")
    while True:
        try:
            global LAST_PREDICTED_TIMESTAMP
            history, current = await asyncio.to_thread(fetch_live_noaa_data)
            
            pred_result = predictor.predict(current, history_df=NOAA_CACHE.get("history_df"))
            
            if current["timestamp"] != LAST_PREDICTED_TIMESTAMP:
                db.log_prediction(pred_result["prediction"]["predicted_speed"], pred_result["prediction"]["storm_risk"])
                
                full_history = history["24h"] + [current]
                db.update_actuals(full_history)
                
                LAST_PREDICTED_TIMESTAMP = current["timestamp"]
                print(f"Background Task: Logged new prediction for {current['timestamp']}")
                
        except Exception as e:
            print(f"Background prediction task error: {e}")
            traceback.print_exc()
            
        # Wait 5 minutes before fetching again
        await asyncio.sleep(300)
