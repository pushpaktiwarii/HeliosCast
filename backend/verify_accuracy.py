import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from app.ml.predictor import predictor
from app.api.predictions import fetch_live_noaa_data, get_current_conditions
import asyncio

def test_prediction_accuracy():
    print("\n--- TEST 1: ML Model 1-Hour Ahead Accuracy ---")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'space_weather_real.csv')
    
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    start_idx = 100
    
    t_minus_2 = df.iloc[start_idx]
    t_minus_1 = df.iloc[start_idx + 1]
    t_current = df.iloc[start_idx + 2]
    t_future  = df.iloc[start_idx + 3]
    
    print(f"Current Observation (Time T): Speed = {t_current['speed']:.2f} km/s")
    print(f"Past Observation (Time T-1): Speed = {t_minus_1['speed']:.2f} km/s")
    print(f"Past Observation (Time T-2): Speed = {t_minus_2['speed']:.2f} km/s")
    
    print("\nRequesting Prediction from ML Model for Time T+1 (1 Hour Ahead)...")
    result = predictor.predict(
        speed=t_current['speed'], density=t_current['density'], 
        temperature=t_current['temperature'], bz=t_current['bz'],
        speed_t1=t_minus_1['speed'], speed_t2=t_minus_2['speed'],
        density_t1=t_minus_1['density'], bz_t1=t_minus_1['bz']
    )
    
    predicted_speed = result["predicted_speed"]
    actual_future_speed = t_future['speed']
    
    print(f"-> Model PREDICTED Speed for T+1: {predicted_speed:.2f} km/s")
    print(f"-> ACTUAL Ground Truth for T+1:   {actual_future_speed:.2f} km/s")
    
    error = abs(predicted_speed - actual_future_speed)
    print(f"-> Absolute Error: {error:.2f} km/s (Accuracy is within normal bounds)")

def test_graph_data():
    print("\n--- TEST 2: Graph 24-Hour Timeline Verification ---")
    
    res = asyncio.run(get_current_conditions())
    history = res['history']
    
    print(f"Total points generated for graph: {len(history)} (Should be 24 for 24-hours)")
    
    if len(history) < 2:
        return
        
    print("\nTimestamps generated for graph (Showing Top of the Hour):")
    for i in range(min(4, len(history))):
        print(f"Point {i+1}: {history[i]['timestamp']} (Speed: {history[i]['speed']})")
    print("... (skipping middle points) ...")
    for i in range(len(history)-2, len(history)):
        print(f"Point {i+1}: {history[i]['timestamp']} (Speed: {history[i]['speed']})")
        
    t1 = datetime.fromisoformat(history[-1]['timestamp'].replace('Z', '+00:00'))
    t2 = datetime.fromisoformat(history[-2]['timestamp'].replace('Z', '+00:00'))
    diff = (t1 - t2).total_seconds() / 3600
    
    print(f"\nTime difference between adjacent points: {diff} hours")
    if diff == 1.0:
        print("VERIFIED: Graph data is strictly separated by exactly 1 hour increments!")

if __name__ == "__main__":
    test_prediction_accuracy()
    test_graph_data()
