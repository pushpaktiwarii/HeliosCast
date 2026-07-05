import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import shap

def train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'space_weather_real.csv')
    
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Run data_pipeline.py first.")
        return
        
    print("Loading real NASA/NOAA data...")
    df = pd.read_csv(data_path)
    
    features = ['speed', 'density', 'temperature', 'bz', 'speed_t-1', 'speed_t-2', 'density_t-1', 'bz_t-1']
    X = df[features]
    y = df['target_speed']
    
    # Chronological split for time series
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    models_dir = os.path.join(current_dir, '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    results = {}
    
    # --- 1. Random Forest ---
    print(f"\nTraining Random Forest on {len(X_train)} records...")
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=15, min_samples_split=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    results['Random Forest'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'MAE': mean_absolute_error(y_test, rf_pred)
    }
    joblib.dump(rf_model, os.path.join(models_dir, 'rf_model.pkl'))
    
    # --- 2. XGBoost ---
    print(f"Training XGBoost on {len(X_train)} records...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    results['XGBoost'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
        'MAE': mean_absolute_error(y_test, xgb_pred)
    }
    joblib.dump(xgb_model, os.path.join(models_dir, 'xgb_model.pkl'))
    
    # --- 3. LSTM ---
    print(f"Training LSTM on {len(X_train)} records...")
    # Reshape for LSTM: [samples, time steps, features]
    X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    lstm_model = Sequential([
        LSTM(32, activation='relu', input_shape=(1, len(features))),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    # Train for a few epochs for demonstration
    lstm_model.fit(X_train_lstm, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=0)
    lstm_pred = lstm_model.predict(X_test_lstm, verbose=0).flatten()
    results['LSTM'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, lstm_pred)),
        'MAE': mean_absolute_error(y_test, lstm_pred)
    }
    lstm_model.save(os.path.join(models_dir, 'lstm_model.keras'))
    
    # --- Print Comparison ---
    print("\nModel Comparison (Real Data):")
    for name, metrics in results.items():
        print(f"{name}: RMSE={metrics['RMSE']:.2f} km/s, MAE={metrics['MAE']:.2f} km/s")
        
    # --- Explainable AI (SHAP) ---
    print("\nGenerating SHAP explainer for Random Forest...")
    # Use Random Forest for SHAP to avoid XGBoost base_score string casting bug
    # RF also had the lowest RMSE!
    explainer = shap.TreeExplainer(rf_model)
    # Save the explainer (joblib can serialize TreeExplainer)
    joblib.dump(explainer, os.path.join(models_dir, 'shap_explainer.pkl'))
    print("SHAP explainer saved.")
    
    print("\nAll models and explainer successfully saved!")

if __name__ == "__main__":
    train()
