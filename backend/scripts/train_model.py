import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
    
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    models_dir = os.path.join(current_dir, '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    results = {}
    
    print(f"\nTraining Random Forest on {len(X_train)} records...")
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=15, min_samples_split=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    results['Random Forest'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'MAE': mean_absolute_error(y_test, rf_pred)
    }
    
    # Save Model Information (Metrics) for Frontend
    model_info = {
        "algorithm": "Random Forest Regressor",
        "dataset": f"OMNI2 (NASA/NOAA) {len(df)} records",
        "features": features,
        "rmse": round(results['Random Forest']['RMSE'], 2),
        "mae": round(results['Random Forest']['MAE'], 2),
        "interpretability": "SHAP (TreeExplainer)"
    }
    import json
    info_path = os.path.join(models_dir, 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f)
        
    joblib.dump(rf_model, os.path.join(models_dir, 'rf_model.pkl'))
    
    print("\nModel Comparison (Real Data):")
    for name, metrics in results.items():
        print(f"{name}: RMSE={metrics['RMSE']:.2f} km/s, MAE={metrics['MAE']:.2f} km/s")
        
    print("\nGenerating SHAP explainer for Random Forest...")
    explainer = shap.TreeExplainer(rf_model)
    joblib.dump(explainer, os.path.join(models_dir, 'shap_explainer.pkl'))
    print("SHAP explainer saved.")
    
    print("\nAll models and explainer successfully saved!")

if __name__ == "__main__":
    train()
