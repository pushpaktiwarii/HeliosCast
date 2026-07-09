import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import shap

def train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'space_weather_real.csv')
    
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Run data_pipeline.py first.")
        return
        
    print("Loading preprocessed high-res OMNI data...")
    df = pd.read_csv(data_path)
    print(f"Dataset Shape: {df.shape}")
    
    # Define features
    features = ['speed', 'density', 'temperature', 'bz', 'bx', 'by', 'bt', 
                'dynamic_pressure', 'electric_field', 'plasma_beta', 'alfven_mach',
                'speed_lag_1h', 'density_lag_1h', 'bz_lag_1h',
                'speed_lag_3h', 'density_lag_3h', 'bz_lag_3h',
                'speed_ma_1h', 'bz_ma_1h', 'speed_roc', 'bz_std_1h']
                
    # Keep only available features
    features = [f for f in features if f in df.columns]
    
    # Handle infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    X = df[features]
    y_reg = df['target_speed']
    y_clf = df['target_risk']
    
    # 70% Train, 15% Validation, 15% Test
    # We do a simple sequential split to avoid data leakage in time series
    n = len(df)
    train_idx = int(n * 0.7)
    val_idx = int(n * 0.85)
    
    X_train, X_val, X_test = X.iloc[:train_idx], X.iloc[train_idx:val_idx], X.iloc[val_idx:]
    y_train_reg, y_val_reg, y_test_reg = y_reg.iloc[:train_idx], y_reg.iloc[train_idx:val_idx], y_reg.iloc[val_idx:]
    y_train_clf, y_val_clf, y_test_clf = y_clf.iloc[:train_idx], y_clf.iloc[train_idx:val_idx], y_clf.iloc[val_idx:]
    
    models_dir = os.path.join(current_dir, '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. REGRESSION MODELS (Solar Wind Speed)
    print("\n--- Training Regression Models ---")
    reg_results = {}
    best_reg_model = None
    best_reg_rmse = float('inf')
    best_reg_name = ""
    
    reg_models = {
        'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    }
    
    for name, model in reg_models.items():
        print(f"Training {name} Regressor...")
        model.fit(X_train, y_train_reg)
        preds = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test_reg, preds))
        mae = mean_absolute_error(y_test_reg, preds)
        r2 = r2_score(y_test_reg, preds)
        
        reg_results[name] = {
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2)
        }
        print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
        
        if rmse < best_reg_rmse:
            best_reg_rmse = rmse
            best_reg_model = model
            best_reg_name = name
            
    joblib.dump(best_reg_model, os.path.join(models_dir, 'best_reg_model.pkl'))
    print(f"Best Regression Model: {best_reg_name} saved.")
    
    # 2. CLASSIFICATION MODELS (Storm Risk)
    print("\n--- Training Classification Models ---")
    clf_results = {}
    best_clf_model = None
    best_clf_acc = 0
    best_clf_name = ""
    
    clf_models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    }
    
    for name, model in clf_models.items():
        print(f"Training {name} Classifier...")
        model.fit(X_train, y_train_clf)
        preds = model.predict(X_test)
        
        acc = accuracy_score(y_test_clf, preds)
        clf_results[name] = {
            'Accuracy': float(acc),
            'F1': float(f1_score(y_test_clf, preds, average='weighted'))
        }
        print(f"  Accuracy: {acc:.2f}")
        
        if acc > best_clf_acc:
            best_clf_acc = acc
            best_clf_model = model
            best_clf_name = name
            
    joblib.dump(best_clf_model, os.path.join(models_dir, 'best_clf_model.pkl'))
    print(f"Best Classification Model: {best_clf_name} saved.")
    
    # SHAP Explainer
    print("\nGenerating SHAP explainer for best models...")
    # Use a small background sample for TreeExplainer
    background = shap.sample(X_train, 100)
    
    # SHAP for Regression
    reg_explainer = shap.TreeExplainer(best_reg_model, feature_perturbation='tree_path_dependent')
    joblib.dump(reg_explainer, os.path.join(models_dir, 'shap_explainer_reg.pkl'))
    
    # Feature Importance Data for UI
    try:
        importances = best_reg_model.feature_importances_
        feature_imp = {features[i]: float(importances[i]) for i in range(len(features))}
    except:
        feature_imp = {}

    model_info = {
        "dataset_records": n,
        "features": features,
        "best_regression": best_reg_name,
        "regression_metrics": reg_results,
        "best_classification": best_clf_name,
        "classification_metrics": clf_results,
        "feature_importance": feature_imp
    }
    
    info_path = os.path.join(models_dir, 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=4)
        
    print("\nAll models and metadata successfully saved!")

if __name__ == "__main__":
    train()
