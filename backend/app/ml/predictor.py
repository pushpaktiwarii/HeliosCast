import joblib
import pandas as pd
import os
import numpy as np

class SolarWindPredictor:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, '..', '..', 'models')
        
        self.xgb_path = os.path.join(models_dir, 'xgb_model.pkl')
        self.rf_path = os.path.join(models_dir, 'rf_model.pkl')
        self.shap_path = os.path.join(models_dir, 'shap_explainer.pkl')
        
        self.xgb_model = None
        self.rf_model = None
        self.shap_explainer = None
        
        self.load_models()
        
    def load_models(self):
        if os.path.exists(self.xgb_path):
            self.xgb_model = joblib.load(self.xgb_path)
        if os.path.exists(self.rf_path):
            self.rf_model = joblib.load(self.rf_path)
        if os.path.exists(self.shap_path):
            self.shap_explainer = joblib.load(self.shap_path)
            
    def predict(self, current_speed, density, temperature, bz, speed_t1, speed_t2, density_t1, bz_t1):
        if not self.rf_model:
            raise ValueError("Random Forest Model is not loaded. Run train_model.py first.")
            
        features_list = ['speed', 'density', 'temperature', 'bz', 'speed_t-1', 'speed_t-2', 'density_t-1', 'bz_t-1']
        features = pd.DataFrame([{
            'speed': current_speed,
            'density': density,
            'temperature': temperature,
            'bz': bz,
            'speed_t-1': speed_t1,
            'speed_t-2': speed_t2,
            'density_t-1': density_t1,
            'bz_t-1': bz_t1
        }], columns=features_list)
        
        # 1. Main Prediction via Random Forest (performed best)
        predicted_speed = float(self.rf_model.predict(features)[0])
        
        # 2. Uncertainty / Confidence Interval via Random Forest standard deviation
        lower_bound = predicted_speed
        upper_bound = predicted_speed
        
        # Get predictions from all trees in the random forest
        tree_preds = [tree.predict(features.values)[0] for tree in self.rf_model.estimators_]
        std_dev = np.std(tree_preds)
        # 95% confidence interval is approx +- 1.96 * std_dev
        margin = 1.96 * std_dev
        lower_bound = round(predicted_speed - margin, 2)
        upper_bound = round(predicted_speed + margin, 2)
            
        # 3. Explainable AI via SHAP
        feature_importance = {}
        if self.shap_explainer:
            shap_values = self.shap_explainer.shap_values(features)[0]
            # Convert to dictionary {feature_name: shap_value}
            for i, feat in enumerate(features_list):
                feature_importance[feat] = round(float(shap_values[i]), 4)
        
        # Risk assessment logic based on speed (km/s)
        if predicted_speed < 400:
            risk = "Low"
        elif predicted_speed < 500:
            risk = "Medium"
        else:
            risk = "High"
            
        return {
            "predicted_speed": round(predicted_speed, 2),
            "risk_level": risk,
            "confidence_interval": {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            },
            "feature_importance": feature_importance
        }

predictor = SolarWindPredictor()
