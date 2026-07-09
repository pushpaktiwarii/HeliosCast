import joblib
import pandas as pd
import os
import numpy as np

class SolarWindPredictor:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, '..', '..', 'models')
        
        self.reg_path = os.path.join(models_dir, 'best_reg_model.pkl')
        self.clf_path = os.path.join(models_dir, 'best_clf_model.pkl')
        self.shap_path = os.path.join(models_dir, 'shap_explainer_reg.pkl')
        
        self.reg_model = None
        self.clf_model = None
        self.shap_explainer = None
        
        self.load_models()
        
    def load_models(self):
        if os.path.exists(self.reg_path):
            self.reg_model = joblib.load(self.reg_path)
        if os.path.exists(self.clf_path):
            self.clf_model = joblib.load(self.clf_path)
        if os.path.exists(self.shap_path):
            self.shap_explainer = joblib.load(self.shap_path)
            
    def predict(self, current_data, history_df=None):
        if not self.reg_model or not self.clf_model:
            raise ValueError("Models are not loaded. Run train_model.py first.")
            
        # Extract base current data
        speed = current_data.get('speed', 400.0)
        density = current_data.get('density', 5.0)
        temperature = current_data.get('temperature', 100000.0)
        bz = current_data.get('bz', 0.0)
        
        # We need more features. Assume defaults or compute if possible
        bx = current_data.get('bx', 0.0)
        by = current_data.get('by', 0.0)
        bt = current_data.get('bt', abs(bz))
        
        dyn_pressure = current_data.get('dynamic_pressure', 1.67e-6 * density * (speed ** 2))
        e_field = current_data.get('electric_field', -speed * bz * 1e-3)
        plasma_beta = current_data.get('plasma_beta', 1.0)
        alfven_mach = current_data.get('alfven_mach', 10.0)
        
        # Derived historical features
        speed_lag_1h = speed
        density_lag_1h = density
        bz_lag_1h = bz
        speed_lag_3h = speed
        density_lag_3h = density
        bz_lag_3h = bz
        speed_ma_1h = speed
        bz_ma_1h = bz
        speed_roc = 0.0
        bz_std_1h = 0.0
        
        if history_df is not None and not history_df.empty:
            # We assume history_df has 'speed', 'density', 'bz' in 5-minute or 1-hour intervals
            if len(history_df) >= 2:
                speed_roc = (speed - history_df['speed'].iloc[-2]) / (history_df['speed'].iloc[-2] + 1e-5)
            if len(history_df) >= 12:  # Assuming 5-min intervals, 12 = 1 hour
                speed_lag_1h = history_df['speed'].iloc[-12]
                density_lag_1h = history_df['density'].iloc[-12]
                bz_lag_1h = history_df['bz'].iloc[-12]
                speed_ma_1h = history_df['speed'].tail(12).mean()
                bz_ma_1h = history_df['bz'].tail(12).mean()
                bz_std_1h = history_df['bz'].tail(12).std()
            if len(history_df) >= 36:
                speed_lag_3h = history_df['speed'].iloc[-36]
                density_lag_3h = history_df['density'].iloc[-36]
                bz_lag_3h = history_df['bz'].iloc[-36]
                
        features_list = ['speed', 'density', 'temperature', 'bz', 'bx', 'by', 'bt', 
                         'dynamic_pressure', 'electric_field', 'plasma_beta', 'alfven_mach',
                         'speed_lag_1h', 'density_lag_1h', 'bz_lag_1h',
                         'speed_lag_3h', 'density_lag_3h', 'bz_lag_3h',
                         'speed_ma_1h', 'bz_ma_1h', 'speed_roc', 'bz_std_1h']
                         
        feature_dict = {
            'speed': speed, 'density': density, 'temperature': temperature, 'bz': bz,
            'bx': bx, 'by': by, 'bt': bt,
            'dynamic_pressure': dyn_pressure, 'electric_field': e_field, 
            'plasma_beta': plasma_beta, 'alfven_mach': alfven_mach,
            'speed_lag_1h': speed_lag_1h, 'density_lag_1h': density_lag_1h, 'bz_lag_1h': bz_lag_1h,
            'speed_lag_3h': speed_lag_3h, 'density_lag_3h': density_lag_3h, 'bz_lag_3h': bz_lag_3h,
            'speed_ma_1h': speed_ma_1h, 'bz_ma_1h': bz_ma_1h, 'speed_roc': speed_roc, 'bz_std_1h': bz_std_1h
        }
        
        # In case some models drop NaNs, fill them
        for k in feature_dict:
            if pd.isna(feature_dict[k]) or feature_dict[k] == np.inf or feature_dict[k] == -np.inf:
                feature_dict[k] = 0.0
                
        features = pd.DataFrame([feature_dict], columns=features_list)
        
        # Regression
        predicted_speed = float(self.reg_model.predict(features)[0])
        
        # Classification
        risk_class_idx = int(self.clf_model.predict(features)[0])
        risk_probs = self.clf_model.predict_proba(features)[0]
        confidence = float(np.max(risk_probs))
        
        risk_mapping = {0: "Low", 1: "Moderate", 2: "High", 3: "Extreme"}
        risk_level = risk_mapping.get(risk_class_idx, "Unknown")
        
        # Confidence interval approximation for regression
        # Not all models have .estimators_ (e.g. XGB, LGBM), so we mock a 5% margin for now
        margin = predicted_speed * 0.05
        lower_bound = round(predicted_speed - margin, 2)
        upper_bound = round(predicted_speed + margin, 2)
            
        # SHAP
        feature_importance = {}
        top_factors = []
        if self.shap_explainer:
            try:
                shap_values = self.shap_explainer.shap_values(features)[0]
                total_shap = np.sum(np.abs(shap_values))
                if total_shap > 0:
                    for i, feat in enumerate(features_list):
                        importance = round(float(np.abs(shap_values[i]) / total_shap * 100), 2)
                        feature_importance[feat] = importance
                        if importance > 5.0:
                            top_factors.append({"name": feat, "value": importance})
                            
                    top_factors = sorted(top_factors, key=lambda x: x['value'], reverse=True)[:5]
            except:
                pass
                
        # Human Readable Explanation logic
        reasons = []
        if bz < -5: reasons.append("Negative Bz")
        if speed_roc > 0.05: reasons.append("Increasing Solar Wind")
        if density > 10: reasons.append("High Density")
        
        return {
            "prediction": {
                "current_speed": speed,
                "predicted_speed": round(predicted_speed, 2),
                "storm_risk": risk_level,
                "risk_probability": round(confidence * 100, 2),
                "confidence": round(confidence * 100, 2)
            },
            "confidence_interval": {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            },
            "xai": {
                "top_factors": top_factors,
                "explanation_reasons": reasons
            }
        }

predictor = SolarWindPredictor()
