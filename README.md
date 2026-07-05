# HelioCast: Advanced Space Weather Intelligence Platform

## Problem Statement
Space weather—comprising solar flares, coronal mass ejections (CMEs), and high-speed solar wind streams—poses a significant threat to modern technological infrastructure. Severe geomagnetic storms can disrupt satellite communications, damage power grids, and endanger astronauts. Accurate, real-time forecasting of solar wind speed and its associated parameters is essential to provide early warnings and mitigate these risks. HelioCast aims to provide an advanced, explainable, and highly accurate forecasting platform for short-term space weather anomalies.

## Dataset
This project utilizes the **OMNI2 Space Weather Dataset** provided by NASA/NOAA. The dataset aggregates multi-spacecraft observations into a continuous timeline.
- **Historical Training Data**: ~100,000 real chronological records encompassing solar wind speed, proton density, plasma temperature, and the Z-component of the interplanetary magnetic field (Bz).
- **Live Inference Data**: Real-time integration with the NOAA Space Weather Prediction Center (SWPC) 1-minute JSON feeds (`plasma-1-day.json` and `mag-1-day.json`).

## Methodology
The platform employs a robust Machine Learning pipeline that forecasts solar wind speed 1-hour into the future:
1. **Feature Engineering**: Incorporation of temporal lag features ($t-1$ and $t-2$) for speed and density to capture time-series momentum and trends.
2. **Model Ensemble Comparison**:
   - **Random Forest Regressor**: Used primarily to extract standard deviations across decision trees for prediction confidence intervals.
   - **XGBoost (Extreme Gradient Boosting)**: Employed as the primary inference engine due to its exceptional performance and speed on tabular time-series data.
   - **LSTM (Long Short-Term Memory Network)**: A deep learning baseline utilizing Keras/TensorFlow to capture complex non-linear temporal dependencies.
3. **Explainable AI (XAI)**: `shap.TreeExplainer` is integrated with the XGBoost model to compute SHAP (SHapley Additive exPlanations) values in real-time, providing transparency into which physical parameters drove a specific forecast.

## Results
The models were evaluated chronologically on a hold-out test set (20%). The primary evaluation metrics are Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE):
- **XGBoost Regressor**: MAE ~ 10.44 km/s | RMSE ~ 17.89 km/s
- **Random Forest**: MAE ~ 10.50 km/s | RMSE ~ 18.10 km/s
- **LSTM Network**: Comparable baseline efficiency, though slightly higher latency for real-time inference.

*Note: With an average solar wind speed of ~400 km/s, an MAE of 10.44 km/s represents an error margin of approximately 2.5%, indicating high forecasting reliability.*

## Architecture
The HelioCast platform is designed as a decoupled, modern web application:
- **Backend**: FastAPI (Python) serving REST endpoints.
  - `/current-conditions`: Fetches and streams live SWPC JSON data.
  - `/predict`: Executes the XGBoost model, computes the 95% confidence interval via Random Forest, and generates SHAP values.
- **Frontend**: A sleek, glassmorphism UI built with Vanilla JS, HTML, and CSS (via Vite), utilizing `Chart.js` for dynamic time-series visualization.
- **Machine Learning**: `scikit-learn`, `xgboost`, `tensorflow`, and `shap`.

## Future Work
- **Geomagnetic Storm Classification**: Expand the predictive targets to include the Kp-Index for direct classification of storm severity (G1 to G5).
- **Multi-step Forecasting**: Extend the forecasting horizon from 1 hour to 24-72 hours using sequence-to-sequence LSTM or Transformer architectures.
- **Solar Image Data**: Incorporate CNNs trained on live SDO (Solar Dynamics Observatory) imagery (e.g., AIA 193 Å) to predict CMEs before the solar wind reaches the L1 Lagrange point.
