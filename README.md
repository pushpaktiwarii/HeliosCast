# HeliosCast v2.0: Advanced Space Weather Intelligence

**An Enterprise-Grade, AI-Powered Telemetry and Prediction Platform for Space Weather (Geomagnetic Storms & Solar Winds).**

HeliosCast bridges the gap between raw astrophysical data and actionable intelligence. Built with a robust microservices architecture, this platform ingests live telemetry from NASA's DSCOVR and ACE satellites (via NOAA APIs), processes the physics parameters, and utilizes ensemble Machine Learning models to forecast Geomagnetic Storm Risks with a 1-hour horizon (T+1).

This project was engineered to meet high-tier enterprise standards, focusing on **Explainable AI (XAI)**, **Cloud-Ready API-First Design**, and **Automated Generative Telemetry Assessment**.

## Live Demo
 **[http://150.230.239.72:8000/](http://150.230.239.72:8000/)**
  **[http://helioscast.pushpaktiwari.tech/)**

 

---

## Key Features

- **Real-Time Data Ingestion:** Automated polling of real-time Solar Plasma metrics (Density, Speed, Temperature, and Magnetic Vectors: Bz, Bt) directly from the NOAA Space Weather Prediction Center.
- **Ensemble ML Prediction Engine:** Utilizes optimized LightGBM and Random Forest architectures trained on historical NASA OMNI datasets. Outputs regression (Solar Wind Speed) and multi-class risk classification (Low, Moderate, High, Extreme).
- **Explainable AI (SHAP):** Integrates SHAP (SHapley Additive exPlanations) to provide feature importance transparency. The system explicitly defines *why* it reached a specific risk conclusion (e.g., negative Bz threshold, dynamic pressure variance).
- **Automated Telemetry Assessment:** Synthesizes raw predictive metrics into a professional, human-readable System Log, mimicking Enterprise Generative AI diagnostics.
- **Auto-Pruning SQLite Database:** Intelligent storage management that automatically caps historical prediction bounds to prevent server memory bloat during deployment.
- **API Developer Portal:** Fully integrated FastAPI backend with interactive Swagger UI documentation for seamless integration into other cloud services.

## Architecture & Tech Stack

- **Backend:** Python 3, FastAPI, Pandas, Scikit-Learn, LightGBM, XGBoost, SHAP, SQLite.
- **Frontend:** Vanilla JavaScript, Vite, Chart.js, HTML5/CSS3 (Glassmorphism UI).
- **Data Source:** NOAA SWPC JSON APIs.

## Local Setup & Deployment

### 1. Start the Backend (API & ML Engine)
```bash
cd backend
python -m venv venv
# On Windows: .\venv\Scripts\activate
# On Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python app/main.py
```
*The backend API and Swagger Docs will run on `http://127.0.0.1:8000`*

### 2. Start the Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
```
*The dashboard will be available at `http://localhost:5173`*

## ML Model Pipeline
The ML pipeline is defined in `backend/scripts/train_model.py`. It uses a rigorous time-series sequential split to prevent data leakage. Features are engineered across 21 parameters, including derived physics (Alfven Mach, Plasma Beta, Dynamic Pressure) and rolling lags.

## Live Alerts
HeliosCast actively listens for real-time unstructured warnings broadcasted by NOAA and renders critical alerts globally on the frontend dashboard.

---
*Built as a showcase for Advanced AI and Data Engineering.*
