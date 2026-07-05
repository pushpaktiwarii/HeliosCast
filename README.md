# HelioCast - AI-Powered Space Weather Intelligence

Live Demo: https://helioscast.onrender.com/

HelioCast is an advanced, real-time Space Weather monitoring and prediction platform. It fetches live solar wind data from official NASA/NOAA satellites (DSCOVR/ACE) and uses an ensemble of Machine Learning models to predict future solar wind speeds and assess geomagnetic storm risks.

---

## Data Sources

This project relies on authentic space weather data streams provided by the US Government:
- Live Real-time Data: Fetched directly from the NOAA Space Weather Prediction Center (SWPC) APIs. The live data originates from the DSCOVR and ACE satellites positioned at the L1 Lagrange point.
  - Plasma API: https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json
  - Magnetic Field API: https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json
- Training Data: The AI models were trained on historical OMNI dataset records to accurately learn solar wind patterns and geomagnetic disturbances.

---

## Key Features

- Real-Time Data Streaming: Uses WebSockets to stream live solar wind metrics (Speed, Density, Temperature, Bz) directly to the dashboard.
- AI/ML Forecasting: Uses a robust ML pipeline (XGBoost & RandomForest) trained on historical OMNI data to predict solar wind speed for the next hour.
- Explainable AI (XAI): Integrates SHAP (SHapley Additive exPlanations) to provide real-time transparency into which space weather parameters (e.g., Density vs. Magnetic Field Bz) are driving the AI's predictions.
- Fault-Tolerant Cache System: Built-in offline fallback mechanism. If the NOAA API goes down, the system gracefully switches to the last recorded cache, preventing the dashboard from crashing while displaying an honest timestamp to the user.
- UI/UX: A dark-mode real-time dashboard built with vanilla JavaScript, Vite, and Chart.js.

## Tech Stack

Backend (AI & API)
- Framework: FastAPI (Python)
- Real-time: WebSockets (Uvicorn)
- Machine Learning: XGBoost, Scikit-Learn, Pandas, NumPy
- Explainability: SHAP

Frontend (Dashboard)
- Bundler: Vite
- UI & Logic: Vanilla JavaScript, HTML, CSS
- Data Visualization: Chart.js

## Local Development Setup

To run this project locally on your machine, follow these steps:

1. Clone the repository
```bash
git clone https://github.com/pushpaktiwarii/HeliosCast.git
cd HeliosCast
```

2. Setup the Backend (FastAPI + AI)
```bash
cd backend
python -m venv venv
# Activate virtual environment (Windows)
.\venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
# Run the server
uvicorn app.main:app --reload --port 8000
```

3. Setup the Frontend (Vite)
Open a new terminal window:
```bash
cd frontend
npm install
# Run the development server
npm run dev
```
The dashboard will be available at http://localhost:5173.

## Deployment (Monolithic)

This project is configured to be deployed as a single web service on platforms like Render. The FastAPI backend is configured to automatically serve the statically built frontend.

1. Build the frontend: cd frontend && npm run build
2. Deploy the root directory as a Python Web Service.
3. Start command: cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT
## Contributing

If you'd like to contribute, you're most welcome! Some ideas include:

- Improve the ML models (XGBoost, LSTM, Transformers)
- Improve forecasting accuracy and feature engineering
- Integrate additional NASA/NOAA datasets
- Add Solar Flare forecasting
- Add Geomagnetic Storm prediction
- Add Satellite Risk Analysis
- Improve the UI/UX and dashboard
- Add tests, optimize performance, and improve documentation

Feel free to open an issue or submit a pull request. https://github.com/pushpaktiwarii/HeliosCast
