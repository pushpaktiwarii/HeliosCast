# ☀️ HelioCast - AI-Powered Space Weather Intelligence

![HelioCast Banner](https://img.shields.io/badge/Status-Live-success?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-XGBoost_|_Scikit--Learn-orange?style=for-the-badge)

**Live Demo:** [https://helioscast.onrender.com](https://helioscast.onrender.com/)

HelioCast is an advanced, real-time Space Weather monitoring and prediction platform. It fetches live solar wind data from official NASA/NOAA satellites (DSCOVR/ACE) and uses an ensemble of Machine Learning models to predict future solar wind speeds and assess geomagnetic storm risks.

---

## 📡 Data Sources

This project relies on authentic space weather data streams provided by the US Government:
- **Live Real-time Data:** Fetched directly from the **NOAA Space Weather Prediction Center (SWPC)** APIs. The live data originates from the DSCOVR and ACE satellites positioned at the L1 Lagrange point.
  - *Plasma API:* `https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json`
  - *Magnetic Field API:* `https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json`
- **Training Data:** The AI models were trained on historical **OMNI** dataset records to accurately learn solar wind patterns and geomagnetic disturbances.

---

## 🚀 Key Features

- **Real-Time Data Streaming:** Uses WebSockets to stream live solar wind metrics (Speed, Density, Temperature, Bz) directly to the dashboard.
- **AI/ML Forecasting:** Uses a robust ML pipeline (XGBoost & RandomForest) trained on historical OMNI data to predict solar wind speed for the next hour.
- **Explainable AI (XAI):** Integrates **SHAP** (SHapley Additive exPlanations) to provide real-time transparency into which space weather parameters (e.g., Density vs. Magnetic Field Bz) are driving the AI\'s predictions.
- **Fault-Tolerant Cache System:** Built-in offline fallback mechanism. If the NOAA API goes down, the system gracefully switches to the last recorded cache, preventing the dashboard from crashing while displaying an honest timestamp to the user.
- **Stunning UI/UX:** A glassmorphism-inspired, dark-mode real-time dashboard built with vanilla JavaScript, Vite, and Chart.js.

## 🛠️ Tech Stack

### Backend (AI & API)
- **Framework:** FastAPI (Python)
- **Real-time:** WebSockets (Uvicorn)
- **Machine Learning:** XGBoost, Scikit-Learn, Pandas, NumPy
- **Explainability:** SHAP

### Frontend (Dashboard)
- **Bundler:** Vite
- **UI & Logic:** Vanilla JavaScript, HTML, CSS (Glassmorphism design)
- **Data Visualization:** Chart.js

## 📁 Project Structure

\\	ext
HeliosCast/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── predictions.py     # FastAPI REST & WebSocket routes
│   │   ├── ml/
│   │   │   └── predictor.py       # ML Inference & SHAP logic
│   │   └── main.py                # FastAPI App & Static File Serving
│   ├── data/                      # Historical CSVs & JSON Caches
│   ├── models/                    # Trained ML Models (.pkl, .keras)
│   ├── scripts/                   # Training & Pipeline Scripts
│   └── requirements.txt           # Python Dependencies
│
├── frontend/
│   ├── src/                       # Frontend JS, CSS, and Assets
│   ├── index.html                 # Dashboard Entry Point
│   ├── package.json               # Node Dependencies
│   └── dist/                      # Production Build (Served by FastAPI)
│
└── .python-version                # Enforces Python 3.10 for Cloud Deployment
\
## 💻 Local Development Setup

To run this project locally on your machine, follow these steps:

### 1. Clone the repository
\\ash
git clone https://github.com/pushpaktiwarii/HeliosCast.git
cd HeliosCast
\
### 2. Setup the Backend (FastAPI + AI)
\\ash
cd backend
python -m venv venv
# Activate virtual environment (Windows)
.\venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
# Run the server
uvicorn app.main:app --reload --port 8000
\
### 3. Setup the Frontend (Vite)
Open a new terminal window:
\\ash
cd frontend
npm install
# Run the development server
npm run dev
\The dashboard will be available at \http://localhost:5173\.

## 🌐 Deployment (Monolithic)

This project is configured to be deployed as a **single web service** on platforms like Render. The FastAPI backend is configured to automatically serve the statically built frontend.

1. Build the frontend: \cd frontend && npm run build2. Deploy the root directory as a Python Web Service.
3. Start command: \cd backend && uvicorn app.main:app --host 0.0.0.0 --port \

---
*Developed by [Pushpak Tiwari](https://github.com/pushpaktiwarii)*
