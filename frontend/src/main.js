import './style.css'
import Chart from 'chart.js/auto';

const protocol = window.location.protocol === 'https:' ? 'https' : 'http';
const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
// Use the current host, defaulting to localhost:8000 for local dev if opened directly as file
const host = window.location.host || 'localhost:8000';

const API_URL = `${protocol}://${host}/api`;
const WS_URL = `${wsProtocol}://${host}/api/ws/current-conditions`;
let speedChartInstance = null;
let currentDataState = null;
let wsSocket = null;
let useRealtime = true;
let reconnectTimeout = null;

// Initialize the app
async function init() {
  await fetchCurrentConditions();
  setupEventListeners();
}

async function fetchCurrentConditions(isAuto = false) {
  try {
    const response = await fetch(`${API_URL}/current-conditions`);
    if (!response.ok) throw new Error('Network response was not ok');
    
    const data = await response.json();
    currentDataState = data;
    
    if (data.is_cached && data.current && data.current.timestamp) {
      const cacheDate = new Date(data.current.timestamp).toLocaleDateString();
      document.querySelector('.status-indicator').innerHTML = `<span class="dot" style="background: orange;"></span> Showing Recorded Data from ${cacheDate} (NOAA Server Down)`;
    } else if (data.is_cached) {
      document.querySelector('.status-indicator').innerHTML = '<span class="dot" style="background: orange;"></span> Showing Recorded Data (NOAA Server Down)';
    } else {
      document.querySelector('.status-indicator').innerHTML = '<span class="dot live"></span> System Online (Real-time)';
    }
    
    updateMetrics(data.current);
    renderChart(data.history);
    
    if (data.history.length >= 2) {
      await fetchPrediction(data.history, data.current, isAuto);
    }
  } catch (error) {
    console.error("Error fetching current conditions:", error);
  }
}

function connectWebSocket() {
  if (wsSocket) {
    wsSocket.close();
  }
  useRealtime = true;
  wsSocket = new WebSocket(WS_URL);
  
  wsSocket.onopen = () => {
    console.log("WebSocket connected");
    document.querySelector('.status-indicator').innerHTML = '<span class="dot live"></span> System Online (Real-time)';
  };
  
  wsSocket.onmessage = async (event) => {
    const data = JSON.parse(event.data);
    
    if (data.error) {
      document.querySelector('.status-indicator').innerHTML = '<span class="dot" style="background: red;"></span> NOAA API Offline';
      return;
    }
    
    if (data.is_cached && data.current && data.current.timestamp) {
      const cacheDate = new Date(data.current.timestamp).toLocaleDateString();
      document.querySelector('.status-indicator').innerHTML = `<span class="dot" style="background: orange;"></span> Showing Recorded Data from ${cacheDate} (NOAA Server Down)`;
    } else if (data.is_cached) {
      document.querySelector('.status-indicator').innerHTML = '<span class="dot" style="background: orange;"></span> Showing Recorded Data (NOAA Server Down)';
    } else {
      document.querySelector('.status-indicator').innerHTML = '<span class="dot live"></span> System Online (Real-time)';
    }
    currentDataState = data;
    
    updateMetrics(data.current);
    renderChart(data.history);
    
    if (data.history && data.history.length >= 2) {
      await fetchPrediction(data.history, data.current, true); // Auto prediction silently
    }
  };
  
  wsSocket.onclose = () => {
    console.log("WebSocket disconnected, retrying in 5s...");
    document.querySelector('.status-indicator').innerHTML = '<span class="dot" style="background: red;"></span> System Offline';
    if (useRealtime) {
      reconnectTimeout = setTimeout(connectWebSocket, 5000);
    }
  };
}

async function fetchPrediction(history, currentData, isAuto = false) {
  const current = currentData;
  const t1 = history[history.length - 1];
  const t2 = history.length > 1 ? history[history.length - 2] : t1;

  const payload = {
    speed: current.speed,
    density: current.density,
    temperature: current.temperature,
    bz: current.bz,
    speed_t1: t1.speed,
    speed_t2: t2.speed,
    density_t1: t1.density,
    bz_t1: t1.bz
  };

  const btn = document.getElementById('btn-refresh');
  let originalText;
  if (!isAuto) {
    originalText = btn.innerText;
    btn.innerText = 'Analyzing...';
    btn.disabled = true;
  }

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (!response.ok) throw new Error('Prediction failed');
    
    const result = await response.json();
    updatePredictionUI(result, isAuto);
  } catch (error) {
    console.error("Error fetching prediction:", error);
  } finally {
    if (!isAuto) {
      btn.innerText = originalText;
      btn.disabled = false;
    }
  }
}

function updateMetrics(current) {
  document.querySelector('#card-speed .metric-value').innerHTML = `${current.speed} <span class="unit">km/s</span>`;
  document.querySelector('#card-density .metric-value').innerHTML = `${current.density} <span class="unit">p/cm³</span>`;
  document.querySelector('#card-temperature .metric-value').innerHTML = `${(current.temperature / 1000).toFixed(1)}k <span class="unit">K</span>`;
  document.querySelector('#card-bz .metric-value').innerHTML = `${current.bz} <span class="unit">nT</span>`;
}

function updatePredictionUI(result, isAuto = false) {
  // Update Forecast Value with Animation
  const speedEl = document.getElementById('forecast-speed');
  const target = result.predicted_speed;
  const duration = isAuto ? 0 : 1000;
  
  if (isAuto) {
    speedEl.innerText = `${target.toFixed(2)} km/s`;
  } else {
    const startTime = performance.now();
    function updateNumber(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const easeOutQuart = 1 - Math.pow(1 - progress, 4);
      const currentVal = (target * easeOutQuart).toFixed(2);
      
      speedEl.innerText = `${currentVal} km/s`;
      
      if (progress < 1) {
        requestAnimationFrame(updateNumber);
      } else {
        speedEl.innerText = `${target.toFixed(2)} km/s`;
      }
    }
    requestAnimationFrame(updateNumber);
  }
  
  // Update Confidence Interval
  if(result.confidence_interval) {
    const spread = (result.confidence_interval.upper_bound - result.predicted_speed).toFixed(1);
    document.getElementById('forecast-interval').innerText = `± ${spread} km/s (95% CI)`;
  }
  
  // Update Risk Badge
  const badge = document.getElementById('risk-badge');
  badge.className = `risk-badge ${result.risk_level.toLowerCase()}`;
  badge.innerText = result.risk_level;
  
  // Update SHAP Feature Importance
  const shapList = document.getElementById('shap-list');
  if(result.feature_importance) {
    // Sort by absolute importance value
    const sortedFeatures = Object.entries(result.feature_importance)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .slice(0, 3); // top 3
      
    shapList.innerHTML = sortedFeatures.map(([feat, val]) => {
      const isPos = val > 0;
      const valClass = isPos ? 'positive' : 'negative';
      const sign = isPos ? '+' : '';
      return `<div class="shap-item"><span class="shap-label">${feat.toUpperCase()}</span><span class="shap-val ${valClass}">${sign}${val.toFixed(2)}</span></div>`;
    }).join('');
  }
}

function renderChart(history) {
  const ctx = document.getElementById('speedChart').getContext('2d');
  const labels = history.map(item => {
    const d = new Date(item.timestamp);
    return `${String(d.getHours()).padStart(2, '0')}:00`;
  });
  const data = history.map(item => item.speed);

  // If chart exists, update it smoothly
  if (speedChartInstance) {
    speedChartInstance.data.labels = labels;
    speedChartInstance.data.datasets[0].data = data;
    speedChartInstance.update('none'); // no animation for smooth update
    return;
  }

  const gradient = ctx.createLinearGradient(0, 0, 0, 400);
  gradient.addColorStop(0, 'rgba(0, 112, 243, 0.2)');
  gradient.addColorStop(1, 'rgba(0, 112, 243, 0.0)');

  Chart.defaults.color = '#a3a3a3';
  Chart.defaults.font.family = "'Inter', sans-serif";

  speedChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Solar Wind Speed (km/s)',
        data: data,
        borderColor: '#0070f3',
        borderWidth: 2,
        backgroundColor: gradient,
        fill: true,
        tension: 0.2,
        pointBackgroundColor: '#121212',
        pointBorderColor: '#0070f3',
        pointBorderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 5
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#050505',
          titleFont: { size: 12, weight: '600' },
          bodyFont: { size: 13 },
          padding: 10,
          borderColor: '#333333',
          borderWidth: 1,
          displayColors: false
        }
      },
      scales: {
        x: { grid: { color: 'transparent', drawBorder: false } },
        y: { 
          grid: { color: '#333333', drawBorder: false },
          suggestedMin: 300,
          suggestedMax: 500
        }
      },
      interaction: {
        intersect: false,
        mode: 'index',
      },
      animation: {
        duration: 0 // Disable initial animation for snappiness
      }
    }
  });
}

let refreshIntervalId = null;

function setupEventListeners() {
  document.getElementById('btn-refresh').addEventListener('click', () => {
    if (currentDataState && currentDataState.history && currentDataState.current) {
      fetchPrediction(currentDataState.history, currentDataState.current);
    }
  });

  // Tab Navigation Logic
  const tabs = ['dashboard', 'analytics', 'settings'];
  tabs.forEach(tab => {
    document.getElementById(`nav-${tab}`).addEventListener('click', (e) => {
      // Deactivate all nav items
      tabs.forEach(t => document.getElementById(`nav-${t}`).classList.remove('active'));
      // Hide all views
      tabs.forEach(t => document.getElementById(`view-${t}`).classList.add('hidden'));
      
      // Activate clicked nav item
      e.target.classList.add('active');
      // Show corresponding view
      document.getElementById(`view-${tab}`).classList.remove('hidden');
    });
  });

  // Settings: Auto-Refresh Interval
  const refreshSelect = document.getElementById('setting-refresh');
  refreshSelect.addEventListener('change', (e) => {
    const val = e.target.value;
    if (refreshIntervalId) clearInterval(refreshIntervalId);
    
    if (val === 'realtime') {
      connectWebSocket();
    } else {
      useRealtime = false;
      if (wsSocket) wsSocket.close();
      clearTimeout(reconnectTimeout);
      
      if (parseInt(val) > 0) {
        refreshIntervalId = setInterval(() => fetchCurrentConditions(true), parseInt(val));
      }
    }
  });
}

// Boot
document.addEventListener('DOMContentLoaded', () => {
  init();
  connectWebSocket(); // Start real-time sync by default
});
