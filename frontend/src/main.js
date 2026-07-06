import './style.css'
import Chart from 'chart.js/auto';

const protocol = window.location.protocol === 'https:' ? 'https' : 'http';
const host = window.location.host || 'localhost:8000';

const API_URL = `${protocol}://${host}/api`;
let speedChartInstance = null;
let currentDataState = null;

async function init() {
  await fetchModelInfo();
  await fetchCurrentConditions();
  setupEventListeners();
}

async function fetchModelInfo() {
  try {
    const response = await fetch(`${API_URL}/model-info`);
    if (response.ok) {
      const info = await response.json();
      document.getElementById('analytics-algorithm').innerText = info.algorithm;
      document.getElementById('analytics-dataset').innerText = info.dataset;
      document.getElementById('analytics-features').innerText = info.features.join(', ');
      document.getElementById('analytics-rmse').innerText = `${info.rmse} km/s`;
      document.getElementById('analytics-mae').innerText = `${info.mae} km/s`;
      document.getElementById('analytics-interpretability').innerText = info.interpretability;
    }
  } catch (error) {
    console.error("Error fetching model info:", error);
  }
}

async function fetchCurrentConditions(isAuto = false) {
  try {
    const response = await fetch(`${API_URL}/current-conditions`, { cache: 'no-store' });
    if (!response.ok) throw new Error('Network response was not ok');
    
    const data = await response.json();
    
    // Check if new data actually arrived
    if (isAuto && currentDataState && currentDataState.current && data.current) {
      if (currentDataState.current.timestamp === data.current.timestamp) {
        return; // No new data, skip UI update
      }
    }
    
    currentDataState = data;
    
    if (data.is_cached && data.current && data.current.timestamp) {
      const cacheDate = new Date(data.current.timestamp).toLocaleString();
      document.querySelector('.status-indicator').innerHTML = `<span class="dot" style="background: orange;"></span> Showing Recorded Data from ${cacheDate} (NOAA Server Down)`;
    } else if (data.is_cached) {
      document.querySelector('.status-indicator').innerHTML = '<span class="dot" style="background: orange;"></span> Showing Recorded Data (NOAA Server Down)';
    } else {
      document.querySelector('.status-indicator').innerHTML = '<span class="dot live"></span> System Online (Latest Feed)';
    }

    const badge = document.querySelector('.live-badge');
    const trends = document.querySelectorAll('.metric-trend');
    const chartBtn = document.getElementById('chart-status-btn');
    
    if (data.is_cached) {
      if (badge) {
        badge.innerText = 'CACHED FEED (OFFLINE)';
        badge.style.background = 'orange';
        badge.style.color = '#000';
      }
      if (chartBtn) {
        chartBtn.innerText = 'Cached Data (Offline)';
        chartBtn.style.color = 'orange';
        chartBtn.style.borderColor = 'orange';
      }
      trends.forEach(t => {
        t.innerText = 'RECORDED OBSERVATION';
        t.style.color = 'orange';
      });
    } else {
      if (badge) {
        badge.innerText = 'LATEST NOAA FEED';
        badge.style.background = 'var(--success-color)';
        badge.style.color = 'var(--bg-color)';
      }
      if (chartBtn) {
        chartBtn.innerText = 'Latest Telemetry';
        chartBtn.style.color = '';
        chartBtn.style.borderColor = '';
      }
      trends.forEach(t => {
        t.innerText = 'LATEST OBSERVATION';
        t.style.color = 'var(--success-color)';
      });
    }
    
    updateMetrics(data.current);
    
    // Append the exact current point to history so the chart isn't stuck at the last hour boundary
    const chartData = [...data.history, data.current];
    renderChart(chartData);
    
    if (data.history.length >= 2) {
      await fetchPrediction(data.history, data.current, isAuto);
    }
  } catch (error) {
    console.error("Error fetching current conditions:", error);
  }
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
  
  if(result.confidence_interval) {
    const spread = (result.confidence_interval.upper_bound - result.predicted_speed).toFixed(1);
    document.getElementById('forecast-interval').innerText = `± ${spread} km/s (95% CI)`;
  }
  
  const badge = document.getElementById('risk-badge');
  badge.className = `risk-badge ${result.risk_level.toLowerCase()}`;
  badge.innerText = result.risk_level;
  
  const shapList = document.getElementById('shap-list');
  if(result.feature_importance) {
    const sortedFeatures = Object.entries(result.feature_importance)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .slice(0, 3);
      
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
  const labels = history.map((item, index) => {
    const d = new Date(item.timestamp);
    const timeStr = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    return index === history.length - 1 ? `${timeStr} (Latest)` : timeStr;
  });
  const data = history.map(item => item.speed);

  if (speedChartInstance) {
    speedChartInstance.data.labels = labels;
    speedChartInstance.data.datasets[0].data = data;
    speedChartInstance.update('none');
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
        duration: 0
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

  const tabs = ['dashboard', 'analytics', 'settings'];
  tabs.forEach(tab => {
    document.getElementById(`nav-${tab}`).addEventListener('click', (e) => {
      tabs.forEach(t => document.getElementById(`nav-${t}`).classList.remove('active'));
      tabs.forEach(t => document.getElementById(`view-${t}`).classList.add('hidden'));
      
      e.target.classList.add('active');
      document.getElementById(`view-${tab}`).classList.remove('hidden');
    });
  });

  const refreshSelect = document.getElementById('setting-refresh');
  refreshSelect.addEventListener('change', (e) => {
    const val = e.target.value;
    if (refreshIntervalId) clearInterval(refreshIntervalId);
    
    if (parseInt(val) > 0) {
      refreshIntervalId = setInterval(() => fetchCurrentConditions(true), parseInt(val));
    }
  });
}

document.addEventListener('DOMContentLoaded', () => {
  init();
  refreshIntervalId = setInterval(() => fetchCurrentConditions(true), 30000);
});
