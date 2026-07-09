import Chart from 'chart.js/auto';

// Global state
let currentData = null;
let historyData = null;
let modelStats = null;
let charts = {};

const factorNames = {
    'speed': 'Current Speed',
    'density': 'Particle Density',
    'bz': 'Magnetic Shield (Bz)',
    'bt': 'Total Magnetism (Bt)',
    'bx': 'Magnetic Field (Bx)',
    'by': 'Magnetic Field (By)',
    'temperature': 'Temperature',
    'dynamic_pressure': 'Wind Pressure',
    'electric_field': 'Electric Field',
    'plasma_beta': 'Plasma Beta',
    'alfven_mach': 'Alfven Mach',
    'speed_lag_1h': 'Previous Hour Speed',
    'speed_lag_3h': 'Speed 3 Hours Ago',
    'density_lag_1h': 'Previous Hour Density',
    'density_lag_3h': 'Density 3 Hours Ago',
    'bz_lag_1h': 'Previous Hour Shield (Bz)',
    'bz_lag_3h': 'Shield 3 Hours Ago (Bz)',
    'speed_ma_1h': 'Average Speed (Last Hour)',
    'bz_ma_1h': 'Average Shield (Last Hour)',
    'speed_roc': 'Speed Acceleration',
    'bz_std_1h': 'Magnetic Shield Fluctuation'
};

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    fetchData();
    setInterval(fetchData, 60000); // Poll every minute
});

function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            const targetId = `tab-${btn.dataset.tab}`;
            document.getElementById(targetId).classList.add('active');
        });
    });
}

async function fetchData() {
    try {
        const [currentRes, predHistRes, modelRes, alertsRes] = await Promise.all([
            fetch('/api/current-conditions'),
            fetch('/api/prediction-history'),
            fetch('/api/model-training-stats'),
            fetch('/api/alerts')
        ]);

        const currentJson = await currentRes.json();
        const predHistJson = await predHistRes.json();
        const modelJson = await modelRes.json();
        const alertsJson = await alertsRes.json();

        currentData = currentJson;
        
        // System Online State
        document.getElementById('status-dot').style.background = 'var(--risk-low)';
        document.getElementById('status-dot').style.boxShadow = '0 0 8px var(--risk-low)';
        document.getElementById('status-text').innerText = 'System Online';

        updateLiveDashboard(currentJson, alertsJson);
        updatePredictionHistory(predHistJson);
        updateModelTraining(modelJson);
        
    } catch (err) {
        console.error("Error fetching data:", err);
        // System Offline State
        document.getElementById('status-dot').style.background = 'var(--risk-high)';
        document.getElementById('status-dot').style.boxShadow = '0 0 8px var(--risk-high)';
        document.getElementById('status-text').innerText = 'System Offline';
    }
}

function updateLiveDashboard(data, alerts) {
    const { current, prediction } = data;
    const pred = prediction.prediction;

    document.getElementById('current-speed').innerText = `${current.speed} km/s`;
    document.getElementById('predicted-speed').innerText = `${pred.predicted_speed} km/s`;
    
    const riskEl = document.getElementById('storm-risk');
    riskEl.innerText = pred.storm_risk;
    riskEl.className = `value risk-level risk-${pred.storm_risk.toLowerCase()}`;
    
    document.getElementById('risk-prob').innerText = pred.risk_probability;
    document.getElementById('conf-interval').innerText = `[${prediction.confidence_interval.lower_bound} - ${prediction.confidence_interval.upper_bound}]`;
    document.getElementById('current-bz').innerText = `${current.bz} nT`;

    const getStatus = (val, thresholds) => {
        if(val > thresholds.high) return `<span class="badge" style="background:rgba(239, 68, 68, 0.2); color:#fca5a5;">High</span>`;
        if(val > thresholds.elevated) return `<span class="badge" style="background:rgba(245, 158, 11, 0.2); color:#fcd34d;">Elevated</span>`;
        return `<span class="badge" style="background:rgba(16, 185, 129, 0.2); color:#6ee7b7;">Normal</span>`;
    };

    document.getElementById('live-density').innerHTML = `${current.density} p/cc ${getStatus(current.density, {elevated: 15, high: 30})}`;
    document.getElementById('live-temp').innerHTML = `${current.temperature} K ${getStatus(current.temperature, {elevated: 300000, high: 500000})}`;
    document.getElementById('live-pressure').innerHTML = `${current.dynamic_pressure} nPa ${getStatus(current.dynamic_pressure, {elevated: 3, high: 6})}`;
    document.getElementById('live-bt').innerHTML = `${current.bt} nT ${getStatus(current.bt, {elevated: 15, high: 25})}`;

    // XAI
    const xai = prediction.xai;
    document.getElementById('xai-summary').innerText = `Primary AI triggers: ${xai.explanation_reasons.join(', ') || 'Normal conditions'}`;
    
    const factorsList = document.getElementById('top-factors');
    factorsList.innerHTML = '';
    xai.top_factors.forEach(f => {
        const simpleName = factorNames[f.name] || f.name.replace(/_/g, ' ');
        factorsList.innerHTML += `
            <div class="factor-item">
                <span>${simpleName}</span>
                <div class="factor-bar-bg"><div class="factor-bar" style="width: ${f.value}%"></div></div>
                <span>${f.value}%</span>
            </div>
        `;
    });

    // TL;DR Banner Update
    const banner = document.getElementById('tldr-banner');
    const bTitle = document.getElementById('tldr-title');
    const bDesc = document.getElementById('tldr-desc');
    const bIcon = document.getElementById('tldr-icon');

    if (pred.storm_risk === 'High' || pred.storm_risk === 'Extreme') {
        banner.style.borderLeftColor = 'var(--risk-high)';
        banner.style.background = 'transparent';
        bIcon.innerText = '!';
        bIcon.style.color = 'var(--risk-high)';
        bTitle.innerText = 'Geomagnetic Storm Warning.';
        bDesc.innerText = 'Fast solar wind and negative magnetic fields are hitting Earth. Power grids and GPS may experience minor issues.';
    } else if (pred.storm_risk === 'Moderate') {
        banner.style.borderLeftColor = 'var(--risk-medium)';
        banner.style.background = 'transparent';
        bIcon.innerText = '!';
        bIcon.style.color = 'var(--risk-medium)';
        bTitle.innerText = 'Space Weather is Active.';
        bDesc.innerText = 'Conditions are slightly elevated. Auroras may be visible at high latitudes, but no major threats to satellites.';
    } else {
        banner.style.borderLeftColor = 'var(--risk-low)';
        banner.style.background = 'transparent';
        bIcon.style.color = 'var(--risk-low)';
        bIcon.innerHTML = '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>';
        bTitle.innerText = 'All Systems Normal';
        bDesc.innerText = 'All solar indicators are currently within safe operational ranges.';
    }
    
    // NOAA Alerts Processing
    const tickerContainer = document.getElementById('alerts-ticker');
    const tickerText = document.getElementById('alerts-text');
    const alertsTitle = document.getElementById('alerts-title');
    
    if (alerts && alerts.length > 0) {
        tickerContainer.style.display = 'block';
        if (alertsTitle) {
            alertsTitle.innerText = `${alerts.length} NOAA Active Alert${alerts.length > 1 ? 's' : ''}`;
        }
        const latestAlerts = alerts.slice(0, 3);
        const alertMessages = latestAlerts.map(a => {
            const time = new Date(a.issue_datetime).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            
            let msg = a.message || '';
            msg = msg.replace(/^Space Weather Message Code:.*$/gm, '');
            msg = msg.replace(/^Serial Number:.*$/gm, '');
            msg = msg.replace(/^Issue Time:.*$/gm, '');
            msg = msg.replace(/^Valid From:.*$/gm, '');
            msg = msg.replace(/^Valid To:.*$/gm, '');
            msg = msg.replace(/^Warning Conditions:.*$/gm, '');
            msg = msg.replace(/^Threshold Reached:.*$/gm, '');
            msg = msg.replace(/^Synoptic Period:.*$/gm, '');
            msg = msg.replace(/^Active Warning:.*$/gm, '');
            msg = msg.replace(/^Comment:.*$/gm, '');
            msg = msg.replace(/NOAA Space Weather Scale descriptions can be found.*$/gm, '');
            
            // Clean up empty lines
            msg = msg.replace(/^\s*[\r\n]/gm, '');
            
            msg = msg.trim().replace(/\n/g, '<br>');
            msg = msg.replace(/(WARNING:|ALERT:|SUMMARY:|WATCH:|CANCELLATION:|Potential Impacts:|Aurora:|Induced Currents:)/g, '<strong style="color: var(--accent-color);">$1</strong>');
            
            return `<div style="margin-bottom: 0.75rem; padding: 0.75rem; background: var(--panel-hover-bg); border-radius: 6px; border-left: 3px solid var(--accent-color);"><span style="color: var(--text-muted); font-size: 0.75rem; font-weight: 600;">[${time}]</span><br><div style="margin-top: 0.4rem; font-size: 0.85rem; line-height: 1.4;">${msg}</div></div>`;
        }).join('');
        tickerText.innerHTML = alertMessages;
    } else {
        tickerContainer.style.display = 'none';
    }

    // Generative AI Analyst Report
    const analystReportEl = document.getElementById('ai-analyst-report');
    let summary = '';
    
    if (pred.storm_risk === 'High' || pred.storm_risk === 'Extreme') {
        summary = `[SEV-1] CRITICAL ADVISORY: Geomagnetic anomaly detected. Solar wind velocity (${current.speed} km/s) exceeds nominal thresholds. Interplanetary Magnetic Field (IMF) Bz vector is south-pointing (${current.bz} nT), indicating magnetospheric vulnerability. Primary predictive driver: ${xai.explanation_reasons[0] || 'elevated velocity'}. Grid infrastructure monitoring recommended.`;
    } else if (pred.storm_risk === 'Moderate') {
        summary = `[SEV-3] ELEVATED ACTIVITY: Space weather perturbations detected. Dynamic pressure recorded at ${current.dynamic_pressure} nPa with ion density at ${current.density} p/cc. 1-hour forecast indicates velocities up to ${pred.predicted_speed} km/s. High-latitude communication disruptions possible; orbital asset charging risks remain marginal.`;
    } else {
        summary = `[SEV-5] NOMINAL: Space weather environment remains stable. Velocity parameters (${current.speed} km/s) and IMF Bz (${current.bz} nT) are within safe operational bounds. Telemetry forecasts continued stability over the T+1 hour horizon. No anomalous signatures detected.`;
    }
    
    analystReportEl.innerHTML = summary.replace(/\*\*(.*?)\*\*/g, '<strong style="color:var(--text-primary);">$1</strong>');
}



function updatePredictionHistory(history) {
    const tbody = document.getElementById('history-tbody');
    tbody.innerHTML = '';
    
    let totalError = 0;
    let worstError = 0;
    let errorCount = 0;
    
    if (history.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding: 2rem; color: var(--text-secondary);">
        ⏳ Predictions are being recorded in the background. The first results will appear here once their 1-hour target time is reached!
        </td></tr>`;
        
        document.getElementById('ea-avg').innerText = '--';
        document.getElementById('ea-worst').innerText = '--';
        return;
    }
    
    history.forEach(row => {
        const tr = document.createElement('tr');
        const d = new Date(row.timestamp);
        const time = d.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        
        // Add 1 hour for Target Time
        d.setHours(d.getHours() + 1);
        const targetTime = d.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});

        const actual = row.actual_speed ? row.actual_speed.toFixed(2) : 'Pending';
        const error = row.error ? row.error.toFixed(2) : '--';
        
        if (row.error) {
            totalError += row.error;
            errorCount++;
            if (row.error > worstError) worstError = row.error;
        }

        tr.innerHTML = `
            <td>${time}</td>
            <td style="color: var(--primary); font-weight: bold;">${targetTime}</td>
            <td>${row.predicted_speed.toFixed(2)}</td>
            <td>${actual}</td>
            <td>${error}</td>
            <td><span class="risk-level risk-${row.risk_class.toLowerCase()}">${row.risk_class}</span></td>
        `;
        tbody.appendChild(tr);
    });

    if (errorCount > 0) {
        document.getElementById('ea-avg').innerText = (totalError / errorCount).toFixed(2);
        document.getElementById('ea-worst').innerText = worstError.toFixed(2);
    }
}

function updateModelTraining(modelJson) {
    if (modelJson.error) return;
    
    document.getElementById('mt-dataset').innerText = `${modelJson.dataset_records} rows`;
    document.getElementById('mt-best').innerText = modelJson.best_regression;

    const regBody = document.getElementById('mt-reg-body');
    regBody.innerHTML = '';
    for (const [name, metrics] of Object.entries(modelJson.regression_metrics)) {
        regBody.innerHTML += `
            <tr>
                <td>${name}</td>
                <td>${metrics.MAE.toFixed(2)}</td>
                <td>${metrics.RMSE.toFixed(2)}</td>
                <td>${metrics.R2.toFixed(3)}</td>
            </tr>
        `;
    }

    const clfBody = document.getElementById('mt-clf-body');
    clfBody.innerHTML = '';
    for (const [name, metrics] of Object.entries(modelJson.classification_metrics)) {
        clfBody.innerHTML += `
            <tr>
                <td>${name}</td>
                <td>${(metrics.Accuracy * 100).toFixed(1)}%</td>
                <td>${(metrics.F1 * 100).toFixed(1)}%</td>
            </tr>
        `;
    }

    // Feature Importance Chart
    if (modelJson.feature_importance) {
        const sortedImp = Object.entries(modelJson.feature_importance)
            .sort((a,b) => b[1] - a[1]).slice(0, 10);
        
        createOrUpdateChart('chart-feature-imp', 'Importance', sortedImp.map(x => x[0]), sortedImp.map(x => x[1]), '#2f81f7');
        
        // Feature List
        const featureList = document.getElementById('feature-list');
        featureList.innerHTML = '';
        if (modelJson.features) {
            modelJson.features.forEach(f => {
                let badgeClass = f.includes('lag') || f.includes('ma_') || f.includes('roc') || f.includes('std') ? 'risk-moderate' : 'risk-low';
                let niceName = factorNames[f] || f;
                featureList.innerHTML += `<span class="badge ${badgeClass}" style="margin:0; font-size: 0.8rem; border: 1px solid rgba(255,255,255,0.1)">${niceName}</span>`;
            });
            document.getElementById('mt-feature-count').innerText = `${modelJson.features.length} Features Used`;
        }
    }
}

// Global chart instances
let chartInstances = {};

function createOrUpdateChart(canvasId, label, labels, data, color) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    if (chartInstances[canvasId]) {
        chartInstances[canvasId].destroy();
    }

    chartInstances[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: data,
                backgroundColor: color,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                }
            }
        }
    });
}
