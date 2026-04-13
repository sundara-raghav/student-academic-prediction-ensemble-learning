/* ─────────────────────────────────────────
   CHART.JS GLOBAL DEFAULTS
───────────────────────────────────────── */
function initChartDefaults() {
    if (typeof Chart === 'undefined') return;
    Chart.defaults.color = '#8892B0';
    Chart.defaults.borderColor = 'rgba(255,255,255,0.07)';
    Chart.defaults.font.family = 'Inter';
}

/* ─────────────────────────────────────────
   STATE
───────────────────────────────────────── */
const charts = {};

/* ─────────────────────────────────────────
   BOOTSTRAP
───────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
    // Chart.js may still be loading (deferred), wait a tick
    setTimeout(() => {
        initChartDefaults();
        initDashboard();
    }, 100);
});

async function initDashboard() {
    // Run all independent fetches IN PARALLEL — not sequential
    await Promise.all([
        fetchModels(),
        fetchFeatureImportance(),
        fetchStats(),
    ]);
    setupFormListener();
}

/* ─────────────────────────────────────────
   MODELS
───────────────────────────────────────── */
async function fetchModels() {
    try {
        const res = await fetch('/models');
        const models = await res.json();
        renderModelCards(models);
        populateModelSelect(models);
        // Defer accuracy chart to idle time so cards paint first
        scheduleIdle(() => renderAccuracyChart(models));
    } catch (e) { console.error('Models fetch failed', e); }
}

function renderModelCards(models) {
    const grid = document.getElementById('models-grid');
    grid.innerHTML = '';
    if (!models?.length) { grid.innerHTML = '<p style="color:var(--text-muted)">No models found.</p>'; return; }

    const best = models[0];
    document.getElementById('best-model-badge').classList.remove('hidden');
    document.getElementById('best-model-name').textContent = best.model_name;
    document.getElementById('best-model-acc').textContent = best.accuracy.toFixed(1);

    models.forEach((m, idx) => {
        const isBest = m.model_name === best.model_name;
        const div = document.createElement('div');
        div.className = `model-card${isBest ? ' best-model-card' : ''}`;
        const barW = Math.max(0, Math.min(100, m.accuracy)).toFixed(1);
        div.innerHTML = `
            <div class="rank">#${idx + 1}</div>
            <h3>${m.model_name}</h3>
            <div class="acc">${m.accuracy.toFixed(1)}<span style="font-size:1rem;font-weight:500;">%</span></div>
            <div class="acc-label">Accuracy</div>
            <div class="acc-bar"><div class="acc-bar-fill" style="width:${barW}%"></div></div>
            <button onclick="selectModel('${m.model_name}')">Select Model</button>`;
        grid.appendChild(div);
    });
}

function selectModel(name) {
    const sel = document.getElementById('modelSelect');
    sel.value = name;
    document.querySelector('.predict-grid')?.scrollIntoView({behavior:'smooth', block:'start'});
}

function populateModelSelect(models) {
    const sel = document.getElementById('modelSelect');
    sel.innerHTML = '<option value="" disabled selected>Select a model…</option>';
    models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.model_name;
        opt.textContent = m.model_name;
        sel.appendChild(opt);
    });
}

/* ─────────────────────────────────────────
   STATS (Pass/Fail distribution)
───────────────────────────────────────── */
async function fetchStats() {
    try {
        const res = await fetch('/stats');
        const d = await res.json();
        document.getElementById('statTotal').textContent = d.total?.toLocaleString() ?? '—';
        document.getElementById('statPass').textContent  = d.pass?.toLocaleString()  ?? '—';
        document.getElementById('statFail').textContent  = d.fail?.toLocaleString()  ?? '—';
        // Defer dist chart to idle
        scheduleIdle(() => renderDistChart(d));
    } catch (e) {}
}

/* ─────────────────────────────────────────
   SCHEDULE IDLE (defers heavy work off main thread paint)
───────────────────────────────────────── */
function scheduleIdle(fn) {
    if ('requestIdleCallback' in window) {
        requestIdleCallback(fn, { timeout: 1500 });
    } else {
        setTimeout(fn, 50);
    }
}

/* ─────────────────────────────────────────
   PREDICTION FORM
───────────────────────────────────────── */
function setupFormListener() {
    const form = document.getElementById('prediction-form');
    if (!form) return;

    // Optimistic INP fix: show spinner on pointer-down, not waiting for submit event
    const btn = document.getElementById('predictBtn');
    btn?.addEventListener('pointerdown', () => {
        document.getElementById('btn-text').classList.add('hidden');
        document.getElementById('predictSpinner').classList.remove('hidden');
    });

    form.addEventListener('submit', async e => {
        e.preventDefault();
        // Spinner already shown on pointerdown — ensure it's visible
        document.getElementById('btn-text').classList.add('hidden');
        document.getElementById('predictSpinner').classList.remove('hidden');

        const payload = {
            model_name:     document.getElementById('modelSelect').value,
            attendance:     document.getElementById('attendance').value,
            study_hours:    document.getElementById('study_hours').value,
            internal_marks: document.getElementById('internal_marks').value,
            assignments:    document.getElementById('assignments').value,
            previous_gpa:   document.getElementById('previous_gpa').value,
        };

        try {
            const res  = await fetch('/predict', {
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (data.error) { alert('Error: '+data.error); return; }
            displayResult(data);
            renderShap(data.shap);
            renderAdvisor(data.advice);
            renderRadar(payload);
        } catch (e) { alert('Prediction failed.'); console.error(e); }
        finally {
            document.getElementById('btn-text').classList.remove('hidden');
            document.getElementById('predictSpinner').classList.add('hidden');
        }
    });
}

/* ─────────────────────────────────────────
   DISPLAY RESULT
───────────────────────────────────────── */
function displayResult(data) {
    document.getElementById('initialMsg').classList.add('hidden');
    document.getElementById('resultContent').classList.remove('hidden');

    const badge = document.getElementById('resultBadge');
    badge.textContent = data.prediction;
    badge.className   = `result-badge ${data.prediction === 'PASS' ? 'pass' : 'fail'}`;

    document.getElementById('res-model-chip').textContent = data.model_used;

    // Probability bars
    animateBar('passBar', data.pass_prob);
    animateBar('failBar', data.fail_prob);
    document.getElementById('res-pass-prob').textContent = data.pass_prob.toFixed(1)+'%';
    document.getElementById('res-fail-prob').textContent = data.fail_prob.toFixed(1)+'%';

    drawGauge(data.confidence, data.prediction);
}

function animateBar(id, pct) {
    const el = document.getElementById(id);
    el.style.width = '0%';
    requestAnimationFrame(() => setTimeout(() => el.style.width = pct+'%', 30));
}

/* ─────────────────────────────────────────
   ADVISOR
───────────────────────────────────────── */
function renderAdvisor(advice) {
    const sec  = document.getElementById('advisorSection');
    const grid = document.getElementById('advisorGrid');
    sec.classList.remove('hidden');
    grid.innerHTML = '';

    const mkCard = (title, icon, cls, items, emptyMsg) => {
        const card = document.createElement('div');
        card.className = `advisor-card ${cls}`;
        card.innerHTML = `<h4><i class="${icon}"></i> ${title}</h4>`;
        if (!items?.length) {
            card.innerHTML += `<p class="advisor-none">${emptyMsg}</p>`;
        } else {
            items.forEach(i => {
                const p = document.createElement('div');
                p.className = 'advisor-item';
                p.textContent = i;
                card.appendChild(p);
            });
        }
        grid.appendChild(card);
    };

    mkCard('Urgent Actions', 'fas fa-exclamation-triangle', 'urgent-card',
           advice.urgent, '✅ No critical issues found.');
    mkCard('Recommendations', 'fas fa-lightbulb', 'tips-card',
           advice.tips, '🌟 All metrics are healthy!');
}

/* ─────────────────────────────────────────
   SHAP CHART
───────────────────────────────────────── */
function renderShap(shap) {
    const sec = document.getElementById('shapSection');
    sec.classList.remove('hidden');

    const ctx = document.getElementById('shapChart');
    if (charts.shap) charts.shap.destroy();

    const colors = shap.contributions.map(v =>
        v > 0 ? 'rgba(34,197,94,0.75)' : 'rgba(239,68,68,0.75)'
    );
    const borders = shap.contributions.map(v =>
        v > 0 ? '#22C55E' : '#EF4444'
    );

    charts.shap = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: shap.labels,
            datasets: [{
                label: 'Contribution to Pass probability (%)',
                data: shap.contributions,
                backgroundColor: colors,
                borderColor: borders,
                borderWidth: 1.5,
                borderRadius: 6,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => `${ctx.raw > 0 ? '+' : ''}${ctx.raw.toFixed(2)}%`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { callback: v => (v > 0 ? '+' : '') + v + '%' }
                },
                y: { grid: { display: false } }
            }
        }
    });
}

/* ─────────────────────────────────────────
   RADAR CHART
───────────────────────────────────────── */
function renderRadar(payload) {
    const panel = document.getElementById('radarPanel');
    panel.classList.remove('hidden');

    const ctx = document.getElementById('radarChart');
    if (charts.radar) charts.radar.destroy();

    const norm = {
        'Attendance':      parseFloat(payload.attendance) / 100 * 10,
        'Study Hours':     parseFloat(payload.study_hours) / 10 * 10,
        'Internal Marks':  parseFloat(payload.internal_marks) / 50 * 10,
        'Assignments':     parseFloat(payload.assignments) / 50 * 10,
        'Previous GPA':    parseFloat(payload.previous_gpa),
    };

    charts.radar = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: Object.keys(norm),
            datasets: [{
                label: 'Student Profile',
                data: Object.values(norm),
                backgroundColor: 'rgba(79,70,229,0.2)',
                borderColor: '#6366F1',
                pointBackgroundColor: '#6366F1',
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                r: {
                    beginAtZero: true, max: 10,
                    grid: { color: 'rgba(255,255,255,0.07)' },
                    pointLabels: { color: '#8892B0', font: { size: 11 } },
                    ticks: { display: false }
                }
            }
        }
    });
}

/* ─────────────────────────────────────────
   GAUGE CHART
───────────────────────────────────────── */
function drawGauge(confidence, prediction) {
    if (typeof Chart === 'undefined') return;
    const ctx   = document.getElementById('gaugeChart');
    const color = prediction === 'PASS' ? '#22C55E' : '#EF4444';
    if (charts.gauge) charts.gauge.destroy();

    charts.gauge = new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [confidence, 100 - confidence],
                backgroundColor: [color, 'rgba(255,255,255,0.05)'],
                borderWidth: 0,
                circumference: 180,
                rotation: 270,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            cutout: '78%',
            plugins: { legend: { display: false }, tooltip: { enabled: false } }
        }
    });

    const label = document.getElementById('gaugeLabel');
    if (label) {
        label.style.color = color;
        label.textContent = confidence.toFixed(1) + '%';
    }
}

/* ─────────────────────────────────────────
   ACCURACY CHART
───────────────────────────────────────── */
function renderAccuracyChart(models) {
    if (typeof Chart === 'undefined') return;
    const ctx = document.getElementById('accuracyChart');
    if (!ctx) return;
    if (charts.accuracy) charts.accuracy.destroy();

    const labels = models.map(m => m.model_name);
    const data   = models.map(m => m.accuracy);
    const bgs    = models.map((_, i) =>
        i === 0 ? 'rgba(34,197,94,0.7)' : 'rgba(79,70,229,0.65)'
    );

    charts.accuracy = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Accuracy (%)',
                data,
                backgroundColor: bgs,
                borderRadius: 6,
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                y: {
                    min: 50, max: 100,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { callback: v => v + '%' }
                },
                x: { grid: { display: false } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

/* ─────────────────────────────────────────
   FEATURE IMPORTANCE CHART
───────────────────────────────────────── */
async function fetchFeatureImportance() {
    try {
        const res = await fetch('/feature_importance');
        const d   = await res.json();
        renderFeatureChart(d);
    } catch (e) {}
}

function renderFeatureChart(d) {
    if (typeof Chart === 'undefined') return;
    const ctx = document.getElementById('featureChart');
    if (!ctx) return;
    if (charts.feature) charts.feature.destroy();

    charts.feature = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: d.labels,
            datasets: [{
                data: d.data,
                backgroundColor: ['#4F46E5','#22C55E','#F59E0B','#EC4899','#3B82F6'],
                borderWidth: 0,
                hoverOffset: 8,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: { position: 'right', labels: { padding: 12, font: { size: 11 } } }
            }
        }
    });
}

/* ─────────────────────────────────────────
   PASS/FAIL DISTRIBUTION CHART
───────────────────────────────────────── */
function renderDistChart(d) {
    if (typeof Chart === 'undefined') return;
    const ctx = document.getElementById('distChart');
    if (!ctx) return;
    if (charts.dist) charts.dist.destroy();

    charts.dist = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Pass', 'Fail'],
            datasets: [{
                data: [d.pass, d.fail],
                backgroundColor: ['rgba(34,197,94,0.75)', 'rgba(239,68,68,0.75)'],
                borderColor: ['#22C55E','#EF4444'],
                borderWidth: 1.5,
                hoverOffset: 8,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: { position: 'right', labels: { padding: 12, font: { size: 11 } } },
                tooltip: {
                    callbacks: {
                        label: ctx => {
                            const pct = ((ctx.raw / d.total) * 100).toFixed(1);
                            return ` ${ctx.raw} students (${pct}%)`;
                        }
                    }
                }
            }
        }
    });
}
