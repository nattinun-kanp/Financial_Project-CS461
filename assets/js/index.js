const META_URL = '/api/model/meta';
const SUMMARY_URL = '/api/summary/risk_counts';
const REBUILD_URL = '/api/admin/rebuild';
let modelMeta = null;

async function fetchModelMeta() {
  const badge = document.getElementById('modelBadge');
  try {
    const res = await fetch(META_URL, { credentials: 'include' });
    if (!res.ok) throw new Error('meta fetch failed');
    const data = await res.json();
    modelMeta = data;
    badge.textContent = `Model: ${data.name} v${data.version}`;
  } catch (e) {
    modelMeta = null;
    badge.textContent = 'Model: unknown';
  }
}

function renderPie(byColor) {
  const canvas = document.getElementById('riskPie');
  const loading = document.getElementById('chartLoading');
  loading.style.display = 'none';
  canvas.style.display = 'block';

  const labels = [
    RISK_LABELS_FULL.green,
    RISK_LABELS_FULL.yellow,
    RISK_LABELS_FULL.orange,
    RISK_LABELS_FULL.red,
  ];
  const data = [
    byColor.green || 0,
    byColor.yellow || 0,
    byColor.orange || 0,
    byColor.red || 0,
  ];
  const colors = [
    RISK_COLORS.green,
    RISK_COLORS.yellow,
    RISK_COLORS.orange,
    RISK_COLORS.red,
  ];

  new Chart(canvas, {
    type: 'pie',
    data: {
      labels,
      datasets: [{ data, backgroundColor: colors }],
    },
    options: {
      plugins: {
        legend: { position: 'bottom', labels: { color: '#0f172a' } },
      },
    },
  });
}

function renderSummaryList(total, byColor) {
  const list = document.getElementById('summaryList');
  const items = [
    { key: 'green', name: RISK_LABELS_FULL.green },
    { key: 'yellow', name: RISK_LABELS_FULL.yellow },
    { key: 'orange', name: RISK_LABELS_FULL.orange },
    { key: 'red', name: RISK_LABELS_FULL.red },
  ];
  list.innerHTML = items.map(({ key, name }) => {
    const count = byColor[key] || 0;
    const pct = total ? (count / total) : 0;
    return `
      <div class="summary-item">
        <div class="summary-left">
          ${riskBadgeHTML(key)}
          <strong>${name}</strong>
        </div>
        <div class="summary-right">
          <span class="summary-count">${formatNumber(count)}</span>
          <span class="summary-pct" style="margin-left:8px;color:#9ca3af">${formatPercent(pct)}</span>
        </div>
      </div>
    `;
  }).join('');
}

async function fetchSummary() {
  const kpiTotal = document.getElementById('kpiTotal');
  const kpiRed = document.getElementById('kpiRed');
  const kpiGreen = document.getElementById('kpiGreen');
  const kpiYellow = document.getElementById('kpiYellow');
  const kpiOrange = document.getElementById('kpiOrange');
  const banner = document.getElementById('summaryError');
  const canvas = document.getElementById('riskPie');
  const loading = document.getElementById('chartLoading');
  try {
    const res = await fetch(SUMMARY_URL, { credentials: 'include' });
    if (!res.ok) throw new Error('summary fetch failed');
    const data = await res.json();
    const { total, by_color: byColor } = data;

    kpiTotal.textContent = formatNumber(total);

    if (kpiRed) kpiRed.textContent = formatNumber(byColor.red || 0);
    if (kpiGreen) kpiGreen.textContent = formatNumber(byColor.green || 0);
    if (kpiYellow) kpiYellow.textContent = formatNumber(byColor.yellow || 0);
    if (kpiOrange) kpiOrange.textContent = formatNumber(byColor.orange || 0);

    renderPie(byColor);
    renderSummaryList(total, byColor);
    banner.style.display = 'none';
  } catch (e) {
    banner.style.display = 'block';
    loading.style.display = 'none';
    canvas.style.display = 'none';
    document.getElementById('summaryList').innerHTML = '';
    kpiTotal.textContent = '-';
    if (kpiRed) kpiRed.textContent = '-';
    if (kpiGreen) kpiGreen.textContent = '-';
    if (kpiYellow) kpiYellow.textContent = '-';
    if (kpiOrange) kpiOrange.textContent = '-';
  }
}

async function triggerRebuild() {
  const refreshBtn = document.getElementById('refreshBtn');
  const originalText = refreshBtn.textContent;
  refreshBtn.textContent = 'กำลังอัปเดต...';
  refreshBtn.disabled = true;
  
  try {
    const res = await fetch(REBUILD_URL, { method: 'POST', credentials: 'include' });
    const data = await res.json();
    
    if (data.success) {
      // Refresh the data after successful rebuild
      await fetchSummary();
      refreshBtn.textContent = 'อัปเดตสำเร็จ!';
      setTimeout(() => {
        refreshBtn.textContent = originalText;
        refreshBtn.disabled = false;
      }, 2000);
    } else {
      throw new Error(data.message || 'Rebuild failed');
    }
  } catch (e) {
    console.error('Rebuild failed:', e);
    refreshBtn.textContent = 'อัปเดตล้มเหลว';
    setTimeout(() => {
      refreshBtn.textContent = originalText;
      refreshBtn.disabled = false;
    }, 2000);
  }
}

function init() {
  fetchModelMeta();
  fetchSummary();
  setupModelModal();
  
  // Setup refresh button
  const refreshBtn = document.getElementById('refreshBtn');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', triggerRebuild);
  }
}

document.addEventListener('DOMContentLoaded', init);

function setupModelModal() {
  const openBtn = document.getElementById('openModelBtn');
  const backdrop = document.getElementById('modelModal');
  const closeBtn = document.getElementById('closeModelBtn');
  const closeBtn2 = document.getElementById('closeModelBtn2');
  const nameEl = document.getElementById('modalModelName');
  const verEl = document.getElementById('modalModelVersion');

  function open() {
    backdrop.style.display = 'flex';
    nameEl.textContent = modelMeta?.name ?? 'unknown';
    verEl.textContent = modelMeta?.version ?? '-';
  }
  function close() { backdrop.style.display = 'none'; }

  openBtn?.addEventListener('click', open);
  closeBtn?.addEventListener('click', close);
  closeBtn2?.addEventListener('click', close);
  backdrop?.addEventListener('click', (e) => {
    if (e.target === backdrop) close();
  });
}
