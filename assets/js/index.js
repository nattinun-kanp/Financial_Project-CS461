// URL ไปยัง API Dashboard
const DASHBOARD_URL = '/api/dashboard/stats?segment=All';
const REBUILD_URL = '/api/admin/rebuild';

// ตัวแปรสำหรับ Demographics Toggle
let demoData = null;
let currentDemoIdx = 0;
const demoKeys = ['sex', 'education', 'marriage'];
const demoTitles = {
  sex: 'สัดส่วน: เพศ (Sex)',
  education: 'สัดส่วน: ระดับการศึกษา',
  marriage: 'สัดส่วน: สถานะสมรส'
};
let demoChartInstance = null;
let riskChartInstance = null;
let trendChartInstance = null;

// ฟังก์ชันหลักดึงข้อมูล
async function fetchDashboardData() {
  try {
    const res = await fetch(DASHBOARD_URL);
    if (!res.ok) throw new Error('Failed to fetch data');
    const data = await res.json();

    renderScorecards(data.metrics);
    renderRiskChart(data.risk_dist, data.metrics.total_debtors);
    
    
    // 1. แสดงตัวเลข Scorecards
    renderScorecards(data.metrics);
    
    // 2. แสดงกราฟวงกลมความเสี่ยง (Risk Chart)
    // ส่งทั้ง risk_dist และ total_debtors ไปคำนวณ %
    renderRiskChart(data.risk_dist, data.metrics.total_debtors);

    // --- แสดงตาราง ---
    renderRiskTable(data.table_data, data.metrics);

    // 3. แสดงกราฟแนวโน้ม (Trend Chart)
    renderTrendChart(data.trends);
    
    // 4. เตรียมข้อมูล Demographics
    demoData = data.demographics;
    renderDemoChart(); // Render ครั้งแรก

  } catch (e) {
    console.error(e);
    // กรณี Error ให้แสดงขีด -
    if(document.getElementById('metricTotal')) {
        document.getElementById('metricTotal').textContent = '-';
    }
  }
}


function renderScorecards(metrics) {
  const elTotal = document.getElementById('metricTotal');
  const elExposure = document.getElementById('metricExposure');
  
  if(elTotal) elTotal.textContent = formatNumber(metrics.total_debtors);
  if(elExposure) elExposure.textContent = formatNumber(metrics.total_exposure);
}

// --- ส่วนที่แก้ไข: จัดการสี 4 ระดับให้ถูกต้อง ---
function renderRiskChart(dist, total) {
  const ctx = document.getElementById('riskDonut');
  if(!ctx) return; // ป้องกัน Error ถ้าหา Element ไม่เจอ

  const legendContainer = document.getElementById('riskLegend');
  
  // ตั้งค่าสีให้ตรงกับ Key ที่ส่งมาจาก app.py ('green', 'yellow', 'orange', 'red')
  const config = {
    // แดง: หนักสุด
    'red':    { label: 'ไม่ชำระ (Red)',    color: '#ef4444' },
    // ส้ม: รองลงมา (เสี่ยงสูง)
    'orange': { label: 'เสี่ยงสูง (Orange)',  color: '#f97316' },
    // เหลือง: เบาลงมา (เริ่มเสี่ยง)
    'yellow': { label: 'เริ่มมีความเสี่ยง (Yellow)', color: '#eab308' },
    // เขียว: ดีที่สุด
    'green':  { label: 'ชำระปกติ (Green)', color: '#22c55e' }
  };
  
  // ข้อมูลจาก API (dist.labels จะเป็น ['green', 'yellow', ...])
  const apiLabels = dist.labels; 
  const values = dist.data;
  
  const chartLabels = apiLabels.map(k => config[k]?.label || k);
  const chartColors = apiLabels.map(k => config[k]?.color || '#ccc');

  if (riskChartInstance) riskChartInstance.destroy();
  
  riskChartInstance = new Chart(ctx.getContext('2d'), {
    type: 'doughnut',
    data: {
      labels: chartLabels,
      datasets: [{
        data: values,
        backgroundColor: chartColors,
        borderWidth: 0,
        hoverOffset: 4
      }]
    },
    options: {
      cutout: '65%',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }, // ซ่อน Legend ของ Chart.js (เราสร้างเอง)
        tooltip: {
          callbacks: {
            label: function(context) {
              const val = context.raw;
              const pct = total ? ((val / total) * 100).toFixed(1) : 0;
              return ` ${context.label}: ${formatNumber(val)} ราย (${pct}%)`;
            }
          }
        }
      }
    }
  });

  // สร้าง Legend ด้านขวาเอง
  if(legendContainer) {
      legendContainer.innerHTML = apiLabels.map((key, i) => {
        const val = values[i];
        const pct = total ? ((val / total) * 100).toFixed(1) : 0;
        const item = config[key] || { label: key, color: '#ccc' };
        
        return `
          <div class="legend-item">
            <span class="dot" style="background:${item.color}"></span>
            <div class="legend-text">
              <div class="legend-title">${item.label}</div>
              <div class="legend-val">${formatNumber(val)} ราย (${pct}%)</div>
            </div>
          </div>
        `;
      }).join('');
  }
}

function renderDemoChart() {
  if (!demoData) return;
  const ctx = document.getElementById('demoDonut');
  if(!ctx) return;
  
  const key = demoKeys[currentDemoIdx];
  const info = demoData[key]; // { labels: [], data: [] }
  
  // อัปเดตหัวข้อกราฟขวา
  const titleEl = document.getElementById('demoTitle');
  if(titleEl) titleEl.textContent = demoTitles[key];

  if (demoChartInstance) demoChartInstance.destroy();

  demoChartInstance = new Chart(ctx.getContext('2d'), {
    type: 'doughnut',
    data: {
      labels: info.labels,
      datasets: [{
        data: info.data,
        backgroundColor: ['#3b82f6', '#8b5cf6', '#ec4899', '#6366f1', '#14b8a6', '#64748b'],
        borderWidth: 0
      }]
    },
    options: {
      cutout: '60%',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { 
            position: 'bottom', 
            labels: { boxWidth: 10, usePointStyle: true, padding: 15 } 
        }
      }
    }
  });
}

// ฟังก์ชันวาดกราฟแท่ง
function renderTrendChart(trends) {
  const ctx = document.getElementById('trendBarChart');
  if(!ctx) return;
  
  if (trendChartInstance) trendChartInstance.destroy();
  
  trendChartInstance = new Chart(ctx.getContext('2d'), {
    type: 'bar',
    data: {
      labels: trends.months, // ['Apr', 'May', ...]
      datasets: [
        {
          label: 'ยอดหนี้ (Bill)',
          data: trends.bill,
          backgroundColor: '#94a3b8', // สีเทา
          borderRadius: 4,
          barPercentage: 0.6,
          categoryPercentage: 0.8
        },
        {
          label: 'ยอดชำระ (Pay)',
          data: trends.pay,
          backgroundColor: '#22c55e', // สีเขียว
          borderRadius: 4,
          barPercentage: 0.6,
          categoryPercentage: 0.8
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }, // เราสร้าง Legend เองแล้ว
        tooltip: {
          mode: 'index',
          intersect: false,
          callbacks: {
            label: function(context) {
              return ` ${context.dataset.label}: ${formatNumber(context.raw)} บาท`;
            }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          grid: { borderDash: [2, 4], color: '#f1f5f9' },
          ticks: { callback: (val) => formatNumber(val, 0, true) } // ย่อตัวเลข เช่น 1M
        },
        x: {
          grid: { display: false }
        }
      }
    }
  });
}

// ฟังก์ชันสร้างตาราง
function renderRiskTable(tableData, metrics) {
  const tbody = document.getElementById('riskTableBody');
  const tfoot = document.getElementById('riskTableFoot');
  if(!tbody) return;

  const totalExposure = metrics.total_exposure || 1;
  const totalDebtors = metrics.total_debtors || 1;

  // Map ชื่อภาษาอังกฤษเป็นไทย
  const nameMap = {
    'High Risk': 'ความเสี่ยงสูง',
    'Medium Risk': 'ความเสี่ยงกลาง',
    'Good Payer': 'ความเสี่ยงต่ำ'
  };

  // สร้าง Rows
  tbody.innerHTML = tableData.map(item => {
    const pctExposure = (item.exposure / totalExposure) * 100;
    const pctCount = (item.count / totalDebtors) * 100;
    const thName = nameMap[item.segment] || item.segment;
    
    // กำหนดสีตัวอักษรตามความเสี่ยง
    let colorClass = '';
    if(item.segment === 'High Risk') colorClass = 'text-danger';
    else if(item.segment === 'Medium Risk') colorClass = 'text-warning';
    else colorClass = 'text-success';

    return `
      <tr>
        <td class="${colorClass} fw-bold">${thName}</td>
        <td class="text-end">${formatNumber(item.exposure)}</td>
        <td class="text-end">${pctExposure.toFixed(2)}%</td>
        <td class="text-end">${pctCount.toFixed(2)}%</td>
      </tr>
    `;
  }).join('');

  // สร้าง Footer (ยอดรวม)
  tfoot.innerHTML = `
    <tr>
      <td>รวม</td>
      <td class="text-end">${formatNumber(totalExposure)}</td>
      <td class="text-end">100.00%</td>
      <td class="text-end">100.00%</td>
    </tr>
  `;
}



// ปุ่มควบคุม Demographics (< >)
const btnPrev = document.getElementById('demoPrev');
const btnNext = document.getElementById('demoNext');

if(btnPrev) {
    btnPrev.addEventListener('click', () => {
      currentDemoIdx = (currentDemoIdx - 1 + demoKeys.length) % demoKeys.length;
      renderDemoChart();
    });
}
if(btnNext) {
    btnNext.addEventListener('click', () => {
      currentDemoIdx = (currentDemoIdx + 1) % demoKeys.length;
      renderDemoChart();
    });
}

// เริ่มต้นทำงาน
document.addEventListener('DOMContentLoaded', () => {
  fetchDashboardData();
  
  // ปุ่ม Refresh อัปเดตข้อมูล
  const refreshBtn = document.getElementById('refreshBtn');
  if(refreshBtn) {
      refreshBtn.addEventListener('click', async () => {
        refreshBtn.disabled = true; 
        const originalText = refreshBtn.textContent;
        refreshBtn.textContent = 'Updating...';
        
        try {
            await fetch(REBUILD_URL, { method: 'POST' });
            await fetchDashboardData();
        } catch(e) {
            console.error(e);
        }
        
        refreshBtn.textContent = originalText;
        refreshBtn.disabled = false;
      });
  }
});