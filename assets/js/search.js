const DEBTORS_URL = '/api/debtors';

const state = {
  query: '',
  risk: '', // '', 'green', 'yellow', 'orange', 'red'
  page: 1,
  pageSize: 50,
  sort: 'ID.asc',
  total: 0,
};

const els = {};

function buildParams() {
  const params = new URLSearchParams();
  params.set('query', state.query);
  params.set('risk', state.risk);
  params.set('page', String(state.page));
  params.set('page_size', String(state.pageSize));
  params.set('sort', state.sort);
  return params.toString();
}

function setLoading(loading) {
  els.loadingTable.style.display = loading ? 'block' : 'none';
  els.table.style.display = loading ? 'none' : 'table';
  els.emptyState.style.display = 'none';
}

async function fetchDebtors() {
  setLoading(true);
  els.searchError.style.display = 'none';
  try {
    const res = await fetch(`${DEBTORS_URL}?${buildParams()}`, { credentials: 'include' });
    if (!res.ok) throw new Error('fetch debtors failed');
    const data = await res.json();
    state.total = data.total || 0;
    renderTable(data.items || []);
    renderFooter();
    updateControlsSummary();
    setLoading(false);
    if (!data.items || data.items.length === 0) {
      els.table.style.display = 'none';
      els.emptyState.style.display = 'block';
    }
  } catch (e) {
    setLoading(false);
    els.table.style.display = 'none';
    els.emptyState.style.display = 'none';
    els.searchError.style.display = 'block';
  }
}

function renderTable(items) {
  els.tbody.innerHTML = items.map((it) => {
    const probPct = formatPercent(it.prob ?? 0, 1);
    return `
      <tr>
        <td>${formatNumber(it.ID)}</td>
        <td>${riskBadgeHTML(it.risk_class)}</td>
        <td>${probPct}</td>
        <td>${formatNumber(it.LIMIT_BAL)}</td>
        <td>${formatNumber(it.AGE)}</td>
        <td>${sexToThai(it.SEX)}</td>
        <td>${educationToThai(it.EDUCATION)}</td>
        <td>${formatNumber(it.MARRIAGE)}</td>
        <td>${formatNumber(it.PAY_0)}</td>
        <td>${formatNumber(it.BILL_AMT1)}</td>
        <td>${formatNumber(it.PAY_AMT1)}</td>
        <td>${formatNumber(it.default_label)}</td>
      </tr>
    `;
  }).join('');
}

function renderFooter() {
  const start = (state.page - 1) * state.pageSize + 1;
  const end = Math.min(state.page * state.pageSize, state.total);
  const z = state.total;
  els.rangeInfo.textContent = z ? `แสดง ${formatNumber(start)}–${formatNumber(end)} จาก ${formatNumber(z)} รายการ` : '';
  els.pageInfo.textContent = `หน้า ${state.page}`;

  els.prevBtn.disabled = state.page <= 1;
  els.nextBtn.disabled = end >= z;
}

function onRiskChange() {
  const val = els.riskSelect.value;
  state.risk = val === 'all' ? '' : val;
  state.page = 1;
  fetchDebtors();
}

function onPageSizeChange() {
  state.pageSize = parseInt(els.pageSizeSelect.value, 10) || 50;
  state.page = 1;
  fetchDebtors();
}

function onSortChange() {
  state.sort = els.sortSelect.value || 'ID.asc';
  state.page = 1;
  fetchDebtors();
}

const onQueryInput = debounce(() => {
  state.query = (els.queryInput.value || '').trim();
  state.page = 1;
  fetchDebtors();
}, 300);

function onPrev() {
  if (state.page > 1) {
    state.page -= 1;
    fetchDebtors();
  }
}

function onNext() {
  const end = Math.min(state.page * state.pageSize, state.total);
  if (end < state.total) {
    state.page += 1;
    fetchDebtors();
  }
}

function onClear() {
  els.queryInput.value = '';
  els.riskSelect.value = 'all';
  els.sortSelect.value = 'ID.asc';
  els.pageSizeSelect.value = String(state.pageSize);
  state.query = '';
  state.risk = '';
  state.sort = 'ID.asc';
  state.page = 1;
  fetchDebtors();
}

function init() {
  els.searchError = document.getElementById('searchError');
  els.queryInput = document.getElementById('queryInput');
  els.riskSelect = document.getElementById('riskSelect');
  els.pageSizeSelect = document.getElementById('pageSizeSelect');
  els.sortSelect = document.getElementById('sortSelect');
  els.clearBtn = document.getElementById('clearBtn');
  els.loadingTable = document.getElementById('loadingTable');
  els.emptyState = document.getElementById('emptyState');
  els.table = document.getElementById('debtorsTable');
  els.tbody = document.getElementById('tableBody');
  els.rangeInfo = document.getElementById('rangeInfo');
  els.prevBtn = document.getElementById('prevBtn');
  els.nextBtn = document.getElementById('nextBtn');
  els.pageInfo = document.getElementById('pageInfo');
  els.controlsSummary = document.getElementById('controlsSummary');

  els.queryInput.addEventListener('input', onQueryInput);
  els.riskSelect.addEventListener('change', onRiskChange);
  els.pageSizeSelect.addEventListener('change', onPageSizeChange);
  els.sortSelect.addEventListener('change', onSortChange);
  els.prevBtn.addEventListener('click', onPrev);
  els.nextBtn.addEventListener('click', onNext);
  els.clearBtn.addEventListener('click', onClear);

  // initial fetch
  fetchDebtors();
}

document.addEventListener('DOMContentLoaded', init);

function updateControlsSummary() {
  const riskText = state.risk ? RISK_BADGE_TH[state.risk] : 'ทุกสี';
  const queryText = state.query ? state.query : '-';
  let sortText = 'ID น้อย→มาก';
  
  switch (state.sort) {
    case 'ID.desc': sortText = 'ID มาก→น้อย'; break;
    case 'prob.asc': sortText = 'ความเสี่ยงสูง→ต่ำ'; break;
    case 'prob.desc': sortText = 'ความเสี่ยงต่ำ→สูง'; break;
  }
  
  els.controlsSummary.textContent = `กรอง: ${riskText} • เรียง: ${sortText} • แถว/หน้า: ${formatNumber(state.pageSize)} • ค้นหา: ${queryText} • รวม: ${formatNumber(state.total)}`;
}