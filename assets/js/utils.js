// Risk colors and labels
const RISK_COLORS = {
  green: '#22c55e',
  yellow: '#eab308',
  orange: '#f97316',
  red: '#ef4444',
};

const RISK_LABELS_FULL = {
  green: 'ชำระตรงเวลา',
  yellow: 'เริ่มมีความเสี่ยง',
  orange: 'มีความเสี่ยงสูง',
  red: 'ไม่ชำระแน่นอน',
};

const RISK_BADGE_TH = {
  green: 'เขียว',
  yellow: 'เหลือง',
  orange: 'ส้ม',
  red: 'แดง',
};

function formatNumber(n) {
  try {
    return new Intl.NumberFormat('th-TH').format(n);
  } catch {
    return String(n);
  }
}

function riskBadgeHTML(color) {
    const map = {
        'green': { bg: '#dcfce7', text: '#166534', label: 'Low' },
        'yellow': { bg: '#fef9c3', text: '#854d0e', label: 'Medium' },
        'orange': { bg: '#ffedd5', text: '#9a3412', label: 'High' },
        'red': { bg: '#fee2e2', text: '#991b1b', label: 'Critical' },
    };
    const c = map[color] || map['green'];
    return `<span class="badge" style="background:${c.bg};color:${c.text}">${c.label}</span>`;
}

function formatPercent(n, digits = 2) {
  const pct = (n * 100);
  try {
    return new Intl.NumberFormat('th-TH', { maximumFractionDigits: digits }).format(pct) + '%';
  } catch {
    return pct.toFixed(digits) + '%';
  }
}

function riskBadgeHTML(risk) {
  const color = RISK_COLORS[risk] || '#334155';
  const label = RISK_BADGE_TH[risk] || risk;
  return `<span class="badge" style="background-color:${color}">${label}</span>`;
}

function debounce(fn, delay = 300) {
  let t = null;
  return function (...args) {
    if (t) clearTimeout(t);
    t = setTimeout(() => fn.apply(this, args), delay);
  };
}

function sexToThai(code) {
  if (code === 1) return 'ชาย';
  if (code === 2) return 'หญิง';
  return 'ไม่ทราบ';
}

function educationToThai(code) {
  switch (code) {
    case 1: return 'บัณฑิตศึกษา';
    case 2: return 'มหาวิทยาลัย';
    case 3: return 'มัธยม';
    case 4: return 'อื่น ๆ';
    default: return 'ไม่ทราบ';
  }
}

function marriageToThai(code) {
  switch (code) {
    case 1: return 'แต่งงาน';
    case 2: return 'โสด';
    case 3: return 'อื่น ๆ';
    default: return 'ไม่ทราบ';
  }
}