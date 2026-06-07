// ── Log Viewer ───────────────────────────────────────────────────

async function renderLogViewer(runId, container) {
  container.innerHTML = '<div class="log-loading">Loading logs...</div>';
  try {
    const data = await fetchRunTimeline(runId);
    const items = data.items || [];
    if (items.length === 0) {
      container.innerHTML = '<div class="log-empty">No logs available for this run.</div>';
      return;
    }
    renderLogItems(items, container, runId);
  } catch (err) {
    container.innerHTML = `<div class="log-error">Failed to load logs: ${escapeHtml(err.message)}</div>`;
  }
}

function renderLogItems(items, container, runId) {
  // Collect unique phases for filter
  const phases = [...new Set(items.map(i => i.phase).filter(Boolean))];
  const types = [...new Set(items.map(i => i.type).filter(Boolean))];

  const phaseOptions = ['<option value="">All phases</option>', ...phases.map(p => `<option value="${p}">${p}</option>`)].join('');
  const typeOptions = ['<option value="">All types</option>', ...types.map(t => `<option value="${t}">${t}</option>`)].join('');

  container.innerHTML = `
    <div class="log-toolbar">
      <select id="log-filter-phase" class="log-select">${phaseOptions}</select>
      <select id="log-filter-type" class="log-select">${typeOptions}</select>
      <input id="log-filter-search" class="log-search" placeholder="Search messages..." />
      <span class="log-count">${items.length} events</span>
    </div>
    ${items.length >= 3000 ? '<div class="log-warning">Showing up to 3000 events. Some older logs may be truncated.</div>' : ''}
    <div class="log-timeline" id="log-timeline"></div>
  `;

  const timeline = container.querySelector('#log-timeline');

  function applyFilters() {
    const phase = container.querySelector('#log-filter-phase').value;
    const type = container.querySelector('#log-filter-type').value;
    const search = container.querySelector('#log-filter-search').value.toLowerCase();
    const filtered = items.filter(i => {
      if (phase && i.phase !== phase) return false;
      if (type && i.type !== type) return false;
      if (search && !(i.message || '').toLowerCase().includes(search)) return false;
      return true;
    });
    container.querySelector('.log-count').textContent = `${filtered.length} events`;
    timeline.innerHTML = filtered.map(item => renderLogRow(item)).join('');
    bindLogRowClicks(timeline);
  }

  container.querySelector('#log-filter-phase').addEventListener('change', applyFilters);
  container.querySelector('#log-filter-type').addEventListener('change', applyFilters);
  container.querySelector('#log-filter-search').addEventListener('input', applyFilters);

  applyFilters();
}

function renderLogRow(item) {
  const time = item.created_at ? new Date(item.created_at * 1000).toLocaleTimeString() : '';
  const type = escapeHtml(item.type || '');
  const phase = escapeHtml(item.phase || '');
  const levelCls = (item.level || 'info').toLowerCase();
  const isLlm = item.source === 'llm' || item.type === 'llm_call';
  const latency = item.latency_ms ? `<span class="log-latency">${item.latency_ms}ms</span>` : '';
  const role = item.role ? `<span class="log-role">${escapeHtml(item.role)}</span>` : '';
  const details = item.details || '';
  const hasDetails = details && details !== '{}' && details !== 'null';

  // ReAct chain items get special visual treatment
  const isReact = phase === 'subagents' && type.startsWith('react_');
  const reactCls = isReact ? 'react-chain-item' : '';
  const reactIcon = isReact ? reactTypeIcon(type) : '';

  return `
    <div class="log-row ${levelCls} ${reactCls}" data-type="${type}" data-phase="${phase}">
      <div class="log-header">
        <span class="log-time">${time}</span>
        <span class="log-phase">${phase}</span>
        <span class="log-type">${type}</span>
        ${reactIcon}
        ${role}
        ${latency}
        <span class="log-message">${escapeHtml(item.message || '')}</span>
        ${hasDetails ? '<button class="log-toggle">+</button>' : ''}
      </div>
      ${hasDetails ? `<pre class="log-details" style="display:none">${escapeHtml(formatDetails(details))}</pre>` : ''}
    </div>
  `;
}

function reactTypeIcon(type) {
  const icons = {
    'react_start': '<span class="react-icon react-start">▶</span>',
    'react_step': '<span class="react-icon react-step">●</span>',
    'react_act': '<span class="react-icon react-act">→</span>',
    'react_observe': '<span class="react-icon react-observe">←</span>',
    'react_guard': '<span class="react-icon react-guard">⚠</span>',
    'react_tool_error': '<span class="react-icon react-error">✕</span>',
    'react_error': '<span class="react-icon react-error">✕</span>',
    'react_complete': '<span class="react-icon react-complete">✓</span>',
    'react_max_steps': '<span class="react-icon react-error">⏹</span>',
  };
  return icons[type] || '';
}

function formatDetails(details) {
  if (typeof details === 'string') {
    try {
      const obj = JSON.parse(details);
      return JSON.stringify(obj, null, 2);
    } catch {
      return details;
    }
  }
  return JSON.stringify(details, null, 2);
}

function bindLogRowClicks(container) {
  container.querySelectorAll('.log-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      const row = btn.closest('.log-row');
      const details = row.querySelector('.log-details');
      if (!details) return;
      const visible = details.style.display !== 'none';
      details.style.display = visible ? 'none' : 'block';
      btn.textContent = visible ? '+' : '-';
    });
  });
}

// ── Integration helpers ─────────────────────────────────────────

function attachLogViewerToReport(runId) {
  const reportActions = document.querySelector('.report-actions');
  if (!reportActions) return;

  // Avoid duplicate button
  if (document.getElementById('btn-view-logs')) return;

  const btn = document.createElement('button');
  btn.id = 'btn-view-logs';
  btn.className = 'btn-secondary';
  btn.textContent = 'View Logs';

  const panel = document.createElement('div');
  panel.id = 'report-log-panel';
  panel.className = 'report-log-panel';
  panel.style.display = 'none';

  btn.addEventListener('click', async () => {
    const isHidden = panel.style.display === 'none';
    panel.style.display = isHidden ? 'block' : 'none';
    btn.textContent = isHidden ? 'Hide Logs' : 'View Logs';
    if (isHidden && panel.children.length === 0) {
      await renderLogViewer(runId, panel);
    }
  });

  reportActions.appendChild(btn);

  const reportSection = document.querySelector('#page-report .center');
  if (reportSection) {
    reportSection.appendChild(panel);
  }
}

function attachLogViewerToHistoryRow(runId, cell) {
  const btn = document.createElement('button');
  btn.className = 'history-log-btn';
  btn.textContent = 'Logs';
  btn.style.cssText = 'margin-left:8px;font-size:11px;padding:2px 6px;background:var(--surface);border:1px solid var(--border);border-radius:4px;cursor:pointer;color:var(--text-secondary);';
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    showLogModal(runId);
  });
  cell.appendChild(btn);
}

function showLogModal(runId) {
  let modal = document.getElementById('log-modal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'log-modal';
    modal.className = 'log-modal';
    modal.innerHTML = `
      <div class="log-modal-overlay"></div>
      <div class="log-modal-content">
        <div class="log-modal-header">
          <h3>Debug Logs</h3>
          <button class="log-modal-close">&times;</button>
        </div>
        <div class="log-modal-body"></div>
      </div>
    `;
    document.body.appendChild(modal);
    modal.querySelector('.log-modal-overlay').addEventListener('click', () => { modal.style.display = 'none'; });
    modal.querySelector('.log-modal-close').addEventListener('click', () => { modal.style.display = 'none'; });
  }
  modal.style.display = 'block';
  const body = modal.querySelector('.log-modal-body');
  body.innerHTML = '';
  renderLogViewer(runId, body);
}
