// ── Input page ───────────────────────────────────────────────────

let inputDepth = 3;
let selectedProvider = '';
let selectedModel = '';

async function initInputPage() {
  // Load config for model selector
  try {
    const cfg = await fetchConfig();
    selectedProvider = cfg.default_provider;
    selectedModel = cfg.default_model;

    const providers = cfg.available_providers || [];
    const select = document.getElementById('model-select');
    select.innerHTML = '';
    if (providers.length === 0) {
      select.innerHTML = '<option>No provider configured</option>';
    } else {
      for (const p of providers) {
        const sel = p === cfg.default_provider ? 'selected' : '';
        select.innerHTML += `<option value="${p}" ${sel}>${p} / ${cfg.default_model}</option>`;
      }
    }
  } catch (e) {
    document.getElementById('model-select').innerHTML = '<option>Unavailable</option>';
  }

  // Depth buttons
  document.querySelectorAll('#depth-group .btn-option').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#depth-group .btn-option').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      inputDepth = parseInt(btn.dataset.depth);
    });
  });

  // Model select
  document.getElementById('model-select').addEventListener('change', () => {
    selectedProvider = document.getElementById('model-select').value;
  });

  // Start button
  document.getElementById('btn-start').addEventListener('click', startResearch);

  // Enter to submit, Shift+Enter for newline; ignore IME composition
  document.getElementById('query-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      startResearch();
    }
  });

  // Load recent
  loadRecent();
}

async function loadRecent() {
  try {
    const data = await fetchHistory();
    const list = document.getElementById('recent-list');
    if (!data.history || data.history.length === 0) {
      list.innerHTML = '<div class="recent-item"><span class="query" style="color:var(--text-tertiary)">No research history yet</span></div>';
      return;
    }
    list.innerHTML = data.history.slice(0, 5).map(r => `
      <div class="recent-item" onclick="viewHistoryReport('${r.run_id}')">
        <span class="query">${escapeHtml(r.query)}</span>
        <span class="meta">${timeAgo(r.started_at)} · ${r.total_sources || 0} sources</span>
      </div>
    `).join('');
  } catch {
    document.getElementById('recent-list').innerHTML = '';
  }
}

function startResearch() {
  const query = document.getElementById('query-input').value.trim();
  if (!query) return;

  const errorEl = document.getElementById('input-error');
  errorEl.style.display = 'none';
  document.getElementById('btn-start').disabled = true;

  // Reset state
  resetState();
  STATE.running = true;
  STATE.startTime = Date.now();

  // Switch to dashboard
  navigateTo('dashboard');

  streamResearch(
    query, inputDepth, selectedProvider, selectedModel,
    handleResearchEvent,
    handleResearchError,
    handleResearchDone,
  );
}

window.startResearch = startResearch;
window.viewHistoryReport = viewHistoryReport;
