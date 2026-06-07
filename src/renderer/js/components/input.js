// ── Input page ───────────────────────────────────────────────────

let inputDepth = 2;
let selectedModel = '';
let outputLanguage = 'zh';
let _selectedCollections = [];

let _streamController = null;

async function startResearch() {
  const query = document.getElementById('query-input').value.trim();
  if (!query) return;

  const errorEl = document.getElementById('input-error');
  errorEl.style.display = 'none';
  document.getElementById('btn-start').disabled = true;

  store.reset();
  store.set('running', true);
  store.set('startTime', Date.now());

  if (_streamController) { _streamController.abort(); _streamController = null; }

  _streamController = startResearchStream(
    query, inputDepth, selectedModel, _selectedCollections, outputLanguage,
    (evt) => {
      if (evt.run_id && !store.get('currentRunId')) {
        store.set('currentRunId', evt.run_id);
      }
      handleResearchEvent(evt);
      if (evt.type === 'complete' || evt.type === 'error') {
        _streamController = null;
        document.getElementById('btn-start').disabled = false;
      }
    },
    (err) => {
      errorEl.textContent = err.message || 'Failed to start research';
      errorEl.style.display = 'block';
      document.getElementById('btn-start').disabled = false;
      store.set('running', false);
      _streamController = null;
    },
    () => {
      document.getElementById('btn-start').disabled = false;
      _streamController = null;
      handleResearchDone();
    }
  );

  navigateTo('dashboard');
}

async function loadRecent() {
  try {
    const data = await fetchHistory();
    const list = document.getElementById('recent-list');
    if (!data.history || data.history.length === 0) {
      list.innerHTML = '<div class="recent-item"><span class="query" style="color:var(--text-tertiary)">No research history yet</span></div>';
      return;
    }
    const completed = data.history.filter(r => r.status === 'completed').slice(0, 5);
    list.innerHTML = completed.map(r => `
      <div class="recent-item" onclick="viewHistoryReport('${r.run_id}')">
        <span class="query">${escapeHtml(r.query)}</span>
        <span class="meta">${timeAgo(r.started_at)} · ${r.total_sources || 0} sources</span>
      </div>
    `).join('');
  } catch {
    document.getElementById('recent-list').innerHTML = '';
  }
}

function initInputPage() {
  const listeners = [];

  function on(el, event, handler) {
    el.addEventListener(event, handler);
    listeners.push(() => el.removeEventListener(event, handler));
  }

  onPageCleanup('input', () => {
    listeners.forEach(fn => fn());
    document.getElementById('btn-start').disabled = false;
  });

  (async () => {
    try {
      const cfg = await fetchConfig();
      selectedModel = cfg.default_model || '';
      document.getElementById('model-display').textContent = selectedModel || 'No model configured';
    } catch (e) {
      document.getElementById('model-display').textContent = 'Unavailable';
    }
  })();

  document.querySelectorAll('#depth-group .btn-option').forEach(btn => {
    on(btn, 'click', () => {
      document.querySelectorAll('#depth-group .btn-option').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      inputDepth = parseInt(btn.dataset.depth);
      // Update depth hint text
      const hintEl = document.getElementById('depth-hint');
      if (hintEl) {
        const hints = {
          1: '快速认知',
          2: '均衡模式',
          3: '深度研究',
        };
        hintEl.textContent = hints[inputDepth] || '均衡模式';
      }
    });
  });

  document.querySelectorAll('#language-group .btn-option').forEach(btn => {
    on(btn, 'click', () => {
      document.querySelectorAll('#language-group .btn-option').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      outputLanguage = btn.dataset.lang;
    });
  });

  on(document.getElementById('btn-start'), 'click', startResearch);

  on(document.getElementById('query-input'), 'keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      startResearch();
    }
  });

  // Collection toggle — default expanded
  const toggleBtn = document.getElementById('btn-toggle-collections');
  const panel = document.getElementById('collection-panel');
  const chevron = document.getElementById('collection-chevron');
  if (toggleBtn && panel) {
    panel.style.display = 'block';
    if (chevron) chevron.textContent = '▾';
    on(toggleBtn, 'click', () => {
      const open = panel.style.display !== 'none';
      panel.style.display = open ? 'none' : 'block';
      if (chevron) chevron.textContent = open ? '▸' : '▾';
      if (!open) loadCollectionsForInput();
    });
  }

  loadRecent();
  loadCollectionsForInput();
}

async function loadCollectionsForInput() {
  const container = document.getElementById('collection-checkboxes');
  const selected = document.getElementById('selected-collections');
  if (!container) return;

  try {
    const data = await fetchCollections();
    const collections = data.collections || [];
    if (collections.length === 0) {
      container.innerHTML = '<span style="color:var(--text-tertiary);font-size:12px">No libraries available</span>';
      if (selected) selected.innerHTML = '';
      return;
    }

    container.innerHTML = `
      <div style="color:var(--text-tertiary);font-size:12px;margin-bottom:8px">Select libraries to include in research</div>
      ${collections.map(c => {
        const checked = _selectedCollections.includes(c.id) ? 'checked' : '';
        return `
          <label class="collection-checkbox">
            <input type="checkbox" value="${c.id}" ${checked}>
            <span>${escapeHtml(c.name)}</span>
          </label>
        `;
      }).join('')}
    `;

    container.querySelectorAll('input[type="checkbox"]').forEach(cb => {
      cb.addEventListener('change', () => {
        if (cb.checked) {
          if (!_selectedCollections.includes(cb.value)) _selectedCollections.push(cb.value);
        } else {
          _selectedCollections = _selectedCollections.filter(id => id !== cb.value);
        }
        renderSelectedCollections();
      });
    });

    renderSelectedCollections();
  } catch {
    container.innerHTML = '<span style="color:var(--text-tertiary);font-size:12px">Failed to load libraries</span>';
  }
}

function renderSelectedCollections() {
  const el = document.getElementById('selected-collections');
  if (!el) return;
  if (_selectedCollections.length === 0) {
    el.innerHTML = '';
    return;
  }
  // Get names from checkboxes
  const names = [];
  document.querySelectorAll('#collection-checkboxes input:checked').forEach(cb => {
    const label = cb.closest('label');
    if (label) names.push(label.querySelector('span')?.textContent || cb.value);
  });
  el.innerHTML = names.map(n => `<span class="collection-tag">${escapeHtml(n)}</span>`).join('');
}

window.startResearch = startResearch;
