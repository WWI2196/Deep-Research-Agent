// ── Input page ───────────────────────────────────────────────────

let inputDepth = 3;
let selectedProvider = '';
let selectedModel = '';

function initInputPage() {
  // Track listeners for cleanup — prevents double-binding on re-init
  const listeners = [];

  function on(el, event, handler) {
    el.addEventListener(event, handler);
    listeners.push(() => el.removeEventListener(event, handler));
  }

  // Register cleanup
  onPageCleanup('input', () => {
    listeners.forEach(fn => fn());
    document.getElementById('btn-start').disabled = false;
  });

  // Load config for model selector
  (async () => {
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
  })();

  // Depth buttons
  document.querySelectorAll('#depth-group .btn-option').forEach(btn => {
    on(btn, 'click', () => {
      document.querySelectorAll('#depth-group .btn-option').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      inputDepth = parseInt(btn.dataset.depth);
    });
  });

  // Model select
  on(document.getElementById('model-select'), 'change', () => {
    selectedProvider = document.getElementById('model-select').value;
  });

  // Start button
  on(document.getElementById('btn-start'), 'click', startResearch);

  // Enter to submit, Shift+Enter for newline; ignore IME composition
  on(document.getElementById('query-input'), 'keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      startResearch();
    }
  });

  // Load recent
  loadRecent();
}
