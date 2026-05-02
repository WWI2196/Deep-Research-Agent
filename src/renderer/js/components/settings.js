// ── Settings page ────────────────────────────────────────────────

async function initSettingsPage() {
  const container = document.getElementById('settings-form');
  try {
    const cfg = await fetchConfig();
    const providers = cfg.available_providers || [];

    const providerOptions = providers.map(p =>
      `<option value="${p}" ${p === cfg.default_provider ? 'selected' : ''}>${p}</option>`
    ).join('');

    container.innerHTML = `
      <div class="settings-section">
        <h3>Default Model</h3>
        <div class="settings-row">
          <label>Provider</label>
          <select id="s-default-provider">${providerOptions}</select>
        </div>
        <div class="settings-row">
          <label>Model</label>
          <input id="s-default-model" type="text" value="${escapeHtml(cfg.default_model || '')}" placeholder="e.g. gpt-4o">
        </div>
        <div class="settings-row">
          <label>Max Iterations</label>
          <input id="s-max-iterations" type="number" min="1" max="10" value="${cfg.max_iterations || 3}">
        </div>
        <div class="settings-row">
          <label>Quality Threshold</label>
          <input id="s-quality-threshold" type="number" min="0" max="1" step="0.05" value="${cfg.quality_threshold || 0.7}">
        </div>
      </div>
      <button class="btn-primary" onclick="saveSettings()" style="align-self:flex-start">Save Settings</button>
      <div id="settings-msg" style="font-size:12px;margin-top:4px"></div>
    `;

  } catch {
    container.innerHTML = '<p style="color:var(--red)">Failed to load settings. Is the backend running?</p>';
  }
}

async function saveSettings() {
  const provider = document.getElementById('s-default-provider').value;
  const model = document.getElementById('s-default-model').value.trim();
  const maxIterations = parseInt(document.getElementById('s-max-iterations').value);
  const qualityThreshold = parseFloat(document.getElementById('s-quality-threshold').value);

  try {
    await saveConfig({
      default_provider: provider,
      default_model: model,
      max_iterations: maxIterations,
      quality_threshold: qualityThreshold,
    });
    document.getElementById('settings-msg').innerHTML = '<span style="color:var(--green)">Settings saved successfully.</span>';
  } catch (e) {
    document.getElementById('settings-msg').innerHTML = `<span style="color:var(--red)">Failed to save: ${e.message}</span>`;
  }
}

window.saveSettings = saveSettings;
