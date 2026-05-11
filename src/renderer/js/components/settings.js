// ── Settings page ────────────────────────────────────────────────

async function initSettingsPage() {
  const container = document.getElementById('settings-form');
  try {
    const cfg = await fetchConfig();

    container.innerHTML = `
      <div class="settings-section">
        <h3>LLM Configuration</h3>
        <div class="settings-row">
          <label>Base URL</label>
          <input id="s-base-url" type="text" value="${escapeHtml(cfg.base_url || '')}" placeholder="https://api.openai.com/v1">
        </div>
        <div class="settings-row">
          <label>API Key</label>
          <input id="s-api-key" type="password" value="${escapeHtml(cfg.api_key || '')}" placeholder="sk-...">
        </div>
        <div class="settings-row">
          <label>Default Model</label>
          <input id="s-default-model" type="text" value="${escapeHtml(cfg.default_model || '')}" placeholder="e.g. gpt-4o">
        </div>
      </div>
      <div class="settings-section">
        <h3>Research Defaults</h3>
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
  const baseUrl = document.getElementById('s-base-url').value.trim();
  const apiKey = document.getElementById('s-api-key').value.trim();
  const model = document.getElementById('s-default-model').value.trim();
  const qualityThreshold = parseFloat(document.getElementById('s-quality-threshold').value);

  try {
    await saveConfig({
      base_url: baseUrl,
      api_key: apiKey,
      default_model: model,
      quality_threshold: qualityThreshold,
    });
    document.getElementById('settings-msg').innerHTML = '<span style="color:var(--green)">Settings saved successfully.</span>';
  } catch (e) {
    document.getElementById('settings-msg').innerHTML = `<span style="color:var(--red)">Failed to save: ${e.message}</span>`;
  }
}

window.saveSettings = saveSettings;
