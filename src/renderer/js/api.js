// ── Backend communication ────────────────────────────────────────

const BACKEND_PORT = 8787; // Default, updated by Electron on startup
const BASE = () => `http://127.0.0.1:${BACKEND_PORT}`;

async function apiGet(path) {
  const res = await fetch(`${BASE()}${path}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(`${BASE()}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

function streamResearch(query, depth, provider, model, onEvent, onError, onDone) {
  const url = `${BASE()}/api/research/stream`;
  fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      max_iterations: depth,
      provider: provider || undefined,
      model: model || undefined,
    }),
  }).then(async (response) => {
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Connection failed' }));
      onError(new Error(err.detail || 'Backend error'));
      return;
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            onEvent(data);
          } catch {}
        }
      }
    }
    onDone();
  }).catch(err => onError(err));
}

async function cancelResearch(runId) {
  return apiPost(`/api/research/${runId}/cancel`, {});
}

async function fetchHistory() {
  return apiGet('/api/research/history');
}

async function fetchConfig() {
  return apiGet('/api/config');
}

async function saveConfig(cfg) {
  return apiPost('/api/config', cfg);
}

async function fetchHealth() {
  return apiGet('/api/health');
}

async function fetchModels() {
  return apiGet('/api/models');
}
