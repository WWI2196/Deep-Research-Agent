// ── Backend communication ────────────────────────────────────────

const BASE = () => '';

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

async function startResearchApi(query, depth, model, documentCollections) {
  return apiPost('/api/research', {
    query,
    max_iterations: depth,
    model: model || undefined,
    document_collections: documentCollections || undefined,
  });
}

function startResearchStream(query, depth, model, documentCollections, onEvent, onError, onDone) {
  const body = JSON.stringify({
    query,
    max_iterations: depth,
    model: model || undefined,
    document_collections: documentCollections || undefined,
  });

  // Native EventSource doesn't support POST body; use fetch + ReadableStream instead
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${BASE()}/api/research/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
        signal: controller.signal,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        onError(new Error(err.detail || `HTTP ${res.status}`));
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        let currentEvent = null;
        let currentData = '';
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            currentData += line.slice(6);
          } else if (line === '' && currentData !== '') {
            try {
              const payload = JSON.parse(currentData);
              // Backend only sends 'data:' lines; read type from JSON payload
              const evtType = currentEvent || payload.type || 'log';
              onEvent({ type: evtType, ...payload });
            } catch {
              // ignore malformed line
            }
            currentEvent = null;
            currentData = '';
          }
        }
      }
      onDone();
    } catch (err) {
      if (err.name !== 'AbortError') onError(err);
    }
  })();

  return controller;
}

async function fetchRunStatus(runId) {
  return apiGet(`/api/research/${runId}/status`);
}

async function cancelResearch(runId) {
  return apiPost(`/api/research/${runId}/cancel`, {});
}

async function fetchHistory() {
  return apiGet('/api/research/history');
}

async function deleteHistory(runId) {
  const res = await fetch(`${BASE()}/api/research/${runId}`, { method: 'DELETE' });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
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

async function fetchReport(runId) {
  return apiGet(`/api/research/${runId}/report`);
}

async function fetchRunLogs(runId, { phase = '', level = '', eventType = '', limit = 500 } = {}) {
  const params = new URLSearchParams();
  if (phase) params.append('phase', phase);
  if (level) params.append('level', level);
  if (eventType) params.append('event_type', eventType);
  params.append('limit', String(limit));
  return apiGet(`/api/research/${runId}/logs?${params.toString()}`);
}

async function fetchRunLlmCalls(runId, { role = '', limit = 500 } = {}) {
  const params = new URLSearchParams();
  if (role) params.append('role', role);
  params.append('limit', String(limit));
  return apiGet(`/api/research/${runId}/llm-calls?${params.toString()}`);
}

async function fetchRunTimeline(runId, limit = 1000) {
  return apiGet(`/api/research/${runId}/timeline?limit=${limit}`);
}

// ── Document collections ────────────────────────────────────────

async function fetchCollections() {
  return apiGet('/api/collections');
}

async function createCollection(name, description = '') {
  return apiPost('/api/collections', { name, description });
}

async function deleteCollection(id) {
  const res = await fetch(`${BASE()}/api/collections/${id}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Delete failed');
  return res.json();
}

async function fetchDocuments(collectionId) {
  return apiGet(`/api/collections/${collectionId}/documents`);
}

async function uploadDocument(collectionId, file) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE()}/api/collections/${collectionId}/documents`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

async function deleteDocument(collectionId, docId) {
  const res = await fetch(`${BASE()}/api/collections/${collectionId}/documents/${docId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Delete failed');
  return res.json();
}

async function reindexDocument(collectionId, docId) {
  const res = await fetch(`${BASE()}/api/collections/${collectionId}/documents/${docId}/reindex`, { method: 'POST' });
  if (!res.ok) throw new Error('Reindex failed');
  return res.json();
}

async function reindexCollection(collectionId) {
  const res = await fetch(`${BASE()}/api/collections/${collectionId}/reindex`, { method: 'POST' });
  if (!res.ok) throw new Error('Reindex failed');
  return res.json();
}
