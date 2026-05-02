// ── Subagent board ───────────────────────────────────────────────

function renderSubagentBoard() {
  const container = document.getElementById('subagent-board');
  if (!container) return;

  const agents = Object.values(STATE.subagents);
  if (agents.length === 0) {
    container.innerHTML = '';
    return;
  }

  container.innerHTML = agents.map(agent => {
    const statusClass = agent.status === 'complete' ? 'done'
      : agent.status === 'error' ? 'failed'
      : agent.status === 'pending' ? 'queued'
      : 'running';

    const statusLabel = agent.status === 'complete' ? 'Done'
      : agent.status === 'error' ? 'Failed'
      : agent.status === 'pending' ? 'Queued'
      : agent.status.replace(/-/g, ' ');

    const sourcesCount = agent.sources.length;
    const searchesCount = agent.searches.length;

    return `
      <div class="subagent-card ${agent.status !== 'complete' && agent.status !== 'pending' && agent.status !== 'error' ? 'active' : ''}">
        <div class="sa-header">
          <span class="sa-title">${escapeHtml(agent.title || agent.id)}</span>
          <span class="sa-status ${statusClass}">${statusLabel}</span>
        </div>
        <div class="sa-meta">
          ${searchesCount > 0 ? `<span>🔍 <span class="val">${searchesCount}</span> searches</span>` : ''}
          ${sourcesCount > 0 ? `<span>📄 <span class="val">${sourcesCount}</span> sources</span>` : ''}
          ${agent.evidenceCount > 0 ? `<span>📊 <span class="val">${agent.evidenceCount}</span> evidence</span>` : ''}
          ${agent.reportLength > 0 ? `<span>📝 <span class="val">${formatNumber(agent.reportLength)}</span> chars</span>` : ''}
        </div>
      </div>
    `;
  }).join('');
}
