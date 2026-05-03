// ── Sources panel (left sidebar) ─────────────────────────────────

function renderSourcesPanel() {
  const container = document.getElementById('sources-panel');
  if (!container) return;

  if (store.get('allSources').length === 0) {
    container.innerHTML = '<div class="sources-header">Sources</div><div style="font-size:11px;color:var(--text-tertiary)">Waiting for sources...</div>';
    return;
  }

  const sorted = [...store.get('allSources')].sort((a, b) => (b.score || 0) - (a.score || 0));

  // Domain counts
  const domainCounts = {};
  sorted.forEach(s => {
    const d = getDomain(s.url);
    domainCounts[d] = (domainCounts[d] || 0) + 1;
  });

  const domainTags = Object.entries(domainCounts)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 8)
    .map(([d, c]) => `<span class="domain-tag">${d} ${c}</span>`)
    .join('');

  const sourceRows = sorted.slice(0, 15).map(s => {
    const pct = Math.round((s.score || 0) * 100);
    const cls = pct >= 70 ? 'high' : pct >= 40 ? 'mid' : 'low';
    return `
      <div class="source-row">
        <div class="quality-bar"><div class="quality-fill ${cls}" style="width:${pct}%"></div></div>
        <a class="source-url" href="${s.url}" title="${escapeHtml(s.title || s.url)}">${escapeHtml(truncate(s.title || getDomain(s.url), 40))}</a>
      </div>
    `;
  }).join('');

  container.innerHTML = `
    <div class="sources-panel">
      <div class="sources-header">Sources (${sorted.length})</div>
      ${domainTags ? `<div class="domain-tags">${domainTags}</div>` : ''}
      ${sourceRows}
    </div>
  `;
}

// ── Agents panel (right sidebar) ─────────────────────────────────

function renderAgentsPanel() {
  const container = document.getElementById('agents-panel');
  if (!container) return;

  const agents = Object.values(store.get('subagents'));
  if (agents.length === 0) {
    container.innerHTML = '';
    return;
  }

  const initial = agents.filter(a => a.iteration <= 1);
  const gapFill = agents.filter(a => a.iteration > 1);

  const renderRow = (agent) => `
    <div class="agent-row">
      <div class="agent-dot ${agent.status === 'complete' ? 'done' : agent.status === 'pending' ? 'queued' : 'running'}"></div>
      <span class="agent-name">${escapeHtml(agent.title || agent.id)}</span>
      ${agent.sources.length > 0 ? `<span class="agent-stat">${agent.sources.length} src</span>` : ''}
    </div>
  `;

  container.innerHTML = `
    <div class="agents-panel">
      <div style="font-size:12px;font-weight:600;color:var(--text-secondary);margin-bottom:4px">
        Agents (${agents.filter(a => a.status === 'complete').length}/${agents.length})
      </div>
      ${initial.map(renderRow).join('')}
      ${gapFill.length > 0 ? `
        <div style="font-size:11px;font-weight:600;color:var(--amber);margin-top:8px;padding-top:8px;border-top:1px solid var(--border)">
          Gap-Fill (${gapFill.filter(a => a.status === 'complete').length}/${gapFill.length})
        </div>
        ${gapFill.map(renderRow).join('')}
      ` : ''}
    </div>
  `;
}
