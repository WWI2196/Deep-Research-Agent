// ── Report page ──────────────────────────────────────────────────

function renderReportPage() {
  const content = store.get('citedReport') || store.get('reportDraft') || '';
  document.getElementById('report-content').innerHTML = renderMarkdown(content);

  // Stats
  if (store.get('completionStats')) {
    const stats = store.get('completionStats');
    document.getElementById('report-stats').innerHTML = `
      <div class="stat"><span class="stat-label">Sources</span><span class="stat-value">${stats.total_sources || 0}</span></div>
      <div class="stat"><span class="stat-label">Reports</span><span class="stat-value">${stats.total_reports || 0}</span></div>
      <div class="stat"><span class="stat-label">Iterations</span><span class="stat-value">${stats.iterations || 0}</span></div>
      <div class="stat"><span class="stat-label">Duration</span><span class="stat-value">${formatDuration(store.get('elapsed'))}</span></div>
    `;
  }

  // Export buttons
  document.getElementById('btn-export-md').onclick = () => {
    const text = store.get('citedReport') || store.get('reportDraft') || '';
    const blob = new Blob([text], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `research-report-${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  document.getElementById('btn-copy').onclick = () => {
    const text = store.get('citedReport') || store.get('reportDraft') || '';
    navigator.clipboard.writeText(text);
  };

  document.getElementById('btn-new-research').onclick = () => {
    navigateTo('input');
  };

  // Also update sources in report sidebar
  renderSourcesInReportSidebar();
}

function renderSourcesInReportSidebar() {
  const sorted = [...store.get('allSources')].sort((a, b) => (b.score || 0) - (a.score || 0));
  if (sorted.length === 0) return;

  const list = sorted.slice(0, 20).map(s => {
    const pct = Math.round((s.score || 0) * 100);
    const cls = pct >= 70 ? 'high' : pct >= 40 ? 'mid' : 'low';
    return `
      <div class="source-row">
        <div class="quality-bar"><div class="quality-fill ${cls}" style="width:${pct}%"></div></div>
        <a class="source-url" href="${s.url}">${escapeHtml(truncate(s.title || getDomain(s.url), 35))}</a>
      </div>
    `;
  }).join('');

  // Append to report sidebar after stats
  const sidebar = document.getElementById('report-sidebar');
  const existing = document.getElementById('report-sources-list');
  if (existing) existing.remove();
  const div = document.createElement('div');
  div.id = 'report-sources-list';
  div.className = 'report-stats';
  div.innerHTML = `<div style="font-size:12px;font-weight:600;color:var(--text-secondary);margin-bottom:8px">Sources (${sorted.length})</div>${list}`;
  sidebar.appendChild(div);
}

async function viewHistoryReport(runId) {
  try {
    const report = await fetchReport(runId);
    store.set('citedReport', report.content || '');
    store.set('reportDraft', '');
    store.set('completionStats', {
      total_sources: report.total_sources || 0,
      total_reports: report.total_reports || 0,
      iterations: report.iterations || 0,
    });
    store.set('allSources', []);
    navigateTo('report');
    renderReportPage();
  } catch {
    navigateTo('report');
    document.getElementById('report-content').innerHTML =
      '<p style="color:var(--text-tertiary)">Failed to load report.</p>';
  }
}
