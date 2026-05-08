// ── History page ─────────────────────────────────────────────────

async function initHistoryPage() {
  const container = document.getElementById('history-list');
  try {
    const data = await fetchHistory();
    if (!data.history || data.history.length === 0) {
      container.innerHTML = '<p style="color:var(--text-tertiary);font-size:13px">No research history yet.</p>';
      return;
    }
    container.innerHTML = `
      <table class="history-table">
        <thead><tr>
          <th>Query</th><th>Status</th><th>Sources</th><th>Reports</th><th>Iterations</th><th>Time</th><th></th>
        </tr></thead>
        <tbody>
          ${data.history.map(r => {
            let onClick = '';
            let rowClass = 'not-clickable';
            let cursorStyle = 'cursor:not-allowed';
            if (r.status === 'completed') {
              onClick = `onclick="viewHistoryReport('${r.run_id}')"`;
              rowClass = 'clickable';
              cursorStyle = 'cursor:pointer';
            } else if (r.status === 'running') {
              onClick = `onclick="viewHistoryProcess('${r.run_id}')"`;
              rowClass = 'clickable';
              cursorStyle = 'cursor:pointer';
            }
            const opacity = (r.status === 'completed' || r.status === 'running') ? '' : 'opacity:0.6';
            return `
            <tr class="${rowClass}" ${onClick} style="${opacity};${cursorStyle}">
              <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escapeHtml(r.query)}</td>
              <td><span class="history-status ${r.status}">${r.status}</span></td>
              <td>${r.total_sources || 0}</td>
              <td>${r.total_reports || 0}</td>
              <td>${r.iterations || 0}</td>
              <td style="color:var(--text-tertiary);font-size:11px">${timeAgo(r.started_at)}</td>
              <td>
                <button class="history-log-btn" onclick="viewHistoryLogs(event, '${r.run_id}')" title="Logs">Logs</button>
                <button class="history-delete-btn" onclick="deleteHistoryItem(event, '${r.run_id}')" title="Delete">×</button>
              </td>
            </tr>`;
          }).join('')}
        </tbody>
      </table>
    `;
  } catch {
    container.innerHTML = '<p style="color:var(--red);font-size:13px">Failed to load history. Is the backend running?</p>';
  }
}

async function deleteHistoryItem(event, runId) {
  event.stopPropagation();
  if (!confirm('Delete this research record?')) return;
  try {
    // If still running, cancel first
    try { await cancelResearch(runId); } catch { /* ignore if not running */ }
    await deleteHistory(runId);
    initHistoryPage();
  } catch {
    alert('Failed to delete. Is the backend running?');
  }
}

function viewHistoryLogs(event, runId) {
  event.stopPropagation();
  showLogModal(runId);
}

window.viewHistoryProcess = async function(runId) {
  store.set('currentRunId', runId);
  store.set('running', true);
  store.set('complete', false);
  // Reset timeline tracking so events are processed from the beginning
  if (typeof lastTimelineLength !== 'undefined') lastTimelineLength = 0;
  try {
    const data = await fetchRunStatus(runId);
    store.set('startTime', (data.started_at || Date.now() / 1000) * 1000);
    store.set('currentPhase', data.phase || '');
    store.set('progressPercent', data.progress_percent || 0);
  } catch {
    store.set('startTime', Date.now());
  }
  navigateTo('dashboard');
};

window.deleteHistoryItem = deleteHistoryItem;
window.viewHistoryLogs = viewHistoryLogs;
