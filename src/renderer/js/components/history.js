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
          <th>Query</th><th>Status</th><th>Sources</th><th>Reports</th><th>Iterations</th><th>Time</th>
        </tr></thead>
        <tbody>
          ${data.history.map(r => `
            <tr onclick="viewHistoryReport('${r.run_id}')">
              <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escapeHtml(r.query)}</td>
              <td><span class="history-status ${r.status}">${r.status}</span></td>
              <td>${r.total_sources || 0}</td>
              <td>${r.total_reports || 0}</td>
              <td>${r.iterations || 0}</td>
              <td style="color:var(--text-tertiary);font-size:11px">${timeAgo(r.started_at)}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;
  } catch {
    container.innerHTML = '<p style="color:var(--red);font-size:13px">Failed to load history. Is the backend running?</p>';
  }
}
