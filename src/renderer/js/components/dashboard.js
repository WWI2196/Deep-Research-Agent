// ── Dashboard event handling & orchestration ──────────────────────

let elapsedTimer = null;
let statusPoller = null;
let timelinePoller = null;
let lastTimelineLength = 0;

function stopPollers() {
  if (statusPoller) { clearInterval(statusPoller); statusPoller = null; }
  if (timelinePoller) { clearInterval(timelinePoller); timelinePoller = null; }
  if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }
}

// Clean up when navigating away from dashboard
onPageCleanup('dashboard', () => {
  stopPollers();
});

function initDashboardPage() {
  const runId = store.get('currentRunId');
  if (!runId) return;

  // Start elapsed timer
  if (!elapsedTimer && !store.get('complete')) {
    startElapsedTimer();
  }

  // Start status polling
  if (!statusPoller) {
    pollStatus(); // immediate first call
    statusPoller = setInterval(pollStatus, 2000);
  }

  // Start timeline polling for detailed events
  if (!timelinePoller) {
    pollTimeline(); // immediate first call
    timelinePoller = setInterval(pollTimeline, 5000);
  }
}

async function pollStatus() {
  const runId = store.get('currentRunId');
  if (!runId) return;

  try {
    const data = await fetchRunStatus(runId);
    store.set('currentPhase', data.phase || 'unknown');
    store.set('progressPercent', data.progress_percent || 0);
    store.set('completionStats', {
      total_sources: data.total_sources || 0,
      total_reports: data.total_reports || 0,
      iterations: data.iteration || 0,
    });

    if (data.status === 'completed') {
      stopPollers();
      store.set('running', false);
      store.set('complete', true);
      try {
        const report = await fetchReport(runId);
        if (report && report.content) {
          store.set('citedReport', report.content);
        }
      } catch {}
      navigateTo('report');
      renderReportPage();
      return;
    }

    if (data.status === 'cancelled' || data.status === 'failed') {
      stopPollers();
      store.set('running', false);
      store.set('error', { error: `Research ${data.status}`, phase: data.phase || '' });
      updateDashboardUI();
      return;
    }

    updateDashboardUI();
  } catch {
    // Silently retry on next poll
  }
}

async function pollTimeline() {
  const runId = store.get('currentRunId');
  if (!runId) return;

  try {
    const data = await fetchRunTimeline(runId);
    const items = data.items || [];
    if (items.length > lastTimelineLength) {
      const newItems = items.slice(lastTimelineLength);
      lastTimelineLength = items.length;
      for (const item of newItems) {
        const evt = convertTimelineItemToEvent(item);
        if (evt) handleResearchEvent(evt);
      }
    }
  } catch {
    // Silently retry on next poll
  }
}

function convertTimelineItemToEvent(item) {
  // Convert trace_log / llm_call timeline items back to dashboard event format
  const t = item.type || item.event_type || '';
  if (t === 'node_enter' || t === 'node_exit') {
    return { type: 'phase-update', phase: item.phase, message: item.message };
  }
  if (t === 'llm_call') {
    return { type: 'llm-call', role: item.role, status: item.message?.includes('error') ? 'error' : 'completed', model: item.model, provider: item.provider };
  }
  if (t === 'error') {
    return { type: 'error', error: item.message, phase: item.phase };
  }
  return null;
}

function handleResearchEvent(evt) {
  const e = evt;

  if (e.run_id && !store.get('currentRunId')) {
    store.set('currentRunId', e.run_id);
  }

  if (e.type === 'phase-update') {
    store.set('currentPhase', e.phase);
  }
  if (e.type === 'plan-generated') {
    store.set('planPreview', e.message || e.plan_preview || '');
  }
  if (e.type === 'subtasks-created') {
    store.set('subtaskList', e.subtasks || []);
  }
  if (e.type === 'progress') {
    store.set('progressPercent', e.percent || 0);
  }

  // Subagent events
  if (e.type === 'subagents-launch') {
    const iter = e.iteration || 1;
    (e.agent_details || []).forEach(a => store.ensureAgent(a.id, a.title, a.description, iter));
  }
  if (e.type === 'subagent-step') {
    const sa = store.ensureAgent(e.subtask_id, e.subtask_title, '');
    sa.status = e.step;
    if (e.evidence_count !== undefined) sa.evidenceCount = e.evidence_count;
  }
  if (e.type === 'subagent-queries') {
    const sa = store.ensureAgent(e.subtask_id, e.subtask_title, '');
    (e.queries || []).forEach(q => { if (!sa.queries.includes(q)) sa.queries.push(q); });
  }
  if (e.type === 'subagent-search') {
    const sa = store.ensureAgent(e.subtask_id, '', '');
    const existing = sa.searches.find(s => s.query === e.query);
    if (existing) {
      existing.status = e.status;
      if (e.results_found !== undefined) existing.results_found = e.results_found;
    } else {
      sa.searches.push({ query: e.query, status: e.status, results_found: e.results_found });
    }
  }
  if (e.type === 'subagent-sources-scored') {
    const sa = store.ensureAgent(e.subtask_id, e.subtask_title, '');
    (e.top_scores || []).forEach(s => {
      store.addSource({ url: s.url, title: s.title, score: s.score, subtask: e.subtask_title || '' });
      if (!sa.sources.find(x => x.url === s.url)) sa.sources.push({ url: s.url, title: s.title, score: s.score });
    });
  }
  if (e.type === 'subagent-extract') {
    const sa = store.ensureAgent(e.subtask_id, '', '');
    const existing = sa.extractions.find(x => x.url === e.url);
    if (existing) existing.status = e.status;
    else sa.extractions.push({ url: e.url, status: e.status });
  }
  if (e.type === 'subagent-complete') {
    const sa = store.ensureAgent(e.subtask_id, e.subtask_title, '');
    sa.status = 'complete';
    sa.reportLength = e.report_length || 0;
    sa.evidenceCount = e.evidence_count || 0;
  }

  // LLM calls
  if (e.type === 'llm-call') {
    if (e.status === 'started') {
      const calls = store.get('llmCalls');
      calls.push({ model: e.model, provider: e.provider, role: e.role, status: 'started', attempt: e.attempt || 1, timestamp: Date.now() });
      store.set('llmCalls', calls);
    } else {
      const calls = store.get('llmCalls');
      const prev = [...calls].reverse().find(c => c.role === e.role && c.status === 'started');
      if (prev) {
        prev.status = e.status;
        prev.error = e.error;
        prev.outputLength = e.output_length;
      }
    }
  }

  // Reflection
  if (e.type === 'reflection-decision') {
    store.set('reflectionInfo', {
      decision: e.decision || (e.research_complete ? 'research-complete' : 'gaps-found'),
      new_subtasks: e.new_subtasks || [],
      iteration: e.iteration || 0,
      total_reports: e.total_reports,
      total_sources: e.total_sources,
    });
  }

  // Report
  if (e.type === 'report-draft') {
    store.set('reportDraft', e.content || '');
  }

  // Final
  if (e.type === 'final-result') {
    store.set('citedReport', e.content || '');
  }

  if (e.type === 'warning') {
    const warnings = store.get('warnings');
    warnings.push(`[${e.phase}] ${e.message}`);
    store.set('warnings', warnings);
  }

  if (e.type === 'error') {
    store.set('error', { error: e.error, hint: e.hint, phase: e.phase });
    stopPollers();
  }

  if (e.type === 'complete') {
    store.set('complete', true);
    store.set('progressPercent', 100);
    store.set('completionStats', {
      total_sources: e.total_sources,
      total_reports: e.total_reports,
      iterations: e.iterations,
      provider: e.provider,
      model: e.model,
    });
    stopPollers();
  }

  // Update UI
  updateDashboardUI();
}

function handleResearchError(err) {
  store.set('error', { error: err.message, hint: '', phase: 'connection' });
  store.set('running', false);
  document.getElementById('btn-start').disabled = false;
  updateDashboardUI();
}

function handleResearchDone() {
  store.set('running', false);
  document.getElementById('btn-start').disabled = false;
  if (store.get('complete')) {
    navigateTo('report');
    renderReportPage();
  }
  updateDashboardUI();
}

function updateDashboardUI() {
  store.set('elapsed', Date.now() - store.get('startTime'));

  renderProgressBar();
  renderPhaseTimeline();
  renderSubagentBoard();
  renderSourcesPanel();
  renderAgentsPanel();
}

// Start elapsed timer on first event
function startElapsedTimer() {
  if (elapsedTimer) return;
  elapsedTimer = setInterval(() => {
    if (store.get('running') || !store.get('complete')) {
      store.set('elapsed', Date.now() - store.get('startTime'));
      updateDashboardUI();
    }
    if (store.get('complete') && elapsedTimer) {
      clearInterval(elapsedTimer);
      elapsedTimer = null;
    }
  }, 1000);
}

// Override handleResearchEvent to start timer
const _originalHandle = handleResearchEvent;
handleResearchEvent = function(evt) {
  if (!elapsedTimer && !store.get('complete')) startElapsedTimer();
  _originalHandle(evt);
};
