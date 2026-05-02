// ── Dashboard event handling & orchestration ──────────────────────

let elapsedTimer = null;

function handleResearchEvent(evt) {
  const e = evt;

  if (e.type === 'phase-update') {
    STATE.currentPhase = e.phase;
  }
  if (e.type === 'plan-generated') {
    STATE.planPreview = e.message || e.plan_preview || '';
  }
  if (e.type === 'subtasks-created') {
    STATE.subtaskList = e.subtasks || [];
  }
  if (e.type === 'scaling-computed') {
    STATE.scalingInfo = e.scaling || e;
  }
  if (e.type === 'progress') {
    STATE.progressPercent = e.percent || 0;
  }

  // Subagent events
  if (e.type === 'subagents-launch') {
    const iter = e.iteration || 1;
    (e.agent_details || []).forEach(a => ensureAgent(a.id, a.title, a.description, iter));
  }
  if (e.type === 'subagent-step') {
    const sa = ensureAgent(e.subtask_id, e.subtask_title, '');
    sa.status = e.step;
    if (e.evidence_count !== undefined) sa.evidenceCount = e.evidence_count;
  }
  if (e.type === 'subagent-queries') {
    const sa = ensureAgent(e.subtask_id, e.subtask_title, '');
    (e.queries || []).forEach(q => { if (!sa.queries.includes(q)) sa.queries.push(q); });
  }
  if (e.type === 'subagent-search') {
    const sa = ensureAgent(e.subtask_id, '', '');
    const existing = sa.searches.find(s => s.query === e.query);
    if (existing) {
      existing.status = e.status;
      if (e.results_found !== undefined) existing.results_found = e.results_found;
    } else {
      sa.searches.push({ query: e.query, status: e.status, results_found: e.results_found });
    }
  }
  if (e.type === 'subagent-sources-scored') {
    const sa = ensureAgent(e.subtask_id, e.subtask_title, '');
    (e.top_scores || []).forEach(s => {
      addSource({ url: s.url, title: s.title, score: s.score, subtask: e.subtask_title || '' });
      if (!sa.sources.find(x => x.url === s.url)) sa.sources.push({ url: s.url, title: s.title, score: s.score });
    });
  }
  if (e.type === 'subagent-extract') {
    const sa = ensureAgent(e.subtask_id, '', '');
    const existing = sa.extractions.find(x => x.url === e.url);
    if (existing) existing.status = e.status;
    else sa.extractions.push({ url: e.url, status: e.status });
  }
  if (e.type === 'subagent-complete') {
    const sa = ensureAgent(e.subtask_id, e.subtask_title, '');
    sa.status = 'complete';
    sa.reportLength = e.report_length || 0;
    sa.evidenceCount = e.evidence_count || 0;
  }

  // LLM calls
  if (e.type === 'llm-call') {
    if (e.status === 'started') {
      STATE.llmCalls.push({ model: e.model, provider: e.provider, role: e.role, status: 'started', attempt: e.attempt || 1, timestamp: Date.now() });
    } else {
      const prev = [...STATE.llmCalls].reverse().find(c => c.role === e.role && c.status === 'started');
      if (prev) {
        prev.status = e.status;
        prev.error = e.error;
        prev.outputLength = e.output_length;
      }
    }
  }

  // Reflection
  if (e.type === 'reflection-decision') {
    STATE.reflectionInfo = {
      decision: e.decision || (e.research_complete ? 'research-complete' : 'gaps-found'),
      new_subtasks: e.new_subtasks || [],
      iteration: e.iteration || 0,
      total_reports: e.total_reports,
      total_sources: e.total_sources,
    };
  }

  // Report
  if (e.type === 'report-draft') {
    STATE.reportDraft = e.content || '';
  }

  // Final
  if (e.type === 'final-result') {
    STATE.citedReport = e.content || '';
  }

  if (e.type === 'warning') {
    STATE.warnings.push(`[${e.phase}] ${e.message}`);
  }

  if (e.type === 'error') {
    STATE.error = { error: e.error, hint: e.hint, phase: e.phase };
  }

  if (e.type === 'complete') {
    STATE.complete = true;
    STATE.progressPercent = 100;
    STATE.completionStats = {
      total_sources: e.total_sources,
      total_reports: e.total_reports,
      iterations: e.iterations,
      provider: e.provider,
      model: e.model,
    };
    if (elapsedTimer) clearInterval(elapsedTimer);
  }

  // Update UI
  updateDashboardUI();
}

function handleResearchError(err) {
  STATE.error = { error: err.message, hint: '', phase: 'connection' };
  STATE.running = false;
  document.getElementById('btn-start').disabled = false;
  updateDashboardUI();
}

function handleResearchDone() {
  STATE.running = false;
  document.getElementById('btn-start').disabled = false;
  if (STATE.complete) {
    navigateTo('report');
    renderReportPage();
  }
  updateDashboardUI();
}

function updateDashboardUI() {
  STATE.elapsed = Date.now() - STATE.startTime;

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
    if (STATE.running || !STATE.complete) {
      STATE.elapsed = Date.now() - STATE.startTime;
      updateDashboardUI();
    }
    if (STATE.complete && elapsedTimer) {
      clearInterval(elapsedTimer);
      elapsedTimer = null;
    }
  }, 1000);
}

// Override handleResearchEvent to start timer
const _originalHandle = handleResearchEvent;
handleResearchEvent = function(evt) {
  if (!elapsedTimer && !STATE.complete) startElapsedTimer();
  _originalHandle(evt);
};
