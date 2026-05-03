// ── Progress bar ─────────────────────────────────────────────────

function renderProgressBar() {
  const container = document.getElementById('progress-bar');
  if (!container) return;

  const pct = Math.min(store.get('progressPercent'), 100);
  const phaseLabel = PHASE_LABELS[store.get('currentPhase')] || '';

  container.innerHTML = `
    <div class="progress-header">
      <span class="label">
        ${store.get('complete') ? 'Research Complete' : (phaseLabel || 'Initializing...')}
      </span>
      <div style="display:flex;align-items:center;gap:12px">
        <span style="font-size:11px;color:var(--text-tertiary);font-family:monospace">${formatDuration(store.get('elapsed'))}</span>
        <span class="pct">${pct}%</span>
      </div>
    </div>
    <div class="progress-track">
      <div class="progress-fill" style="width:${pct}%"></div>
    </div>
  `;
}

// ── Phase timeline ───────────────────────────────────────────────

function renderPhaseTimeline() {
  const container = document.getElementById('phase-timeline');
  if (!container) return;

  const steps = store.getPhaseSteps();
  const completedSteps = steps.filter(s => s.status !== 'pending');

  container.innerHTML = completedSteps.map(step => {
    const iconClass = step.status === 'completed' ? 'done' : step.status === 'active' ? 'active' : 'pending';
    const icon = step.status === 'completed' ? '✓' : step.status === 'active' ? '●' : '○';
    let detail = '';

    if (step.id === 'plan' && store.get('planPreview')) {
      detail = `Plan generated (${formatNumber(store.get('planPreview').length)} chars)`;
    } else if (step.id === 'split' && store.get('subtaskList').length) {
      detail = `${store.get('subtaskList').length} subtasks created`;
    } else if (step.id === 'scale' && store.get('scalingInfo')) {
      detail = `Complexity: ${store.get('scalingInfo').complexity}`;
    } else if (step.id === 'subagents') {
      const agents = Object.values(store.get('subagents'));
      const done = agents.filter(a => a.status === 'complete').length;
      detail = `${done}/${agents.length} agents complete`;
    } else if (step.id === 'reflection' && store.get('reflectionInfo')) {
      detail = store.get('reflectionInfo').decision === 'research-complete'
        ? 'No gaps found'
        : store.get('reflectionInfo').decision === 'max-iterations-reached'
        ? 'Max iterations reached'
        : `${store.get('reflectionInfo').new_subtasks.length} gaps identified`;
    } else if (step.id === 'synthesize' && store.get('reportDraft')) {
      detail = `Report: ${formatNumber(store.get('reportDraft').length)} chars`;
    } else if (step.id === 'cite' && store.get('citedReport')) {
      detail = `${formatNumber(store.get('citedReport').length)} chars cited`;
    }

    return `
      <div class="phase-row">
        <div class="phase-icon ${iconClass}">${icon}</div>
        <div class="phase-info">
          <div class="name">${step.label}</div>
          ${detail ? `<div class="detail">${detail}</div>` : ''}
        </div>
        ${step.status === 'active' ? '<span class="phase-badge running">In Progress</span>' : ''}
      </div>
    `;
  }).join('');
}
