// ── Global state management ──────────────────────────────────────

const PHASE_ORDER = ['init', 'plan', 'split', 'scale', 'subagents', 'reflection', 'synthesize', 'cite'];

const PHASE_LABELS = {
  init: 'Initializing',
  plan: 'Planning Research',
  split: 'Creating Subtasks',
  scale: 'Estimating Complexity',
  subagents: 'Running Subagents',
  reflection: 'Reflecting & Gap Analysis',
  synthesize: 'Synthesizing Report',
  cite: 'Adding Citations',
};

const STATE = {
  // Research state
  running: false,
  runId: null,
  currentPhase: '',
  error: null,

  // Data
  planPreview: '',
  subtaskList: [],
  scalingInfo: null,
  subagents: {},       // Map<subtaskId, SubagentState>
  allSources: [],       // GatheredSource[]
  reflectionInfo: null,
  reportDraft: '',
  citedReport: '',

  // Progress
  progressPercent: 0,
  startTime: 0,
  elapsed: 0,

  // Completion
  complete: false,
  completionStats: null,

  // LLM calls
  llmCalls: [],

  // Warnings
  warnings: [],
};

function resetState() {
  STATE.running = false;
  STATE.runId = null;
  STATE.currentPhase = '';
  STATE.error = null;
  STATE.planPreview = '';
  STATE.subtaskList = [];
  STATE.scalingInfo = null;
  STATE.subagents = {};
  STATE.allSources = [];
  STATE.reflectionInfo = null;
  STATE.reportDraft = '';
  STATE.citedReport = '';
  STATE.progressPercent = 0;
  STATE.startTime = 0;
  STATE.elapsed = 0;
  STATE.complete = false;
  STATE.completionStats = null;
  STATE.llmCalls = [];
  STATE.warnings = [];
}

function ensureAgent(id, title, description, iteration) {
  if (!STATE.subagents[id]) {
    STATE.subagents[id] = {
      id, title: title || id, description: description || '',
      status: 'pending',
      queries: [],
      searches: [],
      sources: [],
      extractions: [],
      evidenceCount: 0,
      reportLength: 0,
      iteration: iteration || 1,
    };
  }
  return STATE.subagents[id];
}

function addSource(source) {
  const exists = STATE.allSources.find(s => s.url === source.url);
  if (!exists) {
    STATE.allSources.push({ ...source, timestamp: Date.now() });
  }
}

function getPhaseSteps() {
  const seen = new Set();
  const steps = PHASE_ORDER.map(phase => {
    let status = 'pending';
    if (STATE.complete) {
      status = phase === 'complete' ? 'completed' : 'completed';
    } else if (phase === STATE.currentPhase) {
      status = 'active';
    } else if (seen.has(phase)) {
      status = 'completed';
    }
    return { id: phase, label: PHASE_LABELS[phase] || phase, status };
  });
  return steps;
}

