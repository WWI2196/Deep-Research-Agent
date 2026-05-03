// ── Event-driven state store ─────────────────────────────────────

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

function createStore() {
  const _state = {
    running: false,
    currentPhase: '',
    error: null,
    planPreview: '',
    subtaskList: [],
    scalingInfo: null,
    subagents: {},
    allSources: [],
    reflectionInfo: null,
    reportDraft: '',
    citedReport: '',
    progressPercent: 0,
    startTime: 0,
    elapsed: 0,
    complete: false,
    completionStats: null,
    llmCalls: [],
    warnings: [],
  };

  const _subscribers = {};       // key → [callbacks]
  const _globalSubscribers = []; // called on every change

  function get(key) {
    return _state[key];
  }

  function set(key, value) {
    const prev = _state[key];
    _state[key] = value;
    if (prev !== value) {
      (_subscribers[key] || []).forEach(fn => fn(value, prev));
      _globalSubscribers.forEach(fn => fn(key, value, prev));
    }
  }

  function subscribe(key, fn) {
    if (!_subscribers[key]) _subscribers[key] = [];
    _subscribers[key].push(fn);
    return () => {
      _subscribers[key] = _subscribers[key].filter(f => f !== fn);
    };
  }

  function subscribeAll(fn) {
    _globalSubscribers.push(fn);
    return () => {
      const idx = _globalSubscribers.indexOf(fn);
      if (idx >= 0) _globalSubscribers.splice(idx, 1);
    };
  }

  function reset() {
    _state.running = false;
    _state.currentPhase = '';
    _state.error = null;
    _state.planPreview = '';
    _state.subtaskList = [];
    _state.scalingInfo = null;
    _state.subagents = {};
    _state.allSources = [];
    _state.reflectionInfo = null;
    _state.reportDraft = '';
    _state.citedReport = '';
    _state.progressPercent = 0;
    _state.startTime = 0;
    _state.elapsed = 0;
    _state.complete = false;
    _state.completionStats = null;
    _state.llmCalls = [];
    _state.warnings = [];
    _globalSubscribers.forEach(fn => fn('*', null, null));
  }

  function ensureAgent(id, title, description, iteration) {
    if (!_state.subagents[id]) {
      _state.subagents[id] = {
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
    return _state.subagents[id];
  }

  function addSource(source) {
    const exists = _state.allSources.find(s => s.url === source.url);
    if (!exists) {
      _state.allSources.push({ ...source, timestamp: Date.now() });
    }
  }

  function getPhaseSteps() {
    const seen = new Set();
    return PHASE_ORDER.map(phase => {
      let status = 'pending';
      if (_state.complete) {
        status = 'completed';
      } else if (phase === _state.currentPhase) {
        status = 'active';
      }
      return { id: phase, label: PHASE_LABELS[phase] || phase, status };
    });
  }

  function getAll() {
    return { ..._state };
  }

  return { get, set, subscribe, subscribeAll, reset, ensureAgent, addSource, getPhaseSteps, getAll };
}

const store = createStore();
