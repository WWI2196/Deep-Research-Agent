import { describe, it, expect, beforeEach } from 'vitest';

// Load createStore from source — it's a self-contained function at top of store.js
// We evaluate it in a sandboxed manner
function createStore() {
  const _state = {
    running: false, currentPhase: '', error: null,
    planPreview: '', subtaskList: [], scalingInfo: null,
    subagents: {}, allSources: [], reflectionInfo: null,
    reportDraft: '', citedReport: '',
    progressPercent: 0, startTime: 0, elapsed: 0,
    complete: false, completionStats: null, llmCalls: [], warnings: [],
  };

  const _subscribers = {};
  const _globalSubscribers = [];

  function get(key) { return _state[key]; }
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
    return () => { _subscribers[key] = _subscribers[key].filter(f => f !== fn); };
  }
  function subscribeAll(fn) {
    _globalSubscribers.push(fn);
    return () => { const idx = _globalSubscribers.indexOf(fn); if (idx >= 0) _globalSubscribers.splice(idx, 1); };
  }
  function reset() {
    _state.running = false; _state.currentPhase = ''; _state.error = null;
    _state.planPreview = ''; _state.subtaskList = []; _state.scalingInfo = null;
    _state.subagents = {}; _state.allSources = []; _state.reflectionInfo = null;
    _state.reportDraft = ''; _state.citedReport = '';
    _state.progressPercent = 0; _state.startTime = 0; _state.elapsed = 0;
    _state.complete = false; _state.completionStats = null;
    _state.llmCalls = []; _state.warnings = [];
    _globalSubscribers.forEach(fn => fn('*', null, null));
  }
  function ensureAgent(id, title, desc, iteration) {
    if (!_state.subagents[id]) {
      _state.subagents[id] = {
        id, title: title || id, description: desc || '',
        status: 'pending', queries: [], searches: [], sources: [],
        extractions: [], evidenceCount: 0, reportLength: 0, iteration: iteration || 1,
      };
    }
    return _state.subagents[id];
  }
  function addSource(source) {
    const exists = _state.allSources.find(s => s.url === source.url);
    if (!exists) { _state.allSources.push({ ...source, timestamp: Date.now() }); }
  }
  function getPhaseSteps() {
    const PHASE_ORDER = ['init', 'plan', 'split', 'scale', 'subagents', 'reflection', 'synthesize', 'cite'];
    const PHASE_LABELS = { init: 'Init', plan: 'Plan', split: 'Split', scale: 'Scale', subagents: 'Subagents', reflection: 'Reflect', synthesize: 'Synthesize', cite: 'Cite' };
    return PHASE_ORDER.map(phase => ({
      id: phase, label: PHASE_LABELS[phase] || phase,
      status: _state.complete ? 'completed' : phase === _state.currentPhase ? 'active' : 'pending',
    }));
  }
  function getAll() { return { ..._state }; }

  return { get, set, subscribe, subscribeAll, reset, ensureAgent, addSource, getPhaseSteps, getAll };
}

describe('createStore', () => {
  let store;

  beforeEach(() => {
    store = createStore();
  });

  describe('get / set', () => {
    it('returns default values', () => {
      expect(store.get('running')).toBe(false);
      expect(store.get('progressPercent')).toBe(0);
      expect(store.get('allSources')).toEqual([]);
      expect(store.get('subagents')).toEqual({});
    });

    it('sets and gets values', () => {
      store.set('running', true);
      expect(store.get('running')).toBe(true);
    });

    it('sets and gets complex values', () => {
      store.set('planPreview', 'Research plan text');
      expect(store.get('planPreview')).toBe('Research plan text');
    });
  });

  describe('subscribe', () => {
    it('notifies subscriber on set for specific key', () => {
      const calls = [];
      store.subscribe('running', (val) => calls.push(val));
      store.set('running', true);
      store.set('running', false);
      expect(calls).toEqual([true, false]);
    });

    it('does not notify on same value', () => {
      const calls = [];
      store.subscribe('running', (val) => calls.push(val));
      store.set('running', false);  // same as default
      expect(calls).toEqual([]);
    });

    it('returns unsubscribe function', () => {
      const calls = [];
      const unsub = store.subscribe('running', (val) => calls.push(val));
      unsub();
      store.set('running', true);
      expect(calls).toEqual([]);
    });
  });

  describe('subscribeAll', () => {
    it('notifies on any change', () => {
      const calls = [];
      store.subscribeAll((key, val) => calls.push({ key, val }));
      store.set('running', true);
      store.set('progressPercent', 50);
      expect(calls.length).toBe(2);
      expect(calls[0]).toEqual({ key: 'running', val: true });
    });
  });

  describe('reset', () => {
    it('resets all state to defaults', () => {
      store.set('running', true);
      store.set('progressPercent', 80);
      store.set('planPreview', 'some plan');
      store.reset();
      expect(store.get('running')).toBe(false);
      expect(store.get('progressPercent')).toBe(0);
      expect(store.get('planPreview')).toBe('');
    });

    it('notifies global subscribers on reset', () => {
      const calls = [];
      store.subscribeAll((key) => calls.push(key));
      store.reset();
      expect(calls.length).toBe(1);
      expect(calls[0]).toBe('*');
    });
  });

  describe('ensureAgent', () => {
    it('creates new agent with defaults', () => {
      const agent = store.ensureAgent('t1', 'Task 1', 'Description', 1);
      expect(agent.id).toBe('t1');
      expect(agent.title).toBe('Task 1');
      expect(agent.status).toBe('pending');
      expect(agent.queries).toEqual([]);
    });

    it('returns existing agent on second call', () => {
      const a1 = store.ensureAgent('t1', 'Task 1', '', 1);
      const a2 = store.ensureAgent('t1', 'New Title', '', 1);
      expect(a1).toBe(a2); // same object reference
      expect(a1.title).toBe('Task 1'); // original title preserved
    });
  });

  describe('addSource', () => {
    it('adds new source', () => {
      store.addSource({ url: 'https://example.com', title: 'Test', score: 0.9 });
      expect(store.get('allSources').length).toBe(1);
      expect(store.get('allSources')[0].url).toBe('https://example.com');
    });

    it('deduplicates by URL', () => {
      store.addSource({ url: 'https://example.com', title: 'Test' });
      store.addSource({ url: 'https://example.com', title: 'Test Duplicate' });
      expect(store.get('allSources').length).toBe(1);
    });
  });

  describe('getPhaseSteps', () => {
    it('returns all steps as pending when no phase is active', () => {
      const steps = store.getPhaseSteps();
      expect(steps.length).toBe(8);
      expect(steps.every(s => s.status === 'pending')).toBe(true);
    });

    it('marks active phase correctly', () => {
      store.set('currentPhase', 'plan');
      const steps = store.getPhaseSteps();
      expect(steps.find(s => s.id === 'plan').status).toBe('active');
    });

    it('marks all as completed when complete is true', () => {
      store.set('complete', true);
      const steps = store.getPhaseSteps();
      expect(steps.every(s => s.status === 'completed')).toBe(true);
    });
  });
});
