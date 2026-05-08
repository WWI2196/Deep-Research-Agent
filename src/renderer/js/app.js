// ── App entry ────────────────────────────────────────────────────

let currentPage = 'input';

// Per-page cleanup functions
const _pageCleanups = {};

function onPageCleanup(page, fn) {
  const prev = _pageCleanups[page];
  _pageCleanups[page] = prev ? () => { prev(); fn(); } : fn;
}

function navigateTo(page) {
  // Destroy current page
  if (_pageCleanups[currentPage]) {
    _pageCleanups[currentPage]();
    delete _pageCleanups[currentPage];
  }

  currentPage = page;

  // Update pages
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  const target = document.getElementById(`page-${page}`);
  if (target) target.classList.add('active');

  // Update nav
  document.querySelectorAll('.nav-btn[data-page]').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.page === page);
  });

  // Show/hide dynamic nav buttons
  const dashNav = document.getElementById('nav-dashboard');
  const reportNav = document.getElementById('nav-report');
  dashNav.style.display = (page === 'dashboard') ? '' : 'none';
  reportNav.style.display = (page === 'report') ? '' : 'none';

  // Init page content
  if (page === 'history') initHistoryPage();
  if (page === 'settings') initSettingsPage();
  if (page === 'library') initLibraryPage();
  if (page === 'dashboard') initDashboardPage();
  if (page === 'input') {
    initInputPage();
    loadRecent();
  }
}

// ── Nav button clicks ────────────────────────────────────────────

document.getElementById('nav').addEventListener('click', (e) => {
  const btn = e.target.closest('.nav-btn[data-page]');
  if (!btn) return;
  const page = btn.dataset.page;
  if (page === 'dashboard' && !store.get('currentRunId')) {
    return;
  }
  navigateTo(page);
});

// ── Initialize ───────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  initInputPage();
});
