// ── App entry ────────────────────────────────────────────────────

let currentPage = 'input';

function navigateTo(page) {
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
  if (page === 'dashboard' && !STATE.running && !STATE.complete) {
    // Don't navigate to empty dashboard
    return;
  }
  navigateTo(page);
});

// ── Initialize ───────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  initInputPage();

  // Listen for port injected by Electron
  if (window.process && window.process.argv) {
    // Running in Electron
    window.addEventListener('message', (event) => {
      if (event.data && event.data.type === 'backend-port') {
        window.BACKEND_PORT_WRITABLE = event.data.port;
      }
    });
  }
});
