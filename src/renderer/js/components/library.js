// ── Library page ─────────────────────────────────────────────────

let _selectedCollectionId = null;

async function initLibraryPage() {
  const listeners = [];
  function on(el, event, handler) {
    if (!el) return;
    el.addEventListener(event, handler);
    listeners.push(() => el.removeEventListener(event, handler));
  }

  // Poll for indexing status every 2 seconds while on library page
  let pollInterval = setInterval(() => {
    if (_selectedCollectionId) {
      loadDocuments(_selectedCollectionId);
    }
    loadCollections();
  }, 2000);

  onPageCleanup('library', () => {
    listeners.forEach(fn => fn());
    clearInterval(pollInterval);
  });

  on(document.getElementById('btn-new-collection'), 'click', createNewCollection);
  on(document.getElementById('btn-upload-doc'), 'click', () => {
    document.getElementById('file-upload-input')?.click();
  });
  on(document.getElementById('file-upload-input'), 'change', handleFileUpload);
  on(document.getElementById('btn-delete-collection'), 'click', deleteSelectedCollection);
  on(document.getElementById('btn-reindex-collection'), 'click', reindexSelectedCollection);

  await loadCollections();
}

async function loadCollections() {
  const list = document.getElementById('collections-list');
  if (!list) return;
  list.innerHTML = '<div style="color:var(--text-tertiary);font-size:12px;padding:8px">Loading...</div>';

  try {
    const data = await fetchCollections();
    const collections = data.collections || [];
    if (collections.length === 0) {
      list.innerHTML = '<div style="color:var(--text-tertiary);font-size:12px;padding:8px">No libraries yet</div>';
      return;
    }

    list.innerHTML = collections.map(c => `
      <div class="collection-item ${c.id === _selectedCollectionId ? 'active' : ''}" data-id="${c.id}">
        <div class="collection-name">${escapeHtml(c.name)}</div>
        <div class="collection-meta">${c.doc_count || 0} docs</div>
      </div>
    `).join('');

    list.querySelectorAll('.collection-item').forEach(el => {
      el.addEventListener('click', () => {
        _selectedCollectionId = el.dataset.id;
        loadCollections(); // re-render to update active state
        loadDocuments(_selectedCollectionId);
      });
    });

    if (_selectedCollectionId && !collections.find(c => c.id === _selectedCollectionId)) {
      _selectedCollectionId = null;
    }
    updateLibraryMain();
  } catch (e) {
    list.innerHTML = `<div style="color:var(--red);font-size:12px;padding:8px">${escapeHtml(e.message)}</div>`;
  }
}

async function loadDocuments(collectionId) {
  const container = document.getElementById('documents-list');
  const title = document.getElementById('library-title');
  if (!container || !title) return;

  title.textContent = 'Loading...';
  container.innerHTML = '<div style="color:var(--text-tertiary);font-size:12px;padding:12px">Loading documents...</div>';

  try {
    const data = await fetchDocuments(collectionId);
    const docs = data.documents || [];

    const colData = await fetchCollections();
    const col = (colData.collections || []).find(c => c.id === collectionId);
    title.textContent = col ? escapeHtml(col.name) : 'Documents';

    if (docs.length === 0) {
      container.innerHTML = '<div style="color:var(--text-tertiary);font-size:12px;padding:12px">No documents in this library</div>';
      return;
    }

    container.innerHTML = docs.map(d => {
      const statusCls = d.status === 'indexed' ? 'green' : d.status === 'failed' ? 'red' : d.status === 'indexing' ? 'amber' : '';
      const statusIcon = d.status === 'indexed' ? '✓' : d.status === 'failed' ? '✗' : d.status === 'indexing' ? '⟳' : '⏳';
      return `
      <div class="document-item">
        <div class="document-info">
          <div class="document-name">${escapeHtml(d.name)}</div>
          <div class="document-meta">${d.file_type || 'file'} · ${d.chunk_count || 0} chunks · <span class="doc-status ${statusCls}">${statusIcon} ${d.status}</span>${d.error_msg ? ' · ' + escapeHtml(d.error_msg) : ''}</div>
        </div>
        <div class="document-actions">
          <button class="btn-icon reindex-doc" data-id="${d.id}" title="Re-index">↻</button>
          <button class="btn-icon delete-doc" data-id="${d.id}" title="Delete">×</button>
        </div>
      </div>
    `;
    }).join('');

    container.querySelectorAll('.delete-doc').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        if (!confirm('Delete this document?')) return;
        await deleteDocument(collectionId, btn.dataset.id);
        await loadDocuments(collectionId);
        await loadCollections();
      });
    });

    container.querySelectorAll('.reindex-doc').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        await reindexDocument(collectionId, btn.dataset.id);
        await loadDocuments(collectionId);
      });
    });
  } catch (e) {
    container.innerHTML = `<div style="color:var(--red);font-size:12px;padding:12px">${escapeHtml(e.message)}</div>`;
  }
  updateLibraryMain();
}

function updateLibraryMain() {
  const actions = document.getElementById('library-actions');
  if (!actions) return;
  actions.style.display = _selectedCollectionId ? '' : 'none';
}

async function createNewCollection() {
  const name = prompt('Library name:');
  if (!name || !name.trim()) return;
  try {
    await createCollection(name.trim());
    await loadCollections();
  } catch (e) {
    alert('Failed: ' + e.message);
  }
}

async function deleteSelectedCollection() {
  if (!_selectedCollectionId) return;
  if (!confirm('Delete this library and all its documents?')) return;
  try {
    await deleteCollection(_selectedCollectionId);
    _selectedCollectionId = null;
    await loadCollections();
    document.getElementById('documents-list').innerHTML = '';
    document.getElementById('library-title').textContent = 'Select a library';
    updateLibraryMain();
  } catch (e) {
    alert('Failed: ' + e.message);
  }
}

async function reindexSelectedCollection() {
  if (!_selectedCollectionId) return;
  if (!confirm('Re-index all documents in this library?')) return;
  try {
    await reindexCollection(_selectedCollectionId);
    await loadDocuments(_selectedCollectionId);
  } catch (e) {
    alert('Failed: ' + e.message);
  }
}

async function handleFileUpload(e) {
  const file = e.target.files[0];
  if (!file || !_selectedCollectionId) return;
  try {
    await uploadDocument(_selectedCollectionId, file);
    await loadDocuments(_selectedCollectionId);
    await loadCollections();
  } catch (err) {
    alert('Upload failed: ' + err.message);
  }
  e.target.value = '';
}
