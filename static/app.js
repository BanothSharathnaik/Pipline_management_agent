/**
 * PipelinePulse AI - Frontend Application Logic
 */

let allRuns = [];
let currentFilter = 'ALL';
let currentSearch = '';
let currentReportMarkdown = '';

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initEventListeners();
  loadData();
});

function initTabs() {
  const tabs = [
    { btn: 'tabBtnTelemetry', content: 'telemetryTab' },
    { btn: 'tabBtnChat', content: 'chatTab' },
    { btn: 'tabBtnDocs', content: 'docsTab' }
  ];

  tabs.forEach(t => {
    const btn = document.getElementById(t.btn);
    if (!btn) return;
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const target = document.getElementById(t.content);
      if (target) target.classList.add('active');
    });
  });
}

function initEventListeners() {
  // Source selector
  const sourceSelect = document.getElementById('dataSourceSelect');
  if (sourceSelect) {
    sourceSelect.addEventListener('change', (e) => {
      loadData(e.target.value);
    });
  }

  // Refresh live button
  const btnRefresh = document.getElementById('btnRefreshLive');
  if (btnRefresh) {
    btnRefresh.addEventListener('click', refreshLiveTelemetry);
  }

  // Search input
  const searchInput = document.getElementById('runSearchInput');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      currentSearch = e.target.value.toLowerCase();
      renderRunsTable();
    });
  }

  // Filter pills
  document.querySelectorAll('.pill-btn').forEach(pill => {
    pill.addEventListener('click', () => {
      document.querySelectorAll('.pill-btn').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      currentFilter = pill.getAttribute('data-filter');
      renderRunsTable();
    });
  });

  // Chat form submit
  const chatForm = document.getElementById('chatForm');
  if (chatForm) {
    chatForm.addEventListener('submit', handleChatSubmit);
  }

  // Quick prompts
  document.querySelectorAll('.prompt-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const query = chip.getAttribute('data-query');
      const input = document.getElementById('chatInput');
      if (input) {
        input.value = query;
        // Auto switch to chat tab if not active
        document.getElementById('tabBtnChat')?.click();
        handleChatSubmit(new Event('submit'));
      }
    });
  });

  // Modal close
  const btnCloseModal = document.getElementById('btnCloseModal');
  const incidentModal = document.getElementById('incidentModal');
  if (btnCloseModal && incidentModal) {
    btnCloseModal.addEventListener('click', () => incidentModal.classList.remove('active'));
    incidentModal.addEventListener('click', (e) => {
      if (e.target === incidentModal) incidentModal.classList.remove('active');
    });
  }

  // Copy report
  const btnCopyReport = document.getElementById('btnCopyReport');
  if (btnCopyReport) {
    btnCopyReport.addEventListener('click', () => {
      if (currentReportMarkdown) {
        navigator.clipboard.writeText(currentReportMarkdown).then(() => {
          showToast('Incident Report copied to clipboard!');
        });
      }
    });
  }
}

async function loadData(source = 'live') {
  const tableBody = document.getElementById('runsTableBody');
  if (tableBody) {
    tableBody.innerHTML = `
      <tr>
        <td colspan="7" class="loading-state">
          <div class="spinner"></div>
          <span>Loading telemetry snapshot (${source})...</span>
        </td>
      </tr>
    `;
  }

  try {
    const res = await fetch(`/api/runs?source=${source}`);
    if (!res.ok) throw new Error('Failed to load telemetry');
    const data = await res.json();
    allRuns = data.runs || [];

    updateMetrics(allRuns, source);
    renderRunsTable();
    loadDocsList();
  } catch (err) {
    console.error(err);
    if (tableBody) {
      tableBody.innerHTML = `
        <tr>
          <td colspan="7" class="loading-state" style="color: #f87171;">
            Failed to connect to backend: ${err.message}. Ensure FastAPI server is running on port 8000.
          </td>
        </tr>
      `;
    }
  }
}

function updateMetrics(runs, source) {
  const total = runs.length;
  const failed = runs.filter(r => (r.status || '').toLowerCase().includes('fail') || (r.status || '').toLowerCase().includes('time')).length;
  const success = runs.filter(r => (r.status || '').toLowerCase() === 'success').length;
  const rate = total > 0 ? Math.round((success / total) * 100) : 0;

  document.getElementById('valTotalRuns').textContent = total;
  document.getElementById('valFailedRuns').textContent = failed;
  document.getElementById('valSuccessRate').textContent = `${rate}%`;
  document.getElementById('valDataSourceTag').textContent = source === 'live' ? '⚡ Live Databricks Data' : '🧪 Mock Dataset';
}

function renderRunsTable() {
  const tbody = document.getElementById('runsTableBody');
  if (!tbody) return;

  const filtered = allRuns.filter(r => {
    const matchesSearch = 
      !currentSearch ||
      String(r.run_id).toLowerCase().includes(currentSearch) ||
      String(r.job_name || '').toLowerCase().includes(currentSearch) ||
      String(r.error_message || '').toLowerCase().includes(currentSearch);

    if (!matchesSearch) return false;

    const status = (r.status || '').toUpperCase();
    if (currentFilter === 'FAILED') return status.includes('FAIL') || status.includes('TIME');
    if (currentFilter === 'SUCCESS') return status === 'SUCCESS';
    return true;
  });

  if (filtered.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="7" class="loading-state">
          No pipeline runs matched your filter criteria.
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = filtered.map(r => {
    const isFailed = (r.status || '').toLowerCase().includes('fail') || (r.status || '').toLowerCase().includes('time');
    const isSuccess = (r.status || '').toLowerCase() === 'success';
    const statusClass = isFailed ? 'failed' : (isSuccess ? 'success' : 'running');

    const formattedTime = r.start_time ? new Date(r.start_time).toLocaleString() : 'N/A';
    const errorSnippet = r.error_message ? `<span class="error-snippet" title="${escapeHtml(r.error_message)}">${escapeHtml(r.error_message)}</span>` : '<span style="color: var(--text-muted);">-</span>';

    return `
      <tr>
        <td><span class="run-id-badge">${r.run_id}</span></td>
        <td><strong>${escapeHtml(r.job_name || 'unnamed_job')}</strong></td>
        <td><span class="status-badge ${statusClass}">${r.status}</span></td>
        <td style="color: var(--text-secondary);">${formattedTime}</td>
        <td>${r.duration_formatted || (r.duration_seconds + 's')}</td>
        <td>${errorSnippet}</td>
        <td class="text-right">
          <button class="btn btn-sm btn-secondary" onclick="openIncidentReport('${r.run_id}')">
            ${isFailed ? '⚡ Diagnose' : '📄 Report'}
          </button>
        </td>
      </tr>
    `;
  }).join('');
}

async function openIncidentReport(runId) {
  const modal = document.getElementById('incidentModal');
  const title = document.getElementById('modalRunTitle');
  const content = document.getElementById('modalReportContent');

  if (!modal || !content) return;

  title.textContent = `Investigation: Run ${runId}`;
  content.innerHTML = `
    <div class="spinner-large"></div>
    <p class="modal-loading-text">Synthesizing telemetry facts and diagnostic SOPs for run ${runId}...</p>
  `;
  modal.classList.add('active');

  try {
    const res = await fetch(`/api/incident-report/${runId}?similarity_threshold=0.5`);
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to generate incident report');
    }

    const data = await res.json();
    currentReportMarkdown = data.report_markdown;

    // Render markdown using marked
    const renderedHtml = typeof marked !== 'undefined' ? marked.parse(data.report_markdown) : `<pre>${escapeHtml(data.report_markdown)}</pre>`;

    content.innerHTML = `
      <div class="report-container">
        ${renderedHtml}
      </div>
    `;
  } catch (err) {
    content.innerHTML = `
      <div style="color: #f87171; padding: 20px; text-align: center;">
        <h3>Error Generating Report</h3>
        <p>${escapeHtml(err.message)}</p>
      </div>
    `;
  }
}

async function refreshLiveTelemetry() {
  const btn = document.getElementById('btnRefreshLive');
  const btnText = document.getElementById('btnRefreshText');
  if (btn) btn.disabled = true;
  if (btnText) btnText.textContent = 'Refreshing...';

  try {
    const res = await fetch('/api/runs/refresh', { method: 'POST' });
    if (!res.ok) throw new Error('Live refresh failed');
    const result = await res.json();
    showToast(`Refreshed ${result.count} runs from Databricks API!`);
    await loadData('live');
  } catch (err) {
    showToast(`Refresh error: ${err.message}`, true);
  } finally {
    if (btn) btn.disabled = false;
    if (btnText) btnText.textContent = 'Refresh Live';
  }
}

async function handleChatSubmit(e) {
  if (e) e.preventDefault();
  const input = document.getElementById('chatInput');
  if (!input) return;

  const query = input.value.trim();
  if (!query) return;

  input.value = '';
  appendChatMessage('user', query);

  const botLoadingId = appendChatLoading();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query, similarity_threshold: 0.5 })
    });

    if (!res.ok) throw new Error('Chat API returned an error');
    const data = await res.json();

    removeChatLoading(botLoadingId);
    appendChatMessage('bot', data.reply, data.evidence);
  } catch (err) {
    removeChatLoading(botLoadingId);
    appendChatMessage('bot', `⚠️ Could not get answer: ${err.message}`);
  }
}

function appendChatMessage(sender, text, evidence = null) {
  const messages = document.getElementById('chatMessages');
  if (!messages) return;

  const bubble = document.createElement('div');
  bubble.className = `chat-bubble ${sender}`;

  const avatar = sender === 'user' ? '👤' : '🤖';
  const renderedContent = typeof marked !== 'undefined' ? marked.parse(text) : `<p>${escapeHtml(text)}</p>`;

  let evidenceHtml = '';
  if (evidence && evidence.length > 0) {
    evidenceHtml = `
      <div style="margin-top: 10px; border-top: 1px solid var(--border-subtle); padding-top: 8px;">
        <small style="color: var(--accent-cyan); font-weight: 600;">Retrieved Evidence Sources (${evidence.length}):</small>
        <ul style="padding-left: 16px; margin-top: 4px; font-size: 0.775rem; color: var(--text-muted);">
          ${evidence.map(e => `<li><strong>[${escapeHtml(e.chunk_id)}]</strong>: score ${e.score !== undefined ? e.score.toFixed(2) : 'n/a'}</li>`).join('')}
        </ul>
      </div>
    `;
  }

  bubble.innerHTML = `
    <div class="bubble-avatar">${avatar}</div>
    <div class="bubble-content">
      ${renderedContent}
      ${evidenceHtml}
    </div>
  `;

  messages.appendChild(bubble);
  messages.scrollTop = messages.scrollHeight;
}

function appendChatLoading() {
  const messages = document.getElementById('chatMessages');
  if (!messages) return null;

  const id = `loading-${Date.now()}`;
  const bubble = document.createElement('div');
  bubble.id = id;
  bubble.className = 'chat-bubble bot';
  bubble.innerHTML = `
    <div class="bubble-avatar">🤖</div>
    <div class="bubble-content">
      <div class="spinner"></div>
      <span style="color: var(--text-muted); font-size: 0.85rem;">Searching diagnostic documents & monitoring rules...</span>
    </div>
  `;
  messages.appendChild(bubble);
  messages.scrollTop = messages.scrollHeight;
  return id;
}

function removeChatLoading(id) {
  if (!id) return;
  const el = document.getElementById(id);
  if (el) el.remove();
}

async function loadDocsList() {
  const docsGrid = document.getElementById('docsGrid');
  if (!docsGrid) return;

  // We can fetch root info to get doc counts
  try {
    const res = await fetch('/');
    const info = await res.json();
    document.getElementById('valDocsCount').textContent = info.docs_count || 12;

    const mockDocs = [
      { id: 'DOC-01', title: 'Databricks Cluster Quota Exceeded (OOM / CPU)', category: 'Infrastructure', excerpt: 'Addresses Azure/AWS quota exhaustion, core provisioning limits, and driver memory auto-scaling policies.' },
      { id: 'DOC-02', title: 'Delta Lake Concurrent Append Conflict', category: 'Storage / ACID', excerpt: 'Resolution workflow for ConcurrentAppendException during concurrent MERGE operations on partition keys.' },
      { id: 'DOC-03', title: 'PySpark Column Schema Mismatch Resolution', category: 'Data Quality', excerpt: 'Handling AnalysisException when upstream JSON payloads introduce new fields or incompatible string types.' },
      { id: 'DOC-04', title: 'JDBC Snowflake Connection Timeout SOP', category: 'Networking', excerpt: 'Diagnosing SSL socket timeouts, security group IP whitelisting, and credential rotation in secret scopes.' },
      { id: 'DOC-05', title: 'Databricks Workspace Job Permissions SOP', category: 'Security / IAM', excerpt: 'IAM role policy configuration, token expiration, and Service Principal access validation.' },
      { id: 'DOC-06', title: 'Streaming Checkpoint Recovery SOP', category: 'Structured Streaming', excerpt: 'Handling corrupted RocksDB state stores and recreating valid streaming checkpoint offsets.' }
    ];

    docsGrid.innerHTML = mockDocs.map(d => `
      <div class="doc-card" onclick="queryDocTopic('${escapeHtml(d.title)}')">
        <div class="doc-title">${escapeHtml(d.title)}</div>
        <div class="doc-meta">${d.id} • ${d.category}</div>
        <div class="doc-preview">${escapeHtml(d.excerpt)}</div>
      </div>
    `).join('');
  } catch (err) {
    console.error(err);
  }
}

function queryDocTopic(topic) {
  const input = document.getElementById('chatInput');
  if (input) {
    input.value = `Explain ${topic}`;
    document.getElementById('tabBtnChat')?.click();
    handleChatSubmit(new Event('submit'));
  }
}

function showToast(msg, isError = false) {
  const toast = document.getElementById('toastNotification');
  if (!toast) return;

  toast.textContent = msg;
  toast.style.borderColor = isError ? 'rgba(239, 68, 68, 0.5)' : 'rgba(99, 102, 241, 0.5)';
  toast.classList.add('show');

  setTimeout(() => {
    toast.classList.remove('show');
  }, 3500);
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
