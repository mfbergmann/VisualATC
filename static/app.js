/* VisualATC – Frontend Application */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let ws = null;
let sessionActive = false;
let isPaused = false;
let inputMode = 'stream';
let transcriptSegments = [];
let flightCards = {};
let events = [];
let transcriptFilter = '';
let eventTypeFilter = '';
let flightCardSort = 'recent';
let startTime = 0;
let audioTimer = null;

const TRACKER_URL_TEMPLATE = 'https://flightaware.com/live/flight/{CALLSIGN}';

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  connectWebSocket();
  loadBookmarks();
  updateStartButton();

  document.getElementById('permission-check').addEventListener('change', updateStartButton);
  document.getElementById('stream-url').addEventListener('input', updateStartButton);
  document.getElementById('file-upload').addEventListener('change', updateStartButton);
});

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------
function connectWebSocket() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${location.host}/ws`);

  ws.onopen = () => {
    console.log('WebSocket connected');
  };

  ws.onmessage = (evt) => {
    const msg = JSON.parse(evt.data);
    handleWSMessage(msg);
  };

  ws.onclose = () => {
    console.log('WebSocket disconnected, reconnecting in 2s...');
    setTimeout(connectWebSocket, 2000);
  };

  ws.onerror = (err) => {
    console.error('WebSocket error:', err);
  };
}

function handleWSMessage(msg) {
  switch (msg.type) {
    case 'transcript':
      addTranscriptSegment(msg.segment, msg.callsigns, msg.fields, msg.events);
      break;
    case 'state_update':
      if (msg.flight_cards) updateFlightCards(msg.flight_cards);
      if (msg.events) updateEvents(msg.events);
      if (msg.stats) updateStats(msg.stats);
      if (msg.transcript) {
        // Initial state on reconnect
        msg.transcript.forEach(seg => {
          addTranscriptSegment(seg, [], {}, []);
        });
      }
      break;
    case 'status':
      showStatus(msg.message);
      break;
    case 'started':
      onSessionStarted(msg.session_id);
      break;
    case 'stopped':
      onSessionStopped(msg);
      break;
    case 'paused':
      isPaused = true;
      updatePauseButton();
      setStatusBadge('paused');
      break;
    case 'resumed':
      isPaused = false;
      updatePauseButton();
      setStatusBadge('running');
      break;
    case 'error':
      showError(msg.message);
      break;
  }
}

// ---------------------------------------------------------------------------
// Session control
// ---------------------------------------------------------------------------
function setInputMode(mode) {
  inputMode = mode;
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });
  document.getElementById('stream-input').style.display = mode === 'stream' ? 'block' : 'none';
  document.getElementById('file-input').style.display = mode === 'file' ? 'block' : 'none';
  updateStartButton();
}

function updateStartButton() {
  const check = document.getElementById('permission-check').checked;
  let hasInput = false;

  if (inputMode === 'stream') {
    hasInput = document.getElementById('stream-url').value.trim().length > 0;
  } else {
    hasInput = document.getElementById('file-upload').files.length > 0;
  }

  document.getElementById('start-btn').disabled = !(check && hasInput);
}

async function startSession() {
  const modelSize = document.getElementById('model-select').value;

  if (inputMode === 'file') {
    const fileInput = document.getElementById('file-upload');
    const file = fileInput.files[0];
    if (!file) return;

    showOverlay('Uploading file and loading model...');
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_size', modelSize);

    try {
      const resp = await fetch('/api/upload', { method: 'POST', body: formData });
      const data = await resp.json();
      if (data.status === 'started') {
        // Will receive 'started' via WebSocket
      } else {
        showError('Failed to start: ' + JSON.stringify(data));
        hideOverlay();
      }
    } catch (err) {
      showError('Upload failed: ' + err.message);
      hideOverlay();
    }
  } else {
    const url = document.getElementById('stream-url').value.trim();
    if (!url) return;

    showOverlay('Connecting to stream and loading model...');
    try {
      const resp = await fetch('/api/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, mode: 'stream', model_size: modelSize }),
      });
      const data = await resp.json();
      if (data.status !== 'started') {
        showError('Failed to start: ' + JSON.stringify(data));
        hideOverlay();
      }
    } catch (err) {
      showError('Start failed: ' + err.message);
      hideOverlay();
    }
  }
}

function onSessionStarted(sessionId) {
  sessionActive = true;
  isPaused = false;
  transcriptSegments = [];
  flightCards = {};
  events = [];
  startTime = Date.now();

  hideOverlay();
  document.getElementById('setup-panel').style.display = 'none';
  document.getElementById('live-view').style.display = 'flex';
  setStatusBadge('running');
  startAudioTimer();
  renderTranscript();
  renderFlightCards();
  renderEvents();
}

function onSessionStopped(msg) {
  sessionActive = false;
  isPaused = false;
  stopAudioTimer();
  setStatusBadge('stopped');

  if (msg.session_dir) {
    console.log('Session saved to:', msg.session_dir);
  }
}

async function togglePause() {
  try {
    const resp = await fetch('/api/pause', { method: 'POST' });
    const data = await resp.json();
    // State update comes via WebSocket
  } catch (err) {
    showError('Pause failed: ' + err.message);
  }
}

async function stopSession() {
  try {
    showOverlay('Stopping and exporting...');
    const resp = await fetch('/api/stop', { method: 'POST' });
    const data = await resp.json();
    hideOverlay();

    // Show setup panel again
    setTimeout(() => {
      document.getElementById('setup-panel').style.display = 'flex';
      document.getElementById('live-view').style.display = 'none';
      setStatusBadge('idle');
    }, 500);
  } catch (err) {
    hideOverlay();
    showError('Stop failed: ' + err.message);
  }
}

function exportTxt() {
  window.open('/api/export/txt', '_blank');
}

function exportJson() {
  window.open('/api/export/json', '_blank');
}

// ---------------------------------------------------------------------------
// Transcript
// ---------------------------------------------------------------------------
function addTranscriptSegment(segment, callsigns, fields, segEvents) {
  // Avoid duplicates
  const isDup = transcriptSegments.some(s =>
    s.ts === segment.ts && s.text === segment.text
  );
  if (isDup) return;

  transcriptSegments.push({ ...segment, callsigns, fields, events: segEvents });
  updateStats();
  appendTranscriptLine(segment, callsigns, segEvents);
}

function appendTranscriptLine(segment, callsigns, segEvents) {
  const feed = document.getElementById('transcript-feed');
  const empty = feed.querySelector('.empty-state');
  if (empty) empty.remove();

  // Apply filter
  if (transcriptFilter) {
    const lower = transcriptFilter.toLowerCase();
    const textMatch = segment.text.toLowerCase().includes(lower);
    const csMatch = (callsigns || []).some(cs =>
      cs.canonical.toLowerCase().includes(lower) ||
      cs.alias.toLowerCase().includes(lower)
    );
    if (!textMatch && !csMatch) return;
  }

  const line = document.createElement('div');
  line.className = 'transcript-line';

  const elapsed = segment.ts - (startTime / 1000);
  const tsStr = formatTime(elapsed > 0 ? elapsed : transcriptSegments.length * 7);

  let textHtml = escapeHtml(segment.text);

  // Highlight callsigns in text
  if (callsigns && callsigns.length > 0) {
    callsigns.forEach(cs => {
      const escapedAlias = escapeHtml(cs.alias);
      const regex = new RegExp(escapeRegex(escapedAlias), 'gi');
      textHtml = textHtml.replace(regex,
        `<span class="callsign-tag" onclick="filterByCallsign('${escapeAttr(cs.canonical)}')" title="${escapeAttr(cs.canonical)}">${escapedAlias}</span>`
      );
    });
  }

  // Add event tags
  if (segEvents && segEvents.length > 0) {
    segEvents.forEach(evt => {
      textHtml += ` <span class="event-tag">${evt.type}</span>`;
    });
  }

  line.innerHTML = `
    <span class="ts">${tsStr}</span>
    <span class="text">${textHtml}</span>
  `;

  feed.appendChild(line);

  // Auto-scroll if near bottom
  if (feed.scrollHeight - feed.scrollTop - feed.clientHeight < 100) {
    feed.scrollTop = feed.scrollHeight;
  }
}

function renderTranscript() {
  const feed = document.getElementById('transcript-feed');
  feed.innerHTML = '';

  if (transcriptSegments.length === 0) {
    feed.innerHTML = '<div class="empty-state">Waiting for audio...</div>';
    return;
  }

  transcriptSegments.forEach(seg => {
    appendTranscriptLine(seg, seg.callsigns, seg.events);
  });
}

function filterTranscript() {
  transcriptFilter = document.getElementById('transcript-filter').value;
  renderTranscript();
}

function clearTranscriptFilter() {
  document.getElementById('transcript-filter').value = '';
  transcriptFilter = '';
  renderTranscript();
}

function filterByCallsign(cs) {
  document.getElementById('transcript-filter').value = cs;
  transcriptFilter = cs;
  renderTranscript();

  // Highlight the flight card
  document.querySelectorAll('.flight-card').forEach(card => {
    card.classList.toggle('highlight', card.dataset.callsign === cs);
  });
}

// ---------------------------------------------------------------------------
// Flight Cards
// ---------------------------------------------------------------------------
function updateFlightCards(cards) {
  flightCards = cards;
  renderFlightCards();
}

function sortFlightCards(mode) {
  flightCardSort = mode;
  document.querySelectorAll('.sort-bar .btn-tiny').forEach(btn => {
    btn.classList.remove('active');
  });
  event.target.classList.add('active');
  renderFlightCards();
}

function renderFlightCards() {
  const container = document.getElementById('flight-cards-list');
  document.getElementById('flight-count').textContent = Object.keys(flightCards).length;

  if (Object.keys(flightCards).length === 0) {
    container.innerHTML = '<div class="empty-state">No flights detected yet</div>';
    return;
  }

  // Sort
  let entries = Object.entries(flightCards);
  if (flightCardSort === 'recent') {
    entries.sort((a, b) => b[1].last_seen - a[1].last_seen);
  } else if (flightCardSort === 'frequent') {
    entries.sort((a, b) => b[1].mention_count - a[1].mention_count);
  } else {
    entries.sort((a, b) => a[0].localeCompare(b[0]));
  }

  container.innerHTML = '';
  entries.forEach(([cs, card]) => {
    container.appendChild(createFlightCardElement(cs, card));
  });
}

function createFlightCardElement(cs, card) {
  const el = document.createElement('div');
  el.className = 'flight-card';
  el.dataset.callsign = cs;

  if (transcriptFilter && transcriptFilter.toLowerCase() === cs.toLowerCase()) {
    el.classList.add('highlight');
  }

  // Fields
  let fieldsHtml = '';
  const f = card.latest_fields || {};
  if (f.runway) fieldsHtml += `<span class="field-badge field-runway">RWY ${f.runway}</span>`;
  if (f.altitude) fieldsHtml += `<span class="field-badge field-altitude">ALT ${f.altitude}</span>`;
  if (f.heading) fieldsHtml += `<span class="field-badge field-heading">HDG ${f.heading}</span>`;
  if (f.speed) fieldsHtml += `<span class="field-badge field-speed">SPD ${f.speed}</span>`;
  if (f.frequency) fieldsHtml += `<span class="field-badge field-frequency">FREQ ${f.frequency}</span>`;

  // Events
  let eventsHtml = '';
  if (card.events && card.events.length > 0) {
    const uniqueTypes = [...new Set(card.events.map(e => e.type))];
    uniqueTypes.forEach(t => {
      eventsHtml += `<span class="event-badge event-badge-${t}">${t.replace(/_/g, ' ')}</span>`;
    });
  }

  // Sparkline
  let sparklineHtml = '';
  if (card.sparkline && card.sparkline.length > 1) {
    sparklineHtml = createSparklineSVG(card.sparkline);
  }

  // Aliases
  const aliasesText = (card.aliases || []).filter(a => a !== cs).join(', ');

  // Last seen
  const lastSeenElapsed = card.last_seen - (startTime / 1000);
  const lastSeenStr = lastSeenElapsed > 0 ? formatTime(lastSeenElapsed) : 'just now';

  // Recent mentions
  let mentionsHtml = '';
  if (card.mentions && card.mentions.length > 0) {
    card.mentions.slice(-5).forEach(m => {
      const mElapsed = m.ts - (startTime / 1000);
      mentionsHtml += `<div class="mention-line"><span class="ts">${formatTime(mElapsed > 0 ? mElapsed : 0)}</span> ${escapeHtml(m.raw_text)}</div>`;
    });
  }

  // FlightAware URL
  const trackerUrl = TRACKER_URL_TEMPLATE.replace('{CALLSIGN}', encodeURIComponent(cs));

  el.innerHTML = `
    <div class="flight-card-header">
      <span class="flight-card-callsign" onclick="filterByCallsign('${escapeAttr(cs)}')">${escapeHtml(cs)}</span>
      <div class="flight-card-meta">
        <span class="flight-card-count">${card.mention_count || 0}x</span>
        <span class="flight-card-time">${lastSeenStr}</span>
      </div>
    </div>
    ${aliasesText ? `<div class="flight-card-aliases">${escapeHtml(aliasesText)}</div>` : ''}
    ${fieldsHtml ? `<div class="flight-card-fields">${fieldsHtml}</div>` : ''}
    ${eventsHtml ? `<div class="flight-card-events">${eventsHtml}</div>` : ''}
    ${sparklineHtml ? `<div class="flight-card-sparkline">${sparklineHtml}</div>` : ''}
    <div class="flight-card-actions">
      <a href="${trackerUrl}" target="_blank" rel="noopener" class="btn btn-tiny btn-outline">FlightAware</a>
      <button class="btn btn-tiny btn-outline" onclick="copyCallsign('${escapeAttr(cs)}')">Copy</button>
      <button class="btn btn-tiny btn-outline" onclick="filterByCallsign('${escapeAttr(cs)}')">Filter</button>
      <button class="btn btn-tiny btn-outline" onclick="toggleCardExpand(this)">Mentions</button>
    </div>
    <div class="flight-card-mentions">${mentionsHtml}</div>
  `;

  return el;
}

function createSparklineSVG(data) {
  if (!data || data.length < 2) return '';
  const width = 280;
  const height = 18;
  const max = Math.max(...data, 1);
  const step = width / (data.length - 1);

  let path = '';
  data.forEach((val, i) => {
    const x = i * step;
    const y = height - (val / max) * (height - 2) - 1;
    path += (i === 0 ? `M ${x},${y}` : ` L ${x},${y}`);
  });

  // Fill area
  const fillPath = path + ` L ${(data.length - 1) * step},${height} L 0,${height} Z`;

  return `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
    <path d="${fillPath}" fill="rgba(88, 166, 255, 0.1)" />
    <path d="${path}" fill="none" stroke="var(--accent)" stroke-width="1.5" />
  </svg>`;
}

function toggleCardExpand(btn) {
  const card = btn.closest('.flight-card');
  card.classList.toggle('expanded');
}

function copyCallsign(cs) {
  navigator.clipboard.writeText(cs).then(() => {
    // Brief visual feedback
  }).catch(() => {
    // Fallback
    const ta = document.createElement('textarea');
    ta.value = cs;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
  });
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------
function updateEvents(newEvents) {
  events = newEvents;
  renderEvents();
}

function filterEvents() {
  eventTypeFilter = document.getElementById('event-type-filter').value;
  renderEvents();
}

function renderEvents() {
  const container = document.getElementById('events-list');
  document.getElementById('event-count').textContent = events.length;

  let filtered = events;
  if (eventTypeFilter) {
    filtered = events.filter(e => e.type === eventTypeFilter);
  }

  if (filtered.length === 0) {
    container.innerHTML = '<div class="empty-state">No events detected yet</div>';
    return;
  }

  container.innerHTML = '';
  // Newest first
  [...filtered].reverse().forEach(evt => {
    const el = document.createElement('div');
    el.className = 'event-item';

    const elapsed = evt.ts - (startTime / 1000);
    const tsStr = formatTime(elapsed > 0 ? elapsed : 0);

    let csHtml = '';
    if (evt.callsign_canonical) {
      csHtml = `<span class="event-callsign" onclick="filterByCallsign('${escapeAttr(evt.callsign_canonical)}')">${escapeHtml(evt.callsign_canonical)}</span> `;
    }

    el.innerHTML = `
      <span class="ts">${tsStr}</span>
      <span class="event-badge event-badge-${evt.type}">${evt.type.replace(/_/g, ' ')}</span>
      <div class="event-info">${csHtml}${escapeHtml(evt.details || '')}</div>
    `;

    container.appendChild(el);
  });
}

// ---------------------------------------------------------------------------
// Stats & UI
// ---------------------------------------------------------------------------
function updateStats(stats) {
  document.getElementById('stat-segments').textContent = transcriptSegments.length;
  document.getElementById('stat-flights').textContent = Object.keys(flightCards).length;
  document.getElementById('stat-events').textContent = events.length;

  if (stats && stats.total_audio_seconds) {
    document.getElementById('stat-audio').textContent = formatTime(stats.total_audio_seconds);
  }
}

function setStatusBadge(status) {
  const badge = document.getElementById('status-badge');
  badge.className = `badge badge-${status}`;
  badge.textContent = status.charAt(0).toUpperCase() + status.slice(1);
}

function updatePauseButton() {
  const btn = document.getElementById('pause-btn');
  btn.textContent = isPaused ? 'Resume' : 'Pause';
}

function startAudioTimer() {
  stopAudioTimer();
  audioTimer = setInterval(() => {
    const elapsed = (Date.now() - startTime) / 1000;
    document.getElementById('audio-time').textContent = formatTime(elapsed);
  }, 1000);
}

function stopAudioTimer() {
  if (audioTimer) {
    clearInterval(audioTimer);
    audioTimer = null;
  }
}

function showOverlay(message) {
  document.getElementById('status-message').textContent = message;
  document.getElementById('status-overlay').style.display = 'flex';
}

function hideOverlay() {
  document.getElementById('status-overlay').style.display = 'none';
}

function showStatus(message) {
  document.getElementById('status-message').textContent = message;
  // Also update overlay if visible
  if (document.getElementById('status-overlay').style.display === 'flex') {
    document.getElementById('status-message').textContent = message;
  }
}

function showError(message) {
  hideOverlay();
  console.error('Error:', message);
  // Simple alert for now
  alert('Error: ' + message);
}

// ---------------------------------------------------------------------------
// Bookmarks
// ---------------------------------------------------------------------------
async function loadBookmarks() {
  try {
    const resp = await fetch('/api/bookmarks');
    const data = await resp.json();
    renderBookmarks(data.bookmarks || []);
  } catch (err) {
    console.error('Failed to load bookmarks:', err);
  }
}

function renderBookmarks(bookmarks) {
  const container = document.getElementById('bookmarks-list');
  container.innerHTML = '';

  bookmarks.forEach(b => {
    const el = document.createElement('div');
    el.className = 'bookmark-item';
    el.innerHTML = `
      <div>
        <span class="label" onclick="useBookmark('${escapeAttr(b.url)}')">${escapeHtml(b.label)}</span>
        <span class="url">${escapeHtml(truncate(b.url, 40))}</span>
      </div>
      <button class="btn btn-tiny btn-outline" onclick="deleteBookmark('${escapeAttr(b.url)}')">Remove</button>
    `;
    container.appendChild(el);
  });
}

function useBookmark(url) {
  setInputMode('stream');
  document.getElementById('stream-url').value = url;
  updateStartButton();
}

function toggleBookmarkForm() {
  const form = document.getElementById('bookmark-add-form');
  form.style.display = form.style.display === 'none' ? 'flex' : 'none';
}

async function saveBookmark() {
  const label = document.getElementById('bookmark-label').value.trim();
  const url = document.getElementById('stream-url').value.trim();
  if (!label || !url) {
    alert('Enter a label and a stream URL first.');
    return;
  }

  try {
    const resp = await fetch(`/api/bookmarks?label=${encodeURIComponent(label)}&url=${encodeURIComponent(url)}`, {
      method: 'POST',
    });
    const data = await resp.json();
    renderBookmarks(data.bookmarks || []);
    document.getElementById('bookmark-label').value = '';
    document.getElementById('bookmark-add-form').style.display = 'none';
  } catch (err) {
    console.error('Failed to save bookmark:', err);
  }
}

async function deleteBookmark(url) {
  try {
    const resp = await fetch(`/api/bookmarks?url=${encodeURIComponent(url)}`, {
      method: 'DELETE',
    });
    const data = await resp.json();
    renderBookmarks(data.bookmarks || []);
  } catch (err) {
    console.error('Failed to delete bookmark:', err);
  }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function formatTime(seconds) {
  if (isNaN(seconds) || seconds < 0) seconds = 0;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) {
    return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  }
  return `${m}:${String(s).padStart(2, '0')}`;
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function escapeAttr(str) {
  if (!str) return '';
  return str.replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

function escapeRegex(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function truncate(str, maxLen) {
  if (!str || str.length <= maxLen) return str;
  return str.substring(0, maxLen) + '...';
}
