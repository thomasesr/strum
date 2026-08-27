'use strict';

const $ = (sel) => document.querySelector(sel);
const drop = $('#drop'), fileInput = $('#file'), picked = $('#picked');
const submit = $('#submit'), errorBox = $('#error'), jobList = $('#jobs');
const metaPanel = $('#meta-panel'), art = $('#art'), artEmpty = $('#art-empty');
const artPick = $('#art-pick'), artFile = $('#art-file');
const META_FIELDS = ['title', 'artist', 'album', 'year', 'genre', 'charter'];

// Set once the staged upload exists server-side. The file itself is uploaded
// exactly once; submitting only sends the id plus the edited metadata.
let uploadId = null;
let config = { allowed_extensions: [], max_upload_mb: 300 };
const jobs = new Map();

fetch('/api/config').then((r) => r.json()).then((cfg) => {
  config = cfg;
  $('#accepted').textContent =
    cfg.allowed_extensions.map((e) => e.replace('.', '')).join(', ') +
    ` · up to ${cfg.max_upload_mb} MB`;
});

// --- file choosing -------------------------------------------------------

async function choose(file) {
  if (!file) return;
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (config.allowed_extensions.length && !config.allowed_extensions.includes(ext)) {
    return fail(`${ext} files are not supported.`);
  }
  if (file.size > config.max_upload_mb * 1024 * 1024) {
    return fail(`That file is ${(file.size / 1048576).toFixed(0)} MB, over the ${config.max_upload_mb} MB limit.`);
  }

  reset();
  picked.textContent = `${file.name} · ${(file.size / 1048576).toFixed(1)} MB · reading tags…`;
  picked.hidden = false;
  errorBox.hidden = true;

  const body = new FormData();
  body.append('file', file);
  try {
    const res = await fetch('/api/uploads', { method: 'POST', body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Upload failed');

    uploadId = data.upload_id;
    for (const name of META_FIELDS) {
      $(`[name="${name}"]`).value = data.metadata[name] || '';
    }
    // Cache-bust: the art for a new upload lives at a different id, but a
    // replaced cover reuses the same URL.
    setArt(data.has_cover ? `/api/uploads/${uploadId}/cover?t=${Date.now()}` : null);
    metaPanel.hidden = false;
    const mins = Math.floor(data.duration_s / 60), secs = Math.round(data.duration_s % 60);
    picked.textContent =
      `${data.filename} · ${mins}:${String(secs).padStart(2, '0')}` +
      (data.has_video ? ' · audio will be extracted from video' : '');
    submit.disabled = false;
  } catch (e) {
    picked.hidden = true;
    fail(e.message);
  }
}

function setArt(url) {
  art.hidden = !url;
  artEmpty.hidden = !!url;
  if (url) art.src = url;
}

function reset() {
  uploadId = null;
  submit.disabled = true;
  metaPanel.hidden = true;
  setArt(null);
}

function fail(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

drop.addEventListener('click', () => fileInput.click());
drop.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener('change', () => choose(fileInput.files[0]));
['dragenter', 'dragover'].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => choose(e.dataTransfer.files[0]));

// --- submitting ----------------------------------------------------------

submit.addEventListener('click', async () => {
  if (!uploadId) return;
  const title = $('[name="title"]').value.trim();
  const artist = $('[name="artist"]').value.trim();
  if (!title || !artist) return fail('Title and artist are required.');
  const year = $('[name="year"]').value.trim();
  if (year && !/^\d{4}$/.test(year)) return fail('Year should be four digits, or blank.');

  const body = new FormData();
  body.append('upload_id', uploadId);
  for (const name of META_FIELDS) body.append(name, $(`[name="${name}"]`).value.trim());
  for (const name of ['drums', 'guitar', 'bass', 'vocals', 'keys', 'stems',
                      'karaoke', 'backing_split']) {
    body.append(name, String($(`[name="${name}"]`).checked));
  }
  body.append('separator', $('[name="separator"]').value);
  const formats = ['zip', 'sng'].filter((f) => $(`[name="${f}"]`).checked);
  if (!formats.length) return fail('Pick at least one package format.');
  body.append('formats', formats.join(','));

  submit.disabled = true;
  submit.textContent = 'Queueing…';
  try {
    const res = await fetch('/api/jobs', { method: 'POST', body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Could not queue the job');
    render(data);
    // The upload is consumed server-side, so the form goes back to empty.
    reset();
    fileInput.value = '';
    picked.hidden = true;
    errorBox.hidden = true;
  } catch (e) {
    fail(e.message);
    submit.disabled = false;
  } finally {
    submit.textContent = 'Chart it';
  }
});

artPick.addEventListener('click', () => artFile.click());
artFile.addEventListener('change', async () => {
  const file = artFile.files[0];
  if (!file || !uploadId) return;
  const body = new FormData();
  body.append('file', file);
  try {
    const res = await fetch(`/api/uploads/${uploadId}/cover`, { method: 'POST', body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Could not read that image');
    setArt(`/api/uploads/${uploadId}/cover?t=${Date.now()}`);
    errorBox.hidden = true;
  } catch (e) {
    fail(e.message);
  } finally {
    artFile.value = '';
  }
});

// --- job list ------------------------------------------------------------

const STATUS_TEXT = {
  queued: 'Queued', running: 'Running', done: 'Done',
  failed: 'Failed', cancelled: 'Cancelled',
};

function render(job) {
  jobs.set(job.id, job);
  $('#empty').hidden = jobs.size > 0;

  let li = document.getElementById(`job-${job.id}`);
  if (!li) {
    li = document.createElement('li');
    li.id = `job-${job.id}`;
    li.className = 'job';
    jobList.prepend(li);
  }
  li.dataset.status = job.status;

  const pct = Math.round(job.progress * 100);
  const done = job.status === 'done';
  const active = job.status === 'queued' || job.status === 'running';

  li.innerHTML = '';
  const head = document.createElement('div');
  head.className = 'job-head';
  const name = document.createElement('span');
  name.className = 'job-name';
  // textContent throughout: filenames and tag values are user-supplied.
  name.textContent = (job.artist && job.title)
    ? `${job.artist} - ${job.title}`
    : job.filename;
  const state = document.createElement('span');
  state.className = 'badge';
  state.textContent = STATUS_TEXT[job.status] || job.status;
  head.append(name, state);

  const detail = document.createElement('p');
  detail.className = 'job-detail';
  detail.textContent = job.error
    ? job.error
    : done ? `Charted in ${Math.round(job.duration_s)}s` : job.stage;

  const bar = document.createElement('div');
  bar.className = 'bar';
  const fill = document.createElement('i');
  fill.style.width = `${done ? 100 : pct}%`;
  bar.append(fill);

  li.append(head, detail);
  if (active || done) li.append(bar);

  const actions = document.createElement('div');
  actions.className = 'job-actions';
  for (const fmt of job.packages || []) {
    const a = document.createElement('a');
    a.className = 'download';
    a.href = `/api/jobs/${job.id}/download/${fmt}`;
    a.textContent = fmt === 'sng' ? 'Download .sng' : 'Download .zip';
    actions.append(a);
  }
  if (active) {
    const cancel = document.createElement('button');
    cancel.className = 'ghost';
    cancel.textContent = 'Cancel';
    cancel.onclick = () => fetch(`/api/jobs/${job.id}/cancel`, { method: 'POST' });
    actions.append(cancel);
  }
  if (actions.childElementCount) li.append(actions);
}

// Downloads start on their own the moment a job finishes, which is the point:
// you upload, walk away, and the packages are waiting in the browser.
const autoDownloaded = new Set();
function autoDownload(job) {
  if (job.status !== 'done' || autoDownloaded.has(job.id)) return;
  autoDownloaded.add(job.id);
  (job.packages || []).forEach((fmt, i) => {
    setTimeout(() => {
      const a = document.createElement('a');
      a.href = `/api/jobs/${job.id}/download/${fmt}`;
      a.download = '';
      document.body.append(a);
      a.click();
      a.remove();
    }, i * 800);   // staggered; browsers drop simultaneous downloads
  });
}

const source = new EventSource('/api/events');
source.onmessage = (e) => {
  const job = JSON.parse(e.data);
  const known = jobs.has(job.id);
  render(job);
  // Only auto-download jobs this tab watched finish, not history on reload.
  if (known) autoDownload(job);
};
