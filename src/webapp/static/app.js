'use strict';

const $ = (sel) => document.querySelector(sel);
const drop = $('#drop'), fileInput = $('#file'), picked = $('#picked');
const submit = $('#submit'), errorBox = $('#error'), jobList = $('#jobs');

let chosen = null;
let config = { allowed_extensions: [], max_upload_mb: 300 };
const jobs = new Map();

fetch('/api/config').then((r) => r.json()).then((cfg) => {
  config = cfg;
  $('#accepted').textContent =
    cfg.allowed_extensions.map((e) => e.replace('.', '')).join(', ') +
    ` · up to ${cfg.max_upload_mb} MB`;
});

// --- file choosing -------------------------------------------------------

function choose(file) {
  if (!file) return;
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (config.allowed_extensions.length && !config.allowed_extensions.includes(ext)) {
    return fail(`${ext} files are not supported.`);
  }
  if (file.size > config.max_upload_mb * 1024 * 1024) {
    return fail(`That file is ${(file.size / 1048576).toFixed(0)} MB, over the ${config.max_upload_mb} MB limit.`);
  }
  chosen = file;
  picked.textContent = `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`;
  picked.hidden = false;
  submit.disabled = false;
  errorBox.hidden = true;
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
  if (!chosen) return;
  const body = new FormData();
  body.append('file', chosen);
  for (const name of ['drums', 'guitar', 'bass', 'vocals', 'keys', 'stems',
                      'karaoke', 'backing_split']) {
    body.append(name, String($(`[name="${name}"]`).checked));
  }
  body.append('separator', $('[name="separator"]').value);
  const formats = ['zip', 'sng'].filter((f) => $(`[name="${f}"]`).checked);
  if (!formats.length) return fail('Pick at least one package format.');
  body.append('formats', formats.join(','));

  submit.disabled = true;
  submit.textContent = 'Uploading…';
  try {
    const res = await fetch('/api/jobs', { method: 'POST', body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Upload failed');
    render(data);
    chosen = null;
    fileInput.value = '';
    picked.hidden = true;
    errorBox.hidden = true;
  } catch (e) {
    fail(e.message);
  } finally {
    submit.textContent = 'Chart it';
    submit.disabled = chosen === null;
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
  name.textContent = job.filename;            // textContent: never trust a filename as HTML
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
