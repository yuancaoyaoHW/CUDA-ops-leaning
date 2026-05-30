const STATUS_LABELS = {
  not_started: "Not Started",
  in_progress: "In Progress",
  done: "Done",
  blocked: "Blocked",
  skipped: "Skipped",
  correctness_stage: "Correctness",
  benchmark_stage: "Benchmark",
  profile_stage: "Profile",
  complete: "Complete",
};

const numberFormat = new Intl.NumberFormat("en-US");
const percentFormat = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 0,
});

let DATA = null;
let EDIT = null;
let DIRTY = false;
let previousFocus = null;
let restoreFocus = true;

const STATE = {
  week: "all",
  status: "all",
  tag: "all",
  query: "",
};

function truthy(value) {
  return value === true || value === "true" || value === "complete";
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[ch]));
}

function allDays() {
  return DATA.weeks.flatMap((week) => week.days);
}

function readInitialData() {
  const node = document.getElementById("initial-data");
  if (!node || !node.textContent.trim()) return null;
  return JSON.parse(node.textContent);
}

async function loadData() {
  const initial = readInitialData();
  try {
    const response = await fetch("/api/progress");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    DATA = await response.json();
  } catch (error) {
    if (!initial) throw error;
    DATA = initial;
  }
}

function updateQueryString() {
  const params = new URLSearchParams();
  if (STATE.week !== "all") params.set("week", STATE.week);
  if (STATE.status !== "all") params.set("status", STATE.status);
  if (STATE.tag !== "all") params.set("tag", STATE.tag);
  if (STATE.query) params.set("q", STATE.query);
  const query = params.toString();
  const nextUrl = `${window.location.pathname}${query ? `?${query}` : ""}`;
  window.history.replaceState(null, "", nextUrl);
}

function readQueryString() {
  const params = new URLSearchParams(window.location.search);
  STATE.week = params.get("week") || "all";
  STATE.status = params.get("status") || "all";
  STATE.tag = params.get("tag") || "all";
  STATE.query = params.get("q") || "";
}

function statusLabel(status) {
  return STATUS_LABELS[status] || status || "Unknown";
}

function statusBadge(status) {
  const safeStatus = escapeHtml(status || "not_started");
  return `<span class="status-badge ${safeStatus}">${escapeHtml(statusLabel(status))}</span>`;
}

function progressBar(done, total, label) {
  const pct = total ? (done / total) * 100 : 0;
  return `<span class="bar-wrap" aria-label="${escapeHtml(label)}: ${percentFormat.format(pct)}%">
    <span class="bar" aria-hidden="true"><span style="width:${pct.toFixed(1)}%"></span></span>
    <strong class="number">${percentFormat.format(pct)}%</strong>
    <small class="number">${numberFormat.format(done)}/${numberFormat.format(total)}</small>
  </span>`;
}

function miniProgress(done, total, label) {
  const pct = total ? (done / total) * 100 : 0;
  return `<span class="mini-progress" aria-label="${escapeHtml(label)}: ${percentFormat.format(pct)}%">
    <span><span style="width:${pct.toFixed(1)}%"></span></span>
    <strong class="number">${percentFormat.format(pct)}%</strong>
  </span>`;
}

function tagsHtml(tags) {
  if (!tags || !tags.length) return "";
  return `<span class="tag-list">${tags.map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("")}</span>`;
}

function statusLine(status) {
  const safeStatus = escapeHtml(status || "not_started");
  return `<span class="state-line ${safeStatus}"><span aria-hidden="true"></span>${escapeHtml(statusLabel(status))}</span>`;
}

function currentWeekNumber() {
  return DATA.current_day ? DATA.current_day.week : 8;
}

function filteredWeeks() {
  const currentWeek = currentWeekNumber();
  const query = STATE.query.trim().toLowerCase();
  return DATA.weeks
    .map((week) => {
      const days = week.days.filter((day) => {
        if (STATE.week !== "all") {
          const targetWeek = STATE.week === "current" ? currentWeek : Number(STATE.week);
          if (day.week !== targetWeek) return false;
        }
        if (STATE.status !== "all" && day.status !== STATE.status) return false;
        if (STATE.tag !== "all" && !(day.jd_tags || []).includes(STATE.tag)) return false;
        if (query) {
          const text = [
            day.title,
            day.status,
            day.verification,
            day.weaknesses,
            day.next_fix,
            day.notes,
            ...(day.jd_tags || []),
          ].join(" ").toLowerCase();
          if (!text.includes(query)) return false;
        }
        return true;
      });
      return {...week, days};
    })
    .filter((week) => week.days.length > 0);
}

function render() {
  const app = document.getElementById("app");
  app.innerHTML = `<section class="dashboard-shell">
    <section class="panel focus-panel" aria-label="Current progress">
      ${renderNextStep()}
      ${renderStats()}
    </section>
    <section class="workbench">
      <div class="plan-column">
        ${renderPlanPanel()}
      </div>
      <aside class="insight-column">
        ${renderRisks()}
        ${renderOperators()}
        ${renderLibraries()}
        ${renderTagCoverage()}
    </aside>
    </section>
  </section>`;
  syncControls();
}

function renderNextStep() {
  const day = DATA.current_day;
  if (!day) {
    return `<div class="next-step">
      <div class="next-title"><h2>Next Step</h2>${statusBadge("done")}</div>
      <p class="muted">All planned days are marked done.</p>
    </div>`;
  }

  const taskItems = Object.entries(day.tasks || {}).slice(0, 6).map(([key, value]) => `
    <li><span class="task-mark ${truthy(value) ? "done" : ""}" aria-hidden="true">✓</span><span>${escapeHtml(key)}</span></li>
  `).join("");
  const artifactItems = Object.entries(day.artifacts || {}).slice(0, 6).map(([key, value]) => `
    <li><span class="task-mark ${truthy(value) ? "done" : ""}" aria-hidden="true">✓</span><span>${escapeHtml(key)}</span></li>
  `).join("");

  return `<div class="next-step priority-panel">
    <div class="next-title">
      <div>
        <p class="section-label">Next Step</p>
        <h2>${escapeHtml(day.title)}</h2>
        <p class="muted">Day ${String(day.num).padStart(2, "0")} · Week ${day.week}</p>
      </div>
      <button type="button" class="primary" data-edit-day="${day.num}">Edit Day ${String(day.num).padStart(2, "0")}</button>
    </div>
    <div class="next-meta">${statusBadge(day.status)}${tagsHtml(day.jd_tags || [])}</div>
    ${day.next_fix ? `<p><strong>Next Fix:</strong> ${escapeHtml(day.next_fix)}</p>` : ""}
    ${day.weaknesses ? `<p><strong>Weakness:</strong> ${escapeHtml(day.weaknesses)}</p>` : ""}
    <div class="metric-strip">
      <div><strong class="number">${day.task_done}/${day.task_total}</strong><span>Tasks</span></div>
      <div><strong class="number">${day.artifact_done}/${day.artifact_total}</strong><span>Artifacts</span></div>
      <div><strong class="number">${day.daily_check || 0}/3</strong><span>Daily Check</span></div>
      <div><strong class="number">${percentFormat.format(day.completion_pct)}%</strong><span>Progress</span></div>
    </div>
    <div class="next-lists">
      <div>
        <p class="group-label">Tasks</p>
        <ul class="task-list">${taskItems || "<li><span>No tasks recorded.</span></li>"}</ul>
      </div>
      <div>
        <p class="group-label">Artifacts</p>
        <ul class="task-list">${artifactItems || "<li><span>No artifacts recorded.</span></li>"}</ul>
      </div>
    </div>
  </div>`;
}

function renderStats() {
  const summary = DATA.summary;
  return `<div class="overview-panel" aria-label="Progress summary">
    <p class="section-label">Overview</p>
    <div class="overview-list">
      <div><span>Days</span><strong class="number">${summary.done_days}/${summary.total_days}</strong>${miniProgress(summary.done_days, summary.total_days, "Days done")}</div>
      <div><span>Tasks</span><strong class="number">${summary.done_tasks}/${summary.total_tasks}</strong>${miniProgress(summary.done_tasks, summary.total_tasks, "Tasks done")}</div>
      <div><span>Artifacts</span><strong class="number">${summary.done_artifacts}/${summary.total_artifacts}</strong>${miniProgress(summary.done_artifacts, summary.total_artifacts, "Artifacts done")}</div>
      <div><span>Avg Daily Check</span><strong class="number">${summary.average_daily_check ?? "—"}</strong><small>0-3 score</small></div>
    </div>
  </div>`;
}

function renderFilters() {
  const weekOptions = [`<option value="all">All Weeks</option>`, `<option value="current">Current Week</option>`]
    .concat(DATA.weeks.map((week) => `<option value="${week.num}">Week ${week.num}</option>`))
    .join("");
  const statusOptions = [`<option value="all">All Statuses</option>`]
    .concat(DATA.options.day_statuses.map((status) => `<option value="${escapeHtml(status)}">${escapeHtml(statusLabel(status))}</option>`))
    .join("");
  const tagOptions = [`<option value="all">All Tags</option>`]
    .concat(DATA.options.tags.map((tag) => `<option value="${escapeHtml(tag)}">${escapeHtml(tag)}</option>`))
    .join("");

  return `<section class="filters" aria-label="Dashboard filters">
    <div class="filter-field">
      <label for="filter-week">Week</label>
      <select id="filter-week">${weekOptions}</select>
    </div>
    <div class="filter-field">
      <label for="filter-status">Status</label>
      <select id="filter-status">${statusOptions}</select>
    </div>
    <div class="filter-field">
      <label for="filter-tag">Tag</label>
      <select id="filter-tag">${tagOptions}</select>
    </div>
    <div class="filter-field search">
      <label for="filter-query">Search</label>
      <input id="filter-query" type="search" autocomplete="off" placeholder="row_softmax, Nsight, blocked…" value="${escapeHtml(STATE.query)}">
    </div>
  </section>`;
}

function renderOperators() {
  const rows = Object.entries(DATA.operator_maturity).map(([name, item]) => `
    <div class="insight-item">
      <div class="insight-main">
        <strong translate="no">${escapeHtml(name)}</strong>
        <small>${statusLine(item.status)} · ${numberFormat.format(item.done)}/${numberFormat.format(item.total)} artifacts</small>
      </div>
      <div class="insight-side">
        ${miniProgress(item.done, item.total, `${name} maturity`)}
        <button type="button" class="small-button" data-edit-operator="${escapeHtml(name)}" aria-label="Edit operator ${escapeHtml(name)}">Edit</button>
      </div>
    </div>
  `).join("");
  return `<section class="panel compact-panel">
    <div class="panel-header">
      <div><h2>Operator Maturity</h2><p class="muted">Six-artifact checklist.</p></div>
    </div>
    <div class="insight-list">${rows || `<p class="muted">No operators recorded.</p>`}</div>
  </section>`;
}

function renderPlanPanel() {
  const weeks = filteredWeeks();
  const visibleDays = weeks.reduce((total, week) => total + week.days.length, 0);
  return `<section class="panel plan-panel">
    <div class="plan-toolbar">
      <div>
        <h2>Plan</h2>
        <p class="muted">${numberFormat.format(visibleDays)} visible days. Use filters to narrow the schedule.</p>
      </div>
      ${renderFilters()}
    </div>
    ${renderWeeks(weeks)}
  </section>`;
}

function renderWeeks(weeks = filteredWeeks()) {
  if (!weeks.length) {
    return `<section class="empty-state"><h2>No Matching Days</h2><p>Adjust the filters to show planned work.</p></section>`;
  }

  const currentWeek = currentWeekNumber();
  const html = weeks.map((week) => {
    const done = week.days.filter((day) => day.status === "done").length;
    const total = week.days.length;
    const open = week.num === currentWeek || STATE.week !== "current" ? " open" : "";
    const rows = week.days.map((day) => `
      <div class="day-row">
        <strong class="number">D${String(day.num).padStart(2, "0")}</strong>
        <div class="day-row-title">
          <strong>${escapeHtml(day.title)}</strong>
          <div class="day-meta">
            ${statusLine(day.status)}
            <span>${day.task_done}/${day.task_total} tasks</span>
            <span>${day.artifact_done}/${day.artifact_total} artifacts</span>
            ${tagsHtml(day.jd_tags || [])}
          </div>
        </div>
        <div class="row-actions">
          <button type="button" class="small-button" data-edit-day="${day.num}" aria-label="Edit day ${day.num}: ${escapeHtml(day.title)}">Edit</button>
        </div>
      </div>
    `).join("");
    return `<details class="week"${open}>
      <summary>
        <h3>Week ${week.num}</h3>
        ${progressBar(done, total, `Week ${week.num} visible days done`)}
      </summary>
      <div class="week-body">${rows}</div>
    </details>`;
  }).join("");
  return `<section class="weeks" aria-label="Week plan">
    ${html}
  </section>`;
}

function renderRisks() {
  const risks = DATA.risks || [];
  return `<section class="panel compact-panel">
    <div class="panel-header"><h2>Risks</h2></div>
    ${risks.length ? `<ul class="risk-list">${risks.map((risk) => `<li>${escapeHtml(risk)}</li>`).join("")}</ul>` : `<p class="muted">No high-priority risks.</p>`}
  </section>`;
}

function renderTagCoverage() {
  const rows = Object.entries(DATA.tag_coverage).map(([tag, item]) => `
    <div class="insight-item">
      <div class="insight-main">
        <strong>${escapeHtml(tag)}</strong>
        <small>${numberFormat.format(item.done)}/${numberFormat.format(item.planned)} days complete</small>
      </div>
      ${miniProgress(item.done, item.planned, `${tag} coverage`)}
    </div>
  `).join("");
  return `<section class="panel compact-panel">
    <div class="panel-header"><h2>JD Tag Coverage</h2></div>
    <div class="insight-list">${rows}</div>
  </section>`;
}

function renderLibraries() {
  const rows = Object.entries(DATA.gpu_libraries).map(([name, info]) => `
    <div class="insight-item">
      <div class="insight-main">
        <strong translate="no">${escapeHtml(name)}</strong>
        <small>${statusLine(info.status)} · ${numberFormat.format((info.evidence || []).length)} evidence</small>
      </div>
      <button type="button" class="small-button" data-edit-library="${escapeHtml(name)}" aria-label="Edit library ${escapeHtml(name)}">Edit</button>
    </div>
  `).join("");
  return `<section class="panel compact-panel">
    <div class="panel-header"><h2>GPU Library Coverage</h2></div>
    <div class="insight-list">${rows}</div>
  </section>`;
}

function syncControls() {
  const week = document.getElementById("filter-week");
  const status = document.getElementById("filter-status");
  const tag = document.getElementById("filter-tag");
  const query = document.getElementById("filter-query");
  if (week) week.value = STATE.week;
  if (status) status.value = STATE.status;
  if (tag) tag.value = STATE.tag;
  if (query) query.value = STATE.query;
}

function setFilter(name, value) {
  STATE[name] = value;
  updateQueryString();
  render();
}

function fieldTemplate({id, label, name, value = "", type = "text", min = "", max = "", options = null, textarea = false}) {
  const errorId = `${id}-error`;
  if (textarea) {
    return `<div class="field">
      <label for="${id}">${escapeHtml(label)}</label>
      <textarea id="${id}" name="${name}" aria-describedby="${errorId}">${escapeHtml(value)}</textarea>
      <div class="field-error" id="${errorId}"></div>
    </div>`;
  }
  if (options) {
    const optionsHtml = options.map((option) => `<option value="${escapeHtml(option)}" ${value === option ? "selected" : ""}>${escapeHtml(statusLabel(option))}</option>`).join("");
    return `<div class="field">
      <label for="${id}">${escapeHtml(label)}</label>
      <select id="${id}" name="${name}" aria-describedby="${errorId}">${optionsHtml}</select>
      <div class="field-error" id="${errorId}"></div>
    </div>`;
  }
  return `<div class="field">
    <label for="${id}">${escapeHtml(label)}</label>
    <input id="${id}" name="${name}" type="${type}" value="${escapeHtml(value)}" ${min !== "" ? `min="${min}"` : ""} ${max !== "" ? `max="${max}"` : ""} aria-describedby="${errorId}">
    <div class="field-error" id="${errorId}"></div>
  </div>`;
}

function checkboxGroup(name, values, prefix) {
  const entries = Object.entries(values || {});
  if (!entries.length) return `<p class="muted">No ${escapeHtml(name)} recorded.</p>`;
  return `<div class="checks">${entries.map(([key, value], index) => {
    const id = `${prefix}-${name}-${index}`;
    return `<label for="${id}"><input id="${id}" type="checkbox" name="${name}" value="${escapeHtml(key)}" ${truthy(value) ? "checked" : ""}> ${escapeHtml(key)}</label>`;
  }).join("")}</div>`;
}

function editorShell(title, body) {
  return `<div class="editor-title-row">
    <div>
      <h2 id="editor-title">${title}</h2>
      <p class="muted" id="editor-help">Save writes to progress.yaml when the dashboard is served locally.</p>
    </div>
    <button type="button" class="ghost" data-close-editor aria-label="Close editor">Close</button>
  </div>
  <div id="editor-error" class="field-error" role="alert"></div>
  ${body}`;
}

function openDayEditor(dayNum) {
  const day = allDays().find((item) => item.num === dayNum);
  EDIT = {type: "day", id: dayNum};
  const prefix = `day-${dayNum}`;
  const body = `
    ${fieldTemplate({id: `${prefix}-status`, label: "Status", name: "status", value: day.status || "not_started", options: DATA.options.day_statuses})}
    ${fieldTemplate({id: `${prefix}-date`, label: "Date", name: "date", value: day.date || "", type: "date"})}
    ${fieldTemplate({id: `${prefix}-daily`, label: "Daily Check", name: "daily_check", value: Number(day.daily_check || 0), type: "number", min: 0, max: 3})}
    ${"weekly_check_score" in day ? fieldTemplate({id: `${prefix}-weekly`, label: "Weekly Check", name: "weekly_check_score", value: Number(day.weekly_check_score || 0), type: "number", min: 0, max: 21}) : ""}
    ${"stage_check_score" in day ? fieldTemplate({id: `${prefix}-stage`, label: "Stage Check", name: "stage_check_score", value: Number(day.stage_check_score || 0), type: "number", min: 0, max: 100}) : ""}
    <div class="field"><p class="group-label">Tasks</p>${checkboxGroup("tasks", day.tasks, prefix)}</div>
    <div class="field"><p class="group-label">Artifacts</p>${checkboxGroup("artifacts", day.artifacts, prefix)}</div>
    ${fieldTemplate({id: `${prefix}-verification`, label: "Verification", name: "verification", value: day.verification || "", textarea: true})}
    ${fieldTemplate({id: `${prefix}-weaknesses`, label: "Weaknesses", name: "weaknesses", value: day.weaknesses || "", textarea: true})}
    ${fieldTemplate({id: `${prefix}-next-fix`, label: "Next Fix", name: "next_fix", value: day.next_fix || "", textarea: true})}
    ${fieldTemplate({id: `${prefix}-notes`, label: "Notes", name: "notes", value: day.notes || "", textarea: true})}
    <div class="actions"><button type="button" data-close-editor>Cancel</button><button class="primary" type="submit">Save Day</button></div>`;
  setEditorHtml(editorShell(`Day ${String(day.num).padStart(2, "0")} · ${escapeHtml(day.title)}`, body));
  openEditor();
}

function openOperatorEditor(name) {
  const op = DATA.operators[name];
  EDIT = {type: "operator", id: name};
  const prefix = `operator-${name.replace(/\W+/g, "-")}`;
  const body = `
    ${fieldTemplate({id: `${prefix}-status`, label: "Status", name: "status", value: op.status || "not_started", options: DATA.options.operator_statuses})}
    <div class="field"><p class="group-label">Artifacts</p>${checkboxGroup("artifacts", op.artifacts, prefix)}</div>
    ${fieldTemplate({id: `${prefix}-notes`, label: "Notes", name: "notes", value: op.notes || "", textarea: true})}
    <div class="actions"><button type="button" data-close-editor>Cancel</button><button class="primary" type="submit">Save Operator</button></div>`;
  setEditorHtml(editorShell(`Operator · ${escapeHtml(name)}`, body));
  openEditor();
}

function openLibraryEditor(name) {
  const library = DATA.gpu_libraries[name];
  EDIT = {type: "library", id: name};
  const prefix = `library-${name.replace(/\W+/g, "-")}`;
  const body = `
    ${fieldTemplate({id: `${prefix}-status`, label: "Status", name: "status", value: library.status || "not_started", options: DATA.options.library_statuses})}
    ${fieldTemplate({id: `${prefix}-evidence`, label: "Evidence, One Per Line", name: "evidence", value: (library.evidence || []).join("\n"), textarea: true})}
    <div class="actions"><button type="button" data-close-editor>Cancel</button><button class="primary" type="submit">Save Library</button></div>`;
  setEditorHtml(editorShell(`GPU Library · ${escapeHtml(name)}`, body));
  openEditor();
}

function setEditorHtml(html) {
  const editor = document.getElementById("editor");
  editor.innerHTML = html;
  editor.addEventListener("input", () => {
    DIRTY = true;
  }, {once: true});
}

function openEditor() {
  previousFocus = document.activeElement;
  DIRTY = false;
  restoreFocus = true;
  const drawer = document.getElementById("drawer");
  drawer.hidden = false;
  document.body.style.overflow = "hidden";
  requestAnimationFrame(() => {
    const first = getFocusable(drawer)[0];
    if (first) first.focus();
  });
}

function closeEditor({force = false, restore = true} = {}) {
  if (!force && DIRTY && !window.confirm("Discard unsaved changes?")) return false;
  document.getElementById("drawer").hidden = true;
  document.body.style.overflow = "";
  EDIT = null;
  DIRTY = false;
  if (restore && previousFocus && typeof previousFocus.focus === "function") {
    previousFocus.focus();
  }
  previousFocus = null;
  return true;
}

function getFocusable(root) {
  return Array.from(root.querySelectorAll("button, input, select, textarea, a[href], [tabindex]:not([tabindex='-1'])"))
    .filter((element) => !element.disabled && !element.hidden && element.offsetParent !== null);
}

function trapFocus(event) {
  const drawer = document.getElementById("drawer");
  if (drawer.hidden || event.key !== "Tab") return;
  const focusable = getFocusable(drawer);
  if (!focusable.length) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

function checkedValues(form, name) {
  const values = {};
  form.querySelectorAll(`input[name="${name}"]`).forEach((input) => {
    values[input.value] = input.checked;
  });
  return values;
}

function valueOf(form, name) {
  return form.elements.namedItem(name)?.value ?? "";
}

function showEditorError(message) {
  const error = document.getElementById("editor-error");
  if (!error) return;
  error.textContent = message;
  error.style.display = "block";
}

async function saveEditor(event) {
  event.preventDefault();
  const form = event.currentTarget;
  if (!form.checkValidity()) {
    form.reportValidity();
    return;
  }

  const submit = form.querySelector("button[type='submit']");
  const originalText = submit.textContent;
  submit.disabled = true;
  submit.textContent = "Saving…";

  try {
    let url;
    let payload;
    if (EDIT.type === "day") {
      const updates = {
        status: valueOf(form, "status"),
        date: valueOf(form, "date"),
        daily_check: Number(valueOf(form, "daily_check") || 0),
        tasks: checkedValues(form, "tasks"),
        artifacts: checkedValues(form, "artifacts"),
        verification: valueOf(form, "verification"),
        weaknesses: valueOf(form, "weaknesses"),
        next_fix: valueOf(form, "next_fix"),
        notes: valueOf(form, "notes"),
        auto_status: true,
      };
      if (form.elements.namedItem("weekly_check_score")) updates.weekly_check_score = Number(valueOf(form, "weekly_check_score") || 0);
      if (form.elements.namedItem("stage_check_score")) updates.stage_check_score = Number(valueOf(form, "stage_check_score") || 0);
      url = "/api/day";
      payload = {day: EDIT.id, updates};
    } else if (EDIT.type === "operator") {
      url = "/api/operator";
      payload = {operator: EDIT.id, updates: {status: valueOf(form, "status"), artifacts: checkedValues(form, "artifacts"), notes: valueOf(form, "notes")}};
    } else {
      url = "/api/library";
      payload = {library: EDIT.id, updates: {status: valueOf(form, "status"), evidence: valueOf(form, "evidence")}};
    }

    const response = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok || !result.ok) {
      throw new Error(result.error || `Save failed with HTTP ${response.status}`);
    }
    DIRTY = false;
    closeEditor({force: true});
    showToast("Saved");
    await refreshData();
  } catch (error) {
    const message = window.location.protocol === "file:"
      ? "Start the dashboard with --serve before saving."
      : error.message;
    showEditorError(message);
    showToast(message);
  } finally {
    submit.disabled = false;
    submit.textContent = originalText;
  }
}

async function refreshData() {
  try {
    const response = await fetch("/api/progress");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    DATA = await response.json();
    render();
    showToast("Data refreshed");
  } catch (error) {
    if (DATA) {
      showToast("Static preview data is shown");
      render();
      return;
    }
    showToast("Unable to load dashboard data");
  }
}

function showToast(message) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("show"), 2200);
}

function bindEvents() {
  document.getElementById("refresh-button").addEventListener("click", refreshData);

  document.addEventListener("click", (event) => {
    const closeButton = event.target.closest("[data-close-editor]");
    const dayButton = event.target.closest("[data-edit-day]");
    const opButton = event.target.closest("[data-edit-operator]");
    const libButton = event.target.closest("[data-edit-library]");

    if (closeButton) {
      closeEditor();
      return;
    }
    if (event.target.id === "drawer") {
      closeEditor();
      return;
    }
    if (dayButton) openDayEditor(Number(dayButton.dataset.editDay));
    if (opButton) openOperatorEditor(opButton.dataset.editOperator);
    if (libButton) openLibraryEditor(libButton.dataset.editLibrary);
  });

  document.addEventListener("change", (event) => {
    if (event.target.id === "filter-week") setFilter("week", event.target.value);
    if (event.target.id === "filter-status") setFilter("status", event.target.value);
    if (event.target.id === "filter-tag") setFilter("tag", event.target.value);
  });

  document.addEventListener("input", (event) => {
    if (event.target.id === "filter-query") setFilter("query", event.target.value);
  });

  document.getElementById("editor").addEventListener("submit", saveEditor);

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !document.getElementById("drawer").hidden) {
      closeEditor();
    }
    trapFocus(event);
  });

  window.addEventListener("beforeunload", (event) => {
    if (!DIRTY) return;
    event.preventDefault();
    event.returnValue = "";
  });
}

async function init() {
  readQueryString();
  bindEvents();
  await loadData();
  render();
}

init().catch((error) => {
  document.getElementById("app").innerHTML = `<section class="empty-state"><h2>Unable To Load Dashboard</h2><p>${escapeHtml(error.message)}</p></section>`;
});
