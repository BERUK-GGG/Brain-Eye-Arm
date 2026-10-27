// Operator HMI: polls the Flask backend (see app/web.py) and drives /pick.

const $ = (id) => document.getElementById(id);

const SWATCHES = { red: "#d9413b", green: "#2fa35a", blue: "#2f6fd6", black: "#111", white: "#f2f2f2" };
const POLL_MS = { parts: 500, status: 1000, history: 1000 };

const state = {
    parts: [],
    pendingJob: null,   // this page's pick: "sending", then its job ID until the outcome arrives
    robotBusy: false,   // a cycle is running (from any client)
    history: [],        // handled objects since the backend started
    historySeq: 0,
};

// ---------- helpers ----------

async function getJSON(url, options) {
    const res = await fetch(url, options);
    const data = await res.json().catch(() => ({}));
    return { ok: res.ok, status: res.status, data };
}

/** Runs fn every ms, never overlapping itself, surviving errors. */
function poll(fn, ms) {
    const tick = async () => {
        try { await fn(); } catch (e) { /* backend unreachable; status shows it */ }
        setTimeout(tick, ms);
    };
    tick();
}

function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
        if (key === "class") node.className = value;
        else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
        else node.setAttribute(key, value);
    }
    for (const child of children) node.append(child);
    return node;
}

function setPill(id, level, text) {
    const pill = $(id);
    pill.className = `pill ${level}`;
    pill.querySelector("b").textContent = text;
}

function age(seconds) {
    if (seconds == null) return "--";
    return seconds < 60 ? `${seconds.toFixed(1)} s ago` : `${Math.round(seconds / 60)} min ago`;
}

function selection() {
    return {
        shape: $("shape-select").value || null,
        color: $("color-select").value || null,
    };
}

function matches(part, sel) {
    return (!sel.shape || part.shape === sel.shape) && (!sel.color || part.color === sel.color);
}

// ---------- setup ----------

async function loadOptions() {
    const [{ data: s }, { data: c }] = await Promise.all([getJSON("/shapes"), getJSON("/colors")]);
    for (const name of s.shapes || []) $("shape-select").append(el("option", { value: name }, name));
    for (const name of c.colors || []) $("color-select").append(el("option", { value: name }, name));
}

// ---------- status bar ----------

async function refreshStatus() {
    const { data } = await getJSON("/status").catch(() => ({ data: null }));
    if (!data) {
        for (const id of ["pill-camera", "pill-vision", "pill-robot"]) setPill(id, "err", "no connection to Pi");
        return;
    }

    const v = data.vision;
    if (v.frame_age == null) setPill("pill-camera", "err", "no frames");
    else if (v.frame_age > 2) setPill("pill-camera", "err", `stalled ${age(v.frame_age)}`);
    else setPill("pill-camera", "ok", "live");

    if (v.paused) setPill("pill-vision", "busy", "paused while picking");
    else if (v.detection_age == null) setPill("pill-vision", "warn", "starting");
    else if (v.detection_age > 10) setPill("pill-vision", "err", `stalled ${age(v.detection_age)}`);
    else setPill("pill-vision", "ok", `${v.detect_seconds.toFixed(2)} s / pass`);
    $("feed-meta").textContent = v.detection_age == null ? "" : `detections updated ${age(v.detection_age)}`;

    const r = data.robot;
    state.robotBusy = r.busy;
    if (r.busy) setPill("pill-robot", "busy", "running cycle");
    else if (r.connected) setPill("pill-robot", "ok", `connected ${r.address}`);
    else setPill("pill-robot", "warn", r.address ? `idle, connects on pick` : "unknown");

    // Our job finished once /status reports its outcome.
    const outcome = data.last_pick;
    if (typeof state.pendingJob === "number" && outcome && outcome.job === state.pendingJob) {
        state.pendingJob = null;
        refreshParts().catch(() => {});
    }
    // Otherwise mirror whatever the robot is doing (e.g. a pick from another browser),
    // without overwriting our own "Picking..." message.
    if (state.pendingJob === null) {
        if (r.busy) showResult({ busy: true, request: r.current || "...", time: "" });
        else if (outcome) showResult(outcome);
    }
    updateButtons();
}

// ---------- detected parts ----------

async function refreshParts() {
    const { data } = await getJSON("/detections");
    state.parts = data.parts || [];
    renderParts();
}

function renderParts() {
    const sel = selection();
    const filtering = Boolean(sel.shape || sel.color);   // with no filter everything matches; don't light it all up
    const body = $("parts-body");
    body.replaceChildren(...state.parts.map((p) => el("tr", { class: filtering && matches(p, sel) ? "match" : "" },
        el("td", {}, `#${p.id}`),
        el("td", {}, p.shape),
        el("td", {},
            p.color ? el("span", { class: "swatch", style: `background:${SWATCHES[p.color] || "transparent"}` }) : "",
            p.color || "?"),
        el("td", { class: "num" }, `${Math.round(p.confidence * 100)}%`),
        el("td", { class: "num" }, p.orientation == null ? "--" : `${p.orientation.toFixed(1)}°`),
        el("td", { class: "num" }, `${p.robot[0].toFixed(1)}, ${p.robot[1].toFixed(1)}`),
        el("td", {}, el("button", {
            class: "btn ghost small pick-one",
            onclick: () => pick({ object_id: p.id }, `${p.color || ""} ${p.shape} #${p.id}`.trim()),
        }, "Pick")),
    )));
    $("parts-empty").hidden = state.parts.length > 0;

    const matching = state.parts.filter((p) => matches(p, sel)).length;
    $("parts-count").textContent = `${state.parts.length} tracked`;
    const what = [sel.color, sel.shape ? `${sel.shape}s` : null].filter(Boolean).join(" ") || "all parts";
    $("btn-pick").textContent = `Pick ${what} (${matching})`;
    $("btn-pick").dataset.matching = matching;
    updateButtons();
}

function updateButtons() {
    const locked = state.pendingJob !== null || state.robotBusy;
    $("btn-pick").disabled = locked || Number($("btn-pick").dataset.matching || 0) === 0;
    for (const button of document.querySelectorAll(".pick-one")) button.disabled = locked;
}

// ---------- picking ----------

function showResult(outcome) {
    const box = $("last-result");
    box.hidden = false;
    const allPlaced = outcome.ok && outcome.results.every((r) => r.status === "placed");
    box.className = `result ${outcome.busy ? "busy" : allPlaced ? "ok" : outcome.ok ? "warn" : "err"}`;
    let text;
    if (outcome.busy) text = `Picking ${outcome.request}...`;
    else if (outcome.ok) text = `${outcome.request}: ` + outcome.results.map((r) =>
        `${r.color ? r.color + " " : ""}${r.shape} ${resultTag(r.status)}` +
        (r.attempts > 1 ? ` (${r.attempts} attempts)` : "")).join(", ");
    else text = `${outcome.request}: ${outcome.error}`;
    box.replaceChildren(el("span", { class: "when" }, outcome.time || ""), text);
}

/** Queues a pick; the robot thread runs it, and refreshStatus() picks up the outcome. */
async function pick(body, label) {
    const now = () => new Date().toLocaleTimeString([], { hour12: false });
    state.pendingJob = "sending";
    updateButtons();
    showResult({ busy: true, request: label, time: now() });
    const { ok, data } = await getJSON("/pick", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    }).catch(() => ({ ok: false, data: { error: "no connection to Pi" } }));
    if (ok) {
        state.pendingJob = data.job;
    } else {
        state.pendingJob = null;   // refused (nothing matches / already busy): nothing was queued
        showResult({ ok: false, request: label, error: data.error, time: now() });
    }
    updateButtons();
    refreshStatus().catch(() => {});
}

// ---------- pick history ----------

const fmtXY = (xy) => (xy ? `${xy[0].toFixed(1)}, ${xy[1].toFixed(1)}` : "--");
const fmtDeg = (deg) => (deg == null ? "--" : `${deg.toFixed(1)}°`);
const resultTag = (status) => ({
    placed: "placed", skipped_no_target: "skipped", not_picked_up: "missed", failed: "failed",
}[status] || status);

function partCell(entry) {
    if (!entry.shape) return entry.request || "--";   // a failed request, not a single part
    return [
        entry.color ? el("span", { class: "swatch", style: `background:${SWATCHES[entry.color] || "transparent"}` }) : "",
        `${entry.color || "?"} ${entry.shape}`,
    ];
}

async function refreshHistory() {
    const { data } = await getJSON(`/history?after=${state.historySeq}`);
    const entries = data.entries || [];
    if (!entries.length) return;
    state.history.push(...entries);
    state.historySeq = entries[entries.length - 1].seq;
    renderHistory();
}

function renderHistory() {
    const body = $("history-body");
    // Newest first, so the latest pick is always in view.
    body.replaceChildren(...[...state.history].reverse().map((e) => el("tr", {},
        el("td", {}, e.time),
        el("td", {}, e.object_id == null ? "--" : `#${e.object_id}`),
        el("td", {}, ...[].concat(partCell(e))),
        el("td", { class: "num" }, fmtXY(e.pick)),
        el("td", { class: "num" }, fmtDeg(e.orientation)),
        el("td", { class: "num" }, fmtDeg(e.rotation)),
        el("td", { class: "num" }, fmtXY(e.place)),
        el("td", {},
            el("span", { class: `tag ${resultTag(e.status)}` }, resultTag(e.status)),
            e.attempts > 1 ? el("span", { class: "attempts" }, `${e.attempts} attempts`) : "",
            e.error ? el("span", { class: "error-text" }, e.error) : ""),
    )));
    $("history-empty").hidden = state.history.length > 0;
    $("btn-export").disabled = state.history.length === 0;

    const count = (status) => state.history.filter((e) => e.status === status).length;
    $("history-summary").textContent =
        `placed ${count("placed")} · missed ${count("not_picked_up")} · skipped ${count("skipped_no_target")} · failed ${count("failed")}`;
}

function exportHistory() {
    const header = ["time", "object_id", "color", "shape", "pick_x", "pick_y", "angle_deg",
        "rotation_deg", "place_x", "place_y", "result", "attempts", "error"];
    const cell = (v) => (v == null ? "" : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
    const rows = state.history.map((e) => [
        e.time, e.object_id, e.color, e.shape ?? e.request, e.pick?.[0], e.pick?.[1], e.orientation,
        e.rotation, e.place?.[0], e.place?.[1], resultTag(e.status), e.attempts, e.error,
    ].map(cell).join(","));
    const blob = new Blob([[header.join(","), ...rows].join("\n") + "\n"], { type: "text/csv" });
    const link = el("a", {
        href: URL.createObjectURL(blob),
        download: `pick_history_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.csv`,
    });
    link.click();
    URL.revokeObjectURL(link.href);
}

// ---------- start ----------

function tickClock() {
    $("clock").textContent = new Date().toLocaleTimeString([], { hour12: false });
}

$("btn-pick").addEventListener("click", () => {
    const sel = selection();
    const label = [sel.color, sel.shape].filter(Boolean).join(" ") || "all parts";
    pick(sel, label);
});
$("shape-select").addEventListener("change", renderParts);
$("color-select").addEventListener("change", renderParts);
$("btn-export").addEventListener("click", exportHistory);

tickClock();
setInterval(tickClock, 1000);
loadOptions().then(renderParts);
poll(refreshParts, POLL_MS.parts);
poll(refreshStatus, POLL_MS.status);
poll(refreshHistory, POLL_MS.history);
