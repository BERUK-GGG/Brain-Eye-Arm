// Operator HMI: polls the Flask backend (see app/web.py) and drives /pick.

const $ = (id) => document.getElementById(id);

const SWATCHES = { red: "#d9413b", green: "#2fa35a", blue: "#2f6fd6", black: "#111", white: "#f2f2f2" };
const POLL_MS = { parts: 500, status: 1000, logs: 1000 };

const state = {
    parts: [],
    picking: false,     // this page is waiting on a /pick
    robotBusy: false,   // a cycle is running (from any client)
    logSeq: 0,
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

    if (v.detection_age == null) setPill("pill-vision", "warn", "starting");
    else if (v.detection_age > 10) setPill("pill-vision", "err", `stalled ${age(v.detection_age)}`);
    else setPill("pill-vision", "ok", `${v.detect_seconds.toFixed(2)} s / pass`);
    $("feed-meta").textContent = v.detection_age == null ? "" : `detections updated ${age(v.detection_age)}`;

    const r = data.robot;
    state.robotBusy = r.busy;
    if (r.busy) setPill("pill-robot", "busy", "running cycle");
    else if (r.connected) setPill("pill-robot", "ok", `connected ${r.address}`);
    else setPill("pill-robot", "warn", r.address ? `idle, connects on pick` : "unknown");

    // Show another client's pick too, but don't overwrite our own in-flight message.
    if (!state.picking && data.last_pick) showResult(data.last_pick);
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
    const body = $("parts-body");
    body.replaceChildren(...state.parts.map((p) => el("tr", { class: matches(p, sel) ? "match" : "" },
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
    const locked = state.picking || state.robotBusy;
    $("btn-pick").disabled = locked || Number($("btn-pick").dataset.matching || 0) === 0;
    for (const button of document.querySelectorAll(".pick-one")) button.disabled = locked;
}

// ---------- picking ----------

function showResult(outcome) {
    const box = $("last-result");
    box.hidden = false;
    box.className = `result ${outcome.busy ? "busy" : outcome.ok ? "ok" : "err"}`;
    let text;
    if (outcome.busy) text = `Picking ${outcome.request}...`;
    else if (outcome.ok) text = `${outcome.request}: ` + outcome.results.map((r) =>
        `${r.color ? r.color + " " : ""}${r.shape} ${r.status === "placed" ? "placed" : "skipped (no place target)"}`).join(", ");
    else text = `${outcome.request}: ${outcome.error}`;
    box.replaceChildren(el("span", { class: "when" }, outcome.time || ""), text);
}

async function pick(body, label) {
    state.picking = true;
    updateButtons();
    showResult({ busy: true, request: label, time: new Date().toLocaleTimeString([], { hour12: false }) });
    try {
        await getJSON("/pick", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
    } finally {
        state.picking = false;
        await refreshStatus().catch(() => {});  // shows the recorded outcome
        refreshParts().catch(() => {});
    }
}

// ---------- log ----------

function logLevel(text) {
    if (/error|failed|traceback|exception|refused/i.test(text)) return "err";
    if (/skipping|nothing currently|stale|warning/i.test(text)) return "warn";
    if (/: done|connected\.|ready/i.test(text)) return "ok";
    if (/^(sending|server reply):/i.test(text)) return "dim";
    return "";
}

async function refreshLogs() {
    const { data } = await getJSON(`/logs?after=${state.logSeq}`);
    const lines = data.lines || [];
    if (!lines.length) return;
    const consoleBox = $("console");
    for (const line of lines) {
        consoleBox.append(el("div", { class: `line ${logLevel(line.text)}` },
            el("span", { class: "t" }, line.time), line.text));
        state.logSeq = line.seq;
    }
    while (consoleBox.childElementCount > 500) consoleBox.firstChild.remove();
    if ($("autoscroll").checked) consoleBox.scrollTop = consoleBox.scrollHeight;
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
$("btn-clear").addEventListener("click", () => $("console").replaceChildren());

tickClock();
setInterval(tickClock, 1000);
loadOptions().then(renderParts);
poll(refreshParts, POLL_MS.parts);
poll(refreshStatus, POLL_MS.status);
poll(refreshLogs, POLL_MS.logs);
