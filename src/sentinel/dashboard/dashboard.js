"use strict";

const $ = (id) => document.getElementById(id);
const svgNS = "http://www.w3.org/2000/svg";
const pageSize = 20;
let key = sessionStorage.getItem("sentinel.api_key") || "";
let offset = 0;
let items = [];
let selected = null;
let timeline = [];
let accountFlags = [];

function element(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined && value !== null) node.textContent = String(value);
  return node;
}

function svgElement(tag, attributes) {
  const node = document.createElementNS(svgNS, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, String(value));
  return node;
}

function message(target, value, error = false) {
  target.textContent = value;
  target.classList.toggle("error", error);
}

function errorText(status) {
  if (status === 401) return "The API key is invalid or has expired. Enter a valid key to continue.";
  if (status === 403) return "This API key does not have the read scope required for the console.";
  if (status === 429) return "Too many requests. Wait a moment, then try again.";
  return `Request failed (${status}). Please try again.`;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "X-API-Key": key, ...(options.headers || {}) },
  });
  if (!response.ok) {
    const failure = new Error(errorText(response.status));
    failure.status = response.status;
    throw failure;
  }
  return response.json();
}

function showSignIn(reason = "") {
  $("console").hidden = true;
  $("sign-in").hidden = false;
  $("sign-out").hidden = true;
  $("audit-strip").hidden = true;
  $("sign-in-error").hidden = !reason;
  message($("sign-in-error"), reason, true);
}

function showConsole() {
  $("sign-in").hidden = true;
  $("console").hidden = false;
  $("sign-out").hidden = false;
}

function reportBadge(state) {
  if (!state.exists) return "";
  return state.grounded ? "GROUNDED REPORT" : "REPORT · CHECK NEEDED";
}

function money(transaction) {
  const number = Number(transaction.amount);
  return `${transaction.currency} ${Number.isFinite(number) ? number.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : transaction.amount}`;
}

function shortDate(value) {
  const time = new Date(value);
  return Number.isNaN(time.valueOf()) ? String(value) : time.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function drawQueue() {
  const list = $("queue");
  list.replaceChildren();
  for (const flag of items) {
    const transaction = flag.transaction;
    const button = element("button", "queue-item");
    button.type = "button";
    button.classList.toggle("active", selected && selected.id === flag.id);
    button.setAttribute("aria-label", `${transaction.merchant_name}, ${money(transaction)}, score ${Math.round(flag.score * 100)} percent`);
    const top = element("div", "queue-top");
    top.append(element("strong", "", transaction.merchant_name), element("span", "", `${Math.round(flag.score * 100)}%`));
    const middle = element("div", "queue-middle");
    middle.append(element("b", "", money(transaction)), element("span", "", shortDate(transaction.timestamp)));
    const bottom = element("div", "queue-bottom");
    bottom.append(element("span", "queue-reason", flag.reasons[0] || "Flagged for review"));
    const badge = element("span", "report-badge", reportBadge(flag.explanation));
    if (flag.explanation.exists && !flag.explanation.grounded) badge.classList.add("unverified");
    bottom.append(badge);
    const bar = element("progress", "score-track");
    bar.max = 1;
    bar.value = flag.score;
    if (flag.score >= 0.8) bar.classList.add("high");
    button.append(top, middle, bottom, bar);
    button.addEventListener("click", () => selectFlag(flag));
    list.append(button);
  }
  $("queue-count").textContent = String(items.length);
  $("page-label").textContent = `Page ${Math.floor(offset / pageSize) + 1}`;
  $("previous").disabled = offset === 0;
  $("next").disabled = items.length < pageSize;
}

async function loadQueue() {
  message($("queue-message"), "Loading flags…");
  const params = new URLSearchParams({ limit: String(pageSize), offset: String(offset), min_score: $("score-filter").value });
  const account = $("account-filter").value.trim();
  if (account) params.set("account_id", account);
  try {
    const page = await api(`/v1/flags?${params}`);
    showConsole();
    items = page.items;
    message($("queue-message"), items.length ? "" : "No flags match these filters.");
    if (!items.length) {
      selected = null;
      $("detail").hidden = true;
      $("empty-detail").hidden = false;
    }
    drawQueue();
    if (items.length && !items.some((item) => selected && item.id === selected.id)) await selectFlag(items[0]);
  } catch (failure) {
    if (failure.status === 401 || failure.status === 403 || $("console").hidden) showSignIn(failure.message);
    else message($("queue-message"), failure.message, true);
  }
}

function drawFacts(transaction) {
  const facts = [
    ["Transaction ID", transaction.transaction_id], ["Account", transaction.account_id],
    ["Amount", money(transaction)], ["Time", shortDate(transaction.timestamp)],
    ["Country", transaction.country], ["Channel", transaction.channel.replaceAll("_", " ")],
    ["Merchant category", transaction.merchant_category], ["Device", transaction.device_id],
    ["Memo", transaction.memo || "—"],
  ];
  const target = $("facts");
  target.replaceChildren();
  for (const [label, value] of facts) {
    const pair = element("div");
    pair.append(element("dt", "", label), element("dd", "", value));
    target.append(pair);
  }
}

async function loadTimeline(flag) {
  const transaction = flag.transaction;
  const params = new URLSearchParams({ limit: "200", before: transaction.timestamp });
  const [history, flags] = await Promise.all([
    api(`/v1/accounts/${encodeURIComponent(transaction.account_id)}/history?${params}`),
    api(`/v1/flags?account_id=${encodeURIComponent(transaction.account_id)}&limit=200`),
  ]);
  const selectedTime = new Date(transaction.timestamp).valueOf();
  if (!selected || selected.id !== flag.id) return;
  timeline = history.items.filter((row) => new Date(row.timestamp).valueOf() <= selectedTime);
  if (!timeline.some((row) => row.transaction_id === transaction.transaction_id)) timeline.push(transaction);
  timeline.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
  accountFlags = flags.items;
  drawChart();
}

function drawChart() {
  const target = $("chart");
  target.replaceChildren();
  if (!timeline.length) {
    target.append(element("p", "muted", "No account activity available."));
    return;
  }
  const width = 800, height = 205, left = 55, right = 20, top = 18, bottom = 36;
  const svg = svgElement("svg", { viewBox: `0 0 ${width} ${height}`, preserveAspectRatio: "none", "aria-hidden": "true" });
  const times = timeline.map((row) => new Date(row.timestamp).valueOf());
  const amounts = timeline.map((row) => Math.max(0.01, Number(row.amount)));
  const log = Math.max(...amounts) / Math.min(...amounts) > 30;
  const values = amounts.map((value) => log ? Math.log10(value) : value);
  const minTime = Math.min(...times), maxTime = Math.max(...times);
  const minValue = Math.min(...values), maxValue = Math.max(...values);
  const x = (time, index) => maxTime === minTime ? left + (index / Math.max(1, times.length - 1)) * (width - left - right) : left + ((time - minTime) / (maxTime - minTime)) * (width - left - right);
  const y = (value) => top + (1 - (value - minValue) / (maxValue - minValue || 1)) * (height - top - bottom);
  for (let index = 0; index < 3; index++) {
    const lineY = top + index * (height - top - bottom) / 2;
    svg.append(svgElement("line", { x1: left, x2: width - right, y1: lineY, y2: lineY, class: "grid" }));
    const value = minValue + (2 - index) * (maxValue - minValue) / 2;
    const label = svgElement("text", { x: left - 7, y: lineY + 3, "text-anchor": "end" });
    label.textContent = log ? `${Math.round(10 ** value)}` : `${Math.round(value)}`;
    svg.append(label);
  }
  svg.append(svgElement("line", { x1: left, x2: width - right, y1: height - bottom, y2: height - bottom, class: "axis" }));
  const polyline = svgElement("polyline", { points: timeline.map((row, index) => `${x(times[index], index)},${y(values[index])}`).join(" "), class: "line" });
  svg.append(polyline);
  const flaggedIds = new Set(accountFlags.map((flag) => flag.transaction_id));
  timeline.forEach((row, index) => {
    const classes = ["point"];
    if (flaggedIds.has(row.transaction_id)) classes.push("flagged");
    if (selected && row.transaction_id === selected.transaction_id) classes.push("selected");
    const point = svgElement("circle", { cx: x(times[index], index), cy: y(values[index]), r: classes.includes("selected") ? 6 : 4, class: classes.join(" ") });
    point.dataset.ref = row.transaction_id;
    const title = svgElement("title", {});
    title.textContent = `${shortDate(row.timestamp)} · ${money(row)} · ${row.merchant_name}`;
    point.append(title);
    svg.append(point);
  });
  for (const [index, time] of [[0, minTime], [1, maxTime]]) {
    const label = svgElement("text", { x: index ? width - right : left, y: height - 9, "text-anchor": index ? "end" : "start" });
    label.textContent = new Date(time).toLocaleDateString(undefined, { month: "short", day: "numeric" });
    svg.append(label);
  }
  target.append(svg);
  target.setAttribute("aria-label", `${timeline.length} account transactions over time. ${log ? "Logarithmic" : "Linear"} amount scale. Selected transaction highlighted.`);
}

function highlight(ref, active) {
  for (const row of $("records").children) if (row.dataset.ref === ref) row.classList.toggle("record-active", active);
  for (const point of $("chart").querySelectorAll("circle")) if (point.dataset.ref === ref) point.classList.toggle("cited", active);
}

function drawReport(payload) {
  const target = $("report"), records = $("records");
  target.replaceChildren();
  records.replaceChildren();
  if (!payload || !payload.explanation) {
    $("report-meta").textContent = payload && payload.failure ? `Report unavailable · ${payload.failure}` : "No report generated";
    target.append(element("p", "report-empty", payload && payload.failure ? "The report could not be completed. Try generating it again." : "Generate a report to review the evidence and cited records."));
    return;
  }
  const report = payload.explanation;
  $("report-meta").textContent = payload.grounded ? "Grounding check passed" : "Grounding check needs review";
  target.append(element("p", "report-summary", report.summary));
  const stats = element("div", "report-stats");
  for (const [label, value] of [["Risk", report.risk_level], ["Action", report.recommended_action.replaceAll("_", " ")], ["Confidence", `${Math.round(report.confidence * 100)}%`]]) {
    const pill = element("span");
    pill.append(element("strong", "", `${label}: `), document.createTextNode(value));
    stats.append(pill);
  }
  target.append(stats, element("h4", "report-section-title", "Evidence claims"));
  const issues = payload.grounding ? payload.grounding.ungrounded_items || [] : [];
  report.evidence.forEach((claim, index) => {
    const box = element("div", "claim");
    const rejected = issues.filter((issue) => issue.claim_index === index + 1);
    if (rejected.length) box.classList.add("rejected");
    box.append(element("span", "claim-text", claim.claim));
    if (rejected.length) box.append(element("div", "rejection", `Grounding check: ${rejected.map((issue) => `${issue.reason} (${issue.span})`).join("; ")}`));
    const chips = element("div", "chip-list");
    for (const ref of claim.refs) {
      const chip = element("button", "chip", ref);
      chip.type = "button";
      chip.addEventListener("mouseenter", () => highlight(ref, true));
      chip.addEventListener("mouseleave", () => highlight(ref, false));
      chip.addEventListener("focus", () => highlight(ref, true));
      chip.addEventListener("blur", () => highlight(ref, false));
      chips.append(chip);
    }
    box.append(chips);
    target.append(box);
  });
  const summaryIssues = issues.filter((issue) => issue.claim_index === 0);
  if (summaryIssues.length) target.prepend(element("div", "rejection", `Summary grounding check: ${summaryIssues.map((issue) => `${issue.reason} (${issue.span})`).join("; ")}`));
  target.append(element("h4", "report-section-title", "Limitations"), element("p", "limitations", report.limitations));
  const cited = new Set(report.evidence.flatMap((claim) => claim.refs));
  for (const ref of cited) {
    const record = payload.cited_records && payload.cited_records[ref];
    const row = element("tr");
    row.dataset.ref = ref;
    row.append(element("td", "", ref), element("td", "record-json", record ? JSON.stringify(record, null, 2) : "Record unavailable"));
    records.append(row);
  }
}

async function selectFlag(flag) {
  selected = flag;
  drawQueue();
  $("empty-detail").hidden = true;
  $("detail").hidden = false;
  $("detail-kicker").textContent = `FLAG #${flag.id} · ${shortDate(flag.created_at)}`;
  $("detail-title").textContent = flag.transaction.merchant_name;
  $("detail-subtitle").textContent = `${flag.transaction.account_id} · ${flag.transaction.transaction_id}`;
  $("detail-score").textContent = `${Math.round(flag.score * 100)}% score`;
  drawFacts(flag.transaction);
  $("reasons").replaceChildren(...flag.reasons.map((reason) => element("li", "", reason)));
  message($("detail-message"), "Loading account context…");
  drawReport(null);
  try {
    const results = await Promise.allSettled([
      loadTimeline(flag),
      flag.explanation.exists ? api(`/v1/flags/${flag.id}/explanation`) : Promise.resolve(null),
    ]);
    if (selected.id !== flag.id) return;
    if (results[0].status === "rejected") throw results[0].reason;
    if (results[1].status === "rejected") throw results[1].reason;
    drawReport(results[1].value);
    message($("detail-message"), "");
  } catch (failure) {
    message($("detail-message"), failure.message || "Unable to load detail.", true);
  }
}

async function generate() {
  if (!selected) return;
  const flagId = selected.id;
  const button = $("generate");
  button.disabled = true;
  let seconds = 0;
  message($("detail-message"), "Generating report · 0 s elapsed");
  const ticker = setInterval(() => {
    seconds += 1;
    message($("detail-message"), `Generating report · ${seconds} s elapsed`);
  }, 1000);
  try {
    const payload = await api(`/v1/flags/${flagId}/explain`, { method: "POST" });
    if (selected && selected.id === flagId) {
      drawReport(payload);
      selected.explanation = { exists: true, grounded: payload.grounded };
      drawQueue();
      message($("detail-message"), payload.failure ? `Report could not be completed: ${payload.failure}` : "Report saved.", Boolean(payload.failure));
    }
  } catch (failure) {
    message($("detail-message"), failure.message || "Report generation failed.", true);
  } finally {
    clearInterval(ticker);
    button.disabled = false;
  }
}

async function checkAudit() {
  try {
    const status = await api("/v1/audit/verify");
    $("audit-strip").hidden = false;
    $("audit-status").textContent = status.ok ? "Verified" : `Broken at entry ${status.first_broken_entry_id}`;
    $("audit-status").classList.toggle("bad", !status.ok);
    $("audit-meta").textContent = `${status.entry_count} entries · head ${status.head_hash ? status.head_hash.slice(0, 12) : "none"}`;
  } catch (failure) {
    $("audit-strip").hidden = true;
  }
}

$("key-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  key = $("api-key").value.trim();
  sessionStorage.setItem("sentinel.api_key", key);
  await loadQueue();
  if (key && $("sign-in").hidden) await checkAudit();
});
$("sign-out").addEventListener("click", () => {
  key = "";
  selected = null;
  sessionStorage.removeItem("sentinel.api_key");
  $("api-key").value = "";
  showSignIn();
});
$("filters").addEventListener("submit", (event) => { event.preventDefault(); offset = 0; loadQueue(); });
$("previous").addEventListener("click", () => { offset = Math.max(0, offset - pageSize); loadQueue(); });
$("next").addEventListener("click", () => { offset += pageSize; loadQueue(); });
$("generate").addEventListener("click", generate);
if (key) {
  showConsole();
  loadQueue();
  checkAudit();
} else showSignIn();
