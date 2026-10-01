"use strict";

const reasonFormatting = (() => {
  const number = (value, digits = 1) => Number(value).toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
  const count = (value) => number(Math.round(Number(value)), 0);
  const transactions = (value) => `${count(value)} ${Math.round(Number(value)) === 1 ? "transaction" : "transactions"}`;
  const currencyAmount = (value, currency) => `${number(value, 2)} ${currency}`.trim();
  const duration = (seconds) => {
    const elapsed = Number(seconds);
    const [value, unit] = elapsed >= 86400 ? [elapsed / 86400, "day"]
      : elapsed >= 3600 ? [elapsed / 3600, "hour"]
        : elapsed >= 60 ? [elapsed / 60, "minute"] : [elapsed, "second"];
    const rounded = Math.round(value * 10) / 10;
    return `${number(rounded, Number.isInteger(rounded) ? 0 : 1)} ${unit}${rounded === 1 ? "" : "s"}`;
  };
  const previous = (value, window) => `${transactions(value)} in the previous ${window}`;
  const spent = (value, currency, window) => `${currencyAmount(value, currency)} spent in the previous ${window}`;

  const featureFormatters = {
    amount: (value, currency) => `Transaction amount: ${currencyAmount(value, currency)}`,
    account_age: (value) => `${count(value)} earlier ${Math.round(Number(value)) === 1 ? "transaction" : "transactions"} on the account`,
    amount_zscore: (value) => Number(value) === 0 ? "At the account's average amount" : `${number(Math.abs(value))} standard deviations ${Number(value) < 0 ? "below" : "above"} the account's average amount`,
    amount_to_p95: (value) => `${number(value)}× the account's 95th-percentile amount`,
    amount_to_median: (value) => `${number(value)}× the account's median amount`,
    count_10m: (value) => previous(value, "10 minutes"),
    count_1h: (value) => previous(value, "hour"),
    count_24h: (value) => previous(value, "24 hours"),
    sum_10m: (value, currency) => spent(value, currency, "10 minutes"),
    sum_1h: (value, currency) => spent(value, currency, "hour"),
    sum_24h: (value, currency) => spent(value, currency, "24 hours"),
    seconds_since_previous: (value) => `${duration(value)} since the previous transaction`,
    days_since_activity: (value) => `${duration(Number(value) * 86400)} since the previous transaction`,
    is_new_device: (value) => Number(value) ? "New device" : null,
    is_new_country: (value) => Number(value) ? "New country" : null,
    is_new_merchant: (value) => Number(value) ? "New merchant" : null,
    device_seen_count: (value) => `Device used in ${transactions(value)} before`,
    country_seen_count: (value) => `Country seen in ${transactions(value)} before`,
    device_age_hours: (value) => `${duration(Number(value) * 3600)} since this device was first used`,
    country_age_hours: (value) => `${duration(Number(value) * 3600)} since this country was first seen`,
    new_device_count_24h: (value) => `${count(value)} new ${Math.round(Number(value)) === 1 ? "device" : "devices"} in the previous 24 hours`,
    hour_deviation: (value) => `${duration(Number(value) * 3600)} outside the account's usual transaction time`,
    is_online: (value) => Number(value) ? "Online purchase" : null,
    is_transfer: (value) => Number(value) ? "Transfer" : null,
    near_threshold: (value) => Number(value) ? "Amount just under the reporting threshold" : null,
    near_threshold_transfers_7d: (value) => `${transactions(value)} just under the reporting threshold in the previous 7 days`,
    usual_max_10m: (value) => `Busiest 10 minutes on record: ${transactions(value)}`,
  };

  function formatRule(reason) {
    let match = /^velocity: (\d+) transactions in 10 minutes \(usual max (\d+)\)$/.exec(reason);
    if (match) return [`${transactions(match[1])} within 10 minutes (previous maximum ${count(match[2])})`];

    match = /^account takeover: device and country first seen within 24 hours; (\d+) supporting signals, amount (-?\d+(?:\.\d+)?)x prior median$/.exec(reason);
    if (match) {
      const signals = ["New device and country within 24 hours", `${count(match[1])} supporting signals`];
      if (Number(match[2]) > 0) signals.push(`Amount: ${number(match[2])}× the account's median amount`);
      return signals;
    }

    match = /^amount spike: (-?\d+(?:\.\d+)?)x prior p95 on \$\d+(?:\.\d+)?$/.exec(reason);
    if (match) return [`Amount is ${number(match[1])}× the account's 95th-percentile amount`];

    match = /^structuring: \$\d+(?:\.\d+)? transfer below \$10000; (\d+) prior near-threshold transfers in 7 days$/.exec(reason);
    if (match) return [
      "Transfer just under the 10,000 reporting threshold",
      `${count(match[1])} similar ${Number(match[1]) === 1 ? "transfer" : "transfers"} in the past 7 days`,
    ];

    match = /^dormant drain: transfer after (-?\d+(?:\.\d+)?) inactive days at (-?\d+(?:\.\d+)?)x prior p95$/.exec(reason);
    if (match) return [`Transfer after ${number(match[1], Number.isInteger(Number(match[1])) ? 0 : 1)} inactive days at ${number(match[2])}× the 95th-percentile amount`];

    return null;
  }

  function readableReason(reason, currency = "") {
    const separator = reason.indexOf(": ");
    const prefix = separator < 0 ? "" : reason.slice(0, separator);
    if (prefix !== "forest model" && prefix !== "supervised model") {
      return { source: "Rule", signals: formatRule(reason) || [reason] };
    }
    const source = prefix === "forest model" ? "Anomaly model" : "Supervised model";
    const signals = reason.slice(separator + 2).split(", ").map((part) => {
      if (part === "multiple features") return "Multiple unusual transaction signals";
      if (part === "transaction profile") return "Transaction profile differs from usual activity";
      const match = /^(.+) (-?\d+(?:\.\d+)?)$/.exec(part);
      if (!match) return part;
      const feature = match[1].replaceAll(" ", "_");
      const formatter = featureFormatters[feature];
      return formatter ? formatter(match[2], currency) : part;
    }).filter(Boolean);
    return { source, signals: signals.length ? signals : ["Transaction profile differs from usual activity"] };
  }

  return { featureFormatters, readableReason };
})();

if (typeof module !== "undefined") module.exports = reasonFormatting;

if (typeof document !== "undefined") {
const $ = (id) => document.getElementById(id);
const svgNS = "http://www.w3.org/2000/svg";
const pageSize = 20;
let key = sessionStorage.getItem("sentinel.api_key") || "";
let offset = 0;
let items = [];
let selected = null;
let timeline = [];
let accountFlags = [];

function reasonNode(reason, currency) {
  const readable = reasonFormatting.readableReason(reason, currency);
  const node = element("li", "reason-item");
  node.append(element("span", "reason-source", readable.source));
  const signals = element("div", "reason-signals");
  for (const signal of readable.signals) signals.append(element("span", "reason-signal", signal));
  node.append(signals);
  return node;
}

function element(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined && value !== null) node.textContent = String(value);
  return node;
}

function placeholder(label, loading = false, kind = "") {
  const block = element("div", `placeholder ${kind}${loading ? " loading" : ""}`.trim());
  if (loading) {
    block.append(element("span", "visually-hidden", label));
    for (let index = 0; index < 3; index++) block.append(element("span", "skeleton-line"));
  } else {
    block.append(element("span", "placeholder-mark", "·"), element("span", "", label));
  }
  return block;
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
  const amount = Number.isFinite(number)
    ? number.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
    : transaction.amount;
  return `${transaction.currency} ${amount}`;
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
    const merchant = element("strong", "", transaction.merchant_name);
    merchant.title = transaction.merchant_name;
    const score = element("span", "queue-score", `${Math.round(flag.score * 100)}%`);
    if (flag.score >= 0.8) score.classList.add("high");
    top.append(element("span", `risk-dot${flag.score >= 0.8 ? " high" : ""}`), merchant, score);
    const middle = element("div", "queue-middle");
    middle.append(element("b", "", money(transaction)), element("span", "", shortDate(transaction.timestamp)));
    const bottom = element("div", "queue-bottom");
    const summary = reasonFormatting.readableReason(flag.reasons[0] || "Flagged for review", transaction.currency);
    const reason = element("span", "queue-reason", `${summary.source} · ${summary.signals.join(" · ")}`);
    reason.title = reason.textContent;
    bottom.append(reason);
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
  $("queue").replaceChildren(...Array.from({ length: 3 }, () => placeholder("Loading flags", true, "queue-placeholder")));
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
    if (!items.length) $("queue").append(placeholder("Try another account or minimum score.", false, "queue-empty"));
    if (items.length && !items.some((item) => selected && item.id === selected.id)) await selectFlag(items[0]);
  } catch (failure) {
    if (failure.status === 401 || failure.status === 403 || $("console").hidden) showSignIn(failure.message);
    else {
      message($("queue-message"), failure.message, true);
      $("queue").replaceChildren(placeholder("The queue could not be loaded.", false, "queue-empty"));
    }
  }
}

function drawFacts(transaction) {
  const facts = [
    ["Time", shortDate(transaction.timestamp)],
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
    target.append(placeholder("No account activity available.", false, "chart-placeholder"));
    target.setAttribute("aria-label", "No account activity available.");
    return;
  }
  const width = 800, height = 205, left = 55, right = 20, top = 18, bottom = 36;
  const svg = svgElement("svg", { viewBox: `0 0 ${width} ${height}`, preserveAspectRatio: "none", "aria-hidden": "true" });
  const defs = svgElement("defs", {});
  const gradient = svgElement("linearGradient", { id: "timeline-fill", x1: 0, y1: 0, x2: 0, y2: 1 });
  gradient.append(svgElement("stop", { offset: 0, class: "area-top" }), svgElement("stop", { offset: 1, class: "area-bottom" }));
  defs.append(gradient);
  svg.append(defs);
  const times = timeline.map((row) => new Date(row.timestamp).valueOf());
  const amounts = timeline.map((row) => Math.max(0.01, Number(row.amount)));
  const log = Math.max(...amounts) / Math.min(...amounts) > 30;
  const values = amounts.map((value) => log ? Math.log10(value) : value);
  const minTime = Math.min(...times), maxTime = Math.max(...times);
  const minValue = Math.min(...values), maxValue = Math.max(...values);
  const x = (time, index) => maxTime === minTime
    ? left + (index / Math.max(1, times.length - 1)) * (width - left - right)
    : left + ((time - minTime) / (maxTime - minTime)) * (width - left - right);
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
  const linePoints = timeline.map((row, index) => `${x(times[index], index)},${y(values[index])}`).join(" ");
  svg.append(svgElement("polygon", {
    points: `${left},${height - bottom} ${linePoints} ${x(times[times.length - 1], times.length - 1)},${height - bottom}`,
    fill: "url(#timeline-fill)", stroke: "none",
  }));
  const polyline = svgElement("polyline", { points: linePoints, class: "line", "stroke-linejoin": "round" });
  svg.append(polyline);
  const flaggedIds = new Set(accountFlags.map((flag) => flag.transaction_id));
  timeline.forEach((row, index) => {
    const classes = ["point"];
    if (flaggedIds.has(row.transaction_id)) classes.push("flagged");
    if (selected && row.transaction_id === selected.transaction_id) classes.push("selected");
    const point = svgElement("circle", {
      cx: x(times[index], index), cy: y(values[index]),
      r: classes.includes("selected") ? 6 : classes.includes("flagged") ? 4 : 2.5,
      class: classes.join(" "),
    });
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
  target.setAttribute("aria-label",
    `${timeline.length} account transactions over time. ${log ? "Logarithmic" : "Linear"} amount scale. Selected transaction highlighted.`);
}

function highlight(ref, active) {
  let transaction = ref;
  for (const row of $("records").children) {
    if (row.dataset.ref !== ref) continue;
    row.classList.toggle("record-active", active);
    transaction = row.dataset.transaction || ref;
  }
  for (const point of $("chart").querySelectorAll("circle")) {
    if (point.dataset.ref === transaction) point.classList.toggle("cited", active);
  }
}

function issueText(issue) {
  const descriptions = {
    unknown_ref: "The cited record is not available for this report",
    ref_mismatch: "This detail appears in another record, but not the cited one",
    not_found: "This detail was not found in the available records",
  };
  return `${descriptions[issue.reason] || issue.reason.replaceAll("_", " ")}: ${issue.span}`;
}

function recordText(record) {
  if (!record) return "Record unavailable";
  if (record.timestamp && record.merchant_category && record.amount !== undefined) {
    const merchant = record.untrusted_text && record.untrusted_text.merchant_name;
    return [
      shortDate(record.timestamp),
      `${record.currency || ""} ${Number(record.amount).toLocaleString(undefined, { minimumFractionDigits: 2 })}`.trim(),
      merchant || record.merchant_category.replaceAll("_", " "),
      record.country,
      record.channel.replaceAll("_", " "),
    ].filter(Boolean).join(" · ");
  }
  if (record.meaning) return `${record.meaning}: ${record.value}`;
  if (record.reasons) {
    const currency = record.currency || selected?.transaction.currency;
    const reasons = record.reasons.map((reason) => reasonFormatting.readableReason(reason, currency).signals.join(" · ")).join("; ");
    return `${shortDate(record.timestamp)} · score ${Math.round(record.score * 100)}% · ${reasons}`;
  }
  if (typeof record.value === "string" && record.ref.startsWith("flag.reason.")) {
    return reasonFormatting.readableReason(record.value, record.currency || selected?.transaction.currency).signals.join(" · ");
  }
  if (record.value !== undefined) {
    const label = record.ref.split(".").slice(1).join(" ").replaceAll("_", " ");
    return `${label}: ${Array.isArray(record.value) ? record.value.join(", ") : record.value}`;
  }
  if (record.untrusted_text) return Object.values(record.untrusted_text).flat().filter(Boolean).join(", ");
  return "Record available";
}

function emptyRecords(text, loading = false) {
  const row = element("tr");
  const cell = element("td", "record-placeholder-cell");
  cell.colSpan = 2;
  cell.append(placeholder(text, loading, "records-placeholder"));
  row.append(cell);
  $("records").replaceChildren(row);
}

function drawReport(payload, loading = false) {
  const target = $("report"), records = $("records");
  target.replaceChildren();
  records.replaceChildren();
  const badge = $("report-grounding");
  badge.hidden = true;
  if (!payload || !payload.explanation) {
    $("report-meta").textContent = payload && payload.failure
      ? `Report unavailable · ${payload.failure}` : loading ? "Loading report…" : "No report generated";
    target.append(placeholder(payload && payload.failure
      ? "The report could not be completed. Try generating it again."
      : loading ? "Loading report and cited records…" : "Generate a report to review the evidence and cited records.",
    loading, "report-placeholder"));
    emptyRecords(loading ? "Loading cited records…" : "No cited records yet.", loading);
    return;
  }
  const report = payload.explanation;
  const latency = payload.latency_ms;
  const duration = typeof latency === "number" && Number.isFinite(latency)
    ? `${(latency / 1000).toFixed(2)} s` : "latency unavailable";
  $("report-meta").textContent = `${payload.model || "Unknown model"} · ${duration}`;
  badge.hidden = false;
  badge.classList.toggle("unverified", !payload.grounded);
  badge.textContent = payload.grounded ? "All claims verified against records" : "Some claims need review";
  const issues = payload.grounding ? payload.grounding.ungrounded_items || [] : [];
  const lead = element("div", "report-lead");
  const summaryIssues = issues.filter((issue) => issue.claim_index === 0);
  if (summaryIssues.length) lead.classList.add("rejected");
  lead.append(element("p", "report-summary", report.summary));
  for (const issue of summaryIssues) lead.append(element("p", "claim-warning", issueText(issue)));
  target.append(lead);
  const stats = element("div", "report-stats");
  const statsItems = [
    ["Risk", report.risk_level],
    ["Action", report.recommended_action.replaceAll("_", " ")],
  ];
  for (const [label, value] of statsItems) {
    const pill = element("span");
    pill.dataset.value = value;
    pill.append(element("strong", "", `${label}: `), document.createTextNode(value));
    stats.append(pill);
  }
  const confidence = element("div", "confidence-stat");
  const confidencePill = element("span");
  confidencePill.append(element("strong", "", "Confidence: "),
    document.createTextNode(`${Math.round(report.confidence * 100)}%`));
  const meter = element("progress", "confidence-meter");
  meter.max = 1;
  meter.value = report.confidence;
  meter.setAttribute("aria-label", "Report confidence");
  confidence.append(confidencePill, meter);
  stats.append(confidence);
  target.append(stats, element("h4", "report-section-title", "Evidence claims"));
  report.evidence.forEach((claim, index) => {
    const box = element("div", "claim");
    const rejected = issues.filter((issue) => issue.claim_index === index + 1);
    if (rejected.length) box.classList.add("rejected");
    const heading = element("div", "claim-heading");
    heading.append(element("span", "claim-number", index + 1), element("span", "claim-text", claim.claim));
    box.append(heading);
    for (const issue of rejected) box.append(element("p", "claim-warning", issueText(issue)));
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
  target.append(element("h4", "report-section-title", "Limitations"), element("p", "limitations", report.limitations));
  const cited = new Set(report.evidence.flatMap((claim) => claim.refs));
  for (const ref of cited) {
    const record = payload.cited_records && payload.cited_records[ref];
    const row = element("tr");
    row.dataset.ref = ref;
    if (record && record.transaction_id) row.dataset.transaction = record.transaction_id;
    const reference = element("td", "", ref);
    const value = element("td", "record-value", recordText(record));
    reference.title = ref;
    value.title = value.textContent;
    row.append(reference, value);
    records.append(row);
  }
  if (!cited.size) emptyRecords("This report cites no records.");
}

async function selectFlag(flag) {
  selected = flag;
  drawQueue();
  $("empty-detail").hidden = true;
  $("detail").hidden = false;
  $("detail-kicker").textContent = `FLAG #${flag.id} · ${shortDate(flag.transaction.timestamp)}`;
  $("detail-title").textContent = flag.transaction.merchant_name;
  $("detail-title").title = flag.transaction.merchant_name;
  $("detail-amount").textContent = money(flag.transaction);
  $("detail-subtitle").textContent = `Account ${flag.transaction.account_id} · Transaction ${flag.transaction.transaction_id}`;
  $("detail-score").textContent = `${Math.round(flag.score * 100)}% score`;
  $("detail-score").classList.toggle("high", flag.score >= 0.8);
  drawFacts(flag.transaction);
  $("reasons").replaceChildren(...flag.reasons.map((reason) => reasonNode(reason, flag.transaction.currency)));
  message($("detail-message"), "Loading account context…");
  $("chart").replaceChildren(placeholder("Loading account activity…", true, "chart-placeholder"));
  $("chart").setAttribute("aria-label", "Loading account activity.");
  drawReport(null, true);
  const results = await Promise.allSettled([
    loadTimeline(flag),
    flag.explanation.exists ? api(`/v1/flags/${flag.id}/explanation`) : Promise.resolve(null),
  ]);
  if (!selected || selected.id !== flag.id) return;
  if (results[0].status === "rejected") {
    $("chart").replaceChildren(placeholder("Account activity could not be loaded.", false, "chart-placeholder"));
    $("chart").setAttribute("aria-label", "Account activity could not be loaded.");
  }
  if (results[1].status === "rejected") {
    drawReport({ failure: results[1].reason.message || "request failed" });
  } else {
    drawReport(results[1].value);
  }
  const failures = results.filter((result) => result.status === "rejected");
  message($("detail-message"), failures.length ? "Some details could not be loaded. Please try again." : "", Boolean(failures.length));
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
  drawReport(null, true);
  try {
    const payload = await api(`/v1/flags/${flagId}/explain`, { method: "POST" });
    if (selected && selected.id === flagId) {
      drawReport(payload);
      selected.explanation = { exists: true, grounded: payload.grounded };
      drawQueue();
      const status = payload.failure ? `Report could not be completed: ${payload.failure}` : "Report saved.";
      message($("detail-message"), status, Boolean(payload.failure));
    }
  } catch (failure) {
    if (selected && selected.id === flagId) drawReport({ failure: failure.message || "request failed" });
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
document.addEventListener("keydown", (event) => {
  if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
  if (!$("sign-in").hidden || !items.length) return;
  const tag = document.activeElement.tagName;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(tag)) return;
  const buttons = [...$("queue").querySelectorAll(".queue-item")];
  const focused = buttons.indexOf(document.activeElement);
  const current = focused >= 0 ? focused : items.findIndex((item) => selected && item.id === selected.id);
  const next = Math.max(0, Math.min(buttons.length - 1, current + (event.key === "ArrowDown" ? 1 : -1)));
  buttons[next].focus();
  event.preventDefault();
});
if (key) {
  showConsole();
  loadQueue();
  checkAudit();
} else showSignIn();
}
