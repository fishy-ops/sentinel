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

const chartMath = (() => {
  const positive = (values) => values.map(Number).filter((value) => Number.isFinite(value) && value > 0);
  function scaleChoice(values) {
    const amounts = positive(values);
    return amounts.length && Math.max(...amounts) / Math.min(...amounts) > 20 ? "log" : "linear";
  }
  function niceTicks(values, scale = scaleChoice(values)) {
    const amounts = positive(values);
    if (!amounts.length) return [];
    const min = Math.min(...amounts), max = Math.max(...amounts);
    if (scale === "log") {
      const ticks = [];
      for (let power = Math.floor(Math.log10(min)); power <= Math.ceil(Math.log10(max)); power++) ticks.push(10 ** power);
      return ticks;
    }
    if (min === max) return min === 0 ? [0, 1] : [Math.max(0, min * 0.5), min, min * 1.5];
    const rough = (max - min) / 4;
    const magnitude = 10 ** Math.floor(Math.log10(rough));
    const step = [1, 2, 5, 10].find((multiple) => multiple * magnitude >= rough) * magnitude;
    const ticks = [];
    for (let index = Math.floor(min / step); index <= Math.ceil(max / step); index++) ticks.push(Number((index * step).toPrecision(12)));
    return ticks;
  }
  function percentile(values, fraction) {
    const sorted = positive(values).sort((a, b) => a - b);
    if (!sorted.length) return null;
    const position = Math.max(0, Math.min(1, fraction)) * (sorted.length - 1);
    const lower = Math.floor(position);
    return sorted[lower] + (sorted[Math.ceil(position)] - sorted[lower]) * (position - lower);
  }
  function dateTicks(start, end) {
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return [];
    const day = 86400000;
    const span = end - start;
    const candidates = span < 2 * day
      ? [5, 15, 30, 60, 120, 180, 360, 720, 1440].map((minutes) => minutes / 1440)
      : span < 14 * day ? [1, 2, 3, 7] : [7, 14, 28, 30, 60, 90, 180, 365];
    let best = [];
    let distance = Infinity;
    for (const days of candidates) {
      const ticks = [];
      if (days < 28) {
        const boundary = days >= 7 ? Date.UTC(1970, 0, 5) : 0;
        const interval = days * day;
        for (let time = Math.ceil((start - boundary) / interval) * interval + boundary; time <= end; time += interval) ticks.push(time);
      } else {
        const months = days >= 365 ? 12 : days >= 180 ? 6 : days >= 90 ? 3 : days >= 60 ? 2 : 1;
        const first = new Date(start);
        for (let year = first.getUTCFullYear(); year <= new Date(end).getUTCFullYear() + 1; year++) {
          for (let month = 0; month < 12; month += months) {
            const time = Date.UTC(year, month, 1);
            if (time >= start && time <= end) ticks.push(time);
          }
        }
      }
      const score = Math.abs(ticks.length - 5) + (ticks.length < 4 ? 2 : 0) + (ticks.length > 6 ? 2 : 0);
      if (score < distance) { best = ticks; distance = score; }
    }
    return best;
  }
  function labelPlacement(x, width, labelWidth, left = 72, right = 16) {
    return Math.max(left, Math.min(x + 9, width - right - labelWidth));
  }
  return { scaleChoice, niceTicks, percentile, dateTicks, labelPlacement };
})();

if (typeof module !== "undefined") module.exports = { ...reasonFormatting, ...chartMath };

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
let activeReference = null;

function reasonNode(reason, currency) {
  const readable = reasonFormatting.readableReason(reason, currency);
  const node = element("li", "reason-item");
  const source = readable.source === "Rule" && reason.includes(": ")
    ? reason.split(": ")[0].replace(/^./, (letter) => letter.toUpperCase()) : readable.source;
  node.append(element("span", "reason-source", `${source}: `), document.createTextNode(readable.signals.join("; ")));
  return node;
}

function element(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined && value !== null) node.textContent = String(value);
  return node;
}

function placeholder(label) { return element("p", "placeholder", label); }

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
  return state.grounded ? "Report verified" : "Report needs review";
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

function queueDate(value) {
  const time = new Date(value);
  return Number.isNaN(time.valueOf()) ? String(value) : time.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
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
    const merchant = element("strong", "queue-merchant", transaction.merchant_name);
    merchant.title = transaction.merchant_name;
    const level = flag.score >= 0.8 ? "high" : flag.score >= 0.5 ? "medium" : "low";
    top.append(element("span", `risk-dot ${level}`), merchant, element("span", "queue-amount", money(transaction)));
    const bottom = element("div", "queue-bottom");
    bottom.append(element("span", "queue-score", `Score ${Math.round(flag.score * 100)}`), element("span", "queue-date", queueDate(transaction.timestamp)));
    const badge = element("span", "report-status", reportBadge(flag.explanation));
    if (flag.explanation.exists && !flag.explanation.grounded) badge.classList.add("unverified");
    bottom.append(badge);
    button.append(top, bottom);
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
  $("queue").replaceChildren();
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
    else {
      message($("queue-message"), failure.message, true);
      $("queue").replaceChildren();
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
    target.append(placeholder("No account activity available."));
    target.setAttribute("aria-label", "No account activity available.");
    return;
  }
  const width = Math.max(240, Math.round(target.clientWidth || 800));
  const height = 260, left = 72, right = 16, top = 30, bottom = 35;
  const plotRight = width - right, plotBottom = height - bottom;
  const svg = svgElement("svg", { width, height, "aria-hidden": "true" });
  const times = timeline.map((row) => new Date(row.timestamp).valueOf());
  const amounts = timeline.map((row) => Math.max(0.01, Number(row.amount)));
  const scale = chartMath.scaleChoice(amounts);
  const ticks = chartMath.niceTicks(amounts, scale);
  const transform = (value) => scale === "log" ? Math.log10(value) : value;
  const lower = transform(ticks[0]), upper = transform(ticks[ticks.length - 1]);
  const minTime = Math.min(...times), maxTime = Math.max(...times);
  const timePadding = maxTime === minTime ? 7200000 : Math.max(3600000, (maxTime - minTime) * 0.04);
  const domainStart = minTime - timePadding, domainEnd = maxTime + timePadding;
  const x = (time) => left + ((time - domainStart) / (domainEnd - domainStart)) * (plotRight - left);
  const y = (value) => plotBottom - ((transform(value) - lower) / (upper - lower || 1)) * (plotBottom - top);
  const compact = (value) => value >= 1000 ? `${Number((value / 1000).toPrecision(3))}k` : `${Number(value.toPrecision(3))}`;
  const title = svgElement("text", { x: 0, y: 12, class: "axis-title" });
  title.textContent = `Amount, ${selected.transaction.currency}`;
  svg.append(title);
  for (const tick of ticks) {
    const lineY = y(tick);
    svg.append(svgElement("line", { x1: left, x2: plotRight, y1: lineY, y2: lineY, class: "grid" }));
    const label = svgElement("text", { x: left - 9, y: lineY + 4, "text-anchor": "end", class: "axis-label" });
    label.textContent = compact(tick);
    svg.append(label);
  }
  svg.append(svgElement("line", { x1: left, x2: plotRight, y1: plotBottom, y2: plotBottom, class: "axis" }));
  const dateTicks = chartMath.dateTicks(domainStart, domainEnd);
  const tickStride = Math.ceil(dateTicks.length / Math.max(2, Math.floor((plotRight - left) / 75)));
  for (const [index, time] of dateTicks.entries()) {
    if (index % tickStride) continue;
    const coordinate = x(time);
    svg.append(svgElement("line", { x1: coordinate, x2: coordinate, y1: plotBottom, y2: plotBottom + 4, class: "axis" }));
    const label = svgElement("text", { x: coordinate, y: height - 9, "text-anchor": "middle", class: "axis-label" });
    label.textContent = domainEnd - domainStart < 2 * 86400000
      ? new Date(time).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", timeZone: "UTC" })
      : new Date(time).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
    svg.append(label);
  }
  const prior = timeline.filter((row) => row.transaction_id !== selected.transaction_id).map((row) => Number(row.amount));
  if (prior.length >= 3) {
    for (const [name, fraction] of [["Median", 0.5], ["95th percentile", 0.95]]) {
      const value = chartMath.percentile(prior, fraction);
      const coordinate = y(value);
      svg.append(svgElement("line", { x1: left, x2: plotRight, y1: coordinate, y2: coordinate, class: "reference-line" }));
      const label = svgElement("text", { x: plotRight - 3, y: coordinate - 5, "text-anchor": "end", class: "reference-label" });
      label.textContent = `${name} ${value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
      svg.append(label);
    }
  }
  const flaggedIds = new Set(accountFlags.map((flag) => flag.transaction_id));
  timeline.forEach((row, index) => {
    const isSelected = row.transaction_id === selected.transaction_id;
    const coordinateX = x(times[index]), coordinateY = y(amounts[index]);
    if (isSelected) svg.append(svgElement("line", { x1: coordinateX, x2: coordinateX, y1: top, y2: plotBottom, class: "selected-guide" }));
    const classes = ["point", isSelected ? "selected" : flaggedIds.has(row.transaction_id) ? "flagged" : "ordinary"];
    const ring = svgElement("circle", { cx: coordinateX, cy: coordinateY, r: 8, class: "cited-ring" });
    ring.dataset.ref = row.transaction_id;
    svg.append(ring);
    const point = svgElement("circle", {
      cx: coordinateX, cy: coordinateY,
      r: isSelected ? 5 : classes.includes("flagged") ? 4 : 3,
      class: classes.join(" "),
    });
    point.dataset.ref = row.transaction_id;
    const tooltip = svgElement("g", { class: "chart-tooltip", visibility: "hidden" });
    const tooltipWidth = Math.min(220, width - left - right);
    const tooltipX = chartMath.labelPlacement(coordinateX, width, tooltipWidth, left, right);
    const tooltipY = coordinateY > 95 ? coordinateY - 74 : coordinateY + 12;
    tooltip.append(svgElement("rect", { x: tooltipX, y: tooltipY, width: tooltipWidth, height: 60, rx: 0 }));
    for (const [line, value] of [[0, shortDate(row.timestamp)], [1, money(row)], [2, row.merchant_name]]) {
      const text = svgElement("text", { x: tooltipX + 8, y: tooltipY + 16 + line * 17 });
      text.textContent = line === 2 && value.length > 28 ? `${value.slice(0, 27)}…` : value;
      tooltip.append(text);
    }
    const show = () => { point.setAttribute("r", isSelected ? 7 : 6); tooltip.setAttribute("visibility", "visible"); };
    const hide = () => { point.setAttribute("r", isSelected ? 5 : classes.includes("flagged") ? 4 : 3); tooltip.setAttribute("visibility", "hidden"); };
    point.addEventListener("mouseenter", show);
    point.addEventListener("mouseleave", hide);
    svg.append(point);
    if (isSelected) {
      const amount = money(row);
      // Sit beside the point at its height; flip to the left when there is no room on the right.
      const fits = coordinateX + 10 + amount.length * 7 <= width - right;
      const label = svgElement("text", {
        x: fits ? coordinateX + 10 : coordinateX - 10,
        y: Math.max(top + 10, coordinateY + 4),
        "text-anchor": fits ? "start" : "end",
        class: "selected-label",
      });
      label.textContent = amount;
      svg.append(label);
    }
    svg.append(tooltip);
  });
  target.append(svg);
  target.setAttribute("aria-label",
    `${timeline.length} account transactions over time. ${scale === "log" ? "Logarithmic" : "Linear"} amount scale. Selected transaction highlighted.`);
  if (activeReference) highlight(activeReference, true);
}

function highlight(ref, active) {
  activeReference = active ? ref : null;
  let transaction = ref;
  for (const row of $("records").children) {
    if (row.dataset.ref !== ref) continue;
    row.classList.toggle("record-active", active);
    transaction = row.dataset.transaction || ref;
  }
  for (const ring of $("chart").querySelectorAll(".cited-ring")) {
    if (ring.dataset.ref === transaction) ring.classList.toggle("active", active);
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
      : loading ? "Loading report and cited records…" : "Generate a report to review the evidence and cited records."));
    emptyRecords(loading ? "Loading cited records…" : "No cited records yet.", loading);
    return;
  }
  const report = payload.explanation;
  const latency = payload.latency_ms;
  const duration = typeof latency === "number" && Number.isFinite(latency)
    ? `${(latency / 1000).toFixed(2)} s` : "latency unavailable";
  $("report-meta").textContent = `${payload.model || "Unknown model"} · ${duration}`;
  badge.hidden = false;
  const issues = payload.grounding ? payload.grounding.ungrounded_items || [] : [];
  const claimCount = report.evidence.length + 1;
  const rejectedClaims = new Set(issues.map((issue) => issue.claim_index));
  badge.classList.toggle("unverified", !payload.grounded);
  badge.textContent = payload.grounded
    ? `Summary and all ${report.evidence.length} evidence claims verified against records`
    : `${rejectedClaims.size} of ${claimCount} statements (summary plus evidence) could not be verified`;
  const lead = element("div", "report-lead");
  const summaryIssues = issues.filter((issue) => issue.claim_index === 0);
  if (summaryIssues.length) lead.classList.add("rejected");
  lead.append(element("p", "report-summary", report.summary));
  for (const issue of summaryIssues) lead.append(element("p", "claim-warning", issueText(issue)));
  target.append(lead);
  const stats = element("p", "report-stats");
  const risk = element("span", `risk-word ${report.risk_level}`, report.risk_level);
  stats.append(document.createTextNode("Risk "), risk, document.createTextNode(` · Recommended action: ${report.recommended_action.replaceAll("_", " ")} · Confidence ${Math.round(report.confidence * 100)}%`));
  target.append(stats, element("h4", "report-section-title", "Evidence"));
  const evidence = element("ol", "evidence-list");
  report.evidence.forEach((claim, index) => {
    const box = element("li", "claim");
    const rejected = issues.filter((issue) => issue.claim_index === index + 1);
    if (rejected.length) box.classList.add("rejected");
    box.append(element("span", "claim-text", claim.claim));
    const refs = element("div", "claim-refs");
    claim.refs.forEach((ref, refIndex) => {
      if (refIndex) refs.append(document.createTextNode(", "));
      const link = element("button", "reference-link", ref);
      link.type = "button";
      link.addEventListener("mouseenter", () => highlight(ref, true));
      link.addEventListener("mouseleave", () => highlight(ref, false));
      link.addEventListener("focus", () => highlight(ref, true));
      link.addEventListener("blur", () => highlight(ref, false));
      refs.append(link);
    });
    box.append(refs);
    for (const issue of rejected) box.append(element("p", "claim-warning", issueText(issue)));
    evidence.append(box);
  });
  target.append(evidence);
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
  activeReference = null;
  drawQueue();
  $("empty-detail").hidden = true;
  $("detail").hidden = false;
  $("detail-title").textContent = flag.transaction.merchant_name;
  $("detail-title").title = flag.transaction.merchant_name;
  $("detail-amount").textContent = money(flag.transaction);
  const subtitle = $("detail-subtitle");
  subtitle.replaceChildren(document.createTextNode(`${shortDate(flag.transaction.timestamp)} · `),
    element("code", "", flag.transaction.account_id), document.createTextNode(" · "),
    element("code", "", flag.transaction.transaction_id));
  const score = element("span", `risk-value ${flag.score >= 0.8 ? "high" : flag.score >= 0.5 ? "medium" : "low"}`, Math.round(flag.score * 100));
  $("detail-score").replaceChildren(document.createTextNode("Risk score "), score);
  $("generate").textContent = flag.explanation.exists ? "Regenerate report" : "Generate report";
  drawFacts(flag.transaction);
  $("reasons").replaceChildren(...flag.reasons.map((reason) => reasonNode(reason, flag.transaction.currency)));
  message($("detail-message"), "Loading account context…");
  $("chart").replaceChildren(placeholder("Loading account activity…"));
  $("chart").setAttribute("aria-label", "Loading account activity.");
  drawReport(null, true);
  const results = await Promise.allSettled([
    loadTimeline(flag),
    flag.explanation.exists ? api(`/v1/flags/${flag.id}/explanation`) : Promise.resolve(null),
  ]);
  if (!selected || selected.id !== flag.id) return;
  if (results[0].status === "rejected") {
    $("chart").replaceChildren(placeholder("Account activity could not be loaded."));
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
      button.textContent = "Regenerate report";
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
    $("audit-status").textContent = status.ok ? "Audit log verified" : `Audit log broken at entry ${status.first_broken_entry_id}`;
    $("audit-strip").classList.toggle("bad", !status.ok);
    $("audit-meta").textContent = ` · ${status.entry_count} entries`;
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
$("score-filter").addEventListener("change", () => { offset = 0; loadQueue(); });
$("account-filter").addEventListener("change", () => { offset = 0; loadQueue(); });
$("previous").addEventListener("click", () => { offset = Math.max(0, offset - pageSize); loadQueue(); });
$("next").addEventListener("click", () => { offset += pageSize; loadQueue(); });
$("generate").addEventListener("click", generate);
// Redraw only when the width changes; drawing changes the height, which would retrigger the observer.
let chartWidth = 0;
if (typeof ResizeObserver !== "undefined") {
  new ResizeObserver((entries) => {
    const width = Math.round(entries[0].contentRect.width);
    if (width === chartWidth) return;
    chartWidth = width;
    if (timeline.length) drawChart();
  }).observe($("chart"));
}
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
