"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const { featureFormatters, readableReason } = require("../src/sentinel/dashboard/dashboard.js");

test("model features use currency, counts, ratios, durations, and short boolean phrases", () => {
  assert.deepEqual(readableReason(
    "forest model: amount to median 218.50, sum 10m 58.56, count 10m 1.00, usual max 10m 7.00, account age 122.00, is online 1.00, is transfer 0.00",
    "GBP",
  ), {
    source: "Anomaly model",
    signals: [
      "218.5× the account's median amount",
      "58.56 GBP spent in the previous 10 minutes",
      "1 transaction in the previous 10 minutes",
      "Busiest 10 minutes on record: 7 transactions",
      "122 earlier transactions on the account",
      "Online purchase",
    ],
  });
  assert.equal(featureFormatters.sum_24h("1006.32", "USD"), "1,006.32 USD spent in the previous 24 hours");
  assert.equal(featureFormatters.seconds_since_previous("90"), "1.5 minutes since the previous transaction");
  assert.equal(featureFormatters.device_age_hours("48"), "2 days since this device was first used");
  assert.equal(featureFormatters.days_since_activity("0.5"), "12 hours since the previous transaction");
  assert.equal(featureFormatters.amount_zscore("-2.5"), "2.5 standard deviations below the account's average amount");
  assert.equal(featureFormatters.near_threshold("0"), null);
});

test("known rule formats remove stored dollar amounts and explain the signals", () => {
  const cases = [
    ["amount spike: 71.7x prior p95 on $7512.02", ["Amount is 71.7× the account's 95th-percentile amount"]],
    ["velocity: 7 transactions in 10 minutes (usual max 2)", ["7 transactions within 10 minutes (previous maximum 2)"]],
    ["account takeover: device and country first seen within 24 hours; 2 supporting signals, amount 1.5x prior median", ["New device and country within 24 hours", "2 supporting signals", "Amount: 1.5× the account's median amount"]],
    ["structuring: $9750.00 transfer below $10000; 2 prior near-threshold transfers in 7 days", ["Transfer just under the 10,000 reporting threshold", "2 similar transfers in the past 7 days"]],
    ["dormant drain: transfer after 17.0 inactive days at 5.9x prior p95", ["Transfer after 17 inactive days at 5.9× the 95th-percentile amount"]],
  ];
  for (const [raw, signals] of cases) {
    assert.deepEqual(readableReason(raw, "GBP"), { source: "Rule", signals });
  }
  assert.deepEqual(readableReason("new rule: unexpected format", "GBP"), {
    source: "Rule", signals: ["new rule: unexpected format"],
  });
});

test("unrecognized model signals retain their raw text", () => {
  assert.deepEqual(readableReason("supervised model: sum 24h 1006.32, is transfer 1.00, future field 8.00", "USD"), {
    source: "Supervised model",
    signals: ["1,006.32 USD spent in the previous 24 hours", "Transfer", "future field 8.00"],
  });
});

test("dashboard renders risk, timeline, and report grounding states", async () => {
  class Node {
    constructor(tag = "div") {
      this.tagName = tag.toUpperCase();
      this.children = [];
      this.dataset = {};
      this.attributes = {};
      this.listeners = {};
      this.className = "";
      this.hidden = false;
      this.value = "";
      this.classList = {
        add: (name) => { this.className = [...new Set([...this.className.split(" "), name])].filter(Boolean).join(" "); },
        toggle: (name, force) => {
          const present = this.className.split(" ").includes(name);
          const enabled = force === undefined ? !present : Boolean(force);
          this.className = this.className.split(" ").filter((item) => item && item !== name).concat(enabled ? name : []).join(" ");
          return enabled;
        },
        contains: (name) => this.className.split(" ").includes(name),
      };
    }
    set textContent(value) { this.children = []; this.text = String(value); }
    get textContent() { return this.text || this.children.map((child) => child.textContent).join(""); }
    append(...nodes) { this.text = ""; this.children.push(...nodes); }
    replaceChildren(...nodes) { this.text = ""; this.children = nodes; }
    setAttribute(name, value) {
      this.attributes[name] = String(value);
      if (name === "class") this.className = String(value);
    }
    addEventListener(event, handler) { this.listeners[event] = handler; }
    querySelectorAll(selector) {
      const matches = (node) => selector.startsWith(".")
        ? node.classList.contains(selector.slice(1)) : node.tagName.toLowerCase() === selector.toLowerCase();
      return this.children.flatMap((child) => [
        ...(matches(child) ? [child] : []), ...child.querySelectorAll(selector),
      ]);
    }
  }
  const ids = [
    "sign-in", "console", "sign-out", "audit-strip", "sign-in-error", "key-form", "api-key",
    "queue", "queue-count", "page-label", "previous", "next", "queue-message", "score-filter",
    "account-filter", "detail", "empty-detail", "facts", "reasons", "chart", "report",
    "records", "report-grounding", "report-meta", "detail-kicker", "detail-title", "detail-amount",
    "detail-subtitle", "detail-score", "detail-message", "generate", "filters", "audit-status", "audit-meta",
  ];
  const nodes = Object.fromEntries(ids.map((id) => [id, new Node()]));
  nodes["score-filter"].value = "0";
  nodes.console.hidden = true;
  const document = {
    getElementById: (id) => nodes[id],
    createElement: (tag) => new Node(tag),
    createElementNS: (_namespace, tag) => new Node(tag),
    addEventListener() {},
  };
  document.createTextNode = (value) => {
    const node = new Node("#text");
    node.textContent = value;
    return node;
  };
  const transaction = (id, amount, time) => ({
    transaction_id: id, account_id: "acct_1", amount, currency: "USD", timestamp: time,
    merchant_name: `Merchant ${id}`, merchant_category: "retail", country: "US",
    channel: "online", device_id: "dev_1", memo: "",
  });
  const history = [
    transaction("tx_1", 10, "2026-10-01T10:00:00Z"),
    transaction("tx_2", 20, "2026-10-01T11:00:00Z"),
    transaction("tx_3", 30, "2026-10-01T12:00:00Z"),
  ];
  const flags = [
    { id: 1, transaction_id: "tx_3", score: 0.9, transaction: history[2], reasons: ["review"], explanation: { exists: true, grounded: false } },
    { id: 2, transaction_id: "tx_2", score: 0.7, transaction: history[1], reasons: ["review"], explanation: { exists: false, grounded: false } },
  ];
  const report = {
    grounded: false, model: "local", latency_ms: 400,
    explanation: {
      summary: "Review the transaction.", risk_level: "high", recommended_action: "block_and_contact",
      confidence: 0.82, limitations: "Limited history.",
      evidence: [{ claim: "Unusual amount.", refs: ["tx_3"] }],
    },
    grounding: { ungrounded_items: [
      { claim_index: 0, reason: "not_found", span: "Review" },
      { claim_index: 1, reason: "ref_mismatch", span: "amount" },
    ] },
    cited_records: { tx_3: { ref: "tx_3", transaction_id: "tx_3", ...history[2] } },
  };
  const fetch = async (url) => {
    let payload;
    if (url.includes("/explanation")) payload = report;
    else if (url.includes("/history")) payload = { items: history };
    else if (url.includes("account_id=")) payload = { items: flags };
    else if (url.startsWith("/v1/flags?")) payload = { items: flags };
    else if (url === "/v1/audit/verify") payload = { ok: true, entry_count: 0, head_hash: null };
    else throw new Error(`Unexpected request: ${url}`);
    return { ok: true, json: async () => payload };
  };
  const script = fs.readFileSync(path.join(__dirname, "../src/sentinel/dashboard/dashboard.js"), "utf8");
  vm.runInNewContext(script, {
    document, fetch, sessionStorage: { getItem: () => "key" }, setInterval, clearInterval,
    URLSearchParams,
  });
  await new Promise(setImmediate);
  await new Promise(setImmediate);

  const queue = nodes.queue.querySelectorAll(".queue-item");
  assert.equal(queue.length, 2);
  assert.equal(queue[0].querySelectorAll(".queue-score")[0].classList.contains("high"), true);
  assert.equal(queue[1].querySelectorAll(".queue-score")[0].classList.contains("high"), false);
  assert.equal(nodes["detail-score"].classList.contains("high"), true);
  assert.equal(nodes["detail-amount"].textContent, "USD 30.00");
  assert.equal(nodes.facts.textContent.includes("Amount"), false);

  const svg = nodes.chart.querySelectorAll("svg")[0];
  const gradient = svg.querySelectorAll("linearGradient")[0];
  assert.equal(gradient.attributes.id, "timeline-fill");
  assert.deepEqual(gradient.children.map((stop) => stop.className), ["area-top", "area-bottom"]);
  const polygon = svg.querySelectorAll("polygon")[0];
  assert.equal(polygon.attributes.fill, "url(#timeline-fill)");
  assert.equal(polygon.attributes.stroke, "none");
  assert.equal(svg.children.indexOf(polygon) < svg.children.indexOf(svg.querySelectorAll("polyline")[0]), true);
  assert.deepEqual(svg.querySelectorAll("circle").map((point) => point.attributes.r), ["2.5", "4", "6"]);

  const stats = nodes.report.querySelectorAll(".report-stats")[0];
  assert.deepEqual(stats.querySelectorAll("span").map((pill) => pill.dataset.value), ["high", "block and contact", undefined]);
  assert.equal(stats.querySelectorAll("progress")[0].value, 0.82);
  assert.equal(nodes.report.querySelectorAll(".claim-number")[0].textContent, "1");
  assert.equal(nodes.report.querySelectorAll(".claim")[0].classList.contains("rejected"), true);
  assert.equal(nodes.report.querySelectorAll(".claim-warning").length, 2);
  assert.equal(nodes.report.querySelectorAll(".chip")[0].textContent, "tx_3");

  await queue[1].listeners.click();
  assert.equal(nodes["detail-score"].classList.contains("high"), false);
  assert.equal(nodes["detail-amount"].textContent, "USD 20.00");
  assert.equal(nodes.queue.querySelectorAll(".queue-item")[1].classList.contains("active"), true);
});
