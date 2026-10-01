"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
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
