// Acceptance-page regression harness — drives the REAL acceptance.html in a
// headless browser and asserts the intake rules' observable behavior. The
// page's logic is client-side JS, so this is the honest test surface: a
// Python re-implementation of the rules would drift from what a buyer's
// browser actually runs.
//
// Fixture: tests/fixtures/acceptance_two_sheets.xlsx is synthetic (SYN-
// prefixed ids, invented values) — two sheets, the maintenance one
// deliberately missing required_part_id so the no-part-link path is covered.
//
// Run locally:
//   AXM_CHROMIUM_PATH=/opt/pw-browsers/chromium node tests/pages/verify_acceptance.mjs
// In CI the `acceptance-page` job installs playwright + chromium first.

import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

async function loadPlaywright() {
  try { return await import("playwright"); }
  catch { return await import("/opt/node22/lib/node_modules/playwright/index.mjs"); }
}

const here = dirname(fileURLToPath(import.meta.url));
const PAGE = resolve(here, "../../acceptance.html");
const XLSX = resolve(here, "../fixtures/acceptance_two_sheets.xlsx");

let failures = 0;
function check(name, ok, detail = "") {
  console.log(`${ok ? "  ok " : "FAIL "} ${name}${!ok && detail ? " — " + detail : ""}`);
  if (!ok) failures++;
}

const { chromium } = await loadPlaywright();
const exe = process.env.AXM_CHROMIUM_PATH;
const browser = await chromium.launch(exe ? { executablePath: exe } : {});
const page = await browser.newPage();

const pageErrors = [];
const netRequests = [];
page.on("pageerror", (e) => pageErrors.push(String(e.message)));
page.on("request", (r) => { if (!r.url().startsWith("file://")) netRequests.push(r.url()); });

await page.goto("file://" + PAGE);
await page.waitForSelector("#findings .verdict", { timeout: 20000 });

// --- the worked example auto-runs and produces the canonical verdicts ---
const kinds = await page.$$eval("#findings .finding .fkind", (es) => es.map((e) => e.innerText.trim()));
check("worked example: 2 masked shortages lead, 1 supportable closes",
  JSON.stringify(kinds) === JSON.stringify([
    "ADMINISTRATIVE STOCK MASKS A PHYSICAL SHORTAGE",
    "ADMINISTRATIVE STOCK MASKS A PHYSICAL SHORTAGE",
    "SUPPORTABLE FROM STOCK ON RECORD",
  ]), JSON.stringify(kinds));

const headline = await page.$eval("#findings .verdict h2", (e) => e.innerText);
check("headline names the masked shortage and the down asset",
  headline.includes("None is physically available") && headline.includes("ASSET-A17"), headline);

check("every finding cites file/row/hash evidence",
  await page.$eval("#findings", (e) => e.innerText.includes("sha-256")));

const boxes = await page.$$eval("#summary .box b", (bs) => bs.map((b) => b.innerText.trim()));
check("surface tallies: 3 pass / 1 pass-with-gaps / 1 fail",
  JSON.stringify(boxes) === JSON.stringify(["3", "1", "1"]), JSON.stringify(boxes));

const statuses = await page.$$eval("#tableWrap tbody tr", (trs) =>
  trs.map((tr) => {
    const td = tr.querySelectorAll("td");
    return td[0].innerText.trim() + "=" + td[1].innerText.trim();
  }));
check("the deliberately-bad export FAILs",
  statuses.some((s) => s.startsWith("bad_availability_export.csv=FAIL")), JSON.stringify(statuses));

const asks = await page.$$eval("#asks li", (es) => es.map((e) => e.innerText));
check("asks are generated and end with the declared-gap standing rule",
  asks.length >= 2 && asks[asks.length - 1].includes("If a field is not tracked"), JSON.stringify(asks.length));

// --- honesty boundary and identity chrome ---
const bodyText = await page.$eval("body", (e) => e.innerText);
check("honesty boundary present (no shard minting claimed)",
  bodyText.includes("does not mint Genesis shards") && bodyText.includes("axm-hybrid1"));
check("dummy-data stamp present",  // .stamp renders uppercase, so match case-insensitively
  /dummy data/i.test(bodyText) && /not cui/i.test(bodyText));
check("no draft framing chrome in the handed-over page",
  (await page.$("#framingBar")) === null);
check("ecosystem section links the live properties",
  (await page.$$eval('section[data-screen-label="The system behind this page"] a', (as) => as.length)) >= 9);

// --- theme toggle ---
const before = await page.$eval("html", (e) => e.dataset.theme);
await page.click("#themeToggle");
const after = await page.$eval("html", (e) => e.dataset.theme);
check("theme toggle flips cream/void", before !== after, `${before} -> ${after}`);

// --- a real XLSX through the dependency-free reader ---
await page.setInputFiles("#fileInput", XLSX);
await page.click("#analyzeBtn");
await page.waitForFunction(
  () => document.getElementById("loadStatus").textContent.includes("Analyzed"),
  null, { timeout: 20000 });

const xlsxTypes = await page.$$eval("#tableWrap tbody tr td:nth-child(3)", (tds) => tds.map((t) => t.innerText.trim()));
check("XLSX: both sheets classified (inventory + maintenance)",
  JSON.stringify(xlsxTypes) === JSON.stringify(["inventory", "maintenance"]), JSON.stringify(xlsxTypes));

const xlsxKinds = await page.$$eval("#findings .finding .fkind", (es) => es.map((e) => e.innerText.trim()));
check("XLSX: missing part linkage surfaces as the finding, not a crash",
  JSON.stringify(xlsxKinds) === JSON.stringify(["BLOCKED WORK WITHOUT PART LINKAGE"]), JSON.stringify(xlsxKinds));

check("export buttons enabled after analysis",
  await page.$eval("#downloadJsonBtn", (b) => !b.disabled));

// --- the page's core promises ---
check("zero non-file network requests (offline by design)",
  netRequests.length === 0, JSON.stringify(netRequests));
check("no JS errors anywhere in the run",
  pageErrors.length === 0, JSON.stringify(pageErrors));

await browser.close();

if (failures) {
  console.error(`\n${failures} assertion(s) failed`);
  process.exit(1);
}
console.log("\nacceptance page: all assertions passed");
