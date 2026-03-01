/**
 * SBSE Assignment 1 — Test Suite Reduction via Random Search (Node.js)
 *
 * Implements exactly TWO random-search variants:
 *   A) Pure Random Sampling (independent samples; best-so-far)
 *   C) Elitist Random Search (mutate incumbent best; accept only if improved)
 *
 * Logs:
 *  - Per run: CSV of best-so-far fitness over evaluations
 *  - Summary: CSV stats across seeds for each algorithm
 *
 * Usage:
 *   node run_experiments.js --data ./dataset.json --budget 10000 --seeds 30 --out ./results
 *
 * Optional knobs:
 *   --lambda 0.35  (time trade-off among feasible solutions)
 *   --p 0.35       (sampling probability used in random subset init for algorithm A and initial solutions)
 */

const fs = require("fs");
const path = require("path");

// ---------------- CLI ----------------
function getArg(name, defVal) {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx === -1) return defVal;
  const val = process.argv[idx + 1];
  return val === undefined ? defVal : val;
}

const DATA_PATH = getArg("data", path.join(__dirname, "dataset.json"));
const BUDGET = parseInt(getArg("budget", "10000"), 10);
const N_SEEDS = parseInt(getArg("seeds", "30"), 10);
const OUT_DIR = getArg("out", path.join(__dirname, "results"));
const LOG_DIR = path.join(OUT_DIR, "logs");

// Fitness weight for time (applies among feasible solutions)
const LAMBDA = parseFloat(getArg("lambda", "0.35"));
// Probability used to include each test when sampling random subset
const P_INCLUDE = parseFloat(getArg("p", "0.35"));

// ---------------- RNG (seeded) ----------------
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------- Helpers ----------------
function ensureDir(p) {
  if (!fs.existsSync(p)) fs.mkdirSync(p, { recursive: true });
}

function popcountBigInt(x) {
  // Brian Kernighan loop for BigInt
  let c = 0;
  while (x !== 0n) {
    x &= x - 1n;
    c++;
  }
  return c;
}

function naturalTestSort(a, b) {
  const na = parseInt(a.replace(/[^\d]/g, ""), 10);
  const nb = parseInt(b.replace(/[^\d]/g, ""), 10);
  return na - nb;
}

function median(arr) {
  const a = [...arr].sort((x, y) => x - y);
  const mid = Math.floor(a.length / 2);
  if (a.length % 2 === 0) return (a[mid - 1] + a[mid]) / 2;
  return a[mid];
}

function min(arr) {
  return arr.reduce((m, v) => (v < m ? v : m), Number.POSITIVE_INFINITY);
}

function max(arr) {
  return arr.reduce((m, v) => (v > m ? v : m), Number.NEGATIVE_INFINITY);
}

// ---------------- Load dataset ----------------
const raw = fs.readFileSync(DATA_PATH, "utf8");
const dataset = JSON.parse(raw);

const elements = dataset.elements; // ["e1".."e20"]
const tests = dataset.tests;       // [{id,time,covers:[...]}, ...]

const T = tests.length;
const E = elements.length;

// element -> bit index
const elemIndex = new Map(elements.map((e, i) => [e, i]));

// Precompute bitmask per test
const covMask = new Array(T).fill(0n);
const times = new Array(T).fill(0);
for (let i = 0; i < T; i++) {
  times[i] = tests[i].time;
  let mask = 0n;
  for (const e of tests[i].covers) {
    const idx = elemIndex.get(e);
    mask |= 1n << BigInt(idx);
  }
  covMask[i] = mask;
}
const FULL_MASK = (1n << BigInt(E)) - 1n;
const TOTAL_TIME_ALL = times.reduce((s, v) => s + v, 0);

// ---------------- Solution representation ----------------
// x: Uint8Array length T, x[i] in {0,1}
function cloneSol(x) {
  return new Uint8Array(x);
}

function solToIndices(x) {
  const idxs = [];
  for (let i = 0; i < T; i++) if (x[i]) idxs.push(i);
  return idxs;
}

function evaluate(x) {
  let mask = 0n;
  let timeSum = 0;
  let size = 0;
  for (let i = 0; i < T; i++) {
    if (x[i]) {
      mask |= covMask[i];
      timeSum += times[i];
      size++;
    }
  }
  const covCount = popcountBigInt(mask);
  const feasible = mask === FULL_MASK;

  // Normalized terms in [0,1]
  const covRatio = covCount / E;
  const timeNorm = TOTAL_TIME_ALL > 0 ? timeSum / TOTAL_TIME_ALL : 0;
  const sizeNorm = T > 0 ? size / T : 0;

  // Scalar fitness (maximize)
  // Feasible solutions ALWAYS outrank infeasible ones:
  // - Infeasible: in (-1, 0] depending on coverage
  // - Feasible: in (0, 1] because starts at ~1 and subtracts small terms
  let fitness;
  if (!feasible) {
    fitness = covRatio - 1.0; // [-1, 0)
  } else {
    // Encourage low time; tiny nudge for smaller size
    fitness = 1.0 - (LAMBDA * timeNorm) - (0.02 * sizeNorm);
  }

  return { fitness, feasible, covCount, timeSum, size, mask };
}

// ---------------- Initialization (random subset) ----------------
// Bernoulli include each test with probability P_INCLUDE; ensure non-empty.
function randomSolution(rng) {
  const x = new Uint8Array(T);
  let any = false;
  for (let i = 0; i < T; i++) {
    if (rng() < P_INCLUDE) {
      x[i] = 1;
      any = true;
    }
  }
  if (!any) {
    const k = Math.floor(rng() * T);
    x[k] = 1;
  }
  return x;
}

// ---------------- Operators (min 2) ----------------
function opFlip1(x, rng) {
  const y = cloneSol(x);
  const i = Math.floor(rng() * T);
  y[i] = y[i] ? 0 : 1;
  return y;
}

// Swap: remove one included test and add one excluded test (if possible)
function opSwap(x, rng) {
  const y = cloneSol(x);
  const inIdx = [];
  const outIdx = [];
  for (let i = 0; i < T; i++) {
    if (y[i]) inIdx.push(i);
    else outIdx.push(i);
  }

  // If subset empty or full, fall back to flip
  if (inIdx.length === 0 || outIdx.length === 0) return opFlip1(y, rng);

  const i = inIdx[Math.floor(rng() * inIdx.length)];
  const j = outIdx[Math.floor(rng() * outIdx.length)];
  y[i] = 0;
  y[j] = 1;

  // Ensure non-empty
  if (inIdx.length === 1) {
    // subset was size 1; after swap it's still size 1, ok
  }
  return y;
}

const OPERATORS = [
  { name: "flip1", fn: opFlip1 },
  { name: "swap", fn: opSwap },
];

function randomOperator(rng) {
  return OPERATORS[Math.floor(rng() * OPERATORS.length)];
}

// ---------------- Algorithm A: Pure Random Sampling ----------------
function runPureRandomSampling(seed) {
  const rng = mulberry32(seed);

  let bestX = null;
  let bestEval = null;

  // CSV log: eval,best_fitness,best_cov,best_time,best_size,is_feasible,best_selected_tests,best_included_elements
  let log = "eval,best_fitness,best_cov,best_time,best_size,is_feasible,best_selected_tests,best_included_elements\n";

  for (let evals = 1; evals <= BUDGET; evals++) {
    const x = randomSolution(rng);
    const ev = evaluate(x);

    if (bestEval === null || ev.fitness > bestEval.fitness) {
      bestX = x;
      bestEval = ev;
    }

    const bestSummary = bestSolutionSummary(bestX, bestEval);
    log += `${evals},${bestEval.fitness.toFixed(6)},${bestEval.covCount},${bestEval.timeSum.toFixed(4)},${bestEval.size},${bestEval.feasible ? 1 : 0},"${bestSummary.selected_tests}","${bestSummary.covered_elements}"\n`;
  }

  return { seed, bestX, bestEval, log };
}

// ---------------- Algorithm C: Elitist Random Search ----------------
function runElitistRandomSearch(seed) {
  const rng = mulberry32(seed);

  let incumbent = randomSolution(rng);
  let incEval = evaluate(incumbent);

  let bestX = incumbent;
  let bestEval = incEval;

  let log = "eval,best_fitness,best_cov,best_time,best_size,is_feasible,best_selected_tests,best_included_elements\n";

  for (let evals = 1; evals <= BUDGET; evals++) {
    const op = randomOperator(rng);
    const candidate = op.fn(incumbent, rng);
    const candEval = evaluate(candidate);

    // Elitist acceptance: accept only if strictly improved
    if (candEval.fitness > incEval.fitness) {
      incumbent = candidate;
      incEval = candEval;
    }

    if (incEval.fitness > bestEval.fitness) {
      bestX = incumbent;
      bestEval = incEval;
    }

    const bestSummary = bestSolutionSummary(bestX, bestEval);
    log += `${evals},${bestEval.fitness.toFixed(6)},${bestEval.covCount},${bestEval.timeSum.toFixed(4)},${bestEval.size},${bestEval.feasible ? 1 : 0},"${bestSummary.selected_tests}","${bestSummary.covered_elements}"\n`;
  }

  return { seed, bestX, bestEval, log };
}

// ---------------- Experiment Runner ----------------
ensureDir(OUT_DIR);
ensureDir(LOG_DIR);

const ALGS = [
  { key: "A_pure_random_sampling", run: runPureRandomSampling },
  { key: "C_elitist_random_search", run: runElitistRandomSearch },
];

function writeRunLog(algKey, seed, csvText) {
  const fp = path.join(LOG_DIR, `${algKey}_seed${seed}.csv`);
  fs.writeFileSync(fp, csvText, "utf8");
  return fp;
}

function bestSolutionSummary(bestX, bestEval) {
  const chosen = solToIndices(bestX).map((i) => tests[i].id).sort(naturalTestSort);
  const covered = [];
  for (let j = 0; j < E; j++) {
    const bit = 1n << BigInt(j);
    if ((bestEval.mask & bit) !== 0n) covered.push(elements[j]);
  }
  const missing = elements.filter((e) => !covered.includes(e));
  return {
    selected_tests: chosen.join(" "),
    covered_elements: covered.join(" "),
    missing_elements: missing.join(" "),
  };
}

const summaryRows = [];
summaryRows.push([
  "algorithm",
  "seed",
  "final_best_fitness",
  "final_best_cov",
  "final_best_time",
  "final_best_size",
  "final_best_feasible",
  "selected_tests",
  "included_elements",
].join(","));

for (const alg of ALGS) {
  console.log(`\n=== Running ${alg.key} for ${N_SEEDS} seeds (budget=${BUDGET}) ===`);

  for (let s = 1; s <= N_SEEDS; s++) {
    const seed = s; // deterministic seeds 1..N
    const res = alg.run(seed);

    writeRunLog(alg.key, seed, res.log);

    const bestSummary = bestSolutionSummary(res.bestX, res.bestEval);

    summaryRows.push([
      alg.key,
      seed,
      res.bestEval.fitness.toFixed(6),
      res.bestEval.covCount,
      res.bestEval.timeSum.toFixed(4),
      res.bestEval.size,
      res.bestEval.feasible ? 1 : 0,
      `"${bestSummary.selected_tests}"`,
      `"${bestSummary.covered_elements}"`,
    ].join(","));

    console.log(
      `seed=${seed} best_cov=${res.bestEval.covCount}/${E} feasible=${res.bestEval.feasible ? "Y" : "N"} time=${res.bestEval.timeSum.toFixed(2)} fitness=${res.bestEval.fitness.toFixed(4)}`
    );
  }
}

// Write summary.csv
const summaryPath = path.join(OUT_DIR, "summary.csv");
fs.writeFileSync(summaryPath, summaryRows.join("\n") + "\n", "utf8");

// Aggregate stats by algorithm (min/median/max)
const summary = fs.readFileSync(summaryPath, "utf8").trim().split("\n").slice(1)
  .map(line => {
    // naive CSV parse (safe enough: selected_tests is quoted, but we only need first 7 columns)
    const parts = line.split(",");
    return {
      algorithm: parts[0],
      seed: parseInt(parts[1], 10),
      fitness: parseFloat(parts[2]),
      cov: parseInt(parts[3], 10),
      time: parseFloat(parts[4]),
      size: parseInt(parts[5], 10),
      feasible: parseInt(parts[6], 10),
    };
  });

console.log("\n=== Aggregate Stats (fitness) ===");
for (const alg of ALGS) {
  const rows = summary.filter(r => r.algorithm === alg.key);
  const fits = rows.map(r => r.fitness);
  const covs = rows.map(r => r.cov);
  const feas = rows.map(r => r.feasible);

  console.log(`\n${alg.key}`);
  console.log(`  fitness min/median/max = ${min(fits).toFixed(4)} / ${median(fits).toFixed(4)} / ${max(fits).toFixed(4)}`);
  console.log(`  coverage min/median/max = ${min(covs)} / ${median(covs)} / ${max(covs)}`);
  console.log(`  feasible runs = ${feas.reduce((s,v)=>s+v,0)} / ${rows.length}`);
}

console.log(`\nWrote logs to: ${LOG_DIR}`);
console.log(`Wrote summary to: ${summaryPath}`);