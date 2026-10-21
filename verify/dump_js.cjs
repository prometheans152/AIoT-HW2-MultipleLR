// Runs the browser's analysis.js in Node and writes every number the report quotes
// to verify/js_results.json, so verify/crosscheck.py can recompute them independently.
const fs = require('node:fs');
const path = require('node:path');
const A = require('../analysis.js');

const ROOT = path.join(__dirname, '..');
const PROBLEMS = [
  { id: 'carprice', file: 'CarPrice_Assignment.csv', target: 'price' },
  { id: 'student', file: 'student-mat.csv', target: 'G3' },
  { id: 'student_noG1G2', file: 'student-mat.csv', target: 'G3', drop: ['G1', 'G2'] }
];

function load(problem) {
  const parsed = A.parseCSV(fs.readFileSync(path.join(ROOT, 'data', problem.file), 'utf8'));
  if (problem.drop) {
    parsed.headers = parsed.headers.filter(h => !problem.drop.includes(h));
    parsed.rows = parsed.rows.map(r => Object.fromEntries(parsed.headers.map(h => [h, r[h]])));
  }
  return A.prepareDataset(parsed, problem.target);
}

function modelSummary(analysis, p, split) {
  const intervals = A.intervalSummary(p.model, split.test, 0.95);
  return {
    k: p.k, cv: p.cv, names: p.names, indices: p.indices,
    intercept: p.model.intercept, coefficients: p.model.coefficients,
    train: { r2: p.training.r2, rmse: p.training.rmse },
    test: { r2: p.testing.r2, rmse: p.testing.rmse, mae: p.testing.mae },
    interval: {
      coverage: intervals.coverage, inside: intervals.inside, t: intervals.t, df: intervals.df,
      sigma: intervals.sigma, meanPIWidth: intervals.meanPIWidth, meanCIWidth: intervals.meanCIWidth,
      items: intervals.items.map(i => ({ id: i.id, fit: i.fit, leverage: i.leverage,
        piLow: i.piLow, piHigh: i.piHigh, ciLow: i.ciLow, ciHigh: i.ciHigh }))
    }
  };
}

const out = {};
for (const problem of PROBLEMS) {
  const started = Date.now();
  const ds = load(problem);
  const split = A.splitDataset(ds, 42);
  const comparison = A.compareAllMethods(ds, 42);
  const methods = {};
  for (const m of comparison.methods) {
    const a = comparison.results[m.id];
    methods[m.id] = {
      name: m.name, bestK: a.bestK, order: a.order.map(j => ds.features[j].name),
      best: modelSummary(a, a.best, split),
      curve: a.progression.map(p => ({ k: p.k, cv: p.cv, testR2: p.testing ? p.testing.r2 : null,
        testRMSE: p.testing ? p.testing.rmse : null, error: p.error }))
    };
  }
  const native = comparison.lassoNative;
  out[problem.id] = {
    file: problem.file, target: problem.target, drop: problem.drop || [],
    rows: ds.rows.length, rawFeatures: ds.columns.length, encodedFeatures: ds.features.length,
    ignored: ds.ignoredReasons, merged: ds.mergedLevels,
    features: ds.features.map(f => ({ name: f.name, source: f.source, kind: f.kind, level: f.level || null })),
    X: ds.rows.map(r => r.x), y: ds.rows.map(r => r.y), ids: ds.rows.map(r => r.id),
    trainIds: split.train.map(r => r.id), testIds: split.test.map(r => r.id),
    cvOrder: A.shuffledIndices(split.train.length, 2026),
    cvFolds: comparison.results.forward.foldCount,
    methods,
    lassoNative: native ? { alpha: native.alpha, k: native.k, cv: native.cv,
      test: { r2: native.testing.r2, rmse: native.testing.rmse } } : null,
    seconds: (Date.now() - started) / 1000
  };
  console.log(`${problem.id}: ${ds.rows.length} rows, ${ds.columns.length} raw / ${ds.features.length} encoded features, ` +
    Object.values(methods).map(m => `${m.name} k=${m.bestK} R2=${m.best.test.r2.toFixed(4)}`).join('; '));
}
fs.writeFileSync(path.join(__dirname, 'js_results.json'), JSON.stringify(out));
console.log('wrote verify/js_results.json');
