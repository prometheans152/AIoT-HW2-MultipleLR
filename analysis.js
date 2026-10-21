/* Dependency-free, reproducible multiple linear regression engine.
 * Works in a browser (window.MLR) and in Node.js unit tests (module.exports).
 */
(function (root) {
  'use strict';

  // Pick the delimiter that appears most often in the header line (outside quotes).
  function detectDelimiter(text) {
    const counts = { ',': 0, ';': 0, '\t': 0 };
    let quoted = false;
    for (const char of text) {
      if (char === '"') quoted = !quoted;
      else if (!quoted && (char === '\n' || char === '\r')) break;
      else if (!quoted && char in counts) counts[char]++;
    }
    return Object.keys(counts).sort((a, b) => counts[b] - counts[a])[0];
  }

  function parseCSV(csv) {
    const text = String(csv || '').replace(/^\uFEFF/, '');
    const delimiter = detectDelimiter(text);
    const records = [];
    let row = [], cell = '', quoted = false;
    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      if (quoted) {
        if (char === '"' && text[i + 1] === '"') { cell += '"'; i++; }
        else if (char === '"') quoted = false;
        else cell += char;
      } else if (char === '"') quoted = true;
      else if (char === delimiter) { row.push(cell.trim()); cell = ''; }
      else if (char === '\n' || char === '\r') {
        if (char === '\r' && text[i + 1] === '\n') i++;
        row.push(cell.trim()); cell = '';
        if (row.some(v => v !== '')) records.push(row);
        row = [];
      } else cell += char;
    }
    if (quoted) throw new Error('CSV 格式錯誤：引號沒有關閉');
    row.push(cell.trim());
    if (row.some(v => v !== '')) records.push(row);
    if (records.length < 4) throw new Error('CSV 至少需要標題列和 3 筆資料');
    const headers = records.shift().map((h, i) => h || `欄位${i + 1}`);
    if (new Set(headers).size !== headers.length) throw new Error('CSV 欄名不可重複');
    const rows = records.map(r => Object.fromEntries(headers.map((h, j) => [h, (r[j] || '').trim()])));
    return { headers, rows, delimiter };
  }

  function num(v) {
    if (v === null || v === undefined || String(v).trim() === '') return NaN;
    const n = Number(String(v).replace(/,/g, ''));
    return Number.isFinite(n) ? n : NaN;
  }
  function mean(v) { return v.reduce((a, b) => a + b, 0) / v.length; }
  function sum(v) { return v.reduce((a, b) => a + b, 0); }
  function std(v) { const m = mean(v); return Math.sqrt(mean(v.map(x => (x - m) ** 2))); }
  function median(v) { const s = [...v].sort((a, b) => a - b); return (s[Math.floor((s.length - 1) / 2)] + s[Math.floor(s.length / 2)]) / 2; }
  function unique(v) { return [...new Set(v)]; }

  const MAX_FEATURES = 60;   // after one-hot encoding
  const MAX_LEVELS = 10;    // categorical columns with more levels are skipped
  const RARE_MIN = 5;       // categories seen fewer times are merged into OTHER_LEVEL
  const OTHER_LEVEL = '其他';

  // Row identifiers (id, car_ID, customer_id ...) carry no signal and are skipped.
  function isIdColumn(name, values) {
    return /(^|[_\s])id$/i.test(name) && unique(values).length === values.length;
  }
  function levelOf(column, value) {
    return column.rare && column.rare.has(value) ? OTHER_LEVEL : value;
  }

  function describeColumns(parsed, targetName) {
    const columns = parsed.headers.map(name => {
      const values = parsed.rows.map(r => r[name]).filter(v => v !== '');
      const validNumbers = values.filter(v => Number.isFinite(num(v)));
      const numeric = values.length > 0 && validNumbers.length / values.length >= 0.95;
      const id = isIdColumn(name, values);
      const counts = new Map();
      for (const v of values) counts.set(v, (counts.get(v) || 0) + 1);
      // Merge rare categories only when there are at least two of them; one rare level keeps its name.
      const rareLevels = numeric ? [] : [...counts].filter(([, c]) => c < RARE_MIN).map(([v]) => v);
      const rare = new Set(rareLevels.length >= 2 ? rareLevels : []);
      const levels = unique(values.map(v => (rare.has(v) ? OTHER_LEVEL : v))).sort();
      const reason = id ? 'ID 欄位' : !numeric && counts.size > MAX_LEVELS ? `類別過多（${counts.size} 種）`
        : !numeric && levels.length < 2 ? '只有一種類別' : null;
      return { name, numeric, levels, rare, merged: [...rare], missing: parsed.rows.length - values.length,
        id, reason, supported: !reason && (numeric || levels.length >= 2) };
    });
    const viableTargets = columns.filter(c => c.numeric && !c.id).map(c => c.name);
    let suggested = viableTargets.find(x => /^profit$|^medv$|^price$|^g3$|^target$|^y$/i.test(x));
    if (!suggested) suggested = viableTargets[viableTargets.length - 1];
    if (targetName && viableTargets.includes(targetName)) suggested = targetName;
    return { columns, viableTargets, suggested };
  }

  function prepareDataset(parsed, targetName) {
    const desc = describeColumns(parsed, targetName);
    const target = targetName || desc.suggested;
    if (!desc.viableTargets.includes(target)) throw new Error('請選擇數值型的預測目標欄位');
    const columns = desc.columns.filter(c => c.name !== target && c.supported);
    if (columns.length < 1) throw new Error('沒有可用的特徵欄位');
    const features = [];
    for (const c of columns) {
      if (c.numeric) {
        features.push({ name: c.name, source: c.name, kind: 'number' });
      } else {
        for (const level of c.levels.slice(1)) {
          features.push({ name: `${c.name}=${level}`, source: c.name,
            kind: 'dummy', level, reference: c.levels[0] });
        }
      }
    }
    if (features.length > MAX_FEATURES) throw new Error(`編碼後有 ${features.length} 個特徵；目前最多支援 ${MAX_FEATURES} 個，請簡化資料欄位`);
    const byName = Object.fromEntries(columns.map(c => [c.name, c]));
    const valid = [], dropped = [];
    for (let index = 0; index < parsed.rows.length; index++) {
      const original = parsed.rows[index];
      const y = num(original[target]);
      const ok = Number.isFinite(y) && columns.every(c => c.numeric
        ? Number.isFinite(num(original[c.name]))
        : c.levels.includes(levelOf(c, original[c.name])));
      if (!ok) { dropped.push(index + 1); continue; }
      const x = features.map(f => f.kind === 'number' ? num(original[f.source])
        : Number(levelOf(byName[f.source], original[f.source]) === f.level));
      valid.push({ id: index + 1, x, y, original });
    }
    if (valid.length < 20) throw new Error(`有效資料只有 ${valid.length} 筆，至少需要 20 筆來建立訓練/測試集`);
    if (valid.length <= features.length + 10) throw new Error('資料筆數太少或特徵數過多，無法穩定評估');
    const skipped = desc.columns.filter(c => c.name !== target && !c.supported);
    return { parsed, target, features, columns, rows: valid, dropped,
      ignored: skipped.map(c => c.name),
      ignoredReasons: skipped.map(c => ({ name: c.name, reason: c.reason })),
      mergedLevels: columns.filter(c => c.merged.length).map(c => ({ name: c.name, levels: c.merged })) };
  }

  function seededRandom(seed) {
    let x = seed >>> 0;
    return () => {
      x += 0x6D2B79F5;
      let t = x;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function shuffledIndices(n, seed) {
    const out = Array.from({ length: n }, (_, i) => i), rnd = seededRandom(seed);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [out[i], out[j]] = [out[j], out[i]];
    }
    return out;
  }
  // NumPy RandomState-compatible MT19937 permutation, matching sklearn train_test_split.
  function numpyPermutation(n, seed = 42) {
    const mt = new Uint32Array(624);
    mt[0] = seed >>> 0;
    for (let i = 1; i < 624; i++) {
      const v = mt[i - 1] ^ (mt[i - 1] >>> 30);
      mt[i] = (Math.imul(1812433253, v) + i) >>> 0;
    }
    let index = 624;
    function nextUint32() {
      if (index === 624) {
        for (let i = 0; i < 624; i++) {
          const y = (mt[i] & 0x80000000) | (mt[(i + 1) % 624] & 0x7fffffff);
          mt[i] = (mt[(i + 397) % 624] ^ (y >>> 1) ^ ((y & 1) ? 0x9908b0df : 0)) >>> 0;
        }
        index = 0;
      }
      let y = mt[index++];
      y ^= y >>> 11;
      y ^= (y << 7) & 0x9d2c5680;
      y ^= (y << 15) & 0xefc60000;
      y ^= y >>> 18;
      return y >>> 0;
    }
    function interval(max) {
      let mask = max >>> 0;
      mask |= mask >>> 1; mask |= mask >>> 2; mask |= mask >>> 4;
      mask |= mask >>> 8; mask |= mask >>> 16;
      let value;
      do { value = nextUint32() & mask; } while (value > max);
      return value;
    }
    const order = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = interval(i);
      [order[i], order[j]] = [order[j], order[i]];
    }
    return order;
  }
  function splitDataset(dataset, seed = 42) {
    const order = numpyPermutation(dataset.rows.length, seed);
    const testN = Math.max(4, Math.ceil(order.length * 0.20));
    return { train: order.slice(testN).map(i => dataset.rows[i]),
      test: order.slice(0, testN).map(i => dataset.rows[i]), seed, testRatio: testN / order.length };
  }

  /** Mean-centered, variance-scaled QR least squares; no train/test leakage. */
  function fitOLS(rows, featureIndices) {
    const p = featureIndices.length, n = rows.length;
    if (n <= p + 1 || !p) throw new Error('OLS 要求資料筆數大於特徵數+1');
    const y = rows.map(r => r.y), yMean = mean(y);
    const mus = featureIndices.map(j => mean(rows.map(r => r.x[j])));
    const scales = featureIndices.map((j, i) => std(rows.map(r => r.x[j])));
    if (scales.some(s => !Number.isFinite(s) || s < 1e-12)) throw new Error('部分特徵沒有變異，無法估計係數');
    const q = [], R = Array.from({ length: p }, () => Array(p).fill(0));
    for (let j = 0; j < p; j++) {
      const v = rows.map(r => (r.x[featureIndices[j]] - mus[j]) / scales[j]);
      // Re-orthogonalize for additional numerical stability.
      for (let pass = 0; pass < 2; pass++) {
        for (let k = 0; k < j; k++) {
          let dot = 0;
          for (let i = 0; i < n; i++) dot += q[k][i] * v[i];
          R[k][j] += dot;
          for (let i = 0; i < n; i++) v[i] -= dot * q[k][i];
        }
      }
      const length = Math.sqrt(sum(v.map(x => x * x)));
      if (length < 1e-8) throw new Error('特徵高度共線，OLS 無法估計；請減少特徵');
      R[j][j] = length;
      q.push(v.map(x => x / length));
    }
    const wScaled = Array(p).fill(0);
    for (let j = p - 1; j >= 0; j--) {
      let rhs = 0;
      for (let i = 0; i < n; i++) rhs += q[j][i] * (y[i] - yMean);
      for (let k = j + 1; k < p; k++) rhs -= R[j][k] * wScaled[k];
      wScaled[j] = rhs / R[j][j];
    }
    const coefficients = wScaled.map((b, i) => b / scales[i]);
    const intercept = yMean - sum(coefficients.map((b, i) => b * mus[i]));
    const model = { featureIndices: [...featureIndices], intercept, coefficients, n, p };
    // Keep what interval estimates need: the R factor of the scaled design and the residual spread.
    const sse = sum(rows.map(r => (r.y - predict(model, r.x)) ** 2));
    const df = n - p - 1;
    return { ...model, R, mus, scales, sse, df, sigma: Math.sqrt(sse / df) };
  }
  function predict(model, x) {
    let prediction = model.intercept;
    for (let i = 0; i < model.featureIndices.length; i++) {
      prediction += model.coefficients[i] * x[model.featureIndices[i]];
    }
    return prediction;
  }

  // ln Γ(x), Lanczos approximation (g = 7, n = 9).
  function logGamma(x) {
    const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
      -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
    x -= 1;
    let a = c[0];
    const t = x + 7.5;
    for (let i = 1; i < 9; i++) a += c[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }
  // Regularized incomplete beta I_x(a, b) by Lentz's continued fraction.
  function incompleteBeta(x, a, b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const front = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    if (x > (a + 1) / (a + b + 2)) return 1 - incompleteBeta(1 - x, b, a);
    let f = 1, c = 1, d = 1 - (a + b) * x / (a + 1);
    d = Math.abs(d) < 1e-30 ? 1e-30 : d;
    d = 1 / d; f = d;
    for (let m = 1; m <= 300; m++) {
      for (const step of [0, 1]) {
        const num = step === 0
          ? m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
          : -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1));
        d = 1 + num * d; d = Math.abs(d) < 1e-30 ? 1e-30 : d; d = 1 / d;
        c = 1 + num / c; c = Math.abs(c) < 1e-30 ? 1e-30 : c;
        f *= c * d;
      }
      if (Math.abs(c * d - 1) < 1e-14) break;
    }
    return front * f / a;
  }
  function studentTCDF(t, df) {
    const tail = 0.5 * incompleteBeta(df / (df + t * t), df / 2, 0.5);
    return t >= 0 ? 1 - tail : tail;
  }
  // Inverse CDF of Student's t by bisection; prob in (0, 1).
  function studentTQuantile(prob, df) {
    if (!(df > 0) || !(prob > 0 && prob < 1)) return NaN;
    if (prob < 0.5) return -studentTQuantile(1 - prob, df);
    let lo = 0, hi = 1;
    while (studentTCDF(hi, df) < prob) hi *= 2;
    for (let i = 0; i < 200; i++) {
      const mid = (lo + hi) / 2;
      if (studentTCDF(mid, df) < prob) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
  }

  /**
   * OLS confidence interval (mean response) and prediction interval (new observation):
   * ŷ ± t(α/2, n−p−1) · s · √h  and  ŷ ± t · s · √(1 + h),
   * with leverage h = 1/n + ‖R⁻ᵀ z‖², z = the scaled, centered feature vector.
   */
  function predictionInterval(model, x, level = 0.95) {
    const fit = predict(model, x);
    const z = model.featureIndices.map((j, i) => (x[j] - model.mus[i]) / model.scales[i]);
    const v = Array(model.p).fill(0);
    for (let i = 0; i < model.p; i++) {
      let s = z[i];
      for (let k = 0; k < i; k++) s -= model.R[k][i] * v[k];
      v[i] = s / model.R[i][i];
    }
    const leverage = 1 / model.n + sum(v.map(t => t * t));
    const t = studentTQuantile(1 - (1 - level) / 2, model.df);
    const seFit = model.sigma * Math.sqrt(leverage), sePred = model.sigma * Math.sqrt(1 + leverage);
    return { fit, level, t, leverage, seFit, sePred,
      ciLow: fit - t * seFit, ciHigh: fit + t * seFit, piLow: fit - t * sePred, piHigh: fit + t * sePred };
  }
  function intervalSummary(model, rows, level = 0.95) {
    const items = rows.map(r => ({ id: r.id, actual: r.y, ...predictionInterval(model, r.x, level) }));
    const inside = items.filter(i => i.actual >= i.piLow && i.actual <= i.piHigh).length;
    return { level, items, coverage: inside / items.length, inside,
      meanPIWidth: mean(items.map(i => i.piHigh - i.piLow)),
      meanCIWidth: mean(items.map(i => i.ciHigh - i.ciLow)),
      t: items[0]?.t, sigma: model.sigma, df: model.df };
  }
  function evaluate(rows, model) {
    const actual = rows.map(r => r.y), predicted = rows.map(r => predict(model, r.x));
    const residual = actual.map((y, i) => y - predicted[i]), yBar = mean(actual);
    const sse = sum(residual.map(e => e ** 2));
    const sst = sum(actual.map(y => (y - yBar) ** 2));
    const r2 = sst > 1e-20 ? 1 - sse / sst : null;
    const nonzero = actual.map((y, i) => [y, residual[i]]).filter(([y]) => Math.abs(y) > 1e-12);
    return { rmse: Math.sqrt(sse / rows.length), mae: mean(residual.map(Math.abs)),
      r2, mape: nonzero.length ? mean(nonzero.map(([y, e]) => Math.abs(e / y))) * 100 : null,
      actual, predicted, residual, sse };
  }

  function cvRMSE(train, features, seed = 2026) {
    const folds = Math.min(5, Math.max(3, Math.floor(train.length / 8)));
    const order = shuffledIndices(train.length, seed);
    let errors = 0, count = 0;
    for (let k = 0; k < folds; k++) {
      const hold = [], fit = [];
      for (let i = 0; i < order.length; i++) {
        (i % folds === k ? hold : fit).push(train[order[i]]);
      }
      const m = fitOLS(fit, features);
      for (const row of hold) { errors += (row.y - predict(m, row.x)) ** 2; count++; }
    }
    return Math.sqrt(errors / count);
  }
  function pearsonAbs(a, b) {
    const am = mean(a), bm = mean(b);
    let cross = 0, aa = 0, bb = 0;
    for (let i = 0; i < a.length; i++) {
      const x = a[i] - am, y = b[i] - bm;
      cross += x * y; aa += x * x; bb += y * y;
    }
    return aa > 0 && bb > 0 ? Math.abs(cross / Math.sqrt(aa * bb)) : 0;
  }
  function fitLasso(rows, featureIndices, alpha, maxIterations = 3500) {
    const n = rows.length, p = featureIndices.length;
    if (n < 4 || !p || !Number.isFinite(alpha) || alpha < 0)
      throw new Error('Lasso 的資料或 alpha 不合法');
    const yMean = mean(rows.map(r => r.y));
    const mus = featureIndices.map(j => mean(rows.map(r => r.x[j])));
    const scales = featureIndices.map((j,i) => std(rows.map(r => r.x[j])));
    const columns = featureIndices.map((j,k) => rows.map(r =>
      scales[k] > 1e-12 ? (r.x[j] - mus[k]) / scales[k] : 0));
    const colNorms = columns.map(x => sum(x.map(v => v * v)) / n);
    const resid = rows.map(r => r.y - yMean);
    const w = Array(p).fill(0);
    for (let iter = 0; iter < maxIterations; iter++) {
      let biggest = 0;
      for (let j = 0; j < p; j++) {
        if (colNorms[j] < 1e-12) continue;
        const column = columns[j];
        let rho = 0;
        for (let i = 0; i < n; i++) rho += column[i] * (resid[i] + column[i] * w[j]);
        rho /= n;
        const next = Math.sign(rho) * Math.max(Math.abs(rho) - alpha, 0) / colNorms[j];
        const delta = next - w[j];
        if (delta !== 0) {
          w[j] = next;
          for (let i = 0; i < n; i++) resid[i] -= column[i] * delta;
          biggest = Math.max(biggest, Math.abs(delta));
        }
      }
      if (biggest < 1e-7) break;
    }
    const coefficients = w.map((v,j) => scales[j] > 1e-12 ? v / scales[j] : 0);
    const intercept = yMean - sum(coefficients.map((v,j) => v * mus[j]));
    return { featureIndices: [...featureIndices], intercept, coefficients,
      standardizedCoefficients: w, alpha, n, p, type: 'LassoL1' };
  }

  function lassoCrossValidation(train, p, seed = 42) {
    const indices = Array.from({length:p},(_,i)=>i);
    const yMean = mean(train.map(r => r.y));
    const xStats = indices.map(j => ({mu: mean(train.map(r => r.x[j])),
      scale: std(train.map(r => r.x[j]))}));
    const alphaMax = Math.max(...indices.map((j,k) => {
      const s = xStats[k].scale;
      return s > 1e-12 ? Math.abs(sum(train.map(r =>
        ((r.x[j] - xStats[k].mu) / s) * (r.y - yMean))) / train.length) : 0;
    }));
    if (!Number.isFinite(alphaMax) || alphaMax < 1e-10)
      throw new Error('LassoCV 無法處理沒有變異的資料');
    // 35 alpha values on a standard logarithmic Lasso path (alpha_max ... 0.001*alpha_max).
    // Training-only fold membership is fixed; the held-out test cannot leak into tuning.
    const grid = Array.from({length:35},(_,i) => alphaMax * Math.pow(0.001, i/34));
    const folds = Math.min(5, Math.max(3, Math.floor(train.length / 8)));
    const order = shuffledIndices(train.length, seed + 1984);
    const partitions = Array.from({length:folds},()=>({fit:[],hold:[]}));
    for (let k = 0; k < folds; k++) {
      for (let i = 0; i < order.length; i++) {
        partitions[k][i % folds === k ? 'hold' : 'fit'].push(train[order[i]]);
      }
    }
    let best = {cv:Infinity,alpha:grid[0]}, path=[];
    for (const alpha of grid) {
      let squared = 0, count = 0;
      for (const fold of partitions) {
        const m = fitLasso(fold.fit, indices, alpha, 1800);
        for (const row of fold.hold) { squared += (row.y - predict(m,row.x)) ** 2; count++; }
      }
      const cv = Math.sqrt(squared/count);
      path.push({alpha,cv});
      if (cv < best.cv) best = {alpha,cv};
    }
    const model = fitLasso(train, indices, best.alpha, 5000);
    const selected = model.standardizedCoefficients.filter(v => Math.abs(v) > 1e-5).length;
    return { ...best, model, selected, path, folds };
  }

  function selectFeatureOrder(train, p, method = 'forward', lasso = null) {
    const available = Array.from({ length: p }, (_, i) => i);
    if (method === 'correlation' || method === 'pearson' || method === 'selectkbest') {
      const y = train.map(r => r.y);
      function score(j) {
        const corr = pearsonAbs(train.map(r => r.x[j]), y);
        if (method !== 'selectkbest') return corr;
        const r2 = Math.min(corr*corr,1-1e-14);
        // f_regression: F = (r²/(1-r²)) * (n-2), univariate regression with intercept.
        return r2/(1-r2)*(train.length-2);
      }
      return available.sort((a, b) => score(b)-score(a) || a-b);
    }
    if (method === 'rfe') {
      let retained = [...available], discarded = [];
      while (retained.length > 1) {
        let model;
        try { model = fitOLS(train, retained); }
        catch (_) { // gracefully drop a collinear feature first if necessary
          discarded.push(retained.pop()); continue;
        }
        // RFE(LinearRegression) on standardized columns: squared scaled coefficients.
        const scores = retained.map((j,k) => ({j,score:Math.abs(model.coefficients[k]) *
          std(train.map(r => r.x[j]))}));
        scores.sort((a,b) => a.score - b.score || b.j-a.j);
        discarded.push(scores[0].j);
        retained = retained.filter(j => j !== scores[0].j);
      }
      return [...retained, ...discarded.reverse()];
    }
    if (method === 'lasso') {
      if (!lasso || !lasso.model) throw new Error('Lasso 需要已訓練的 LassoCV 模型');
      const weights = lasso.model.standardizedCoefficients;
      const y = train.map(r => r.y);
      return available.sort((a,b) => Math.abs(weights[b]) - Math.abs(weights[a])
        || pearsonAbs(train.map(r=>r.x[b]),y)-pearsonAbs(train.map(r=>r.x[a]),y)
        || a-b);
    }
    const cache = new Map();
    function loss(indices) {
      const key = [...indices].sort((a, b) => a - b).join(',');
      if (cache.has(key)) return cache.get(key);
      let score;
      try { score = cvRMSE(train, indices); }
      catch (_) { score = Infinity; }
      cache.set(key, score);
      return score;
    }
    if (method === 'backward') {
      let current = [...available];
      const removed = [];
      while (current.length > 1) {
        const candidates = current.map(j => ({ j, score: loss(current.filter(x => x !== j)) }));
        candidates.sort((a, b) => a.score - b.score || a.j - b.j);
        const chosen = candidates[0].j;
        current = current.filter(j => j !== chosen);
        removed.push(chosen);
      }
      return [...current, ...removed.reverse()];
    }
    if (method !== 'forward') throw new Error('不支援的特徵選取方法');
    let current = [], remaining = [...available];
    while (remaining.length) {
      const candidates = remaining.map(j => ({ j, score: loss([...current, j]) }));
      candidates.sort((a, b) => a.score - b.score || a.j - b.j);
      if (!Number.isFinite(candidates[0].score)) {
        current.push(...remaining); break; // full-size model may be rank deficient
      }
      current.push(candidates[0].j);
      remaining = remaining.filter(j => j !== candidates[0].j);
    }
    return current;
  }

  function analyze(dataset, method = 'forward', seed = 42) {
    const split = splitDataset(dataset, seed);
    const lasso = method === 'lasso' ? lassoCrossValidation(split.train,dataset.features.length,seed) : null;
    const order = selectFeatureOrder(split.train, dataset.features.length, method, lasso);
    const progression = order.map((_, k) => {
      const indices = order.slice(0, k + 1);
      let model = null, training = null, testing = null, cv = null, error = null;
      try {
        model = fitOLS(split.train, indices);
        training = evaluate(split.train, model);
        testing = evaluate(split.test, model);
        cv = cvRMSE(split.train, indices);
      } catch (e) { error = e.message; }
      return { k: k + 1, indices, names: indices.map(i => dataset.features[i].name),
        model, training, testing, cv, error };
    });
    const eligible = progression.filter(p => Number.isFinite(p.cv));
    if (!eligible.length) throw new Error('所有特徵組合均無法建立有效模型，請檢查資料');
    const best = [...eligible].sort((a, b) => a.cv - b.cv || a.k - b.k)[0];
    return { dataset, split, method, seed, order, progression, bestK: best.k,
      best, foldCount: Math.min(5, Math.max(3, Math.floor(split.train.length / 8))),
      lassoCV: lasso ? {alpha:lasso.alpha,cv:lasso.cv,k:lasso.selected,
        testing:evaluate(split.test,lasso.model),model:lasso.model} : null };
  }
  /**
   * Five-step baseline feature progression, evaluated with a reproducible
   * NumPy/sklearn-compatible 80:20 split and random_state=0.
   * Administration is a non-redundant fifth predictor; adding the California
   * dummy after the other two state dummies would be collinear with the intercept.
   */
  function baselineRegression(dataset, seed = 0) {
    const split = splitDataset(dataset, seed);
    const originalNames = [
      'R&D Spend', 'Marketing Spend', 'State=New York',
      'State=Florida', 'Administration'
    ];
    const exactOrder = dataset.target === 'Profit'
      && originalNames.every(name => dataset.features.some(f => f.name === name));
    const ordered = exactOrder
      ? originalNames.map(name => dataset.features.findIndex(f => f.name === name))
      : selectFeatureOrder(split.train, dataset.features.length, 'pearson').slice(0, 5);
    const progression = ordered.map((_, k) => {
      const indices = ordered.slice(0, k + 1);
      const names = indices.map(i => dataset.features[i].name);
      try {
        const model = fitOLS(split.train, indices);
        return { k: k + 1, indices, names, model, training: evaluate(split.train, model),
          testing: evaluate(split.test, model), error: null };
      } catch (e) {
        return { k: k + 1, indices, names, model: null, training: null,
          testing: null, error: e.message };
      }
    });
    return { dataset, seed, split, order: ordered, exactOrder, progression,
      featureNote: exactOrder
        ? '由於模型含截距及另外兩個州別編碼，State_California 會產生完全共線性；第五個獨立特徵使用 Administration。' : null };
  }
  const COMPARISON_METHODS = [
    {id:'pearson',name:'Pearson (Filter)',color:'#31679f'},
    {id:'selectkbest',name:'SelectKBest (Filter)',color:'#df9a2c'},
    {id:'rfe',name:'RFE (Wrapper)',color:'#2b8a54'},
    {id:'forward',name:'Forward SFS (Wrapper)',color:'#b2565a'},
    {id:'lasso',name:'Lasso (Embedded)',color:'#8760ab'}
  ];
  function compareAllMethods(dataset,seed=42) {
    const results = {};
    for (const method of COMPARISON_METHODS) results[method.id] = analyze(dataset,method.id,seed);
    return {methods:COMPARISON_METHODS,results,seed,
      lassoNative:results.lasso.lassoCV};
  }

  function calculateVIF(rows, indices) {
    return indices.map(j => {
      if (indices.length === 1) return 1;
      const others = indices.filter(i => i !== j);
      try {
        const training = rows.map(r => ({ x: r.x, y: r.x[j] }));
        const m = fitOLS(training, others), res = evaluate(training, m);
        return res.r2 === null || res.r2 >= 1 ? Infinity : 1 / (1 - res.r2);
      } catch (_) { return Infinity; }
    });
  }
  function rowFromInputs(dataset, inputs) {
    return dataset.features.map(f => f.kind === 'number' ? num(inputs[f.source])
      : Number(inputs[f.source] === f.level));
  }

  const api = { parseCSV, describeColumns, prepareDataset, splitDataset, shuffledIndices, numpyPermutation,
    fitOLS, fitLasso, lassoCrossValidation, predict, evaluate, cvRMSE, selectFeatureOrder,
    analyze, compareAllMethods, baselineRegression, COMPARISON_METHODS, calculateVIF,
    rowFromInputs, num, mean, median, studentTCDF, studentTQuantile, predictionInterval, intervalSummary,
    MAX_FEATURES, OTHER_LEVEL };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.MLR = api;
})(typeof window !== 'undefined' ? window : globalThis);
