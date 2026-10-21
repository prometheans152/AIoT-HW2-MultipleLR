const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const A = require('../analysis.js');

let assertions = 0;
function ok(condition, label) {assert.ok(condition, label);assertions++;}
function close(a, b, tol, label) {ok(Math.abs(a - b) <= tol, `${label}: ${a} vs ${b}`);}

const quoted = A.parseCSV('"a,b",target,category\r\n"1,000",3,"A"\r\n4,5,"B"\r\n7,8,"A"\r\n');
ok(quoted.headers[0] === 'a,b', 'Quoted heading');
ok(quoted.rows[0]['a,b'] === '1,000', 'Quoted comma value');

const synthetic = Array.from({length:75},(_,i)=>({x:[i,i*i+2],y:10 + 3*i - 0.25*(i*i+2)}));
const model = A.fitOLS(synthetic,[0,1]);
close(model.intercept, 10, 1e-6, 'OLS intercept');
close(model.coefficients[0], 3, 1e-8, 'OLS slope 1');
close(model.coefficients[1], -0.25, 1e-8, 'OLS slope 2');
close(A.evaluate(synthetic,model).rmse,0,1e-8,'OLS training residual');
ok(A.numpyPermutation(50,42).slice(0,10).join() ===
  [13,39,30,45,17,48,26,25,32,19].join(),
  'NumPy RandomState(42).permutation(50) compatibility');
const strongerL1 = A.fitLasso(synthetic,[0,1],100000);
ok(strongerL1.coefficients.every(x=>Math.abs(x)<1e-10),
  'A sufficiently high L1 penalty zeroes coefficients');

// Delimiter detection and the categorical rules (ID skip, rare-level merge).
const semi = A.parseCSV('a;b;y\n1;2;3\n4;5;6\n7;8;9\n');
ok(semi.delimiter === ';' && semi.headers.join() === 'a,b,y', 'Semicolon CSV detected');
{
  const levels = ['A','A','A','A','A','B','B','B','B','B','C','D'];
  const lines = ['row_id,color,x,y', ...levels.map((c,i)=>`${i+1},${c},${i},${2*i+(c==='A'?1:0)}`)];
  const lines40 = [...lines, ...Array.from({length:28},(_,i)=>`${i+13},${i%2?'A':'B'},${i+12},${2*i+30}`)];
  const ds = A.prepareDataset(A.parseCSV(lines40.join('\n')),'y');
  ok(ds.ignored.includes('row_id'), 'Unique *_id column is skipped');
  ok(ds.features.some(f=>f.name==='color='+A.OTHER_LEVEL), 'Levels seen < 5 times are merged into 其他');
  ok(ds.mergedLevels[0].levels.join()==='C,D', 'Merged levels are reported');
}

// Student-t quantiles against scipy.stats.t.ppf.
close(A.studentTQuantile(0.975,1),12.706204736174694,1e-8,'t(0.975, df=1)');
close(A.studentTQuantile(0.975,10),2.2281388519862744,1e-9,'t(0.975, df=10)');
close(A.studentTQuantile(0.975,135),1.9776922772392527,1e-9,'t(0.975, df=135)');
close(A.studentTQuantile(0.95,30),1.697260886593957,1e-9,'t(0.95, df=30)');

// Intervals against the simple-regression closed form:
// PI = ŷ ± t·s·√(1 + 1/n + (x0−x̄)²/Sxx), CI drops the leading 1.
{
  const rows = Array.from({length:30},(_,i)=>({x:[i],y:5+2*i+Math.sin(i*1.7)*3}));
  const m = A.fitOLS(rows,[0]), x0 = 40, n = rows.length;
  const xs = rows.map(r=>r.x[0]), xbar = xs.reduce((a,b)=>a+b)/n;
  const sxx = xs.reduce((a,x)=>a+(x-xbar)**2,0);
  const s = Math.sqrt(rows.reduce((a,r)=>a+(r.y-A.predict(m,r.x))**2,0)/(n-2));
  const t = A.studentTQuantile(0.975,n-2), fit = A.predict(m,[x0]);
  const h = 1/n + (x0-xbar)**2/sxx;
  const pi = A.predictionInterval(m,[x0],0.95);
  close(pi.leverage,h,1e-12,'Leverage h');
  close(pi.piLow,fit-t*s*Math.sqrt(1+h),1e-9,'PI lower bound');
  close(pi.piHigh,fit+t*s*Math.sqrt(1+h),1e-9,'PI upper bound');
  close(pi.ciHigh-pi.ciLow,2*t*s*Math.sqrt(h),1e-9,'CI width');
  ok(pi.piHigh-pi.piLow > pi.ciHigh-pi.ciLow,'PI is wider than CI');
}

// [file, target, rows, raw features used, encoded predictors]
const datasetPairs = [['50_Startups.csv','Profit',50,4,5],['CarPrice_Assignment.csv','price',205,23,38],
  ['student-mat.csv','G3',395,32,41]];
for(const [file,target,n,raw,p] of datasetPairs){
  const csv=fs.readFileSync(path.join(__dirname,'..','data',file),'utf8');
  const parsed=A.parseCSV(csv), ds=A.prepareDataset(parsed,target);
  ok(parsed.rows.length===n,`${file}: CSV row count`);
  ok(ds.rows.length===n,`${file}: usable row count`);
  ok(ds.columns.length===raw,`${file}: raw predictor count`);
  ok(ds.features.length===p,`${file}: encoded predictor count`);
  const s1=A.splitDataset(ds),s2=A.splitDataset(ds);
  ok(s1.test.map(x=>x.id).join()===s2.test.map(x=>x.id).join(),`${file}: deterministic split`);
  ok(s1.test.every(x=>!s1.train.includes(x)),`${file}: train/test do not overlap`);
  let prior;
  const comparison=A.compareAllMethods(ds,42);
  ok(comparison.methods.length===5,`${file}: five selection methods`);
  for(const method of comparison.methods){
    const res=comparison.results[method.id];
    ok(res.progression.length===p,`${file}: every k explored for ${method.id}`);
    ok(res.best.k>=1 && res.best.k<=p,`${file}: chosen k exists`);
    ok(Number.isFinite(res.best.cv) && Number.isFinite(res.best.testing.rmse),`${file}: metrics are finite`);
    ok(res.best.testing.actual.length===s1.test.length,`${file}: holdout sized correctly`);
    ok(res.split.test.map(x=>x.id).join()===s1.test.map(x=>x.id).join(),`${file}: same test samples across algorithms`);
    if(prior) ok(res.split.test.map(x=>x.id).join()===prior,`${file}: algorithm-independent split`);
    prior=res.split.test.map(x=>x.id).join();
    console.log(`${file} | ${method.id.padEnd(11)} | best k=${res.bestK} | CV RMSE=${res.best.cv.toFixed(4)} | holdout RMSE=${res.best.testing.rmse.toFixed(4)} | R²=${res.best.testing.r2.toFixed(4)}`);
  }
  ok(comparison.lassoNative.k>=1 && comparison.lassoNative.k<=p,`${file}: native LassoCV nonzero k`);
  ok(Number.isFinite(comparison.lassoNative.testing.rmse),`${file}: native LassoCV RMSE finite`);
  ok(comparison.lassoNative.model.type==='LassoL1',`${file}: native LassoCV uses actual L1 model`);
  const rp=comparison.results.pearson.progression;
  const rs=comparison.results.selectkbest.progression;
  // Large k on collinear one-hot columns can be inestimable; both rankings must agree on that too.
  rp.forEach((v,i)=>v.testing ? close(v.testing.rmse,rs[i].testing.rmse,1e-8,`${file}: Pearson equals f_regression ranking at k=${i+1}`)
    : ok(!rs[i].testing,`${file}: both rankings inestimable at k=${i+1}`));
  if(file==='CarPrice_Assignment.csv'){
    ok(ds.ignored.join()==='car_ID,CarName','Car Price: ID and 147-level CarName are skipped');
    const f=comparison.results.forward, pe=comparison.results.pearson;
    ok(f.bestK===28 && pe.bestK===18,'Car Price: CV-chosen k (forward 28, Pearson 18)');
    close(f.best.testing.rmse,4026.0013838525297,1e-6,'Car Price forward holdout RMSE (= sklearn)');
    close(pe.best.testing.r2,0.8466269321608535,1e-9,'Car Price Pearson holdout R² (= sklearn)');
    const iv=A.intervalSummary(f.best.model,s1.test,0.95);
    ok(iv.inside===35 && iv.items.length===41,'Car Price forward model: 35 of 41 inside 95% PI');
    close(iv.t,1.9776922772392527,1e-9,'Car Price t with df=135');
  }
  if(file==='student-mat.csv'){
    ok(parsed.delimiter===';','Student CSV uses semicolons');
    const f=comparison.results.forward;
    ok(f.bestK===19,'Student: forward CV-chosen k');
    close(f.best.testing.r2,0.7499165329502806,1e-9,'Student forward holdout R² (= sklearn)');
    const iv=A.intervalSummary(f.best.model,s1.test,0.95);
    ok(iv.inside===74 && iv.items.length===79,'Student forward model: 74 of 79 inside 95% PI');
  }
  if(file==='50_Startups.csv'){
    ok(s1.test.map(r=>r.id).join() === [14,40,31,46,18,49,27,26,33,20].join(),
      '50 Startups holdout matches sklearn train_test_split random_state=42');
    close(rp[0].testing.rmse,7714.334890830185,1e-6,'Seed 42 comparison k=1 RMSE');
    close(rp[0].testing.r2,0.9265108109341951,1e-9,'Seed 42 comparison k=1 R2');
    close(rp[1].testing.rmse,8206.3288,1e-3,'Seed 42 Pearson k=2 RMSE');
    close(rp[2].testing.rmse,8242.783,1e-3,'Seed 42 Pearson k=3 RMSE');
    close(comparison.results.rfe.progression[2].testing.rmse,8995.9058,1e-3,
      'Seed 42 RFE k=3 RMSE');
  }
  const baseline = A.baselineRegression(ds,0);
  ok(baseline.seed===0,`${file}: baseline uses seed 0`);
  ok(baseline.progression.length===Math.min(5,p),`${file}: baseline line has up to five points`);
  ok(baseline.progression.every(row=>row.testing && Number.isFinite(row.testing.rmse)
    && Number.isFinite(row.testing.r2)),`${file}: computed baseline values finite`);
  if(file==='50_Startups.csv'){
    const expectedRMSE=[8274.868018,8198.797191,8309.059683,8409.916714,9137.990153];
    const expectedR2=[0.946459,0.947439,0.946015,0.944697,0.934707];
    const order=['R&D Spend','Marketing Spend','State=New York','State=Florida','Administration'];
    ok(baseline.split.test.map(row=>row.id).join()===
      [29,12,11,42,3,28,39,32,23,5].join(),
      'Baseline matches sklearn train_test_split random_state=0');
    ok(baseline.exactOrder, 'Baseline uses expected five-feature order');
    for(let i=0;i<5;i++){
      close(baseline.progression[i].testing.rmse,expectedRMSE[i],0.0000005,
        `Baseline RMSE k=${i+1} (six decimal places)`);
      close(baseline.progression[i].testing.r2,expectedR2[i],0.0000005,
        `Baseline R² k=${i+1} (six decimal places)`);
      ok(baseline.progression[i].names[i]===order[i],
        `Baseline k=${i+1} feature name corresponds to model input`);
    }
    // A redundant California dummy adds no independent information once the
    // other state indicators and intercept are already included.
    ok(baseline.progression[4].names[4]==='Administration',
      'Baseline k=5 selects an independent predictor');
    ok(baseline.featureNote.includes('State_California'),
      'Document the dummy-variable collinearity constraint');
  }
}
console.log(`PASS (${assertions} assertions)`);
