/* Interactive regression platform: built-in datasets, custom CSV, models, and charts. */
(function () {
  'use strict';
  const A = window.MLR;
  const el = id => document.getElementById(id);
  const state = { parsed: null, dataset: null, analysis: null, comparison: null, baseline: null,
    selectedK: null, baselineSelectedK: null, nativeLassoSelected: false,
    sourceName: '', key: 'carprice', custom: null, defaults: {} };

  // The two HW2 problems offered in the dropdown. Both come from Kaggle and have 14–39 raw features.
  const PROBLEMS = {
    carprice: { file: 'CarPrice_Assignment.csv', prefix: '$', unit: '美元 ($)', target: 'price',
      title: 'Car Price Prediction', goal: '依車輛規格（引擎、車身尺寸、油耗等）預測美國市場車價 price。',
      kaggle: 'https://www.kaggle.com/datasets/hellbuoy/car-price-prediction' },
    student: { file: 'student-mat.csv', prefix: '', unit: '分（0–20）', target: 'G3',
      title: 'Student Performance (Math)', goal: '依學生背景、家庭、學習習慣與前兩次成績，預測數學期末成績 G3。',
      kaggle: 'https://www.kaggle.com/datasets/uciml/student-alcohol-consumption' }
  };

  function esc(v) { return String(v).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[c]); }
  function csvCell(v) {
    const s = String(v ?? '');
    return /[,\r\n"]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  }
  function fmt(n, places = 2) {
    if (n === null || n === undefined || !Number.isFinite(n)) return '—';
    return Number(n).toLocaleString('en-US', { maximumFractionDigits: places, minimumFractionDigits: places });
  }
  function currency(n) { return (PROBLEMS[state.key]?.prefix || '') + fmt(n, 2); }
  function unitName() { return PROBLEMS[state.key]?.unit || '原資料單位'; }
  function status(message, error = false) {
    el('dataStatus').textContent = message;
    el('dataStatus').classList.toggle('error', error);
    el('sourceBadge').textContent = error ? 'CHECK YOUR CSV' : 'MODEL READY';
    el('sourceBadge').style.background = error ? '#fff2f1' : '#edf7f3';
    el('sourceBadge').style.color = error ? '#bd5655' : '#23795e';
  }
  function getRaw(key) {
    if (key === 'custom') return state.custom;
    return { text: window.REGRESSION_BUILTIN[key], name: PROBLEMS[key].file };
  }
  function populateTarget(parsed, targetName) {
    const desc = A.describeColumns(parsed, targetName);
    const select = el('targetSelect');
    select.innerHTML = desc.viableTargets.map(x => `<option value="${esc(x)}">${esc(x)}</option>`).join('');
    if (!desc.viableTargets.length) throw new Error('沒有數值型目標欄位');
    select.value = desc.suggested;
  }
  function loadProblem(key, message) {
    const raw = getRaw(key);
    if (!raw || !raw.text) return;
    try {
      state.key = key;
      state.parsed = A.parseCSV(raw.text);
      state.sourceName = raw.name;
      populateTarget(state.parsed);
      rebuildDataset();
      if (message) status(message);
    } catch (e) { status(`載入失敗：${e.message}`, true); }
  }
  function rebuildDataset() {
    try {
      state.dataset = A.prepareDataset(state.parsed, el('targetSelect').value);
      runAnalysis();
    } catch (e) { status(`資料準備失敗：${e.message}`, true); }
  }
  function runAnalysis() {
    if (!state.dataset) return;
    const button = el('runButton');
    button.disabled = true;
    button.textContent = '正在分析資料…';
    // Leave an event-loop tick so large custom datasets show the progress label.
    window.setTimeout(() => {
      try {
        state.baseline = A.baselineRegression(state.dataset,0);
        state.comparison = A.compareAllMethods(state.dataset,42);
        state.analysis = state.comparison.results[el('methodSelect').value];
        state.selectedK = state.analysis.bestK;
        state.baselineSelectedK = null;
        state.nativeLassoSelected = false;
        syncKSelector();
        renderDatasetSummary();
        renderAll();
        status(`已完成基準分析 (seed=0) 與五方法比較 (seed=42)：${state.sourceName} · ${state.dataset.rows.length} 筆有效資料`);
      } catch (e) { status(`模型計算失敗：${e.message}`, true); console.error(e); }
      finally { button.disabled = false; button.innerHTML = '重新計算五種方法與全部圖表 <span aria-hidden="true">→</span>'; }
    }, 10);
  }
  function syncKSelector() {
    const select = el('kSelect');
    select.innerHTML = state.analysis.progression.map(x =>
      `<option value="${x.k}">k = ${x.k}${x.k === state.analysis.bestK ? ' ★ CV 最佳' : ''}</option>`).join('');
    select.value = String(state.selectedK);
  }
  function changeMethod() {
    if (!state.comparison) return;
    state.analysis = state.comparison.results[el('methodSelect').value];
    if (!state.analysis) return;
    state.selectedK = state.analysis.bestK;
    state.nativeLassoSelected = false;
    syncKSelector();
    renderAll();
  }
  function renderDatasetSummary() {
    const d = state.dataset, a = state.analysis;
    el('dataSummary').innerHTML = [
      `<span>Dataset<strong>${esc(state.sourceName)}</strong></span>`,
      `<span>有效樣本<strong>${d.rows.length}</strong></span>`,
      `<span>Train / Test<strong>${a.split.train.length} / ${a.split.test.length}</strong></span>`,
      `<span>原始特徵<strong>${d.columns.length}</strong></span>`,
      `<span>編碼後特徵<strong>${d.features.length}</strong></span>`,
      `<span>目標 Y<strong>${esc(d.target)}</strong></span>`,
      d.dropped.length ? `<span class="notice">已排除 ${d.dropped.length} 筆缺失／不合法資料。</span>` : '',
      d.ignoredReasons.length ? `<span class="notice">已略過欄位：${esc(d.ignoredReasons.map(c => `${c.name}（${c.reason}）`).join('、'))}</span>` : '',
      d.mergedLevels.length ? `<span class="notice">出現少於 5 次的類別併為「${esc(A.OTHER_LEVEL)}」：${esc(d.mergedLevels.map(c => `${c.name}：${c.levels.join('/')}`).join('；'))}</span>` : ''
    ].join('');
  }
  function current() { return state.analysis?.progression.find(x => x.k === state.selectedK); }
  function renderAll() {
    renderBaseline();
    renderMetricCards();
    renderProgression();
    renderCharts();
    renderComparisonTable();
    renderCoefficients();
    renderScatter();
    renderResiduals();
    renderIntervals();
    renderPredictor();
    renderCrispDM();
  }
  function formatFeatureName(name) {
    return name.replace(/^State=/,'State_');
  }
  function baselineLineChart(metric) {
    const baseline=state.baseline;
    const title=metric==='rmse'?'RMSE':'R-squared';
    const points=baseline.progression.map(p=>p.testing?.[metric]);
    const good=points.map((v,i)=>({k:i+1,value:v})).filter(x=>Number.isFinite(x.value));
    if(!good.length)return `<p>${esc(title)} 沒有可用的測試數據</p>`;
    const W=670,H=354,L=78,R=24,T=18,B=56;
    const min=Math.min(...good.map(p=>p.value)),max=Math.max(...good.map(p=>p.value));
    const slack=Math.max((max-min)*.09,Math.abs(max)*.0001,0.000001);
    const lo=min-slack, hi=max+slack;
    const x=k=>L+(k-1)/(Math.max(1,points.length-1))*(W-L-R);
    const y=v=>H-B-(v-lo)/(hi-lo)*(H-T-B);
    const tickLabel=v=>metric==='rmse' ? v.toFixed(Math.abs(v)<100?1:0) : v.toFixed(3);
    let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" xmlns="http://www.w3.org/2000/svg" aria-label="${esc(title)} by Number of Features — baseline regression">`;
    for(let j=0;j<=5;j++){
      const v=lo+(hi-lo)*j/5;
      svg+=`<line x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}" stroke="#e6e8ec" stroke-width="1"/>
        <text x="${L-11}" y="${y(v)+4}" text-anchor="end" fill="#687688" font-size="11">${esc(tickLabel(v))}</text>`;
    }
    svg+=`<line x1="${L}" y1="${T}" x2="${L}" y2="${H-B}" stroke="#8390a5"/>
      <line x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}" stroke="#8390a5"/>`;
    for(let k=1;k<=points.length;k++){
      svg+=`<text x="${x(k)}" y="${H-B+19}" text-anchor="middle" font-size="12" fill="#556477">${k}</text>`;
    }
    const series=good.map((p,i)=>`${i?'L':'M'}${x(p.k).toFixed(2)} ${y(p.value).toFixed(2)}`).join(' ');
    svg+=`<path d="${series}" fill="none" stroke="#1f77b4" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>`;
    for(const p of good){
      const selected=state.baselineSelectedK===p.k;
      const label=`基準分析 k=${p.k}，${title} ${p.value.toFixed(6)}。按 Enter 查看資料`;
      svg+=`<g class="chart-point ${selected?'chart-point-active':''}" data-chart-point="baseline" data-k="${p.k}"
        role="button" tabindex="0" aria-pressed="${selected}" aria-label="${esc(label)}">
        <circle class="chart-hit-area" cx="${x(p.k)}" cy="${y(p.value)}" r="15"/>
        ${selected?`<circle class="chart-selected-ring" cx="${x(p.k)}" cy="${y(p.value)}" r="9" stroke="#1f77b4"/>`:''}
        <circle cx="${x(p.k)}" cy="${y(p.value)}" r="4.3" fill="#1f77b4"/>
        <title>${esc(label)}</title></g>`;
    }
    svg+=`<text x="${(L+W-R)/2}" y="${H-9}" text-anchor="middle" font-size="13" fill="#384c69">Number of Features</text>
      <text x="19" y="${H/2}" transform="rotate(-90 19 ${H/2})" text-anchor="middle" font-size="13" fill="#384c69">${esc(title)}</text></svg>`;
    return svg;
  }
  function renderBaseline() {
    const baseline=state.baseline;
    if(!baseline)return;
    el('referenceChartRMSE').innerHTML=baselineLineChart('rmse');
    el('referenceChartR2').innerHTML=baselineLineChart('r2');
    el('referenceTable').innerHTML=baseline.progression.map(p=>`<tr class="${state.baselineSelectedK===p.k?'selected-row':''}">
      <td><button type="button" class="row-action benchmark-row-button" data-baseline-k="${p.k}" aria-pressed="${state.baselineSelectedK===p.k}" aria-label="檢視基準分析 k=${p.k}">${p.k}</button></td>
      <td class="feature-list">[${esc(p.names.map(formatFeatureName).join(', '))}]</td>
      <td class="benchmark-number">${Number.isFinite(p.testing?.rmse)?p.testing.rmse.toFixed(6):'—'}</td>
      <td class="benchmark-number">${Number.isFinite(p.testing?.r2)?p.testing.r2.toFixed(6):'—'}</td>
    </tr>`).join('');
    const original=state.key==='startups'&&baseline.exactOrder;
    el('benchmarkBadge').textContent=original?'BASELINE · SEED 0':'CUSTOM BASELINE · SEED 0';
    el('referenceIntro').textContent=original
      ? '50 Startups 多元線性回歸：80/20 資料切分、random_state=0，以逐步增加特徵的 OLS 模型呈現 RMSE 與 R²。'
      : `使用相同基準分析流程處理 ${state.sourceName}，以 Pearson 排序前 ${baseline.progression.length} 個特徵。不同資料集的指標不適合直接比較。`;
    el('referenceSplit').textContent=`random_state=0 · test=${baseline.split.test.length}`;
    el('referenceNote').textContent=original
      ? '模型設計說明：已有 State_New York、State_Florida 與截距時，再加入 State_California 會造成完全共線性。第五個獨立特徵採用 Administration；所有數值依實際模型輸入重算。'
      : '圖表由目前載入的資料集重新訓練，數值會隨資料與目標欄位變動；無法建立 OLS 的特徵組合會顯示 —。';
    const selected=baseline.progression.find(p=>p.k===state.baselineSelectedK);
    el('referencePointDetail').innerHTML=selected?.testing
      ? `<strong>基準分析 · k=${selected.k}</strong><span>RMSE <b>${fmt(selected.testing.rmse,6)}</b></span>
         <span>R² <b>${fmt(selected.testing.r2,6)}</b></span>
         <span>選取特徵：${esc(selected.names.map(formatFeatureName).join('、'))}</span>`
      : '點擊任一資料點，即可查看對應 k 的 RMSE、R² 和選取特徵，並在下方表格同步反白。';
  }
  function selectBaselineK(k) {
    if(!Number.isInteger(k) || !state.baseline?.progression.some(p=>p.k===k && p.testing)) return;
    state.baselineSelectedK=k;
    renderBaseline();
  }
  function renderMetricCards() {
    const p = current(), a = state.analysis;
    el('metricRMSE').textContent = p?.testing ? currency(p.testing.rmse) : '—';
    el('metricR2').textContent = p?.testing ? fmt(p.testing.r2, 4) : '—';
    el('metricCV').textContent = p?.cv != null ? currency(p.cv) : '—';
    el('metricMAE').textContent = p?.testing ? currency(p.testing.mae) : '—';
    const best = a.best;
    el('bestHint').innerHTML = `訓練集交叉驗證建議 <strong>k = ${best.k}</strong>（CV RMSE = ${esc(currency(best.cv))}）。
      最佳模型特徵：<strong>${esc(best.names.join('、'))}</strong>。
      目前檢視 k = ${p.k}；${p.k === best.k ? '這是依 CV 決定的模型。' : '目前選的是比較用模型，而非 CV 最佳模型。'}
      <span class="micro">測試集指標只供獨立評估，不參與選 k。</span>`;
  }
  function renderProgression() {
    const a = state.analysis;
    el('progressBody').innerHTML = a.progression.map(p => `<tr class="${p.k === state.selectedK ? 'selected-row' : ''} ${p.k === a.bestK ? 'best-row' : ''}">
      <td><button class="row-action" type="button" data-k="${p.k}">k = ${p.k}</button></td>
      <td class="feature-list" title="${esc(p.names.join(' · '))}">${p.k === 1 ? '' : '+ '}${esc(p.names[p.names.length - 1])}</td>
      <td>${fmt(p.cv)}</td><td>${p.testing ? esc(currency(p.testing.rmse)) : '—'}</td>
      <td>${p.testing ? fmt(p.testing.r2, 4) : '—'}</td><td>${p.training ? fmt(p.training.r2, 4) : '—'}</td>
      <td>${p.k === a.bestK ? '★ CV 最佳' : p.error ? esc(p.error) : '可比較'}</td>
      </tr>`).join('');
    for (const b of el('progressBody').querySelectorAll('button[data-k]')) {
      b.addEventListener('click', () => selectK(Number(b.dataset.k)));
    }
  }
  function selectK(k) {
    if (!state.analysis?.progression.some(p => p.k === k)) return;
    state.selectedK = k;
    state.nativeLassoSelected = false;
    el('kSelect').value = String(k);
    renderAll();
  }
  function selectAdvancedPoint(methodId, k) {
    if(methodId==='native-lasso') {
      if(!state.comparison?.lassoNative) return;
      state.nativeLassoSelected=true;
      renderCharts();
      return;
    }
    const analysis=state.comparison?.results?.[methodId];
    if(!analysis || !Number.isInteger(k) || !analysis.progression.some(p=>p.k===k && p.testing)) return;
    state.analysis=analysis;
    state.selectedK=k;
    state.nativeLassoSelected=false;
    el('methodSelect').value=methodId;
    syncKSelector();
    renderAll();
  }
  function chartPointAction(event) {
    if(event.type==='keydown' && event.key!=='Enter' && event.key!==' ' && event.key!=='Spacebar') return;
    const point=event.target.closest?.('[data-chart-point]');
    if(!point || !event.currentTarget.contains(point)) return;
    if(event.type==='keydown') event.preventDefault();
    const type=point.dataset.chartPoint, k=Number(point.dataset.k), method=point.dataset.method;
    const host=event.currentTarget;
    if(type==='baseline') selectBaselineK(k);
    else if(type==='advanced') selectAdvancedPoint(method,k);
    else if(type==='native-lasso') selectAdvancedPoint('native-lasso',k);
    // Re-rendering replaces SVG elements. Restore focus for keyboard users.
    if(event.type==='keydown') {
      const next=[...host.querySelectorAll('[data-chart-point]')].find(p=>
        p.dataset.chartPoint===type && Number(p.dataset.k)===k && p.dataset.method===method);
      next?.focus?.({preventScroll:true});
    }
  }
  function chartLegendAction(event) {
    const button=event.target.closest?.('[data-chart-method]');
    if(!button || !event.currentTarget.contains(button)) return;
    const method=button.dataset.chartMethod;
    selectAdvancedPoint(method,method==='native-lasso'?state.comparison?.lassoNative?.k:state.selectedK);
  }
  function comparisonChart(which) {
    const compare = state.comparison;
    if (!compare) return '<p>尚無可顯示資料。</p>';
    const isRMSE = which === 'rmse', key = isRMSE ? 'rmse' : 'r2';
    const count = state.dataset.features.length;
    const all = compare.methods.map(method => ({ ...method,
      points: compare.results[method.id].progression.map(p => p.testing?.[key]) }));
    const native = compare.lassoNative;
    const extra = native && native.k >= 1 && native.k <= count ? native.testing[key] : null;
    const values = [...all.flatMap(v => v.points),extra].filter(Number.isFinite);
    if (!values.length) return '<p>沒有可顯示的測試結果。</p>';
    const W = 660, H = 330, L = 65, R = 21, T = 22, B = 48;
    const min = Math.min(...values), max = Math.max(...values);
    const span = Math.max(max-min, Math.abs(max)*0.01,0.0001);
    const lo = min-span*0.12, hi = max+span*0.12;
    const x = k => L + (count < 2 ? 0 : (k-1)/(count-1)*(W-L-R));
    const y = v => H-B - (v-lo)/(hi-lo)*(H-T-B);
    const label = v => isRMSE ? (Math.abs(v)>=1000 ? (v/1000).toFixed(1)+'k' : v.toFixed(1)) : v.toFixed(3);
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="五種方法 ${isRMSE?'RMSE':'R-squared'} 對特徵數的比較圖">`;
    for(let tick=0;tick<=5;tick++) {
      const value=lo+(hi-lo)*tick/5;
      svg+=`<line x1="${L}" x2="${W-R}" y1="${y(value)}" y2="${y(value)}" stroke="#e5eaf3"/>
        <text x="${L-10}" y="${y(value)+4}" text-anchor="end" fill="#8998ac" font-size="12">${esc(label(value))}</text>`;
    }
    for(let k=1;k<=count;k++) {
      svg+=`<text x="${x(k)}" y="${H-B+20}" text-anchor="middle" fill="#718197" font-size="11">${k}</text>`;
    }
    svg+=`<text x="${(L+W-R)/2}" y="${H-5}" text-anchor="middle" fill="#748199" font-size="13">Number of Features (k)</text>
      <line x1="${x(state.selectedK)}" x2="${x(state.selectedK)}" y1="${T}" y2="${H-B}" stroke="#a8b4d2" stroke-dasharray="3 6"/>`;
    function shape(k,v,type,color) {
      const cx=x(k),cy=y(v),r=4.2;
      if(type===1)return `<rect x="${cx-r}" y="${cy-r}" width="${r*2}" height="${r*2}" fill="#fff" stroke="${color}" stroke-width="2"/>`;
      if(type===2)return `<polygon points="${cx},${cy-r-1} ${cx+r+1},${cy+r} ${cx-r-1},${cy+r}" fill="#fff" stroke="${color}" stroke-width="2"/>`;
      if(type===3)return `<polygon points="${cx},${cy-r-1} ${cx+r+1},${cy} ${cx},${cy+r+1} ${cx-r-1},${cy}" fill="#fff" stroke="${color}" stroke-width="2"/>`;
      return `<circle cx="${cx}" cy="${cy}" r="${r}" fill="#fff" stroke="${color}" stroke-width="2"/>`;
    }
    // Draw the selected method last so overlapping points remain clickable.
    const paintOrder=all.map((m,index)=>({m,index})).sort((a,b)=>
      Number(a.m.id===state.analysis.method)-Number(b.m.id===state.analysis.method));
    for(const {m,index:methodIdx} of paintOrder) {
      const selected=m.id===state.analysis.method;
      const valid=m.points.map((v,i)=>({k:i+1,v})).filter(p=>Number.isFinite(p.v));
      if(!valid.length)continue;
      const points=valid.map((p,i)=>`${i?'L':'M'}${x(p.k).toFixed(2)} ${y(p.v).toFixed(2)}`).join(' ');
      svg+=`<path d="${points}" fill="none" stroke="${m.color}" stroke-width="${selected?3.1:2}" stroke-opacity="${selected?1:.77}" ${methodIdx===2?'stroke-dasharray="8 5"':methodIdx===3?'stroke-dasharray="3 5"':''} stroke-linejoin="round"/>`;
      for(const p of valid) {
        const active=selected && p.k===state.selectedK && !state.nativeLassoSelected;
        const label=`${m.name}，k=${p.k}，${isRMSE?'RMSE':'R²'} ${p.v.toFixed(5)}。按 Enter 選取此模型`;
        svg+=`<g class="chart-point ${active?'chart-point-active':''}" data-chart-point="advanced"
          data-method="${m.id}" data-k="${p.k}" role="button" tabindex="0"
          aria-pressed="${active}" aria-label="${esc(label)}">
          ${active?`<circle class="chart-selected-ring" cx="${x(p.k)}" cy="${y(p.v)}" r="9" stroke="${m.color}"/>`:''}
          ${shape(p.k,p.v,methodIdx,m.color)}
          <circle class="chart-hit-area" cx="${x(p.k)}" cy="${y(p.v)}" r="12"/>
          <title>${esc(label)}</title></g>`;
      }
    }
    if (Number.isFinite(extra)) {
      const centerX=x(native.k), centerY=y(extra);
      const star=Array.from({length:10},(_,i)=>{
        const angle=i*Math.PI/5-Math.PI/2,rad=i%2===0?9:4;
        return `${(centerX+Math.cos(angle)*rad).toFixed(2)},${(centerY+Math.sin(angle)*rad).toFixed(2)}`;
      }).join(' ');
      const nativeActive=state.nativeLassoSelected;
      const nativeLabel=`LassoCV 原生 L1 收縮模型，k=${native.k}，${isRMSE?'RMSE':'R²'} ${extra.toFixed(5)}。按 Enter 查看原生模型`;
      svg+=`<g class="chart-point ${nativeActive?'chart-point-active':''}" data-chart-point="native-lasso"
        data-k="${native.k}" role="button" tabindex="0" aria-pressed="${nativeActive}" aria-label="${esc(nativeLabel)}">
        ${nativeActive?`<circle class="chart-selected-ring" cx="${centerX}" cy="${centerY}" r="14" stroke="#bc8a1c"/>`:''}
        <polygon points="${star}" fill="#f8c34d" stroke="#694819" stroke-width="1.8"/>
        <circle class="chart-hit-area" cx="${centerX}" cy="${centerY}" r="13"/>
        <title>${esc(nativeLabel)}</title></g>`;
    }
    return svg+'</svg>';
  }
  function renderCharts() {
    el('chartRMSE').innerHTML = comparisonChart('rmse');
    el('chartR2').innerHTML = comparisonChart('r2');
    const legend = state.comparison.methods.map(method =>
      `<button type="button" class="comparison-legend-btn ${method.id===state.analysis.method && !state.nativeLassoSelected?'legend-active':''}"
        data-chart-method="${method.id}" aria-pressed="${method.id===state.analysis.method && !state.nativeLassoSelected}">
        <i style="background:${method.color}"></i>${esc(method.name)}</button>`).join('')
      + `<button type="button" class="comparison-legend-btn ${state.nativeLassoSelected?'legend-active':''}"
        data-chart-method="native-lasso" aria-pressed="${state.nativeLassoSelected}">
        <b class="legend-star">★</b>LassoCV Native Shrinkage</button>`;
    el('legendRMSE').innerHTML = legend;
    el('legendR2').innerHTML = legend;
    const native=state.comparison.lassoNative;
    const p=current();
    if(state.nativeLassoSelected && native){
      const active=native.model.featureIndices.filter((_,i)=>Math.abs(native.model.standardizedCoefficients[i])>1e-5)
        .map(j=>state.dataset.features[j].name);
      el('advancedPointDetail').innerHTML=`<strong>★ LassoCV 原生 L1 模型 · k=${native.k}</strong>
        <span>RMSE <b>${fmt(native.testing.rmse,6)}</b></span>
        <span>R² <b>${fmt(native.testing.r2,6)}</b></span>
        <span>α <b>${fmt(native.alpha,4)}</b> · 特徵：${esc(active.join('、'))}</span>
        <span class="chart-detail-note">金色星號使用原生 L1 正則化係數，與下方 OLS 指標卡片不同。</span>`;
    } else {
      const selected=state.comparison.methods.find(m=>m.id===state.analysis.method);
      el('advancedPointDetail').innerHTML=p?.testing?
        `<strong>${esc(selected?.name||state.analysis.method)} · k=${p.k}</strong>
         <span>RMSE <b>${fmt(p.testing.rmse,6)}</b></span>
         <span>R² <b>${fmt(p.testing.r2,6)}</b></span>
         <span>特徵：${esc(p.names.join('、'))}</span>`:
        '點擊圖表上的資料點，可選取演算法與 k 並查看詳細指標。';
    }
  }
  function renderComparisonTable() {
    const c = state.comparison;
    el('comparisonBody').innerHTML=c.methods.map(method=>{
      const result=c.results[method.id],best=result.best,p=best;
      const selected = method.id === state.analysis.method;
      return `<tr class="${selected?'selected-row':''}"><td><span class="method-color" style="background:${method.color}"></span>${esc(method.name)}</td>
        <td>${best.k}</td><td>${fmt(best.cv)}</td><td>${fmt(p.testing?.rmse)}</td>
        <td>${fmt(p.testing?.r2,4)}</td><td class="feature-list">${esc(p.names.join(' · '))}</td></tr>`;
    }).join('');
    const native=c.lassoNative;
    if(native) {
      const active=native.model.featureIndices.filter((_,i)=>Math.abs(native.model.standardizedCoefficients[i])>1e-5)
        .map(j=>state.dataset.features[j].name);
      el('comparisonBody').innerHTML+=`<tr class="native-lasso-row"><td><b class="legend-star">★</b> LassoCV Native Shrinkage</td>
        <td>${native.k}</td><td>${fmt(native.cv)}</td><td>${fmt(native.testing.rmse)}</td>
        <td>${fmt(native.testing.r2,4)}</td><td class="feature-list">${esc(active.join(' · '))}（原生 L1 模型）</td></tr>`;
      el('lassoNativeSummary').textContent=`LassoCV 由訓練集交叉驗證選 alpha=${fmt(native.alpha,4)}，實際非零係數 ${native.k} 個。
        星號是含 L1 正則化的直接預測結果；折線是按特徵排序選 k 後重新擬合的 OLS，兩者不可直接解讀為同一種模型。`;
    }
  }
  function renderCoefficients() {
    const p = current(), d = state.dataset;
    if (!p.model) { el('equation').textContent = p.error; el('coeffBody').innerHTML = ''; return; }
    const m = p.model;
    el('equation').textContent = `${d.target} = ${fmt(m.intercept, 4)} ` + m.coefficients.map((v,i) =>
      `${v >= 0 ? '+' : '−'} ${fmt(Math.abs(v), 5)} × (${d.features[p.indices[i]].name})`).join(' ');
    el('activeFeatures').innerHTML = p.names.map(x => `<strong>${esc(x)}</strong>`).join(' · ');
    const vif = A.calculateVIF(state.analysis.split.train, p.indices);
    el('coeffBody').innerHTML = `<tr><td>Intercept (β₀)</td><td>${fmt(m.intercept, 6)}</td><td>—</td></tr>`
      + m.coefficients.map((v,i) => `<tr><td>${esc(p.names[i])}</td><td>${fmt(v,6)}</td>
        <td class="${vif[i]>5?'bad-value':'good-value'}">${fmt(vif[i], 2)}</td></tr>`).join('');
  }
  function renderScatter() {
    const p = current();
    if (!p.testing) { el('chartScatter').innerHTML = '模型不可用'; return; }
    const actual = p.testing.actual, predicted = p.testing.predicted;
    const W=615,H=293,L=67,R=24,T=22,B=49;
    const min = Math.min(...actual,...predicted), max = Math.max(...actual,...predicted);
    const pad = (max-min || 1)*.08, lo = min-pad, hi = max+pad;
    const x = v => L + (v-lo)/(hi-lo)*(W-L-R), y = v => H-B - (v-lo)/(hi-lo)*(H-T-B);
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="測試集真實值與預測值散佈圖">`;
    for(let i=0;i<=4;i++){
      const v=lo+i*(hi-lo)/4;
      svg += `<path d="M${x(v)} ${T} V${H-B} M${L} ${y(v)} H${W-R}" stroke="#e8edf6"/>
        <text x="${x(v)}" y="${H-B+20}" text-anchor="middle" fill="#9ba5b5" font-size="10">${esc(shortNum(v))}</text>
        <text x="${L-8}" y="${y(v)+4}" text-anchor="end" fill="#9ba5b5" font-size="10">${esc(shortNum(v))}</text>`;
    }
    svg += `<path d="M${x(lo)} ${y(lo)} L${x(hi)} ${y(hi)}" stroke="#8f9bc2" stroke-width="2" stroke-dasharray="6 5"/>`;
    for(let i=0;i<actual.length;i++){
      svg += `<circle cx="${x(actual[i])}" cy="${y(predicted[i])}" r="${actual.length>200?3:4}" fill="#5067e6" fill-opacity=".67"><title>Actual ${fmt(actual[i])}; Predicted ${fmt(predicted[i])}</title></circle>`;
    }
    svg += `<text x="${(L+W-R)/2}" y="${H-3}" text-anchor="middle" fill="#8491a7" font-size="12">Actual value</text>
      <text x="16" y="${H/2}" text-anchor="middle" fill="#8491a7" font-size="12" transform="rotate(-90 16 ${H/2})">Predicted value</text></svg>`;
    el('chartScatter').innerHTML=svg;
    el('scatterSummary').innerHTML = `測試樣本 <strong>${actual.length}</strong> 筆；R² = <strong>${fmt(p.testing.r2,4)}</strong>，RMSE = <strong>${esc(currency(p.testing.rmse))}</strong>（${esc(unitName())}）。`;
  }
  function shortNum(n) { return Math.abs(n)>=1000 ? (n/1000).toFixed(1)+'k' : n.toFixed(Math.abs(n)<10?1:0); }
  function renderResiduals() {
    const p = current();
    if(!p.testing) { el('residualBody').innerHTML=''; return; }
    const pairs = state.analysis.split.test.map((row,i) => ({ id: row.id, y:row.y,
      pred: p.testing.predicted[i], err: row.y-p.testing.predicted[i] }));
    pairs.sort((a,b)=>Math.abs(b.err)-Math.abs(a.err));
    el('residualBody').innerHTML = pairs.slice(0,20).map(r=>`<tr><td>#${r.id}</td><td>${fmt(r.y)}</td><td>${fmt(r.pred)}</td>
      <td class="${Math.abs(r.err)>p.testing.rmse*1.6?'bad-value':''}">${r.err>=0?'+':''}${fmt(r.err)}</td>
      <td>${fmt(Math.abs(r.err))}</td><td>${Math.abs(r.y)>1e-12?fmt(Math.abs(r.err/r.y)*100,1)+'%':'—'}</td></tr>`).join('');
    el('residualCount').textContent = `${pairs.length} TEST ROWS · TOP ${Math.min(20,pairs.length)}`;
  }
  // Test samples sorted by prediction, with 95% confidence (mean) and prediction (new value) bands.
  function intervalChart(s) {
    const items = [...s.items].sort((a, b) => a.fit - b.fit);
    const W = 1000, H = 360, L = 74, R = 20, T = 34, B = 48;
    const values = items.flatMap(i => [i.piLow, i.piHigh, i.actual]);
    const min = Math.min(...values), max = Math.max(...values), pad = (max - min || 1) * .05;
    const lo = min - pad, hi = max + pad;
    const x = i => L + (items.length < 2 ? 0 : i / (items.length - 1) * (W - L - R));
    const y = v => H - B - (v - lo) / (hi - lo) * (H - T - B);
    const band = (a, b) => items.map((it, i) => `${x(i).toFixed(1)},${y(it[a]).toFixed(1)}`).join(' ') + ' '
      + items.map((it, i) => `${x(i).toFixed(1)},${y(it[b]).toFixed(1)}`).reverse().join(' ');
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="測試集預測值與 95% 信賴區間、預測區間">`;
    for (let j = 0; j <= 5; j++) {
      const v = lo + (hi - lo) * j / 5;
      svg += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="#e5eaf3"/>
        <text x="${L - 9}" y="${y(v) + 4}" text-anchor="end" fill="#8998ac" font-size="12">${esc(shortNum(v))}</text>`;
    }
    svg += `<polygon points="${band('piLow', 'piHigh')}" fill="#5b8def" fill-opacity=".16"/>
      <polygon points="${band('ciLow', 'ciHigh')}" fill="#5b8def" fill-opacity=".38"/>
      <polyline points="${items.map((it, i) => `${x(i).toFixed(1)},${y(it.fit).toFixed(1)}`).join(' ')}" fill="none" stroke="#1f4fbf" stroke-width="2.2"/>`;
    items.forEach((it, i) => {
      const inside = it.actual >= it.piLow && it.actual <= it.piHigh;
      svg += `<circle cx="${x(i)}" cy="${y(it.actual)}" r="${items.length > 120 ? 2.6 : 3.6}" fill="${inside ? '#2b8a54' : '#d0443e'}">
        <title>資料列 #${it.id}：真實 ${fmt(it.actual)}；預測 ${fmt(it.fit)}；95% PI [${fmt(it.piLow)}, ${fmt(it.piHigh)}]</title></circle>`;
    });
    const legend = [['#5b8def', .16, '95% 預測區間 PI'], ['#5b8def', .38, '95% 信賴區間 CI'], ['#1f4fbf', 1, '預測值 ŷ']];
    legend.forEach(([c, o, t], i) => {
      svg += `<rect x="${L + i * 170}" y="9" width="16" height="12" fill="${c}" fill-opacity="${o}"/>
        <text x="${L + i * 170 + 22}" y="20" font-size="12" fill="#556477">${t}</text>`;
    });
    svg += `<circle cx="${L + 518}" cy="15" r="4" fill="#2b8a54"/><text x="${L + 528}" y="20" font-size="12" fill="#556477">真實值（在 PI 內）</text>
      <circle cx="${L + 668}" cy="15" r="4" fill="#d0443e"/><text x="${L + 678}" y="20" font-size="12" fill="#556477">真實值（在 PI 外）</text>
      <text x="${(L + W - R) / 2}" y="${H - 10}" text-anchor="middle" fill="#748199" font-size="13">測試樣本（依預測值由小到大排序）</text>
      <text x="18" y="${H / 2}" transform="rotate(-90 18 ${H / 2})" text-anchor="middle" fill="#748199" font-size="13">${esc(state.dataset.target)}</text></svg>`;
    return svg;
  }
  function renderIntervals() {
    const p = current();
    if (!p?.model) { el('intervalChart').innerHTML = '模型不可用'; return; }
    const s = A.intervalSummary(p.model, state.analysis.split.test, 0.95);
    state.intervals = s;
    el('piCoverage').textContent = `${fmt(s.coverage * 100, 1)}%`;
    el('piWidth').textContent = currency(s.meanPIWidth);
    el('ciWidth').textContent = currency(s.meanCIWidth);
    el('piT').textContent = fmt(s.t, 3);
    el('piStats').textContent = `df = n − p − 1 = ${s.df}；殘差標準差 s = ${currency(s.sigma)}`;
    el('intervalChart').innerHTML = intervalChart(s);
    const note = s.coverage < 0.9 ? '低於名目的 95%：區間太窄，代表模型低估了不確定性（例如誤差會隨數值變大，或遺漏了重要變數）。'
      : s.coverage > 0.99 ? '高於 95%：區間偏保守。' : '接近名目的 95%，區間大致可信。';
    el('intervalSummary').innerHTML = `測試集 ${s.items.length} 筆中有 <strong>${s.inside}</strong> 筆的真實值落在 95% 預測區間內（覆蓋率 ${fmt(s.coverage * 100, 1)}%）。${esc(note)}`;
  }
  function renderCrispDM() {
    const d = state.dataset, a = state.analysis, p = current(), info = PROBLEMS[state.key];
    const ys = d.rows.map(r => r.y), missing = d.parsed.headers.reduce((n, h) => n + d.parsed.rows.filter(r => r[h] === '').length, 0);
    const categorical = d.columns.filter(c => !c.numeric);
    const top = state.comparison.results.pearson.order.slice(0, 3).map(i => d.features[i].name);
    const methodName = state.comparison.methods.find(m => m.id === a.method)?.name || a.method;
    const set = (id, html) => { el(id).innerHTML = html; };
    set('crispBU', `${esc(info ? info.goal : `預測自訂資料集的 ${d.target}。`)}這是回歸問題（目標是連續數值），成功標準是測試集 RMSE 小、R² 高，並能給出 95% 預測區間。`);
    set('crispDU', `${d.parsed.rows.length} 筆資料、${d.columns.length} 個原始特徵，缺值 ${missing} 格。${esc(d.target)} 平均 ${esc(currency(A.mean(ys)))}，範圍 ${esc(currency(Math.min(...ys)))}～${esc(currency(Math.max(...ys)))}。訓練集中和目標最相關的特徵：${esc(top.join('、'))}。`);
    set('crispDP', `${d.ignoredReasons.length ? `略過 ${esc(d.ignoredReasons.map(c => c.name).join('、'))}；` : ''}${categorical.length} 個類別欄位做 One-hot（每欄捨棄一個基準類別，避免完全共線）${d.mergedLevels.length ? '，出現少於 5 次的類別先合併' : ''}，編碼後共 ${d.features.length} 個特徵。80／20 切分（random_state = 42），標準化參數只用訓練集計算。`);
    set('crispMO', `OLS 多元線性回歸。用五種方法排序特徵（Pearson、SelectKBest、RFE、Forward SFS、Lasso），每個 k 都重新擬合 OLS，以訓練集 5-fold CV RMSE 選 k；另列 LassoCV 原生模型對照。目前檢視：${esc(methodName)}，CV 建議 k = ${a.bestK}。`);
    set('crispEV', p?.testing ? `目前模型（k = ${p.k}）測試集 RMSE ${esc(currency(p.testing.rmse))}、MAE ${esc(currency(p.testing.mae))}、R² ${fmt(p.testing.r2, 4)}；95% 預測區間覆蓋率 ${fmt((state.intervals?.coverage || 0) * 100, 1)}%。並檢查 VIF 與殘差。` : '目前模型無法評估。');
    set('crispDE', '部署為這個 GitHub Pages 網頁：下拉選單切換兩個題目，也可上傳 CSV 或輸入公開 CSV 網址；可即時輸入特徵得到預測值與 95% 信賴／預測區間，並匯出預測 CSV 與 JSON 報告。');
  }
  function renderPredictor() {
    const d = state.dataset, p = current();
    const defaults = {};
    el('predictForm').innerHTML = d.columns.map(c => {
      if (c.numeric) {
        const med = A.median(d.rows.map(r=>A.num(r.original[c.name])));
        defaults[c.name] = med;
        return `<div class="input-field"><label for="input-${esc(c.name)}">${esc(c.name)}</label>
          <input data-feature="${esc(c.name)}" id="input-${esc(c.name)}" type="number" step="any" value="${med}" required></div>`;
      }
      defaults[c.name] = c.levels[0];
      return `<div class="input-field"><label for="input-${esc(c.name)}">${esc(c.name)}</label>
        <select data-feature="${esc(c.name)}" id="input-${esc(c.name)}">${c.levels.map(l=>`<option value="${esc(l)}">${esc(l)}</option>`).join('')}</select></div>`;
    }).join('');
    state.defaults = defaults;
    el('predTitle')?.textContent;
    el('predictTitle').textContent = `預測 ${d.target}`;
    el('predModel').textContent = p.model ? `OLS · k=${p.k} · ${unitName()}` : '模型不可用';
    for (const input of el('predictForm').querySelectorAll('[data-feature]')) {
      input.addEventListener('input', updatePrediction);
      input.addEventListener('change', updatePrediction);
    }
    updatePrediction();
  }
  function updatePrediction() {
    const p = current();
    if (!p?.model) { el('predNumber').textContent='—'; return; }
    const inputs = {};
    for (const input of el('predictForm').querySelectorAll('[data-feature]')) {
      inputs[input.dataset.feature] = input.value;
    }
    const x = A.rowFromInputs(state.dataset, inputs);
    if (x.some(v=>!Number.isFinite(v))) { el('predNumber').textContent='請輸入有效數值'; return; }
    const r = A.predictionInterval(p.model, x, 0.95);
    el('predNumber').textContent = currency(r.fit);
    el('predPI').textContent = `[${currency(r.piLow)}, ${currency(r.piHigh)}]`;
    el('predCI').textContent = `[${currency(r.ciLow)}, ${currency(r.ciHigh)}]`;
  }
  function download(name, content, mime) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href=url; a.download=name;
    document.body.appendChild(a); a.click(); a.remove();
    window.setTimeout(()=>URL.revokeObjectURL(url),1500);
  }
  function downloadCSV() {
    if (!state.analysis) return;
    const p=current(), rows=state.analysis.split.test;
    if(!p.model) return;
    const lines = [['sample_id','actual','predicted','residual','abs_error','ci95_low','ci95_high','pi95_low','pi95_high',...state.dataset.columns.map(c=>c.name)].map(csvCell).join(',')];
    for (const row of rows) {
      const r=A.predictionInterval(p.model,row.x,0.95), e=row.y-r.fit;
      lines.push([row.id,row.y,r.fit,e,Math.abs(e),r.ciLow,r.ciHigh,r.piLow,r.piHigh,...state.dataset.columns.map(c=>row.original[c.name])].map(csvCell).join(','));
    }
    download(`regression_${state.key}_k${p.k}_test_predictions.csv`,lines.join('\r\n'),'text/csv;charset=utf-8');
  }
  function downloadReport() {
    if(!state.analysis) return;
    const a=state.analysis,p=current(),d=state.dataset;
    const s=state.intervals;
    const report={title:'HW2 Multiple Linear Regression — Model Analysis',
      generatedAt:new Date().toISOString(),source:state.sourceName,datasetKey:state.key,
      kaggle:PROBLEMS[state.key]?.kaggle||null,
      target:d.target,units:unitName(),totalRows:d.parsed.rows.length,validRows:d.rows.length,
      droppedRows:d.dropped,ignoredColumns:d.ignoredReasons,mergedRareLevels:d.mergedLevels,encodedFeatures:d.features,
      predictionInterval95:s&&{coverage:s.coverage,inside:s.inside,testRows:s.items.length,
        meanPIWidth:s.meanPIWidth,meanCIWidth:s.meanCIWidth,t:s.t,df:s.df,residualSD:s.sigma},
      split:{seed:42,train:a.split.train.length,test:a.split.test.length,
        splitRule:'NumPy RandomState MT19937, sklearn train_test_split test_size=0.2 random_state=42',
        holdoutRowNumbers:a.split.test.map(r=>r.id)},
      modeling:{type:'Ordinary Least Squares with centered/scaled QR',method:a.method,
        folds:a.foldCount,featureSelectionOn:'training set only',criterion:'min CV RMSE',
        recommendedK:a.bestK,displayedK:p.k},
      featureSelectionComparison:state.comparison.methods.map(method=>({method:method.name,
        bestK:state.comparison.results[method.id].bestK,
        progression:state.comparison.results[method.id].progression.map(v=>({k:v.k,
          features:v.names,cvRMSE:v.cv,testRMSE:v.testing?.rmse,testR2:v.testing?.r2}))})),
      lassoNative:state.comparison.lassoNative && {
        description:'Native L1 shrinkage, not refitted OLS',
        alpha:state.comparison.lassoNative.alpha,
        k:state.comparison.lassoNative.k,
        cvRMSE:state.comparison.lassoNative.cv,
        testRMSE:state.comparison.lassoNative.testing.rmse,
        testR2:state.comparison.lassoNative.testing.r2},
      progression:a.progression.map(r=>({k:r.k,features:r.names,cvRMSE:r.cv,
        train:r.training&&{rmse:r.training.rmse,mae:r.training.mae,r2:r.training.r2},
        test:r.testing&&{rmse:r.testing.rmse,mae:r.testing.mae,r2:r.testing.r2},error:r.error})),
      displayedModel:p.model&&{intercept:p.model.intercept,
        features:p.names,coefficients:p.model.coefficients,
        vif:A.calculateVIF(a.split.train,p.indices),
        testMetrics:{rmse:p.testing.rmse,mae:p.testing.mae,r2:p.testing.r2,mape:p.testing.mape}},
      limitations:['Holdout results depend on dataset and chosen split',
        'Feature-selection CV is an internal selection criterion, not an unbiased final estimate',
        'Prediction intervals assume independent, normally distributed errors with constant variance']};
    download(`regression_${state.key}_analysis_report.json`,JSON.stringify(report,null,2),'application/json;charset=utf-8');
  }
  function onCustomCSV(text,name) {
    state.custom={text,name};
    const select=el('datasetSelect');
    if(!select.querySelector('option[value="custom"]')) {
      const opt=document.createElement('option');opt.value='custom';opt.textContent='自訂資料集 — '+name;select.add(opt);
    } else select.querySelector('option[value="custom"]').textContent='自訂資料集 — '+name;
    select.value='custom';
    loadProblem('custom',`已匯入 ${name}。可改變預測目標欄位並重新執行模型。`);
  }
  async function loadCSVUrl() {
    const url=el('csvUrl').value.trim();
    if (!/^https?:\/\//i.test(url)) { status('請輸入以 https:// 或 http:// 開頭的公開 CSV 網址',true); return; }
    const button=el('loadUrlButton');button.disabled=true;button.textContent='讀取中…';
    try {
      const response=await fetch(url,{cache:'no-store'});
      if(!response.ok) throw new Error(`HTTP ${response.status}`);
      const text=await response.text();
      if(text.length>8e6) throw new Error('檔案超過 8MB；請先精簡');
      onCustomCSV(text,decodeURIComponent(url.split('/').pop().split('?')[0])||'remote.csv');
    } catch(e) { status(`網址載入失敗：${e.message}。若網站未開放 CORS，請先下載 CSV 再使用上傳功能。`,true); }
    finally{button.disabled=false;button.textContent='載入 URL';}
  }
  function initialize() {
    el('datasetSelect').addEventListener('change',e=>loadProblem(e.target.value));
    el('targetSelect').addEventListener('change',rebuildDataset);
    el('methodSelect').addEventListener('change',changeMethod);
    el('kSelect').addEventListener('change',e=>selectK(Number(e.target.value)));
    for(const id of ['referenceChartRMSE','referenceChartR2','chartRMSE','chartR2']) {
      el(id).addEventListener('click',chartPointAction);
      el(id).addEventListener('keydown',chartPointAction);
    }
    el('referenceTable').addEventListener('click',event=>{
      const button=event.target.closest?.('[data-baseline-k]');
      if(button) selectBaselineK(Number(button.dataset.baselineK));
    });
    for(const id of ['legendRMSE','legendR2']) {
      el(id).addEventListener('click',chartLegendAction);
    }
    el('runButton').addEventListener('click',runAnalysis);
    el('loadUrlButton').addEventListener('click',loadCSVUrl);
    el('csvFile').addEventListener('change',async e=>{
      const file=e.target.files?.[0];if(!file)return;
      if(file.size>8e6){status('檔案超過 8MB；請精簡資料後重新上傳',true);return;}
      try{onCustomCSV(await file.text(),file.name);}catch(ex){status(`讀取失敗：${ex.message}`,true);}
    });
    el('csvButton').addEventListener('click',downloadCSV);
    el('reportButton').addEventListener('click',downloadReport);
    el('predictForm').addEventListener('submit',e=>e.preventDefault());
    el('exampleUrlButton').addEventListener('click',()=>{
      el('csvUrl').value='https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv';
      loadCSVUrl();
    });
    loadProblem('carprice');
  }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',initialize);
  else initialize();
})();
