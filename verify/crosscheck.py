"""Independent recomputation of the web app's numbers with pandas / scikit-learn / SciPy.

Run `node verify/dump_js.cjs` first, then `python verify/crosscheck.py`.
Checks: encoding against the raw CSV, the 80/20 split against sklearn's
train_test_split(random_state=42), OLS coefficients and test metrics against
LinearRegression, 95% CI / PI against the textbook formula with scipy's t,
pooled 5-fold CV RMSE, the Pearson / f_regression ranking and the forward-selection path.
Writes verify/crosscheck_result.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
js = json.loads((HERE / "js_results.json").read_text(encoding="utf-8"))
report = {}
failures = []


def check(cond, label):
    if not cond:
        failures.append(label)
    return bool(cond)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


def cv_rmse(X, y, cols, order, folds):
    """Pooled CV RMSE with the app's fold assignment (position i goes to fold i % folds)."""
    err, count = 0.0, 0
    order = np.asarray(order)
    for k in range(folds):
        hold = order[np.arange(len(order)) % folds == k]
        fit = order[np.arange(len(order)) % folds != k]
        m = LinearRegression().fit(X[np.ix_(fit, cols)], y[fit])
        err += float(np.sum((y[hold] - m.predict(X[np.ix_(hold, cols)])) ** 2))
        count += len(hold)
    return (err / count) ** 0.5


def intervals(Xtr, ytr, Xte, level=0.95):
    n, p = Xtr.shape
    A = np.column_stack([np.ones(n), Xtr])
    beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    resid = ytr - A @ beta
    df = n - p - 1
    s = np.sqrt(resid @ resid / df)
    XtX_inv = np.linalg.pinv(A.T @ A)
    B = np.column_stack([np.ones(len(Xte)), Xte])
    h = np.einsum("ij,jk,ik->i", B, XtX_inv, B)
    fit = B @ beta
    t = stats.t.ppf(1 - (1 - level) / 2, df)
    return fit, h, t, s, df, fit - t * s * np.sqrt(h), fit + t * s * np.sqrt(h), \
        fit - t * s * np.sqrt(1 + h), fit + t * s * np.sqrt(1 + h)


for pid, r in js.items():
    out = {}
    sep = ";" if r["file"].startswith("student") else ","
    raw = pd.read_csv(DATA / r["file"], sep=sep)
    raw = raw.drop(columns=r["drop"])
    X = np.asarray(r["X"], dtype=float)
    y = np.asarray(r["y"], dtype=float)
    ids = np.asarray(r["ids"])

    # 1. Encoding: every column the app built must match the raw CSV.
    merged = {m["name"]: set(m["levels"]) for m in r["merged"]}
    enc_ok = check(len(ids) == len(raw), f"{pid}: all rows kept")
    for j, f in enumerate(r["features"]):
        col = raw[f["source"]].iloc[ids - 1]
        if f["kind"] == "number":
            expect = col.astype(float).to_numpy()
        elif f["level"] == "其他":
            expect = col.astype(str).isin(merged[f["source"]]).astype(float).to_numpy()
        else:
            expect = (col.astype(str) == f["level"]).astype(float).to_numpy()
        enc_ok &= check(np.allclose(X[:, j], expect), f"{pid}: encoding of {f['name']}")
    out["encoding_matches_csv"] = bool(enc_ok)
    out["y_matches_csv"] = check(np.allclose(y, raw[r["target"]].iloc[ids - 1].astype(float)), f"{pid}: target")

    # 2. Split: same rows as sklearn train_test_split(test_size=0.2, random_state=42).
    tr_idx, te_idx = train_test_split(np.arange(len(ids)), test_size=0.2, random_state=42)
    out["split_matches_sklearn"] = check(
        list(ids[te_idx]) == r["testIds"] and list(ids[tr_idx]) == r["trainIds"], f"{pid}: split")
    pos = {v: i for i, v in enumerate(ids)}
    tr = np.array([pos[v] for v in r["trainIds"]])
    te = np.array([pos[v] for v in r["testIds"]])
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    names = [f["name"] for f in r["features"]]
    col_of = {n: i for i, n in enumerate(names)}

    # 3. Filter rankings: Pearson |r| and sklearn f_regression on the training set.
    F, _ = f_regression(Xtr, ytr)
    corr = np.abs([np.corrcoef(Xtr[:, j], ytr)[0, 1] for j in range(X.shape[1])])
    k = r["methods"]["pearson"]["bestK"]
    out["pearson_topk_matches"] = check(
        [names[j] for j in np.argsort(-corr, kind="stable")[:k]] == r["methods"]["pearson"]["order"][:k],
        f"{pid}: pearson order")
    k = r["methods"]["selectkbest"]["bestK"]
    out["f_regression_topk_matches"] = check(
        [names[j] for j in np.argsort(-F, kind="stable")[:k]] == r["methods"]["selectkbest"]["order"][:k],
        f"{pid}: f_regression order")

    # 4. Forward selection path with the same CV loss, up to the chosen k.
    order, folds = r["cvOrder"], r["cvFolds"]
    fwd = r["methods"]["forward"]
    chosen, remaining = [], list(range(X.shape[1]))
    for _ in range(fwd["bestK"]):
        scores = []
        for j in remaining:
            try:
                scores.append((cv_rmse(Xtr, ytr, chosen + [j], order, folds), j))
            except np.linalg.LinAlgError:
                continue
        best = min(scores)
        chosen.append(best[1])
        remaining.remove(best[1])
    out["forward_path_matches"] = check([names[j] for j in chosen] == fwd["order"][:fwd["bestK"]],
                                        f"{pid}: forward path")

    # 5. Every method's chosen model: coefficients, metrics, CV RMSE and intervals.
    out["methods"] = {}
    for mid, m in r["methods"].items():
        b = m["best"]
        cols = [col_of[n] for n in b["names"]]
        lr = LinearRegression().fit(Xtr[:, cols], ytr)
        pred = lr.predict(Xte[:, cols])
        fit, h, t, s, df, ciL, ciH, piL, piH = intervals(Xtr[:, cols], ytr, Xte[:, cols])
        js_items = b["interval"]["items"]
        res = {
            "k": b["k"],
            "max_rel_coef_diff": max(rel(a, c) for a, c in zip(b["coefficients"], lr.coef_)),
            "intercept_rel_diff": rel(b["intercept"], lr.intercept_),
            "test_r2_sklearn": r2_score(yte, pred), "test_r2_js": b["test"]["r2"],
            "test_rmse_sklearn": mean_squared_error(yte, pred) ** 0.5, "test_rmse_js": b["test"]["rmse"],
            "test_mae_sklearn": mean_absolute_error(yte, pred), "test_mae_js": b["test"]["mae"],
            "cv_rmse_python": cv_rmse(Xtr, ytr, cols, order, folds), "cv_rmse_js": b["cv"],
            "t_scipy": float(t), "t_js": b["interval"]["t"], "df": int(df),
            "max_pi_bound_rel_diff": max(max(rel(i["piLow"], lo), rel(i["piHigh"], hi))
                                         for i, lo, hi in zip(js_items, piL, piH)),
            "max_ci_bound_rel_diff": max(max(rel(i["ciLow"], lo), rel(i["ciHigh"], hi))
                                         for i, lo, hi in zip(js_items, ciL, ciH)),
            "pi_coverage_python": float(np.mean((yte >= piL) & (yte <= piH))),
            "pi_coverage_js": b["interval"]["coverage"],
        }
        tol = 1e-6
        check(res["max_rel_coef_diff"] < tol and res["intercept_rel_diff"] < tol, f"{pid}/{mid}: coefficients")
        check(abs(res["test_r2_sklearn"] - res["test_r2_js"]) < tol, f"{pid}/{mid}: R2")
        check(rel(res["test_rmse_sklearn"], res["test_rmse_js"]) < tol, f"{pid}/{mid}: RMSE")
        check(rel(res["cv_rmse_python"], res["cv_rmse_js"]) < tol, f"{pid}/{mid}: CV RMSE")
        check(abs(res["t_scipy"] - res["t_js"]) < 1e-9, f"{pid}/{mid}: t quantile")
        check(res["max_pi_bound_rel_diff"] < tol and res["max_ci_bound_rel_diff"] < tol, f"{pid}/{mid}: intervals")
        check(res["pi_coverage_python"] == res["pi_coverage_js"], f"{pid}/{mid}: coverage")
        out["methods"][mid] = res

    # 6. Reference point: OLS on every usable feature (lstsq drops nothing; pinv handles collinearity).
    full_cols = list(range(X.shape[1]))
    rank = np.linalg.matrix_rank(np.column_stack([np.ones(len(tr)), Xtr]))
    fit, h, t, s, df, ciL, ciH, piL, piH = intervals(Xtr, ytr, Xte)
    out["all_features"] = {"p": len(full_cols), "rank_with_intercept": int(rank),
                           "test_r2": r2_score(yte, fit), "test_rmse": mean_squared_error(yte, fit) ** 0.5,
                           "pi_coverage": float(np.mean((yte >= piL) & (yte <= piH)))}
    report[pid] = out
    print(f"\n== {pid} ==")
    print({k: v for k, v in out.items() if k not in ("methods",)})
    for mid, res in out["methods"].items():
        print(f"  {mid:12s} k={res['k']:2d}  R2 js={res['test_r2_js']:.4f} sk={res['test_r2_sklearn']:.4f}"
              f"  coefΔ={res['max_rel_coef_diff']:.1e}  PIΔ={res['max_pi_bound_rel_diff']:.1e}"
              f"  CVΔ={rel(res['cv_rmse_python'], res['cv_rmse_js']):.1e}  cover={res['pi_coverage_js']:.3f}")

report["failures"] = failures
(HERE / "crosscheck_result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nFAILURES:", failures or "none")
