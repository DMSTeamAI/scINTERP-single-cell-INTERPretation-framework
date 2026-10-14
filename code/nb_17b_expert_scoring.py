"""nb_17b_expert_scoring - score the returned expert sheets (run after raters save *_DONE.xlsx in reanalysis/expert/).

  * fine-grained accuracy per method (curated Y/N), pooled and per dataset, Wilson 95% CI, exact McNemar vs scINTERP
  * narrator audit: per rater and condition - fraction of texts with any unsupported claim, mean n_unsupported;
    Cohen's kappa (any_unsupported) and quadratic-weighted kappa (n_unsupported) between the two raters;
    constrained vs unconstrained (Fisher exact, rater consensus = both raters Y); agreement of raters with the LLM judge.
Runner:  exec(open("/lakehouse/default/Files/scRNA/code/nb_17b_expert_scoring.py").read())
"""
from pathlib import Path
import json
import numpy as np, pandas as pd
from scipy.stats import fisher_exact, binomtest
ROOT = Path("/lakehouse/default/Files/scRNA"); OUT = ROOT/"reanalysis"/"expert"

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k/n; den = 1+z*z/n; c = (p+z*z/(2*n))/den; h = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return (round(c-h, 3), round(c+h, 3))
def kappa(a, b, w=None, cats=None):
    a = np.asarray(a); b = np.asarray(b); cats = sorted(set(a) | set(b)) if cats is None else cats; k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}; O = np.zeros((k, k))
    for x, y in zip(a, b): O[idx[x], idx[y]] += 1
    O /= O.sum(); E = np.outer(O.sum(1), O.sum(0))
    W = np.zeros((k, k))
    for i in range(k):
        for j in range(k): W[i, j] = 0 if i == j else (1 if w is None else ((i-j)/(k-1))**2)
    den = (W*E).sum()
    return float(1-(W*O).sum()/den) if den > 0 else float("nan")
def yn(s): return s.astype(str).str.strip().str.upper().str[0].map({"Y": 1, "N": 0})
def ypn(s): return s.astype(str).str.strip().str.upper().str[0].map({"Y": 1, "P": 0.5, "N": 0})   # P = parent type right, fine label not

# ---------------- fine-grained (two curators) ----------------
# Input: finegrained_expert_DONE.xlsx. Either (i) Sheet1 = original columns + curator A block (curated_1..4, notes) + curator B block
# (4 curated columns, notes), read by POSITION; or (ii) a single sheet named REVIEW (one curator). Optional sheet ADJUDICATED
# (dataset, cluster, curated_1..4) resolves disagreements; without it the disagreements are listed and bounds are reported.
f = OUT/"finegrained_expert_DONE.xlsx"
if f.exists():
    print("="*70); print("FINE-GRAINED ACCURACY (expert-curated)"); print("="*70)
    xl = pd.ExcelFile(f); print("sheets:", xl.sheet_names)
    if "REVIEW" in xl.sheet_names:
        D = xl.parse("REVIEW"); curators = {"A": [f"curated_{i}" for i in range(1, 5)]}; blocks = {"A": D}
    else:
        D = xl.parse(xl.sheet_names[0]); base = list(D.columns)
        ci = [j for j, c in enumerate(base) if str(c).strip().lower().replace(" ", "_").startswith("curated")]
        print("curated-like column positions:", ci)
        assert len(ci) == 8, "expected 8 curated columns (4 per curator) in the first sheet; found %d" % len(ci)
        D = D.copy(); D.columns = [str(c) for c in D.columns]
        names = list(D.columns)
        for k in range(8): names[ci[k]] = f"{'A' if k < 4 else 'B'}_curated_{k % 4 + 1}"
        D.columns = names; D["cluster"] = D["cluster"].astype(str)
        curators = {"A": [f"A_curated_{i}" for i in range(1, 5)], "B": [f"B_curated_{i}" for i in range(1, 5)]}
    D["cluster"] = D["cluster"].astype(str)
    K = pd.read_csv(OUT/"finegrained_key.csv"); K["cluster"] = K["cluster"].astype(str)
    def long(cols):
        L = []
        for i, c in enumerate(cols, 1):
            t = D[["dataset", "cluster", c]].rename(columns={c: "ok"}); t["slot"] = i; L.append(t)
        L = pd.concat(L).merge(K, on=["dataset", "cluster", "slot"]); L["raw"] = ypn(L["ok"]); return L
    V = {c: long(cols) for c, cols in curators.items()}
    if "ADJUDICATED" in xl.sheet_names:
        adj = xl.parse("ADJUDICATED"); adj["cluster"] = adj["cluster"].astype(str)
        L = []
        for i in range(1, 5):
            t = adj[["dataset", "cluster", f"curated_{i}"]].rename(columns={f"curated_{i}": "ok"}); t["slot"] = i; L.append(t)
        V["adjudicated"] = pd.concat(L).merge(K, on=["dataset", "cluster", "slot"]); V["adjudicated"]["raw"] = ypn(V["adjudicated"]["ok"])
    if "A" in V and "B" in V:
        M2 = V["A"][["dataset", "cluster", "slot", "method", "raw"]].merge(V["B"][["dataset", "cluster", "slot", "raw"]], on=["dataset", "cluster", "slot"], suffixes=("_A", "_B")).dropna()
        ca, cb = M2.raw_A.map({1: "Y", 0.5: "P", 0: "N"}), M2.raw_B.map({1: "Y", 0.5: "P", 0: "N"})
        print(); print(f"INTER-CURATOR AGREEMENT over {len(M2)} call slots")
        print(f"  raw agreement (Y/P/N) = {(ca == cb).mean():.3f}; Cohen kappa (3 categories) = {kappa(ca, cb):.3f}; "
              f"quadratic-weighted kappa = {kappa(ca.map({'N':0,'P':1,'Y':2}), cb.map({'N':0,'P':1,'Y':2}), w='quad', cats=[0,1,2]):.3f}")
        sa, sb = (M2.raw_A == 1).astype(int), (M2.raw_B == 1).astype(int)
        print(f"  strict (Y vs not-Y): agreement = {(sa == sb).mean():.3f}; kappa = {kappa(sa, sb):.3f}")
        ta, tb = (M2.raw_A >= .5).astype(int), (M2.raw_B >= .5).astype(int)
        print(f"  tolerant (Y/P vs N): agreement = {(ta == tb).mean():.3f}; kappa = {kappa(ta, tb):.3f}")
        print("  confusion A (rows) x B (cols):"); print(pd.crosstab(ca, cb).to_string())
        dis = M2[ca != cb].assign(A=ca[ca != cb], B=cb[ca != cb])[["dataset", "cluster", "slot", "A", "B"]]
        dis.to_csv(OUT/"finegrained_disagreements.csv", index=False)
        print(f"  {len(dis)} disagreements -> finegrained_disagreements.csv (de-blinded method column NOT included; adjudicate blind, add sheet ADJUDICATED)")
        # derived scorings
        mk = lambda r: r.copy()
        Vc = M2.copy(); Vc["ok"] = ""   # placeholders
        both_strict = ((M2.raw_A == 1) & (M2.raw_B == 1)); either_strict = ((M2.raw_A == 1) | (M2.raw_B == 1))
        both_tol = ((M2.raw_A >= .5) & (M2.raw_B >= .5)); either_tol = ((M2.raw_A >= .5) | (M2.raw_B >= .5))
        base = M2[["dataset", "cluster", "slot", "method"]].copy()
        V["both_agree(conservative)"] = base.assign(raw=np.where(both_strict, 1.0, np.where(both_tol, 0.5, 0.0)))
        V["either(lenient)"] = base.assign(raw=np.where(either_strict, 1.0, np.where(either_tol, 0.5, 0.0)))
    out = []
    for name, L in V.items():
        miss = int(L.raw.isna().sum())
        if miss: print(f"WARNING [{name}]: {miss} cells without Y/P/N - dropped")
        L = L.dropna(subset=["raw"]).copy(); L["ok"] = (L.raw == 1).astype(int); L["tol"] = (L.raw >= 0.5).astype(int); V[name] = L
        for m in ["scINTERP", "CellTypist", "naive_LLM", "SingleR"]:
            for scope, sub in [("pooled", L[L.method == m])] + [(ds, L[(L.method == m) & (L.dataset == ds)]) for ds in sorted(L.dataset.unique())]:
                k, n = int(sub.ok.sum()), len(sub); lo, hi = wilson(k, n); kt = int(sub.tol.sum()); lot, hit = wilson(kt, n)
                out.append(dict(rating=name, method=m, scope=scope, n=n, strict_correct=k, strict_acc=round(k/n, 3) if n else np.nan, strict_ci_lo=lo, strict_ci_hi=hi,
                                tolerant_acc=round(kt/n, 3) if n else np.nan, tol_ci_lo=lot, tol_ci_hi=hit))
    R = pd.DataFrame(out); R.to_csv(OUT/"finegrained_results.csv", index=False)
    print(); print("POOLED fine-grained accuracy per rating scheme"); print(R[R.scope == "pooled"].drop(columns="scope").to_string(index=False))
    print(); print("PER DATASET (strict accuracy)"); print(R[R.scope != "pooled"].pivot_table(index=["rating", "scope"], columns="method", values="strict_acc").round(3).to_string())
    for name, L in V.items():
        P = L.pivot_table(index=["dataset", "cluster"], columns="method", values="ok")
        for m in ["CellTypist", "naive_LLM", "SingleR"]:
            if m in P and "scINTERP" in P:
                b = int(((P.scINTERP == 1) & (P[m] == 0)).sum()); c = int(((P.scINTERP == 0) & (P[m] == 1)).sum())
                p = binomtest(min(b, c), b+c, 0.5).pvalue if b+c else 1.0
                print(f"McNemar [{name}] strict, scINTERP vs {m}: b={b} c={c} p={p:.3f}")
else: print("finegrained_expert_DONE.xlsx not found - skipped")

# ---------------- narrator ----------------
fa, fb = OUT/"narrator_rater_A_DONE.xlsx", OUT/"narrator_rater_B_DONE.xlsx"
if fa.exists() and fb.exists():
    print(); print("="*70); print("NARRATOR AUDIT (two independent raters)"); print("="*70)
    key = pd.read_csv(OUT/"narrator_key.csv"); key["judge_any"] = key.judge_unsupported.apply(lambda s: int(len(json.loads(s)) > 0) if isinstance(s, str) else 0)
    A = pd.read_excel(fa, sheet_name="REVIEW"); B = pd.read_excel(fb, sheet_name="REVIEW")
    def prep(X, t):
        X = X[["item_id", "n_unsupported", "any_unsupported", "suggested_markers_only"]].copy()
        X["n"] = pd.to_numeric(X.n_unsupported, errors="coerce"); X["any"] = yn(X.any_unsupported)
        X["any"] = X["any"].fillna((X["n"] > 0).astype(float).where(X["n"].notna()))
        return X.rename(columns={"n": f"n_{t}", "any": f"any_{t}", "suggested_markers_only": f"sug_{t}"})[["item_id", f"n_{t}", f"any_{t}", f"sug_{t}"]]
    M = key.merge(prep(A, "A"), on="item_id").merge(prep(B, "B"), on="item_id").dropna(subset=["any_A", "any_B", "n_A", "n_B"])
    print(f"items rated by both: {len(M)}")
    print(f"Cohen kappa (any unsupported claim) = {kappa(M.any_A.astype(int), M.any_B.astype(int)):.3f};  "
          f"quadratic-weighted kappa on counts = {kappa(M.n_A.clip(0, 5).astype(int), M.n_B.clip(0, 5).astype(int), w='quad', cats=list(range(6))):.3f};  raw agreement = {(M.any_A == M.any_B).mean():.3f}")
    rows = []
    for cond in ["constrained", "unconstrained"]:
        s = M[M.condition == cond]
        rows.append(dict(condition=cond, n_texts=len(s), any_A=round(s.any_A.mean(), 3), any_B=round(s.any_B.mean(), 3),
                         mean_n_A=round(s.n_A.mean(), 2), mean_n_B=round(s.n_B.mean(), 2),
                         consensus_any=round(((s.any_A == 1) & (s.any_B == 1)).mean(), 3), judge_any=round(s.judge_any.mean(), 3)))
    R = pd.DataFrame(rows); print(R.to_string(index=False)); R.to_csv(OUT/"narrator_results.csv", index=False)
    c = M[M.condition == "constrained"]; u = M[M.condition == "unconstrained"]
    for tag in ["A", "B"]:
        t = [[int(c[f"any_{tag}"].sum()), len(c)-int(c[f"any_{tag}"].sum())], [int(u[f"any_{tag}"].sum()), len(u)-int(u[f"any_{tag}"].sum())]]
        print(f"Fisher exact constrained vs unconstrained (rater {tag}, texts with any unsupported claim): p={fisher_exact(t)[1]:.3f}  table={t}")
    cons = ((M.any_A == 1) & (M.any_B == 1)).astype(int)
    print(f"judge vs rater-consensus: kappa = {kappa(M.judge_any, cons):.3f}, judge flags {M.judge_any.mean():.2f} of texts vs raters' consensus {cons.mean():.2f}")
    print("texts where the only extra genes are suggested validation markers (rater A / B):", int((M.sug_A.astype(str).str.upper().str[0] == 'Y').sum()), "/", int((M.sug_B.astype(str).str.upper().str[0] == 'Y').sum()))
else: print(); print("narrator_rater_A_DONE.xlsx / narrator_rater_B_DONE.xlsx not both found - skipped")
print(); print("NOTEBOOK 17b COMPLETE")
