"""nb_12_reanalysis - CSB Reports revision: post-hoc re-analysis of EXISTING outputs (no LLM key, no new pipeline run).

Answers reviewer requests (CSBJ-D-26-00878):
  R1-4 / R2 / R3  accuracy beyond coarse lineage: confusion matrix, balanced accuracy, macro-F1,
                  per-lineage recall, Wilson/bootstrap CIs, McNemar vs baselines, fine-grained template
  R2 / R3         lineage-map sensitivity (2-class, 3-class, unmapped handling)
  R3              Brier, ECE, risk-coverage / AURC, predefined >=0.95 threshold
  R1-7 / R3       sensitivity of confidence weights, agreement/conflict multipliers, abstention threshold

Reads (per dataset):
  agent_evidence/<DS>/orchestrated.json, benchmark_10a.json, llm_baseline_narrator.json (optional),
  agent_evidence/<DS>/*singler*.json (optional), DEG/<DS>/cluster_ground_truth.csv
Writes: ROOT/reanalysis/*.csv, reanalysis_12.json

Runner:
    DATASETS = ["GSE131907", "GSE132465"]
    exec(open("/lakehouse/default/Files/scRNA/code/nb_12_reanalysis.py").read())
"""
import json, glob, itertools, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

DATASETS = globals().get("DATASETS", ["GSE131907", "GSE132465"])
ROOT = Path("/lakehouse/default/Files/scRNA")
OUT = ROOT/"reanalysis"; OUT.mkdir(exist_ok=True)
FAMS = ["lymphoid", "myeloid", "epithelial", "stromal"]
NBOOT = 5000
rng = np.random.default_rng(0)

# ---- same lineage map as nb_10a (copied; keep in sync) ----
_FAM = [("alveolar macrophage","myeloid"),("goblet","epithelial"),("ductal","epithelial"),
 ("secretory","epithelial"),("tuft","epithelial"),("enterocyte","epithelial"),("colonocyte","epithelial"),
 ("paneth","epithelial"),("enteroendocrine","epithelial"),("ciliated","epithelial"),("alveolar","epithelial"),
 ("club","epithelial"),("surfactant","epithelial"),("luminal","epithelial"),("basal cell","epithelial"),
 ("lung cell","epithelial"),("lung epithel","epithelial"),("cancer","epithelial"),("malignant","epithelial"),
 ("carcinoma","epithelial"),("epitheli","epithelial"),("mast","myeloid"),("plasmacytoid dendritic","myeloid"),
 ("plasmacytoid","myeloid"),("dendritic","myeloid"),("macrophage","myeloid"),("classical monocyte","myeloid"),
 ("monocyte","myeloid"),("neutrophil","myeloid"),("microglia","myeloid"),("myeloid","myeloid"),
 ("fibroblast","stromal"),("myofibroblast","stromal"),("endotheli","stromal"),("mesenchym","stromal"),
 ("pericyte","stromal"),("smooth muscle","stromal"),("emt","stromal"),("stromal","stromal"),
 ("t/nk","lymphoid"),("nk/t","lymphoid"),("natural killer","lymphoid"),("nkt","lymphoid"),("nk cell","lymphoid"),
 ("plasma cell","lymphoid"),("plasma","lymphoid"),("cd4","lymphoid"),("cd8","lymphoid"),("cytotoxic t","lymphoid"),
 ("regulatory t","lymphoid"),("treg","lymphoid"),("mature b","lymphoid"),("b cell","lymphoid"),
 ("b lymph","lymphoid"),("t lymph","lymphoid"),("t cell","lymphoid"),("lymphocyte","lymphoid"),("t-cell","lymphoid"),("b-cell","lymphoid"),("caf","stromal"),("pvl","stromal"),("perivascular","stromal")]
def family(label):
    t = (label or "").lower()
    for k, f in _FAM:
        if k in t: return f
    return None
COARSE2 = {"lymphoid":"immune","myeloid":"immune","epithelial":"nonimmune","stromal":"nonimmune"}
COARSE3 = {"lymphoid":"immune","myeloid":"immune","epithelial":"epithelial","stromal":"stromal"}

# ---------------------------------------------------------------- load
def load(ds):
    ev = ROOT/"agent_evidence"/ds
    orch = json.load(open(ev/"orchestrated.json"))
    gt = pd.read_csv(ROOT/"DEG"/ds/"cluster_ground_truth.csv", index_col=0); gt.index = gt.index.astype(str)
    bench = json.load(open(ev/"benchmark_10a.json")) if (ev/"benchmark_10a.json").exists() else {}
    calls = {"scINTERP": {cl: o["marker_call"] for cl, o in orch.items()},
             "CellTypist": bench.get("celltypist_calls", {})}
    p = ev/"llm_baseline_narrator.json"
    if p.exists(): calls["naive_LLM"] = json.load(open(p)).get("naive_baseline", {})
    for f in glob.glob(str(ev/"*singler*.json")):
        d = json.load(open(f))
        d = d.get("singler_calls", d.get("calls", d)) if isinstance(d, dict) else {}
        if d and all(isinstance(v, str) for v in d.values()): calls["SingleR"] = d
    rows = []
    for cl, o in orch.items():
        a = gt.loc[cl, "majority_author_label"] if cl in gt.index else None
        r = dict(dataset=ds, cluster=cl, author=a, author_fam=family(a), conf=o["confidence"],
                 c_marker=o["c_marker"], c_pathway=o["c_pathway"], c_disease=o["c_disease"],
                 agree=bool(o["agreement"]), conflict=any(f.startswith("AGENT_DISAGREEMENT") for f in o["flags"]),
                 abstain=bool(o["abstain"]))
        for m, cc in calls.items():
            r[f"call_{m}"] = cc.get(cl); r[f"fam_{m}"] = family(cc.get(cl))
            r[f"ok_{m}"] = (r["author_fam"] is not None and r[f"fam_{m}"] == r["author_fam"])
        rows.append(r)
    return pd.DataFrame(rows), list(calls.keys())

frames, methods = [], None
for ds in DATASETS:
    d, m = load(ds); frames.append(d); methods = m if methods is None else [x for x in methods if x in m]
df = pd.concat(frames, ignore_index=True)
print(f"clusters: {len(df)} across {DATASETS}; methods with calls: {methods}")
unm = df[df.author_fam.isna()]
print(f"clusters whose AUTHOR label is unmapped by the lineage map: {len(unm)}", list(unm.author.unique()))

# ---------------------------------------------------------------- 1. accuracy suite
def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k/n; den = 1+z*z/n; c = (p+z*z/(2*n))/den; h = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return (c-h, c+h)

def _macro(y, p, labs):
    rec, f1 = [], []
    for f in labs:
        tp = np.sum((y == f) & (p == f)); fn = np.sum((y == f) & (p != f)); fp = np.sum((y != f) & (p == f))
        r = tp/(tp+fn) if tp+fn else 0.0; pr = tp/(tp+fp) if tp+fp else 0.0
        rec.append(r); f1.append(2*pr*r/(pr+r) if pr+r else 0.0)
    return float(np.mean(rec)), float(np.mean(f1))

def suite(sub, m):
    y = sub["author_fam"].values.astype(object); p = sub[f"fam_{m}"].fillna("none").values.astype(object)
    ok = sub[f"ok_{m}"].values
    labs = [f for f in FAMS if f in set(y)]
    bal, mf1 = _macro(y, p, labs)
    bb, bf = [], []
    for _ in range(NBOOT):
        i = rng.integers(0, len(y), len(y))
        l2 = [f for f in FAMS if f in set(y[i])]
        a, b = _macro(y[i], p[i], l2); bb.append(a); bf.append(b)
    wl = wilson(int(ok.sum()), len(ok))
    per = {f: (float(np.mean(p[y == f] == f)) if (y == f).any() else None) for f in FAMS}
    return dict(n=len(y), acc=float(ok.mean()), acc_ci=[wl[0], wl[1]], balanced_acc=bal,
                bal_ci=[float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))],
                macro_f1=mf1, f1_ci=[float(np.percentile(bf, 2.5)), float(np.percentile(bf, 97.5))],
                per_lineage_recall=per, support={f: int((y == f).sum()) for f in FAMS})

dfm = df[df.author_fam.notna()].copy()
res = {}
print("\n" + "="*72 + "\n1. ACCURACY SUITE (cluster-level; lineage equivalence)\n" + "="*72)
for scope, sub in [("pooled", dfm)] + [(ds, dfm[dfm.dataset == ds]) for ds in DATASETS]:
    for m in methods:
        s = suite(sub, m); res[f"{scope}|{m}"] = s
        print(f"{scope:10} {m:11} n={s['n']:>2} acc={s['acc']:.3f} [{s['acc_ci'][0]:.2f},{s['acc_ci'][1]:.2f}] "
              f"bal={s['balanced_acc']:.3f} [{s['bal_ci'][0]:.2f},{s['bal_ci'][1]:.2f}] "
              f"macroF1={s['macro_f1']:.3f} [{s['f1_ci'][0]:.2f},{s['f1_ci'][1]:.2f}]")
        print("            per-lineage recall:", {k: (None if v is None else round(v, 2)) for k, v in s["per_lineage_recall"].items()},
              "support:", s["support"])

# confusion matrices
for scope, sub in [("pooled", dfm)] + [(ds, dfm[dfm.dataset == ds]) for ds in DATASETS]:
    for m in methods:
        cm = pd.crosstab(sub["author_fam"], sub[f"fam_{m}"].fillna("none"))
        cm.to_csv(OUT/f"confusion_{scope}_{m}.csv")
print("\nconfusion matrices -> reanalysis/confusion_*.csv")

# McNemar (exact) framework vs each baseline, pooled
print("\nMcNemar exact (pooled; discordant b = scINTERP right/baseline wrong, c = reverse):")
for m in methods:
    if m == "scINTERP": continue
    b = int(((dfm.ok_scINTERP) & (~dfm[f"ok_{m}"])).sum()); c = int(((~dfm.ok_scINTERP) & (dfm[f"ok_{m}"])).sum())
    p = binomtest(min(b, c), b+c, 0.5).pvalue if b+c > 0 else 1.0
    res[f"mcnemar|{m}"] = dict(b=b, c=c, p=float(p))
    print(f"  vs {m:11} b={b} c={c} p={p:.3f}")

# ---------------------------------------------------------------- 2. mapping sensitivity + fine-grained
print("\n" + "="*72 + "\n2. LINEAGE-MAP SENSITIVITY AND FINE-GRAINED TEMPLATE\n" + "="*72)
def score_variant(name, fn_a, unmapped_wrong=True):
    out = {}
    for m in methods:
        ok, n = 0, 0
        for _, r in df.iterrows():
            a = fn_a(r["author_fam"]); p = fn_a(r[f"fam_{m}"])
            if a is None:
                if unmapped_wrong: n += 1
                continue
            n += 1; ok += int(a == p)
        out[m] = ok/n if n else np.nan
    return out
variants = {"4-class (primary)": score_variant("4", lambda f: f),
            "3-class (immune/epithelial/stromal)": score_variant("3", lambda f: COARSE3.get(f)),
            "2-class (immune/non-immune)": score_variant("2", lambda f: COARSE2.get(f)),
            "4-class, unmapped author labels dropped": score_variant("4d", lambda f: f, unmapped_wrong=False)}
# extended synonym map: tissue-specific epithelial names the primary map leaves unmapped (disclosed as a sensitivity variant,
# NOT the primary result; added after seeing that SingleR calls "pancreatic acinar cell" / "lactocyte" were scored as unmapped)
_EXT = [("lactocyte","epithelial"),("acinar","epithelial"),("mammary","epithelial"),("hepatocyte","epithelial")]
def _fam_ext(x):
    f = family(x)
    if f: return f
    t = (x or "").lower()
    for k, v in _EXT:
        if k in t: return v
    return None
_ext = {}
for m in methods:
    ok = n = 0
    for _, r in df.iterrows():
        a = r["author_fam"]; n += 1
        if a is None: continue
        ok += int(_fam_ext(r[f"call_{m}"]) == a)
    _ext[m] = ok/n if n else np.nan
variants["4-class, extended epithelial synonyms (acinar/lactocyte/mammary)"] = _ext
vt = pd.DataFrame(variants).T.round(3); print(vt.to_string()); vt.to_csv(OUT/"mapping_sensitivity.csv")
res["mapping_sensitivity"] = vt.to_dict()

tpl = df[["dataset", "cluster", "author"] + [f"call_{m}" for m in methods]].copy()
def norm(s): return "".join(ch for ch in str(s).lower() if ch.isalnum() or ch == " ").replace("cells", "cell").strip()
tpl["auto_exact_or_contained"] = [int(norm(a) in norm(c) or norm(c) in norm(a)) if a and c else 0
                                  for a, c in zip(df.author, df["call_scINTERP"])]
tpl["curated_equivalent_Y_N"] = ""     # <-- a domain expert fills Y/N: is the call the same cell type/subtype as the author label?
tpl.to_csv(OUT/"finegrained_template.csv", index=False)
print("fine-grained template written: reanalysis/finegrained_template.csv  (fill curated_equivalent_Y_N, re-run to score)")
cur = OUT/"finegrained_curated.csv"
if cur.exists():
    c = pd.read_csv(cur)
    if "curated_equivalent_Y_N" in c:
        v = c["curated_equivalent_Y_N"].astype(str).str.upper().str[0].map({"Y": 1, "N": 0}).dropna()
        res["finegrained_scINTERP"] = dict(n=int(len(v)), acc=float(v.mean()), ci=list(wilson(int(v.sum()), len(v))))
        print("fine-grained (curated) accuracy:", res["finegrained_scINTERP"])

# ---------------------------------------------------------------- 3. confidence quality
print("\n" + "="*72 + "\n3. CONFIDENCE QUALITY (integrated score vs. held-out correctness)\n" + "="*72)
y = dfm.ok_scINTERP.astype(int).values; cf = dfm.conf.values
auroc = roc_auc_score(y, cf) if len(set(y)) > 1 else np.nan
brier = float(np.mean((cf-y)**2)); base_brier = float(np.mean((y.mean()-y)**2))
def ece(cf, y, bins, mode):
    if mode == "width": edges = np.linspace(0, 1, bins+1)
    else: edges = np.unique(np.quantile(cf, np.linspace(0, 1, bins+1)))
    idx = np.clip(np.digitize(cf, edges[1:-1], right=True), 0, len(edges)-2)
    tot = 0.0; table = []
    for b in range(len(edges)-1):
        s = idx == b
        if s.any():
            tot += s.mean()*abs(cf[s].mean()-y[s].mean()); table.append((b, int(s.sum()), float(cf[s].mean()), float(y[s].mean())))
    return float(tot), table
e_w, t_w = ece(cf, y, 5, "width"); e_q, t_q = ece(cf, y, 5, "mass")
print(f"AUROC={auroc:.3f}  Brier={brier:.3f} (constant-rate reference {base_brier:.3f})  ECE(5 equal-width)={e_w:.3f}  ECE(5 equal-mass)={e_q:.3f}")
print("NOTE: the score is an evidence-integration index, not a probability; Brier/ECE are descriptive and expected to be poor in absolute terms.")
print("reliability (equal-mass): bin, n, mean_conf, frac_correct"); [print("  ", r) for r in t_q]
# risk-coverage
order = np.argsort(-cf, kind="stable"); ys = y[order]
risk = 1-np.cumsum(ys)/np.arange(1, len(ys)+1); cov = np.arange(1, len(ys)+1)/len(ys)
aurc = float(np.mean(risk)); rand_aurc = float(1-y.mean())
print(f"AURC={aurc:.3f} (random ordering {rand_aurc:.3f}; lower is better)")
for c in [0.5, 0.7, 0.9, 1.0]:
    k = max(1, int(round(c*len(ys)))); print(f"  coverage {c:.0%}: selective accuracy {ys[:k].mean():.3f} (n={k})")
pd.DataFrame(dict(coverage=cov, selective_risk=risk, conf_cut=cf[order])).to_csv(OUT/"risk_coverage.csv", index=False)
kept = dfm[dfm.conf >= 0.95]
print(f"Predefined threshold conf>=0.95 (declared a priori in the Methods): kept {len(kept)}/{len(dfm)}, accuracy {kept.ok_scINTERP.mean():.3f}")
res["confidence"] = dict(auroc=float(auroc), brier=brier, brier_ref=base_brier, ece_width=e_w, ece_mass=e_q,
                         aurc=aurc, aurc_random=rand_aurc, kept_095=int(len(kept)), acc_095=float(kept.ok_scINTERP.mean()) if len(kept) else None)

# ---------------------------------------------------------------- 4. parameter sensitivity
print("\n" + "="*72 + "\n4. SENSITIVITY OF WEIGHTS / MULTIPLIERS / THRESHOLD\n" + "="*72)
print("The cell-type LABEL is the marker agent's top-1 call, so accuracy is invariant to every parameter below;")
print("these parameters only change the confidence score, hence AUROC, abstention and the >=threshold subset.")
def _auc(y, c):
    pos = c[y == 1]; neg = c[y == 0]
    if len(pos) == 0 or len(neg) == 0: return np.nan
    return float((np.sum(pos[:, None] > neg[None, :]) + 0.5*np.sum(pos[:, None] == neg[None, :]))/(len(pos)*len(neg)))
def conf_of(w, boost, pen, d=dfm):
    base = w[0]*d.c_marker + w[1]*d.c_pathway + w[2]*d.c_disease
    mult = np.where(d.agree, boost, np.where(d.conflict, pen, 1.0))
    return np.clip(base*mult, 0, 1).values
grid = []
steps = np.round(np.arange(0.05, 0.95, 0.05), 2)
for wm in steps:
    for wp in steps:
        wd = round(1-wm-wp, 2)
        if wd < 0.05: continue
        for boost in [1.0, 1.05, 1.10, 1.20]:
            for pen in [0.6, 0.8, 1.0]:
                c = conf_of((wm, wp, wd), boost, pen)
                au = _auc(y, c)
                grid.append(dict(wm=wm, wp=wp, wd=wd, boost=boost, pen=pen, auroc=au))
g = pd.DataFrame(grid); g.to_csv(OUT/"weight_grid.csv", index=False)
print(f"grid points: {len(g)}  AUROC min/median/max = {g.auroc.min():.3f} / {g.auroc.median():.3f} / {g.auroc.max():.3f}")
print(f"fraction of grid with AUROC >= 0.5: {(g.auroc >= 0.5).mean():.2f}; >= 0.70: {(g.auroc >= 0.70).mean():.2f}")
chk = float(np.max(np.abs(conf_of((0.5, 0.35, 0.15), 1.10, 0.80) - dfm.conf.values)))
print(f"self-check: max |recomputed - stored confidence| at default settings = {chk:.4f} (should be ~0.002 or less; rounding only)")
named = {"default 0.50/0.35/0.15": (0.5, 0.35, 0.15), "equal 1/3 each": (1/3, 1/3, 1/3),
         "marker-only": (1.0, 0.0, 0.0), "marker+pathway 0.6/0.4/0": (0.6, 0.4, 0.0), "no-disease 0.5/0.5": (0.5, 0.5, 0.0)}
for nm, w in named.items():
    c = conf_of(w, 1.10, 0.80); au = roc_auc_score(y, c) if len(set(y)) > 1 else np.nan
    print(f"  {nm:28} AUROC={au:.3f}")
no_mult = conf_of((0.5, 0.35, 0.15), 1.0, 1.0)
print(f"  default weights, multipliers disabled   AUROC={roc_auc_score(y, no_mult):.3f}")
print("\nAbstention threshold sweep (default weights) - n abstaining / accuracy of retained:")
c0 = conf_of((0.5, 0.35, 0.15), 1.10, 0.80); thr_rows = []
for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    k = c0 >= t; thr_rows.append(dict(threshold=t, n_abstain=int((~k).sum()), n_kept=int(k.sum()),
                                      acc_kept=float(y[k].mean()) if k.any() else None))
    print("  ", thr_rows[-1])
res["sensitivity"] = dict(grid_auroc_min=float(g.auroc.min()), grid_auroc_median=float(g.auroc.median()),
                          grid_auroc_max=float(g.auroc.max()), threshold_sweep=thr_rows)

json.dump(res, open(OUT/"reanalysis_12.json", "w"), indent=1, default=str)
df.to_csv(OUT/"per_cluster_all_methods.csv", index=False)
print("\n" + "="*60 + f"\nNOTEBOOK 12 COMPLETE -> {OUT}\n" + "="*60)
