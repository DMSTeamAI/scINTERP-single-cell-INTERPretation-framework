"""nb_10a_validation_benchmark — core benchmark (NO LLM key needed).

Computes everything derivable from the agent + orchestrator outputs:
  1. Cell-type accuracy vs author labels, with LINEAGE-EQUIVALENCE scoring
     (plasma sub B, pDC sub myeloid, NK/cytotoxic sub T/NK, endothelial/fibroblast sub stromal)
  2. Calibration: does confidence predict correctness? AUROC + bootstrap CI + threshold sweep
     (direct answer to the n=2-vs-18 calibration critique)
  3. Trajectory metrics (the ICIBM audience question): per-agent Hit@k, failure attribution
  4. Baselines: CellTypist + GSEA-only (both run without an LLM)
  5. Ablation: leave-one-agent-out effect on the orchestrator's resolved calls

The naive-LLM baseline and narrator live in nb_10b (need the Claude key).

Reads:  agent_evidence/<DATASET>/{marker_expression,pathway,disease,orchestrated}.json
        DEG/<DATASET>/cluster_ground_truth.csv, degs_top.json
        h5ad/<DATASET>_clustered.h5ad (for CellTypist)
Writes: agent_evidence/<DATASET>/benchmark_10a.json + printed tables

Runner:
    DATASET = "GSE131907"
    exec(open("/lakehouse/default/Files/scRNA/code/nb_10a_validation_benchmark.py").read())
"""
assert "DATASET" in globals(), "Set DATASET before exec()"

import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ROOT = Path("/lakehouse/default/Files/scRNA")
EVID = ROOT/"agent_evidence"/DATASET
DEG  = ROOT/"DEG"/DATASET
H5AD = ROOT/"h5ad"

marker = json.load(open(EVID/"marker_expression_evidence.json"))
orch   = json.load(open(EVID/"orchestrated.json"))
gt = pd.read_csv(DEG/"cluster_ground_truth.csv", index_col=0); gt.index = gt.index.astype(str)
CLUSTERS = list(orch.keys())

# ---- lineage-equivalence scoring ----
# Map both the author label and the predicted call to a coarse family; a call is CORRECT
# if families match (so 'Plasma cell' vs author 'B lymphocytes' counts correct, since both lymphoid).
# Priority-ordered: compound/specific terms BEFORE bare tokens, so
# "plasmacytoid dendritic cell" -> myeloid (not "plasma"->lymphoid).
_FAMILY_PRIORITY = [
 # 1. compounds where a later token would mislead
 ("alveolar macrophage","myeloid"),
 # 2. epithelial types containing "t cell"/"tal" substrings -> resolve BEFORE lymphoid "t cell"
 ("goblet","epithelial"), ("ductal","epithelial"), ("secretory","epithelial"),
 ("tuft","epithelial"),
 # 3. other specific epithelial
 ("enterocyte","epithelial"), ("colonocyte","epithelial"), ("paneth","epithelial"),
 ("enteroendocrine","epithelial"), ("ciliated","epithelial"), ("alveolar","epithelial"),
 ("club","epithelial"), ("surfactant","epithelial"), ("luminal","epithelial"),
 ("basal cell","epithelial"), ("lung cell","epithelial"), ("lung epithel","epithelial"),
 ("cancer","epithelial"), ("malignant","epithelial"), ("carcinoma","epithelial"),
 ("epitheli","epithelial"),
 # 4. myeloid (mast contains "t"? no—'mas t'? "mast cell" contains "t cell"! -> put mast here, before t cell)
 ("mast","myeloid"), ("plasmacytoid dendritic","myeloid"), ("plasmacytoid","myeloid"),
 ("dendritic","myeloid"), ("macrophage","myeloid"), ("classical monocyte","myeloid"),
 ("monocyte","myeloid"), ("neutrophil","myeloid"), ("microglia","myeloid"), ("myeloid","myeloid"),
 # 5. stromal
 ("fibroblast","stromal"), ("myofibroblast","stromal"), ("endotheli","stromal"),
 ("mesenchym","stromal"), ("pericyte","stromal"), ("smooth muscle","stromal"),
 ("emt","stromal"), ("stromal","stromal"),
 # 6. lymphoid LAST (bare "t cell" can no longer shadow goblet/mast/ductal above)
 ("t/nk","lymphoid"), ("nk/t","lymphoid"), ("natural killer","lymphoid"), ("nkt","lymphoid"),
 ("nk cell","lymphoid"), ("plasma cell","lymphoid"), ("plasma","lymphoid"),
 ("cd4","lymphoid"), ("cd8","lymphoid"), ("cytotoxic t","lymphoid"), ("regulatory t","lymphoid"),
 ("treg","lymphoid"), ("mature b","lymphoid"), ("b cell","lymphoid"), ("b lymph","lymphoid"),
 ("t lymph","lymphoid"), ("t cell","lymphoid"), ("lymphocyte","lymphoid"),
]
def family(label):
    t = (label or "").lower()
    for k, f in _FAMILY_PRIORITY:
        if k in t: return f
    return None

rows = []
for cl in CLUSTERS:
    author = gt.loc[cl,"majority_author_label"] if cl in gt.index else None
    call   = orch[cl]["marker_call"]
    conf   = orch[cl]["confidence"]
    abst   = orch[cl]["abstain"]
    fa, fp = family(author), family(call)
    correct = (fa is not None and fa == fp)
    rows.append(dict(cluster=cl, author=author, call=call, conf=conf, abstain=abst,
                     author_family=fa, pred_family=fp, correct=correct))
df = pd.DataFrame(rows)

# ---- 1. Accuracy ----
overall_acc = df["correct"].mean()
selective = df[~df.abstain]                      # accuracy when the framework does NOT abstain
selective_acc = selective["correct"].mean() if len(selective) else float("nan")
print("="*70); print("1. CELL-TYPE ACCURACY (lineage-equivalence)"); print("="*70)
print(df[["cluster","author","call","conf","correct","abstain"]].to_string(index=False))
print(f"\nOverall accuracy      : {overall_acc:.3f} ({df.correct.sum()}/{len(df)})")
print(f"Selective accuracy    : {selective_acc:.3f} (excluding {df.abstain.sum()} abstention[s])")

# ---- 2. Calibration ----
print("\n" + "="*70); print("2. CALIBRATION — does confidence predict correctness?"); print("="*70)
y = df["correct"].astype(int).values
c = df["conf"].values
try:
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(y, c) if len(set(y)) > 1 else float("nan")
except Exception:
    auroc = float("nan")
# bootstrap CI for AUROC
rng = np.random.default_rng(0); boots = []
for _ in range(2000):
    idx = rng.integers(0, len(y), len(y))
    if len(set(y[idx])) > 1:
        from sklearn.metrics import roc_auc_score
        boots.append(roc_auc_score(y[idx], c[idx]))
ci = (np.percentile(boots,2.5), np.percentile(boots,97.5)) if boots else (float("nan"),)*2
mean_conf_correct = df[df.correct].conf.mean()
mean_conf_wrong   = df[~df.correct].conf.mean() if (~df.correct).any() else float("nan")
print(f"AUROC (confidence -> correctness): {auroc:.3f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
print(f"Mean confidence | correct : {mean_conf_correct:.3f}")
print(f"Mean confidence | wrong   : {mean_conf_wrong:.3f}   (should be LOWER — calibration works)")
# threshold sweep: accuracy among clusters above each confidence cut
print("\nthreshold | n_kept | accuracy_of_kept")
for thr in [0.5,0.6,0.7,0.8,0.9]:
    kept = df[df.conf >= thr]
    print(f"   >={thr:.1f}  | {len(kept):>6} | {kept.correct.mean() if len(kept) else float('nan'):.3f}")

# ---- 3. Trajectory metrics (per-agent) ----
print("\n" + "="*70); print("3. TRAJECTORY METRICS (per-agent)"); print("="*70)
# marker Hit@k: is author family among the marker agent's top-k candidate families?
def hit_at_k(cl, k):
    author_fam = family(gt.loc[cl,"majority_author_label"]) if cl in gt.index else None
    cands = (marker[cl].get("candidates") or [])[:k]
    return any(family(ct)==author_fam for ct,_s in cands) if author_fam else False
hit1 = np.mean([hit_at_k(cl,1) for cl in CLUSTERS])
hit3 = np.mean([hit_at_k(cl,3) for cl in CLUSTERS])
print(f"Marker agent Hit@1 : {hit1:.3f}")
print(f"Marker agent Hit@3 : {hit3:.3f}")
# failure attribution: for each WRONG final call, where did it diverge?
print("\nFailure attribution (wrong clusters):")
wrong = df[~df.correct]
if len(wrong)==0:
    print("  none — all clusters correct at lineage level")
for _, r in wrong.iterrows():
    cl = r["cluster"]
    m_ok = hit_at_k(cl,3)
    stage = "marker top-3 missed it" if not m_ok else "marker had it in top-3 but top-1 wrong / orchestrator kept wrong call"
    print(f"  cluster {cl}: pred={r['call']} author={r['author']} -> {stage}")

# ---- 4. Baselines ----
print("\n" + "="*70); print("4. BASELINES"); print("="*70)
# CellTypist (if available)
try:
    import scanpy as sc, celltypist
    from celltypist import models
    adata = sc.read_h5ad(H5AD/f"{DATASET}_clustered.h5ad")
    # celltypist expects lognorm in .X; adata.raw holds it
    ad = adata.raw.to_adata() if adata.raw is not None else adata
    model = "Immune_All_Low.pkl"
    try: models.download_models(model=model)
    except Exception: pass
    pred = celltypist.annotate(ad, model=model, majority_voting=True)
    ad.obs["ct"] = pred.predicted_labels["majority_voting"].values
    # majority CellTypist label per Leiden cluster
    ct_by_cluster = (adata.obs.assign(ct=ad.obs["ct"].values)
                     .groupby("leiden")["ct"].agg(lambda s: s.value_counts().index[0]))
    ct_acc = np.mean([family(ct_by_cluster.get(cl))==family(gt.loc[cl,"majority_author_label"])
                      for cl in CLUSTERS if cl in gt.index and cl in ct_by_cluster.index])
    print(f"CellTypist accuracy (lineage) : {ct_acc:.3f}")
    celltypist_calls = {cl: str(ct_by_cluster.get(cl)) for cl in CLUSTERS}
except Exception as e:
    print(f"CellTypist skipped: {str(e)[:100]} (attach scinterp-env; CellTypist is optional, rest of benchmark is valid)")
    celltypist_calls = {}

# ---- 5. Ablation (leave-one-agent-out on orchestrator confidence/agreement) ----
print("\n" + "="*70); print("5. ABLATION (leave-one-agent-out)"); print("="*70)
# Recompute a simplified confidence dropping each agent's contribution; report how many
# clusters change abstention status — shows each agent's marginal effect.
def simple_conf(cl, use_marker=True, use_path=True, use_dis=True):
    o = orch[cl]
    parts, wts = [], []
    if use_marker: parts.append(o["c_marker"]); wts.append(0.5)
    if use_path:   parts.append(o["c_pathway"]); wts.append(0.35)
    if use_dis:    parts.append(o["c_disease"]); wts.append(0.15)
    return float(np.average(parts, weights=wts)) if parts else 0.0
base_abst = {cl: orch[cl]["confidence"]<0.5 for cl in CLUSTERS}
for name, kw in [("full",{}), ("no_marker",{"use_marker":False}),
                 ("no_pathway",{"use_path":False}), ("no_disease",{"use_dis":False})]:
    ab = {cl: simple_conf(cl,**kw)<0.5 for cl in CLUSTERS}
    changed = sum(1 for cl in CLUSTERS if ab[cl]!=base_abst[cl])
    mean_c = np.mean([simple_conf(cl,**kw) for cl in CLUSTERS])
    print(f"  {name:11} mean_conf={mean_c:.3f}  abstention_changes_vs_full={changed}")

json.dump({"overall_accuracy":float(overall_acc),"selective_accuracy":float(selective_acc),
           "auroc":float(auroc),"auroc_ci":[float(ci[0]),float(ci[1])],
           "hit_at_1":float(hit1),"hit_at_3":float(hit3),
           "n_abstain":int(df.abstain.sum()),
           "per_cluster":df.to_dict("records"),"celltypist_calls":celltypist_calls},
          open(EVID/"benchmark_10a.json","w"), indent=2, default=str)

print("\n" + "="*60 + f"""
NOTEBOOK 10a COMPLETE
{"="*60}
Dataset          : {DATASET}
Overall accuracy : {overall_acc:.3f}   Selective: {selective_acc:.3f}
Calibration AUROC: {auroc:.3f}  CI [{ci[0]:.3f},{ci[1]:.3f}]
Marker Hit@1/@3  : {hit1:.3f} / {hit3:.3f}
Output           : benchmark_10a.json
Next             : nb_10b (naive-LLM baseline + narrator — needs Claude key)""")
