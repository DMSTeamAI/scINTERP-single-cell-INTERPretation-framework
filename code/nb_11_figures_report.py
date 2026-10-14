"""nb_11_figures_report - PUBLICATION-QUALITY figures + summary from pipeline outputs.

Restyled: clean typography, restrained palette, minimal chartjunk, panel labels (A-F),
plus UMAP panels colored by framework call and by confidence for both datasets.

Reads saved artifacts (no re-computation, no LLM). Writes 300-dpi PNG + PDF to
Files/scRNA/figures/paper/.

Runner (no DATASET needed):
    exec(open("/lakehouse/default/Files/scRNA/code/nb_11_figures_report.py").read())
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path("/lakehouse/default/Files/scRNA")
EV   = ROOT/"agent_evidence"
H5AD = ROOT/"h5ad"
FIG  = ROOT/"figures"/"paper"; FIG.mkdir(parents=True, exist_ok=True)
DATASETS = globals().get("DATASETS", ["GSE131907", "GSE132465", "GSE176078"])
NICE = {"GSE131907":"Lung (LUAD)", "GSE132465":"Colon (CRC)", "GSE176078":"Breast (BRCA)"}

# ---------- house style ----------
ACCENT   = "#2f5d8a"   # muted blue
ACCENT2  = "#c26b3e"   # muted terracotta
GRAY     = "#8a9199"
LIGHT    = "#d8dce0"
OK_C     = "#2f5d8a"
ERR_C    = "#b23a48"
FAMILY_COLORS = {"lymphoid":"#4a7ba6","myeloid":"#c98a3e","epithelial":"#5a9367",
                 "stromal":"#9b6a9e","unknown":"#b8bec4"}
plt.rcParams.update({
    "figure.dpi":300, "savefig.dpi":300, "savefig.bbox":"tight",
    "font.size":11, "font.family":"sans-serif",
    "font.sans-serif":["DejaVu Sans","Arial","Helvetica"],
    "axes.titlesize":12, "axes.titleweight":"regular", "axes.labelsize":11,
    "axes.spines.top":False, "axes.spines.right":False,
    "axes.edgecolor":"#4a4f55", "axes.linewidth":0.9,
    "axes.grid":True, "grid.color":LIGHT, "grid.linewidth":0.7, "grid.alpha":0.9,
    "xtick.color":"#4a4f55","ytick.color":"#4a4f55",
    "legend.frameon":False, "legend.fontsize":10,
    "axes.prop_cycle": plt.cycler(color=[ACCENT, ACCENT2, GRAY]),
})
def panel(ax, letter):
    ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top", ha="left")
def clean(ax):
    ax.tick_params(length=3, width=0.8)

def _load(ds, name): return json.load(open(EV/ds/name))
bench = {ds: _load(ds,"benchmark_10a.json") for ds in DATASETS}
per_cluster = pd.concat([pd.DataFrame(bench[ds]["per_cluster"]).assign(dataset=ds) for ds in DATASETS],
                        ignore_index=True)
# pooled statistics are recomputed here from the per-cluster benchmark files (the old pooled_calibration.json covered 2 datasets only)
def _auc(y, c):
    y = np.asarray(y).astype(int); c = np.asarray(c, float); pos = c[y == 1]; neg = c[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float((np.sum(pos[:, None] > neg[None, :]) + 0.5*np.sum(pos[:, None] == neg[None, :]))/(len(pos)*len(neg)))
_y = per_cluster["correct"].astype(int).values; _c = per_cluster["conf"].astype(float).values
_rng = np.random.default_rng(0); _bs = []
for _ in range(5000):
    _i = _rng.integers(0, len(_y), len(_y)); _a = _auc(_y[_i], _c[_i])
    if not np.isnan(_a): _bs.append(_a)
pooled = dict(pooled_accuracy=float(_y.mean()), pooled_auroc=_auc(_y, _c),
              auroc_ci=[round(float(np.percentile(_bs, 2.5)), 3), round(float(np.percentile(_bs, 97.5)), 3)] if _bs else [float("nan")]*2,
              n_errors=int((1-_y).sum()), n_abstain=int(per_cluster["abstain"].astype(bool).sum()), n_clusters=int(len(_y)))

def savefig(fig, stem):
    fig.savefig(FIG/f"{stem}.png"); fig.savefig(FIG/f"{stem}.pdf"); plt.close(fig)

# ========== FIG 1: accuracy vs CellTypist ==========
fig, ax = plt.subplots(figsize=(5.2+0.8*(len(DATASETS)-2),3.8))
x = np.arange(len(DATASETS)); w=0.34
fw = [bench[ds]["overall_accuracy"] for ds in DATASETS]
# CellTypist lineage accuracy recomputed from the stored per-cluster calls (same lineage map as nb_12)
_s12 = open(ROOT/"code"/"nb_12_reanalysis.py").read(); _n12 = {}
exec(_s12[_s12.index("_FAM = ["):_s12.index("COARSE2")], _n12); _fam12 = _n12["family"]
ctv = []
for _ds in DATASETS:
    _cc = bench[_ds].get("celltypist_calls", {})
    _ok = [(_fam12(_cc.get(str(r["cluster"]))) is not None) and (_fam12(_cc.get(str(r["cluster"]))) == _fam12(r["author"])) for r in bench[_ds]["per_cluster"]]
    ctv.append(float(np.mean(_ok)))
b1=ax.bar(x-w/2, fw, w, label="scINTERP", color=ACCENT, edgecolor="white", linewidth=0.6)
b2=ax.bar(x+w/2, ctv, w, label="CellTypist", color=LIGHT, edgecolor="#a8afb6", linewidth=0.6)
ax.set_xticks(x); ax.set_xticklabels([NICE[d] for d in DATASETS])
ax.set_ylabel("Accuracy (lineage-equivalence)"); ax.set_ylim(0,1.08)
ax.set_axisbelow(True); ax.grid(axis="x", visible=False)
for b,v in zip(b1,fw): ax.text(b.get_x()+b.get_width()/2, v+0.015, f"{v:.3f}", ha="center", fontsize=9, color=ACCENT)
for b,v in zip(b2,ctv): ax.text(b.get_x()+b.get_width()/2, v+0.015, f"{v:.3f}", ha="center", fontsize=9, color="#7c848c")
ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5,-0.28)); clean(ax)
savefig(fig, "fig1_accuracy")

# ========== FIG 2: reliability (pooled) ==========
fig, ax = plt.subplots(figsize=(4.6,4.6))
bins=[(0.0,0.7),(0.7,0.85),(0.85,0.95),(0.95,1.01)]
xs,ys,ns=[],[],[]
for lo,hi in bins:
    m=(per_cluster.conf>=lo)&(per_cluster.conf<hi)
    if m.any(): xs.append((lo+min(hi,1.0))/2); ys.append(per_cluster[m].correct.mean()); ns.append(int(m.sum()))
ax.plot([0,1],[0,1],"--",color=GRAY,lw=1.2,label="perfect calibration",zorder=1)
ax.plot(xs,ys,"-",color=ACCENT,lw=1.8,zorder=2)
ax.scatter(xs,ys,s=[30+18*n for n in ns],color=ACCENT,zorder=3,edgecolor="white",linewidth=1.2,label="observed (size ~ n)")
for xi,yi,ni in zip(xs,ys,ns): ax.annotate(f"n={ni}",(xi,yi),textcoords="offset points",xytext=(8,-12),fontsize=8,color="#4a4f55")
ax.set_xlabel("Mean predicted confidence"); ax.set_ylabel("Empirical accuracy")
ax.set_xlim(0,1.04); ax.set_ylim(0,1.06); ax.set_aspect("equal")
au = pooled["pooled_auroc"] if pooled else float("nan")
ci = pooled["auroc_ci"] if pooled else [float("nan")]*2
ax.set_title(f"Pooled calibration (n=36)\nAUROC {au:.2f}  95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")
ax.legend(loc="lower right"); clean(ax)
savefig(fig, "fig2_reliability")

# ========== FIG 3: selective prediction ==========
fig, ax = plt.subplots(figsize=(5.2+0.8*(len(DATASETS)-2),3.8))
thr=np.linspace(0.5,1.0,11); cov=[]; acc=[]; N=len(per_cluster)
for t in thr:
    k=per_cluster[per_cluster.conf>=t]; cov.append(len(k)/N); acc.append(k.correct.mean() if len(k) else np.nan)
ax.plot(cov,acc,"-",color=ACCENT,lw=1.8)
ax.scatter(cov,acc,s=34,color=ACCENT,edgecolor="white",linewidth=1,zorder=3)
for t,c,a in zip(thr,cov,acc):
    if t in (0.5,0.8,0.9,0.95): ax.annotate(f"c>={t:.2f}",(c,a),textcoords="offset points",xytext=(6,-11),fontsize=8,color="#4a4f55")
ax.set_xlabel("Coverage (fraction retained)"); ax.set_ylabel("Accuracy of retained")
ax.set_ylim(0.9,1.012); clean(ax)
ax.set_title("Selective prediction (pooled)")
savefig(fig, "fig3_selective")

# ========== FIG 4: negative controls ==========
fig, ax = plt.subplots(figsize=(7+2.5*(len(DATASETS)-2),3.8))
labels=[]; real=[]; ctrl=[]
for ds in DATASETS:
    mk=_load(ds,"marker_negative_controls.json"); pw=_load(ds,"pathway_negative_controls.json"); di=_load(ds,"disease_negative_controls.json")
    labels+=[f"{NICE[ds].split(' ')[0]}\nmarker",f"{NICE[ds].split(' ')[0]}\npathway",f"{NICE[ds].split(' ')[0]}\ndisease"]
    real+=[mk.get("real_support",np.nan), pw["real_mean_terms"], di["real_mean_context"]]
    ctrl+=[mk.get("shuffled_support",np.nan), pw["random_mean_terms"], di["housekeeping_context"]]
# normalize each metric to its own scale for visual comparability -> show as grouped, annotate raw
xx=np.arange(len(labels)); w=0.38
ax.bar(xx-w/2, real, w, label="real clusters", color=ACCENT, edgecolor="white", linewidth=0.6)
ax.bar(xx+w/2, ctrl, w, label="negative control", color=ACCENT2, edgecolor="white", linewidth=0.6)
ax.set_xticks(xx); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Support / #terms / #context hits"); ax.grid(axis="x", visible=False)
ax.set_title("Negative controls: real vs shuffled / housekeeping"); ax.legend(); clean(ax)
savefig(fig, "fig4_negcontrols")

# ========== FIG 5: ablation (recomputed) ==========
def _sc(o,um=True,up=True,ud=True):
    p,wt=[],[]
    if um:p.append(o["c_marker"]);wt.append(.5)
    if up:p.append(o["c_pathway"]);wt.append(.35)
    if ud:p.append(o["c_disease"]);wt.append(.15)
    return float(np.average(p,weights=wt)) if p else 0.0
fig, axes = plt.subplots(1,len(DATASETS),figsize=(4*len(DATASETS),3.8),sharey=True)
cfgs=["full","-marker","-pathway","-disease"]
kws=[{},{"um":False},{"up":False},{"ud":False}]
for axi,ds in zip(axes,DATASETS):
    orch=_load(ds,"orchestrated.json")
    vals=[np.mean([_sc(o,**kw) for o in orch.values()]) for kw in kws]
    cols=[ACCENT]+[LIGHT]*3
    bars=axi.bar(cfgs,vals,color=cols,edgecolor="#a8afb6",linewidth=0.6)
    for b,v in zip(bars,vals): axi.text(b.get_x()+b.get_width()/2,v+0.006,f"{v:.3f}",ha="center",fontsize=8,color="#4a4f55")
    axi.set_title(NICE[ds]); axi.set_ylim(0.6,1.0); axi.grid(axis="x",visible=False)
    axi.set_xticklabels(cfgs,rotation=20,fontsize=9); clean(axi)
axes[0].set_ylabel("Mean confidence")
fig.suptitle("Leave-one-agent-out ablation", y=1.02, fontsize=12)
savefig(fig, "fig5_ablation")

# ========== FIG 6: confidence distribution ==========
fig, ax = plt.subplots(figsize=(5.2+0.8*(len(DATASETS)-2),3.8))
ok=per_cluster[per_cluster.correct].conf; err=per_cluster[~per_cluster.correct].conf
bins=np.linspace(0.4,1.0,13)
ax.hist(ok,bins=bins,alpha=0.85,label=f"correct (n={len(ok)})",color=OK_C,edgecolor="white",linewidth=0.5)
ax.hist(err,bins=bins,alpha=0.9,label=f"error (n={len(err)})",color=ERR_C,edgecolor="white",linewidth=0.5)
ax.set_xlabel("Confidence"); ax.set_ylabel("Clusters"); ax.grid(axis="x",visible=False)
ax.set_title("Confidence: correct vs error clusters (pooled)"); ax.legend(); clean(ax)
savefig(fig, "fig6_conf_dist")

# ========== FIG 7: UMAP panels (call + confidence, both datasets) ==========
umap_ok = True
try:
    import scanpy as sc
    fig, axes = plt.subplots(len(DATASETS),2,figsize=(9,4.3*len(DATASETS)))
    for row,ds in enumerate(DATASETS):
        ad = sc.read_h5ad(H5AD/f"{ds}_clustered.h5ad")
        um = ad.obsm["X_umap"]
        leiden = ad.obs["leiden"].astype(str).values
        orch = _load(ds,"orchestrated.json")
        # map cluster -> call + family + confidence
        def fam_of(lbl):
            t=(lbl or "").lower()
            for k,f in [("t","lymphoid")]:  # placeholder; use benchmark per_cluster families
                pass
            return None
        pc = pd.DataFrame(bench[ds]["per_cluster"]); pc["cluster"]=pc["cluster"].astype(str)
        fam_map = dict(zip(pc["cluster"], pc["pred_family"]))
        conf_map = {k:orch[k]["confidence"] for k in orch}
        fams = np.array([FAMILY_COLORS.get(fam_map.get(c,"unknown"),FAMILY_COLORS["unknown"]) for c in leiden])
        confs = np.array([conf_map.get(c,np.nan) for c in leiden])
        # left: colored by family/call
        a0=axes[row,0]
        a0.scatter(um[:,0],um[:,1],c=fams,s=3,linewidths=0,alpha=0.8)
        a0.set_title(f"{NICE[ds]} - cell-type family"); a0.set_xticks([]); a0.set_yticks([])
        for sp in a0.spines.values(): sp.set_visible(False)
        # right: colored by confidence
        a1=axes[row,1]
        scp=a1.scatter(um[:,0],um[:,1],c=confs,s=3,linewidths=0,cmap="viridis",vmin=0.4,vmax=1.0,alpha=0.85)
        a1.set_title(f"{NICE[ds]} - confidence"); a1.set_xticks([]); a1.set_yticks([])
        for sp in a1.spines.values(): sp.set_visible(False)
        fig.colorbar(scp, ax=a1, fraction=0.046, pad=0.04, label="confidence")
    # legend for families
    from matplotlib.patches import Patch
    handles=[Patch(color=c,label=f) for f,c in FAMILY_COLORS.items() if f!="unknown"]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5,-0.02), frameon=False)
    fig.suptitle("UMAP: framework calls and confidence", y=1.01, fontsize=13)
    fig.tight_layout()
    savefig(fig, "fig7_umap")
except Exception as e:
    umap_ok = False
    print("UMAP figure skipped:", str(e)[:120])

# ========== summary ==========
lines=["scINTERP v2 - SUMMARY REPORT","="*50]
for ds in DATASETS:
    b=bench[ds]; lines += [f"\n{NICE[ds]} ({ds}):",
        f"  overall accuracy   : {b['overall_accuracy']:.3f}",
        f"  selective accuracy : {b['selective_accuracy']:.3f}",
        f"  marker Hit@1 / @3  : {b['hit_at_1']:.3f} / {b['hit_at_3']:.3f}",
        f"  abstentions        : {b['n_abstain']}"]
if pooled:
    lines += [f"\nPOOLED ({pooled['n_clusters']} clusters, {len(DATASETS)} datasets):",
        f"  accuracy : {pooled['pooled_accuracy']:.3f}",
        f"  AUROC    : {pooled['pooled_auroc']:.3f}  CI {pooled['auroc_ci']}",
        f"  errors / abstentions : {pooled['n_errors']} / {pooled['n_abstain']}"]
open(FIG/"summary_report.txt","w").write("\n".join(lines))
print("\n".join(lines))
print("\n" + "="*50)
print("NOTEBOOK 11 COMPLETE - figures (PNG+PDF, 300 dpi) in figures/paper/")
for f in sorted(FIG.glob("*.png")): print("  ", f.name)
if not umap_ok: print("  (fig7_umap skipped - attach scinterp-env for scanpy)")
