"""nb_14_ora_robustness - CSB Reports revision (R1-6, R3): how much do the pathway evidence and the integrated
confidence depend on (i) the ORA gene universe and (ii) the top-50 cutoff?

Configurations (all use the SAME three libraries and the SAME generic-term down-ranking as nb_07):
  enrichr_top50      the original run (Enrichr default universe)           [read from pathway_evidence.json]
  tested_top{20,50,100,200}   own hypergeometric ORA, background = all genes tested in the DE step
  expressed_top50    background = genes detected in >=10% of the cluster's cells (if pct column present)
  tested_allsig      all significant DEGs (FDR/logFC thresholds of nb_05), capped at 500 by score
  gsea (optional)    prerank GSEA on the full ranking (set RUN_GSEA_CFG = True; slow)
For each configuration the orchestrator (nb_09, unchanged logic) is re-run on the alternative pathway evidence;
we report pathway-lineage agreement with the original, confidence correlation, AUROC, and conflict/abstention counts.
The cell-type call is the marker agent's top-1 and is by construction unaffected.

Runner:
    DATASETS = ["GSE131907", "GSE132465"]
    RUN_GSEA_CFG = False
    exec(open("/lakehouse/default/Files/scRNA/code/nb_14_ora_robustness.py").read())
Writes: agent_evidence/<DS>/robust/<cfg>_{pathway,orchestrated}.json, ROOT/reanalysis/ora_robustness_<DS>.csv
"""
import json, io, contextlib, re, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import hypergeom, spearmanr
warnings.filterwarnings("ignore")
import gseapy as gp

DATASETS = globals().get("DATASETS", ["GSE131907", "GSE132465"])
RUN_GSEA_CFG = globals().get("RUN_GSEA_CFG", False)
ROOT = Path("/lakehouse/default/Files/scRNA"); (ROOT/"reanalysis").mkdir(exist_ok=True)
CODE = ROOT/"code"
LIBS = {"Reactome": "Reactome_Pathways_2024", "GO_BP": "GO_Biological_Process_2023", "Hallmark": "MSigDB_Hallmark_2020"}
FDR_ORA, TOP_TERMS = 0.05, 10
LOG2FC_MIN, FDR_MAX = 0.5, 0.05

# reuse nb_07's generic-term list and ranking key verbatim
_src = open(CODE/"nb_07_pathway_agent.py").read()
_blk = _src[_src.index("_GENERIC_PATTERNS = ["):_src.index("def run_cluster(")]
_ns = {}; exec(_blk, _ns); _rank_key = _ns["_rank_key"]

_LIBCACHE = {}
def library(name):
    if name not in _LIBCACHE: _LIBCACHE[name] = {t: set(g) for t, g in gp.get_library(name).items()}
    return _LIBCACHE[name]

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.empty(n)
    q = p[o]*n/np.arange(1, n+1); q = np.minimum.accumulate(q[::-1])[::-1]; r[o] = np.minimum(q, 1); return r

def ora(genes, background):
    genes = set(genes) & background; N = len(background); n = len(genes); rows = []
    if n < 5: return rows
    for lab, name in LIBS.items():
        L = library(name); terms, ps, ov = [], [], []
        for t, tg in L.items():
            tg = tg & background; K = len(tg)
            if K < 5: continue
            x = len(genes & tg)
            if x < 2: continue
            terms.append(t); ps.append(hypergeom.sf(x-1, N, K, n)); ov.append(x)
        if not terms: continue
        q = bh(ps)
        for t, p_, q_, x in zip(terms, ps, q, ov):
            if q_ <= FDR_ORA: rows.append(dict(library=lab, Term=t, P=p_, FDR=q_, overlap=x))
    return rows

def to_records(cl, rows, etype="pathway_ora"):
    rows = sorted(rows, key=lambda r: r["FDR"])[:TOP_TERMS]
    recs = [dict(evidence_type=etype, subject=cl, predicate="enriched_for", object=r["Term"], fdr=float(r["FDR"]),
                 database=r["library"], details={"overlap": r.get("overlap")}) for r in rows]
    recs.sort(key=_rank_key)
    return dict(agent="pathway", cluster_id=cl, n_records=len(recs), n_ora=len(recs) if etype == "pathway_ora" else 0,
                n_gsea=len(recs) if etype == "pathway_gsea" else 0, conflicts=[], notes=[] if recs else ["NO_SIGNIFICANT_ORA"], records=recs)

def auc(y, c):
    y = np.asarray(y); c = np.asarray(c, float); pos = c[y == 1]; neg = c[y == 0]
    if len(pos) == 0 or len(neg) == 0: return np.nan
    return float((np.sum(pos[:, None] > neg[None, :]) + 0.5*np.sum(pos[:, None] == neg[None, :]))/(len(pos)*len(neg)))

# same lineage map as nb_10a/nb_12 for scoring correctness
_FAM_SRC = open(CODE/"nb_12_reanalysis.py").read()
_fb = _FAM_SRC[_FAM_SRC.index("_FAM = ["):_FAM_SRC.index("COARSE2")]
_fns = {}; exec(_fb, _fns); family = _fns["family"]

orch_src = open(CODE/"nb_09_orchestrator.py").read()
def run_orch(ds, pathway_file, out_file):
    ns = dict(DATASET=ds, PATHWAY_FILE=pathway_file, OUT_FILE=out_file)
    with contextlib.redirect_stdout(io.StringIO()): exec(orch_src, ns)
    return json.load(open(out_file))

for DATASET in DATASETS:
    print("\n" + "#"*70 + f"\n# {DATASET}\n" + "#"*70)
    DEG = ROOT/"DEG"/DATASET; EVID = ROOT/"agent_evidence"/DATASET; RB = EVID/"robust"; RB.mkdir(exist_ok=True)
    ranked = pd.read_parquet(DEG/"degs_ranked.parquet"); tops = json.load(open(DEG/"degs_top.json")); CL = list(tops.keys())
    gt = pd.read_csv(DEG/"cluster_ground_truth.csv", index_col=0); gt.index = gt.index.astype(str)
    base = json.load(open(EVID/"orchestrated.json"))
    ok = [int(family(base[cl]["marker_call"]) == family(gt.loc[cl, "majority_author_label"])) for cl in CL]
    pct_col = next((c for c in ["pct_nz_group", "pts"] if c in ranked.columns), None)
    sig = ranked[(ranked.logfoldchanges >= LOG2FC_MIN) & (ranked.pvals_adj <= FDR_MAX)]
    configs = {}
    def top_n(cl, n): return sig[sig.cluster == cl].sort_values("scores", ascending=False).gene.head(n).tolist()
    bg_tested = {cl: set(ranked[ranked.cluster == cl].gene) for cl in CL}
    bg_expr = {cl: set(ranked[(ranked.cluster == cl) & (ranked[pct_col] >= 0.10)].gene) for cl in CL} if pct_col else None
    for n in [20, 50, 100, 200]:
        configs[f"tested_top{n}"] = lambda cl, n=n: to_records(cl, ora(top_n(cl, n), bg_tested[cl]))
    if bg_expr: configs["expressed_top50"] = lambda cl: to_records(cl, ora(top_n(cl, 50), bg_expr[cl]))
    configs["tested_allsig_cap500"] = lambda cl: to_records(cl, ora(top_n(cl, 500), bg_tested[cl]))
    if RUN_GSEA_CFG:
        def gsea_cfg(cl):
            sub = ranked[ranked.cluster == cl][["gene", "scores"]].dropna().drop_duplicates("gene")
            rnk = sub.set_index("gene")["scores"].sort_values(ascending=False); rows = []
            for lab, name in LIBS.items():
                try:
                    r = gp.prerank(rnk=rnk, gene_sets=name, permutation_num=200, outdir=None, seed=0, min_size=5, max_size=500).res2d
                    fc = "FDR q-val" if "FDR q-val" in r else "fdr"
                    for x in r[r[fc] <= 0.25].to_dict("records"): rows.append(dict(library=lab, Term=x["Term"], FDR=float(x[fc]), P=np.nan, overlap=None))
                except Exception as e: print("prerank fail", lab, str(e)[:60])
            return to_records(cl, rows, "pathway_gsea")
        configs["gsea_prerank"] = gsea_cfg

    summary = []
    # original run row
    c0 = np.array([base[cl]["confidence"] for cl in CL])
    summary.append(dict(config="enrichr_top50 (original)", mean_terms=np.mean([json.load(open(EVID/'pathway_evidence.json'))[cl]['n_records'] for cl in CL]),
                        lineage_agree_with_original=1.0, spearman_conf=1.0, auroc=auc(ok, c0), n_agree=sum(base[cl]["agreement"] for cl in CL),
                        n_conflict=sum(any(f.startswith("AGENT_DISAGREEMENT") for f in base[cl]["flags"]) for cl in CL),
                        n_abstain=sum(base[cl]["abstain"] for cl in CL)))
    for name, fn in configs.items():
        t0 = time.time(); pw = {cl: fn(cl) for cl in CL}
        pf = RB/f"{name}_pathway.json"; json.dump(pw, open(pf, "w"), indent=1, default=str)
        o = run_orch(DATASET, pf, RB/f"{name}_orchestrated.json")
        c = np.array([o[cl]["confidence"] for cl in CL])
        same_lin = np.mean([(o[cl]["pathway_lineage"] == base[cl]["pathway_lineage"]) for cl in CL])
        rho = spearmanr(c, c0).correlation if np.std(c) > 0 else np.nan
        summary.append(dict(config=name, mean_terms=np.mean([pw[cl]["n_records"] for cl in CL]), lineage_agree_with_original=same_lin,
                            spearman_conf=rho, auroc=auc(ok, c), n_agree=sum(o[cl]["agreement"] for cl in CL),
                            n_conflict=sum(any(f.startswith("AGENT_DISAGREEMENT") for f in o[cl]["flags"]) for cl in CL),
                            n_abstain=sum(o[cl]["abstain"] for cl in CL)))
        print(f"  {name:22} done in {time.time()-t0:.0f}s")
    S = pd.DataFrame(summary).round(3); S.to_csv(ROOT/"reanalysis"/f"ora_robustness_{DATASET}.csv", index=False)
    print("\n" + S.to_string(index=False))
    print("\nReading guide: cell-type accuracy is unchanged by construction (marker top-1). Report how stable the pathway lineage,")
    print("confidence ranking (Spearman), AUROC and conflict/abstention counts are across universes and cutoffs.")
print("\nNOTEBOOK 14 COMPLETE")
