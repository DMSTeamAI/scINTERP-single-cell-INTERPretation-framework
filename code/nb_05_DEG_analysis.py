"""nb_05_DEG_analysis — executed from the skeleton notebook via exec(); code lives in Lakehouse Files."""

# ======================================================================
# # nb_05 — DEG analysis
# 
# **Main function:** Wilcoxon rank-sum marker identification per Leiden cluster (the statistical step — all significance comes from THIS dataset, agents only add external biological consistency evidence). Exports top-N lists for the agents/ORA **and** the full ranking for prerank GSEA (no arbitrary top-50 cutoff).
# 
# **Input:** `{DATASET}_clustered.h5ad` · **Output:** `DEG/{DATASET}/degs_top.json`, `degs_ranked.parquet`, `degs_summary.csv`.
# ======================================================================

# DATASET is defined by the calling notebook cell (before exec)
assert "DATASET" in globals(), "Set DATASET before exec()"

# ---- Shared paths & per-dataset config ----
import os, glob, json
from pathlib import Path

ROOT = Path("/lakehouse/default/Files/scRNA")
RAW, H5AD = ROOT/"raw"/DATASET, ROOT/"h5ad"
DEG_DIR, FIG = ROOT/"DEG"/DATASET, ROOT/"figures"/DATASET
for d in (H5AD, DEG_DIR, FIG): d.mkdir(parents=True, exist_ok=True)

CONFIG = {
  "GSE131907": dict(
    tissue="lung",
    matrix_glob="*raw_UMI*matrix*.txt.gz",
    annot_glob="*cell_annotation*.txt*",
    subset_col_candidates=["Sample_Origin"], subset_values=["tLung"],   # primary tumor lung (matches v1)
    celltype_col_candidates=["Cell_type.refined","Cell_type","Cell_subtype"]),
  "GSE132465": dict(
    tissue="colon",
    matrix_glob="*UMI*count*matrix*.txt.gz",
    annot_glob="*cell_annotation*.txt*",
    subset_col_candidates=["Class"], subset_values=["Tumor"],
    celltype_col_candidates=["Cell_type","Cell_subtype"]),
}[DATASET]

QC = dict(min_genes=200, max_genes=5000, max_mito_pct=15, min_cells_per_gene=3,
          subsample=10000, seed=0)                     # identical across datasets — any change goes in Methods
CLUST = dict(n_hvg=2000, n_pcs=30, n_neighbors=15, leiden_resolution=0.5)
DEGP  = dict(method="wilcoxon", top_ora=50, top_agents=20, log2fc_min=0.5, fdr_max=0.05)
print(f"{DATASET}: raw={RAW}")


import scanpy as sc, pandas as pd, numpy as np
adata = sc.read_h5ad(H5AD/f"{DATASET}_clustered.h5ad")
sc.tl.rank_genes_groups(adata, "leiden", method=DEGP["method"], use_raw=True, pts=True)
res = adata.uns["rank_genes_groups"]
clusters = list(res["names"].dtype.names)
print(f"Wilcoxon DEGs computed for {len(clusters)} clusters")


# ---- Long-format table: every gene, every cluster (full ranking) ----
rows = []
for cl in clusters:
    df = sc.get.rank_genes_groups_df(adata, group=cl)
    df["cluster"] = cl
    rows.append(df)
ranked = pd.concat(rows, ignore_index=True).rename(columns={"names":"gene"})
ranked.to_parquet(DEG_DIR/"degs_ranked.parquet", index=False)
print(f"full ranking: {len(ranked):,} rows -> degs_ranked.parquet")


# ---- Significant top-N lists for agents (top 20) and ORA (top 50) ----
sig = ranked[(ranked.logfoldchanges >= DEGP["log2fc_min"]) & (ranked.pvals_adj <= DEGP["fdr_max"])]
tops = {}
for cl in clusters:
    s = sig[sig.cluster == cl].sort_values("scores", ascending=False)
    tops[cl] = {"top_agents": s.gene.head(DEGP["top_agents"]).tolist(),
                "top_ora":    s.gene.head(DEGP["top_ora"]).tolist(),
                "n_significant": int(len(s))}
json.dump(tops, open(DEG_DIR/"degs_top.json","w"), indent=1)

summary = pd.DataFrame({cl: {"n_significant": v["n_significant"],
                             "top5": ", ".join(v["top_agents"][:5])} for cl, v in tops.items()}).T
gt = pd.read_csv(DEG_DIR/"cluster_ground_truth.csv", index_col=0)
summary = summary.join(gt[["majority_author_label","purity"]].astype({"purity":float}), how="left")
summary.to_csv(DEG_DIR/"degs_summary.csv")
print(summary)
empty = [cl for cl, v in tops.items() if v["n_significant"] == 0]
if empty: print(f"WARNING: clusters with zero significant DEGs: {empty}")


print("="*60 + f"""
NOTEBOOK 05 COMPLETE
{"="*60}
Main function : Wilcoxon DEG identification and export
Dataset       : {DATASET}
Clusters      : {len(clusters)}
Median sig DEGs/cluster : {int(np.median([v['n_significant'] for v in tops.values()]))}
Outputs       : degs_top.json, degs_ranked.parquet, degs_summary.csv
Next notebook : nb_06_DEG_evidence_agent""")

