"""nb_04_scanpy_clustering — executed from the skeleton notebook via exec(); code lives in Lakehouse Files."""

# ======================================================================
# # nb_04 — Clustering
# 
# **Main function:** normalization → HVG → PCA → neighbors → Leiden → UMAP, plus the cluster ↔ author-label crosstab that becomes the ground-truth mapping for every downstream evaluation.
# 
# **Input:** `{DATASET}_qc.h5ad` · **Output:** `{DATASET}_clustered.h5ad`, `cluster_ground_truth.csv`, UMAPs.
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
  "GSE176078": dict(
    tissue="breast",
    matrix_glob="*.mtx*",
    annot_glob="*metadata*.csv*",
    subset_col_candidates=[], subset_values=[],
    celltype_col_candidates=["celltype_major"]),
}[DATASET]

QC = dict(min_genes=200, max_genes=5000, max_mito_pct=15, min_cells_per_gene=3,
          subsample=10000, seed=0)                     # identical across datasets — any change goes in Methods
CLUST = dict(n_hvg=2000, n_pcs=30, n_neighbors=15, leiden_resolution=0.5)
DEGP  = dict(method="wilcoxon", top_ora=50, top_agents=20, log2fc_min=0.5, fdr_max=0.05)
print(f"{DATASET}: raw={RAW}")


import scanpy as sc, pandas as pd, numpy as np
sc.settings.figdir = str(FIG)
adata = sc.read_h5ad(H5AD/f"{DATASET}_qc.h5ad")

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata                                   # keep lognorm full matrix for DEG
sc.pp.highly_variable_genes(adata, n_top_genes=CLUST["n_hvg"], flavor="seurat")
adata = adata[:, adata.var.highly_variable].copy()
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=CLUST["n_pcs"], svd_solver="arpack", random_state=0)
sc.pp.neighbors(adata, n_neighbors=CLUST["n_neighbors"], n_pcs=CLUST["n_pcs"], random_state=0)
sc.tl.leiden(adata, resolution=CLUST["leiden_resolution"], random_state=0, key_added="leiden")
sc.tl.umap(adata, random_state=0)
print(f"{adata.obs.leiden.nunique()} Leiden clusters at resolution {CLUST['leiden_resolution']}")


sc.pl.umap(adata, color="leiden", legend_loc="on data", save=f"_{DATASET}_leiden.png", show=False)
sc.pl.umap(adata, color="author_cell_type", save=f"_{DATASET}_author.png", show=False)


# ---- Ground-truth mapping: majority author label + purity per cluster ----
ct = pd.crosstab(adata.obs["leiden"], adata.obs["author_cell_type"])
gt = pd.DataFrame({
    "cluster": ct.index,
    "n_cells": ct.sum(1).values,
    "majority_author_label": ct.idxmax(1).values,
    "purity": (ct.max(1)/ct.sum(1)).round(3).values,
}).set_index("cluster")
gt.to_csv(DEG_DIR/"cluster_ground_truth.csv")
print(gt)
low = gt[gt.purity < 0.5]
if len(low): print(f"\nNOTE: {len(low)} cluster(s) with purity <0.5 (mixed) — candidates for the conflict/abstention analysis: {list(low.index)}")


out = H5AD/f"{DATASET}_clustered.h5ad"
adata.write(out)
print("="*60 + f"""
NOTEBOOK 04 COMPLETE
{"="*60}
Main function : Clustering + ground-truth mapping
Dataset       : {DATASET}
Clusters      : {adata.obs.leiden.nunique()} (Leiden r={CLUST['leiden_resolution']})
Mean purity   : {gt.purity.mean():.3f}
Output        : {out}
Next notebook : nb_05_DEG_analysis""")

