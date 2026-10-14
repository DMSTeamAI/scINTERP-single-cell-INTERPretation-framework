"""nb_03_QC — executed from the skeleton notebook via exec(); code lives in Lakehouse Files."""

# ======================================================================
# # nb_03 — QC
# 
# **Main function:** Cell/gene QC (counts, genes, mitochondrial %), filtering with identical thresholds across datasets, optional fixed-seed subsample for cross-version comparability.
# 
# **Input:** `{DATASET}_raw.h5ad` · **Output:** `{DATASET}_qc.h5ad` + QC figures.
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


import scanpy as sc, numpy as np
sc.settings.figdir = str(FIG)
adata = sc.read_h5ad(H5AD/f"{DATASET}_raw.h5ad")
print(adata)


# ---- QC metrics ----
adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
sc.pl.violin(adata, ["n_genes_by_counts","total_counts","pct_counts_mt"],
             jitter=0.3, multi_panel=True, save=f"_{DATASET}_qc_violin.png", show=False)
print(adata.obs[["n_genes_by_counts","total_counts","pct_counts_mt"]].describe().round(1))


# ---- Filter (identical thresholds for both datasets; see QC dict) ----
n0 = adata.n_obs
sc.pp.filter_cells(adata, min_genes=QC["min_genes"])
adata = adata[adata.obs.n_genes_by_counts <= QC["max_genes"]]
adata = adata[adata.obs.pct_counts_mt <= QC["max_mito_pct"]].copy()
sc.pp.filter_genes(adata, min_cells=QC["min_cells_per_gene"])
print(f"cells {n0:,} -> {adata.n_obs:,}   genes -> {adata.n_vars:,}")


# ---- Fixed-seed subsample (comparability with v1 design; report in Methods) ----
if QC["subsample"] and adata.n_obs > QC["subsample"]:
    np.random.seed(QC["seed"])
    idx = np.random.choice(adata.n_obs, QC["subsample"], replace=False)
    adata = adata[np.sort(idx)].copy()
    print(f"subsampled to {adata.n_obs:,} cells (seed={QC['seed']})")
print(adata.obs["author_cell_type"].value_counts())


out = H5AD/f"{DATASET}_qc.h5ad"
adata.write(out)
print("="*60 + f"""
NOTEBOOK 03 COMPLETE
{"="*60}
Main function : QC and filtering
Dataset       : {DATASET}
Cells x Genes : {adata.n_obs:,} x {adata.n_vars:,}
Thresholds    : min_genes={QC['min_genes']}, max_genes={QC['max_genes']}, mito<={QC['max_mito_pct']}%
Output        : {out}
Next notebook : nb_04_scanpy_clustering""")

