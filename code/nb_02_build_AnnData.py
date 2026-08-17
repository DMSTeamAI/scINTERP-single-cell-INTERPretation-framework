"""nb_02_build_AnnData — executed from the skeleton notebook via exec(); code lives in Lakehouse Files."""

# ======================================================================
# # nb_02 — Build AnnData
# 
# **Main function:** Convert the GEO UMI matrix (genes × cells, tab-delimited) + published cell annotation into a sparse cells × genes AnnData with author labels attached, subset to the primary-tumor compartment.
# 
# **Input:** `Files/scRNA/raw/{DATASET}/` (from nb_01) · **Output:** `Files/scRNA/h5ad/{DATASET}_raw.h5ad`
# 
# Memory note: the GSE131907 matrix is ~29.6k × 208.5k — it is read in gene-chunks and converted to sparse int32 on the fly (never densified in full). Runs on a Medium (56 GB) node.
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


# ---- Locate input files ----
mats = sorted(glob.glob(str(RAW/CONFIG["matrix_glob"])))
anns = sorted(glob.glob(str(RAW/CONFIG["annot_glob"])))
assert mats, f"No matrix matching {CONFIG['matrix_glob']} in {RAW}"
assert anns, f"No annotation matching {CONFIG['annot_glob']} in {RAW}"
MATRIX, ANNOT = mats[0], anns[0]
print("matrix    :", os.path.basename(MATRIX))
print("annotation:", os.path.basename(ANNOT))


# ---- Chunked sparse read (genes x cells -> cells x genes CSR) ----
import numpy as np, pandas as pd, scipy.sparse as sp

def read_umi_sparse(path, chunk_rows=2000):
    chunks, genes, cells = [], [], None
    for i, ch in enumerate(pd.read_csv(path, sep="\t", index_col=0, chunksize=chunk_rows)):
        if cells is None:
            cells = ch.columns.to_numpy()
        genes.extend(ch.index.astype(str))
        arr = ch.to_numpy(dtype="float32")
        chunks.append(sp.csr_matrix(arr))
        print(f"   chunk {i+1:>3}: {len(genes):>7,} genes read", end="\r")
    X = sp.vstack(chunks).T.tocsr()          # -> cells x genes
    print()
    return X, np.asarray(genes), cells

X, gene_names, cell_names = read_umi_sparse(MATRIX)
is_int = np.allclose(X.data[:100000] % 1, 0)
print(f"shape (cells x genes): {X.shape},  nnz={X.nnz:,},  integer counts: {is_int}")
if is_int: X.data = X.data.astype("int32", copy=False)


# ---- Load author annotation and align to matrix barcodes ----
ann = pd.read_csv(ANNOT, sep="\t")
print("annotation columns:", list(ann.columns))

# auto-detect the barcode/ID column: the one overlapping matrix cell names most
overlaps = {c: len(set(ann[c].astype(str)) & set(cell_names)) for c in ann.columns
            if ann[c].dtype == object or str(ann[c].dtype).startswith("str")}
id_col = max(overlaps, key=overlaps.get)
print(f"barcode column -> '{id_col}' (overlap {overlaps[id_col]:,}/{len(cell_names):,})")
assert overlaps[id_col] > 0.9*len(cell_names), "Barcode overlap <90% — check files"
ann = ann.set_index(ann[id_col].astype(str)).reindex(cell_names)


# ---- Assemble AnnData, attach author labels, subset to tumor compartment ----
import anndata as ad
adata = ad.AnnData(X=X, obs=ann, var=pd.DataFrame(index=gene_names))
adata.var_names_make_unique()

ct_col = next((c for c in CONFIG["celltype_col_candidates"] if c in adata.obs), None)
assert ct_col, f"No author cell-type column among {CONFIG['celltype_col_candidates']}"
adata.obs["author_cell_type"] = adata.obs[ct_col].astype(str)

sub_col = next((c for c in CONFIG["subset_col_candidates"] if c in adata.obs), None)
if sub_col:
    before = adata.n_obs
    adata = adata[adata.obs[sub_col].astype(str).isin(CONFIG["subset_values"])].copy()
    print(f"subset {sub_col} in {CONFIG['subset_values']}: {before:,} -> {adata.n_obs:,} cells")
else:
    print(f"WARNING: none of {CONFIG['subset_col_candidates']} found — keeping ALL cells; "
          f"available columns: {list(adata.obs.columns)}")
print(adata.obs["author_cell_type"].value_counts())


# ---- Save ----
out = H5AD/f"{DATASET}_raw.h5ad"
adata.write(out)
print("="*60 + f"""
NOTEBOOK 02 COMPLETE
{"="*60}
Main function : Build sparse AnnData with author labels
Dataset       : {DATASET}
Cells x Genes : {adata.n_obs:,} x {adata.n_vars:,}
Author labels : {adata.obs['author_cell_type'].nunique()} cell types (col '{ct_col}')
Output        : {out}
Next notebook : nb_03_QC""")

