"""nb_02c_build_AnnData_GSE176078 - third, independent dataset for the CSB Reports revision.

Wu et al. 2021 (Nat Genet), human breast cancer atlas, GEO GSE176078: ~100k cells, 26 primary tumours
(ER+, HER2+, TNBC), author annotation at three levels (celltype_major / minor / subset).
Same downstream pipeline as the lung and colon datasets: identical QC / clustering / DE parameters (nb_03-05).

Expected raw file (GEO supplementary): GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz placed in
  Files/scRNA/raw/GSE176078/   (extracted automatically; mtx, genes, barcodes and metadata.csv are located by glob)
VERIFY after download: the script prints the matrix shape/orientation and the metadata columns; if the
column names differ from celltype_major, set AUTHOR_COL before exec().

Runner:
    DATASET = "GSE176078"
    exec(open("/lakehouse/default/Files/scRNA/code/nb_02c_build_AnnData_GSE176078.py").read())
"""
assert DATASET == "GSE176078", "this builder is for GSE176078 only"
import glob, os, tarfile
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, scipy.io as sio
import anndata as ad

ROOT = Path("/lakehouse/default/Files/scRNA"); RAW = ROOT/"raw"/DATASET; H5AD = ROOT/"h5ad"
for d in (H5AD, ROOT/"DEG"/DATASET, ROOT/"figures"/DATASET): d.mkdir(parents=True, exist_ok=True)
AUTHOR_COL = globals().get("AUTHOR_COL", "celltype_major")

for t in glob.glob(str(RAW/"*.tar*")):
    if not glob.glob(str(RAW/"**"/"*.mtx*"), recursive=True):
        print("extracting", os.path.basename(t)); tarfile.open(t).extractall(RAW)
def find(pat):
    h = sorted(glob.glob(str(RAW/"**"/pat), recursive=True)); assert h, f"no file matching {pat} under {RAW}"; return h[0]
mtx, genes_f, bars_f, meta_f = find("*.mtx*"), find("*gene*.tsv*"), find("*barcode*.tsv*"), find("*metadata*.csv*")
print("files:", [os.path.basename(x) for x in (mtx, genes_f, bars_f, meta_f)])

genes = pd.read_csv(genes_f, sep="\t", header=None).iloc[:, -1 if pd.read_csv(genes_f, sep="\t", header=None).shape[1] > 1 else 0].astype(str).values
cells = pd.read_csv(bars_f, sep="\t", header=None).iloc[:, 0].astype(str).values
M = sio.mmread(mtx).tocsr()
if M.shape == (len(genes), len(cells)): X = M.T.tocsr()
elif M.shape == (len(cells), len(genes)): X = M
else: raise AssertionError(f"matrix {M.shape} matches neither genes={len(genes)} x cells={len(cells)}")
print(f"matrix (cells x genes): {X.shape}, nnz={X.nnz:,}")
is_int = np.allclose(X.data[:100000] % 1, 0); print("integer counts:", is_int)
if is_int: X.data = X.data.astype("int32", copy=False)

meta = pd.read_csv(meta_f)
idc = max(meta.columns, key=lambda c: len(set(meta[c].astype(str)) & set(cells)))
print(f"metadata id column -> '{idc}'; columns: {list(meta.columns)}")
meta = meta.set_index(meta[idc].astype(str)).reindex(cells)
assert meta[AUTHOR_COL].notna().mean() > 0.9, f"{AUTHOR_COL} missing for >10% of cells; set AUTHOR_COL"

adata = ad.AnnData(X=X, obs=meta, var=pd.DataFrame(index=genes)); adata.var_names_make_unique()
adata.obs["author_cell_type"] = adata.obs[AUTHOR_COL].astype(str)
# all samples are primary tumours (no normal compartment) -> no subsetting; patient/subtype columns are kept in obs
print(adata.obs["author_cell_type"].value_counts())
for c in ("orig.ident", "subtype"):
    if c in adata.obs: print(c, adata.obs[c].nunique(), "levels")
out = H5AD/f"{DATASET}_raw.h5ad"; adata.write(out)
print(f"NOTEBOOK 02c COMPLETE: {adata.n_obs:,} x {adata.n_vars:,}; {adata.obs['author_cell_type'].nunique()} author labels -> {out}")
