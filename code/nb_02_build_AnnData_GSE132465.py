"""nb_02 build AnnData — GSE132465 (colon). Reads counts + annotation, subsets to Tumor.
Runner:
    DATASET = "GSE132465"
    exec(open("/lakehouse/default/Files/scRNA/code/nb_02_build_AnnData_GSE132465.py").read())
"""
assert DATASET == "GSE132465", "This build script is for GSE132465"
import glob, hashlib
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

ROOT = Path("/lakehouse/default/Files/scRNA")
RAW, H5AD = ROOT/"raw"/DATASET, ROOT/"h5ad"; H5AD.mkdir(parents=True, exist_ok=True)

mats = [f for f in glob.glob(str(RAW/"*count*matrix*")) if f.endswith((".txt.gz",".txt",".tsv.gz"))]
anns = glob.glob(str(RAW/"*cell_annotation*"))
assert mats and anns, f"missing files in {RAW}: mats={mats}, anns={anns}"
MATRIX, ANNOT = sorted(mats)[0], sorted(anns)[0]
print("matrix    :", Path(MATRIX).name)
print("annotation:", Path(ANNOT).name)

def read_umi_sparse(path, chunk_rows=2000):
    chunks, genes, cells = [], [], None
    for i, ch in enumerate(pd.read_csv(path, sep="\t", index_col=0, chunksize=chunk_rows)):
        if cells is None: cells = ch.columns.to_numpy()
        genes.extend(ch.index.astype(str))
        chunks.append(sp.csr_matrix(ch.to_numpy(dtype="float32")))
        print(f"   chunk {i+1:>3}: {len(genes):>7,} genes", end="\r")
    print(); X = sp.vstack(chunks).T.tocsr()
    return X, np.asarray(genes), cells

X, gene_names, cell_names = read_umi_sparse(MATRIX)
print(f"shape (cells x genes): {X.shape}, nnz={X.nnz:,}")
if np.allclose(X.data[:100000] % 1, 0): X.data = X.data.astype("int32", copy=False)

ann = pd.read_csv(ANNOT, sep="\t")
overlaps = {c: len(set(ann[c].astype(str)) & set(cell_names)) for c in ann.columns if ann[c].dtype==object}
id_col = max(overlaps, key=overlaps.get)
print(f"barcode column -> '{id_col}' (overlap {overlaps[id_col]:,}/{len(cell_names):,})")
ann = ann.set_index(ann[id_col].astype(str)).reindex(cell_names)

adata = ad.AnnData(X=X, obs=ann, var=pd.DataFrame(index=gene_names)); adata.var_names_make_unique()
adata.obs["author_cell_type"]    = adata.obs["Cell_type"].astype(str)
adata.obs["author_cell_subtype"] = adata.obs["Cell_subtype"].astype(str)

before = adata.n_obs
adata = adata[adata.obs["Class"].astype(str) == "Tumor"].copy()
print(f"subset Class==Tumor: {before:,} -> {adata.n_obs:,} cells")
print(adata.obs["author_cell_type"].value_counts())

out = H5AD/f"{DATASET}_raw.h5ad"; adata.write(out)
print("\nNOTEBOOK 02 COMPLETE —", f"{adata.n_obs:,} x {adata.n_vars:,} ->", out)
