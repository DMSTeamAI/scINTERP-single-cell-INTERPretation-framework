"""
Day 1 — Data download + Scanpy preprocessing pipeline
=======================================================
Dataset : GSE131907 (Kim et al. 2020, Nat Comms)
          208,506 cells from lung adenocarcinoma (LUAD)
          tLung = primary tumor lung samples

Run:
    python day1_download_preprocess.py

Outputs:
    data/GSE131907_Lung_cell_annotation.csv.gz       (~2 MB)
    data/GSE131907_Lung_normalized_...matrix.txt.gz  (~1-3 GB, takes time)
    results/gse131907_tlung_processed.h5ad
    figures/figure1_umap.png / .pdf

Demo mode (no download needed):
    Set DEMO_MODE = True  →  uses scanpy's built-in pbmc3k to test pipeline
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gzip
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
import requests
from pathlib import Path
from tqdm import tqdm

from config import (
    DATA_DIR, RESULTS_DIR, FIGURES_DIR,
    COUNTS_FILE, ANNOT_FILE, GEO_BASE_URL,
    COUNTS_FILENAME, ANNOT_FILENAME,
    TARGET_TISSUE, MAX_CELLS,
    MIN_GENES, MAX_GENES, MAX_MITO_PCT, MIN_CELLS,
    N_HVG, N_PCS, N_NEIGHBORS, LEIDEN_RESOLUTION,
)

# ── Toggle this ───────────────────────────────────────────────────────────────
# True  = use pbmc3k (built-in, instant) to test pipeline TODAY
# False = use the real GSE131907 data (requires download below)
DEMO_MODE = not COUNTS_FILE.exists()

sc.settings.verbosity = 2
sc.settings.figdir = str(FIGURES_DIR)
plt.rcParams.update({"figure.figsize": (8, 6), "figure.dpi": 150})


# ─────────────────────────────────────────────────────────────────────────────
# PART A — Download
# ─────────────────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, desc: str = "Downloading") -> bool:
    """Stream download with progress bar; skip if file already exists."""
    dest = Path(dest)
    if dest.exists():
        print(f"  Already exists: {dest.name}  — skipping")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=desc
        ) as bar:
            for chunk in r.iter_content(chunk_size=65_536):
                f.write(chunk)
                bar.update(len(chunk))
        tmp.rename(dest)
        print(f"  Saved → {dest.name}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        if tmp.exists():
            tmp.unlink()
        return False


def download_geo_files():
    """Download annotation (fast) and count matrix (slow) from GEO FTP."""
    print("\n[Step 1/2] Cell annotation — small file, fast download")
    ok = download_file(
        f"{GEO_BASE_URL}/{ANNOT_FILENAME}", ANNOT_FILE, desc="Annotation"
    )
    if not ok:
        raise RuntimeError("Failed to download annotation file. Check internet.")

    print("\n[Step 2/2] Count matrix — 1-3 GB, may take 15-30 min")
    print("  TIP: Run with DEMO_MODE=True in another terminal to test pipeline now.")
    ok = download_file(
        f"{GEO_BASE_URL}/{COUNTS_FILENAME}", COUNTS_FILE, desc="Counts"
    )
    if not ok:
        print("  Count matrix download failed — falling back to DEMO_MODE")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# PART B — Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_demo_pbmc3k() -> sc.AnnData:
    """Built-in 3k PBMC dataset — no download needed, pipeline test only."""
    print("DEMO MODE: loading scanpy pbmc3k (2,700 cells × 32,738 genes)")
    print("NOTE: Replace with GSE131907 for the real analysis.\n")
    adata = sc.datasets.pbmc3k()
    adata.var_names_make_unique()
    return adata


def _read_annotation() -> pd.DataFrame:
    """Load GSE131907 cell annotation CSV."""
    ann = pd.read_csv(ANNOT_FILE, index_col=0)
    print(f"  Annotation: {len(ann)} cells, columns: {list(ann.columns)}")
    return ann


def _select_tlung_cells(ann: pd.DataFrame) -> list:
    """Return tLung cell IDs from annotation dataframe."""
    # Try common column names used in GSE131907
    for col in ["Sample", "sample", "tissue", "Tissue", "Index"]:
        if col in ann.columns and TARGET_TISSUE in ann[col].values:
            cells = ann[ann[col] == TARGET_TISSUE].index.tolist()
            print(f"  Found {len(cells)} {TARGET_TISSUE} cells via column '{col}'")
            return cells

    # Fallback: cell names often start with "tLung_"
    cells = [c for c in ann.index if c.startswith(TARGET_TISSUE + "_")]
    if cells:
        print(f"  Found {len(cells)} {TARGET_TISSUE} cells via barcode prefix")
        return cells

    print(f"  Warning: could not isolate {TARGET_TISSUE} — using all {len(ann)} cells")
    return ann.index.tolist()


def load_gse131907() -> sc.AnnData:
    """
    Load GSE131907 count matrix for tLung cells only.
    The matrix is genes × cells (rows = genes, cols = cell barcodes).
    We transpose to cells × genes for scanpy.
    """
    print("Loading GSE131907...")

    ann = _read_annotation()
    target_cells = _select_tlung_cells(ann)

    # Subsample if needed
    if len(target_cells) > MAX_CELLS:
        np.random.seed(42)
        target_cells = np.random.choice(
            target_cells, MAX_CELLS, replace=False
        ).tolist()
        print(f"  Subsampled to {MAX_CELLS} cells (set MAX_CELLS in config.py to increase)")

    # Read header to map cell IDs → column indices
    print(f"\nReading count matrix header...")
    with gzip.open(COUNTS_FILE, "rt") as f:
        header = f.readline().strip().split("\t")
    all_cell_ids = header[1:]   # col 0 is gene-name column
    print(f"  Matrix has {len(all_cell_ids)} cells total, {len(header)-1} columns")

    cell_set   = set(target_cells)
    col_idx    = [0]             # always keep gene-name column (index 0)
    kept_cells = []
    for i, cid in enumerate(all_cell_ids):
        if cid in cell_set:
            col_idx.append(i + 1)   # +1 offset for gene-name col
            kept_cells.append(cid)

    print(f"  Loading {len(kept_cells)} matching cells...")
    df = pd.read_csv(
        COUNTS_FILE, sep="\t", index_col=0,
        usecols=col_idx, compression="gzip"
    )
    print(f"  Raw matrix: {df.shape[0]} genes × {df.shape[1]} cells")

    # Build AnnData (cells × genes)
    adata = sc.AnnData(
        X=df.T.astype("float32"),
        obs=pd.DataFrame(index=kept_cells),
        var=pd.DataFrame(index=df.index),
    )
    # Attach author metadata
    common = [c for c in kept_cells if c in ann.index]
    adata = adata[common].copy()
    adata.obs = ann.loc[common].copy()

    print(f"\nAnnData ready: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


# ─────────────────────────────────────────────────────────────────────────────
# PART C — QC
# ─────────────────────────────────────────────────────────────────────────────

def run_qc(adata: sc.AnnData) -> sc.AnnData:
    print("\n--- Quality Control ---")
    adata.var_names_make_unique()

    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    print(f"  Before QC: {adata.n_obs:,} cells")

    # Violin plot (saved to figures/)
    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=0.4, multi_panel=True,
        save="_qc_before_filter.png", show=False,
    )

    sc.pp.filter_cells(adata, min_genes=MIN_GENES)
    sc.pp.filter_cells(adata, max_genes=MAX_GENES)
    adata = adata[adata.obs["pct_counts_mt"] < MAX_MITO_PCT].copy()
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS)

    print(f"  After QC:  {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    return adata


# ─────────────────────────────────────────────────────────────────────────────
# PART D — Normalise, cluster, UMAP
# ─────────────────────────────────────────────────────────────────────────────

def normalize_and_cluster(adata: sc.AnnData, is_demo: bool) -> sc.AnnData:
    print("\n--- Normalization & Clustering ---")

    if is_demo:
        # pbmc3k is raw counts → normalize
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        # GSE131907 is already log-normalized — just log1p guard
        import scipy.sparse as sp
        x = adata.X
        if sp.issparse(x):
            x = x.toarray()
        if x.max() > 50:
            print("  Data appears raw — applying normalize + log1p")
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        else:
            print("  Data appears pre-normalized — skipping normalize step")

    adata.raw = adata  # store for DEG analysis later

    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, subset=True)
    print(f"  HVGs selected: {adata.n_vars}")

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=N_PCS, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS)
    sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION)
    sc.tl.umap(adata)

    n_clusters = adata.obs["leiden"].nunique()
    print(f"  Leiden clusters: {n_clusters}")
    return adata


# ─────────────────────────────────────────────────────────────────────────────
# PART E — Figure 1: UMAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure1(adata: sc.AnnData, is_demo: bool):
    print("\n--- Figure 1: UMAP ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc.pl.umap(
        adata, color="leiden", ax=axes[0], show=False,
        title="Leiden clusters",
        legend_loc="on data", legend_fontsize=9,
    )

    # Second panel: cell type if available, else n_genes
    ct_col = next(
        (c for c in ["celltype_major", "CellType", "cell_type", "cell_label"]
         if c in adata.obs.columns),
        None,
    )
    if not is_demo and ct_col:
        sc.pl.umap(
            adata, color=ct_col, ax=axes[1], show=False,
            title="Author cell-type annotation",
            legend_loc="right margin", legend_fontsize=8,
        )
    else:
        sc.pl.umap(
            adata, color="n_genes_by_counts", ax=axes[1], show=False,
            title="Genes per cell", color_map="viridis",
        )

    fig.suptitle(
        "GSE131907 LUAD — primary tumor (tLung)"
        if not is_demo else "PBMC 3k demo dataset",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        p = FIGURES_DIR / f"figure1_umap.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → figures/figure1_umap.png / .pdf")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Day 1 — Download + Preprocessing")
    print(f"Mode: {'DEMO (pbmc3k)' if DEMO_MODE else 'GSE131907 real data'}")
    print("=" * 60)

    # ── Download ──────────────────────────────────────────────────────────────
    if not DEMO_MODE:
        download_geo_files()
    else:
        print("\nSkipping download (DEMO_MODE=True)")
        print("To download real data run:")
        print("  python day1_download_preprocess.py  [with DEMO_MODE=False]\n")

    # ── Load ─────────────────────────────────────────────────────────────────
    if DEMO_MODE or not COUNTS_FILE.exists():
        adata = load_demo_pbmc3k()
        is_demo = True
    else:
        adata = load_gse131907()
        is_demo = False

    # ── Pipeline ─────────────────────────────────────────────────────────────
    adata = run_qc(adata)
    adata = normalize_and_cluster(adata, is_demo=is_demo)
    plot_figure1(adata, is_demo=is_demo)

    # ── Save ─────────────────────────────────────────────────────────────────
    fname = "pbmc3k_processed.h5ad" if is_demo else "gse131907_tlung_processed.h5ad"
    out = RESULTS_DIR / fname
    adata.write_h5ad(out)

    print("\n" + "=" * 60)
    print("Day 1 DONE")
    print(f"  Cells: {adata.n_obs:,}   Genes: {adata.n_vars:,}   Clusters: {adata.obs['leiden'].nunique()}")
    print(f"  Saved → {out.name}")
    print(f"  Figure → figures/figure1_umap.png")
    print("\nNext step: python day2_deg_pathway_baseline.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
