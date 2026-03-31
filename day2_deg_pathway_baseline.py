"""
Day 2 — DEG analysis · pathway enrichment · GPT-4o naive baseline
===================================================================
Reads: results/gse131907_tlung_processed.h5ad  (or pbmc3k demo)
Writes:
  results/all_degs.csv
  results/degs/cluster_N_degs.csv
  results/pathways/cluster_N_pathways.csv
  results/cluster_summary_table.csv
  results/baseline_results.json         ← input for Day 3 agents
  figures/figure2_pathway_heatmap.png

Run:
    export OPENAI_API_KEY="sk-..."
    python day2_deg_pathway_baseline.py

GPT-4o calls are cached in cache/ — re-runs are free.

Hallucination audit note (paper methods):
  We reuse the 403-sentence expert-verified set from Hu et al. Nat Methods
  2025 (Suppl Table 4) as the reference for hallucination scoring.
  No separate expert panel is required; sentences are loaded from
  data/hu2025_verified_sentences.csv  (see README for source).
"""

import sys, os, json, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import gseapy as gp
from tqdm import tqdm
from openai import OpenAI
from pathlib import Path

from config import (
    RESULTS_DIR, FIGURES_DIR, CACHE_DIR,
    OPENAI_API_KEY, OPENAI_MODEL,
    TOP_N_DEGS, N_DEGS_FOR_LLM, MIN_LOGFC,
    GENE_SETS, PVAL_CUTOFF, TOP_N_PATHWAYS,
    LLM_TEMPERATURE, LLM_MAX_TOKENS,
)

# Determine which processed file to load
_REAL = RESULTS_DIR / "gse131907_tlung_processed.h5ad"
_DEMO = RESULTS_DIR / "pbmc3k_processed.h5ad"
PROCESSED_FILE = _REAL if _REAL.exists() else _DEMO

# OpenAI client (fails gracefully if key missing)
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    print("WARNING: OPENAI_API_KEY not set — GPT-4o baseline will be skipped.")
    print("  Set it with:  export OPENAI_API_KEY='sk-...'")

DEG_DIR      = RESULTS_DIR / "degs"
PATHWAY_DIR  = RESULTS_DIR / "pathways"
DEG_DIR.mkdir(exist_ok=True)
PATHWAY_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DEG Analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_deg_analysis(adata: sc.AnnData) -> dict[str, pd.DataFrame]:
    """
    Wilcoxon rank-sum test, one-vs-rest per Leiden cluster.
    Returns dict: cluster_id → DataFrame with columns
        names, scores, pvals, pvals_adj, logfoldchanges, pct_nz_group, pct_nz_reference
    """
    print("\n[1/3] DEG Analysis (Wilcoxon one-vs-rest)")

    # Use raw counts stored in adata.raw
    if adata.raw is not None:
        adata_deg = adata.raw.to_adata()
        adata_deg.obs = adata.obs.copy()
    else:
        adata_deg = adata.copy()

    sc.tl.rank_genes_groups(
        adata_deg, groupby="leiden", method="wilcoxon",
        n_genes=TOP_N_DEGS, tie_correct=True, pts=True,
    )

    clusters = sorted(adata.obs["leiden"].unique().tolist(), key=int)
    results  = {}
    all_rows = []

    for cl in clusters:
        df = sc.get.rank_genes_groups_df(
            adata_deg, group=str(cl),
            pval_cutoff=0.05,
            log2fc_min=MIN_LOGFC,
        ).head(TOP_N_DEGS)
        results[cl] = df
        df_out = df.copy()
        df_out.insert(0, "cluster", cl)
        all_rows.append(df_out)
        df.to_csv(DEG_DIR / f"cluster_{cl}_degs.csv", index=False)
        print(f"  Cluster {cl:>2s}: {len(df):>3d} DEGs")

    pd.concat(all_rows).to_csv(RESULTS_DIR / "all_degs.csv", index=False)
    print(f"  Saved → results/all_degs.csv + results/degs/cluster_*.csv")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pathway Enrichment
# ─────────────────────────────────────────────────────────────────────────────

def run_pathway_enrichment(deg_results: dict) -> dict[str, pd.DataFrame]:
    """
    gseapy Enrichr on top DEGs per cluster.
    Gene sets: MSigDB Hallmark (50 curated sets) + KEGG 2021.
    """
    print("\n[2/3] Pathway Enrichment (gseapy Enrichr — MSigDB Hallmark + KEGG)")
    print("  NOTE: first run downloads gene-set libraries (~20 MB) automatically.")

    pathway_results = {}

    for cl, degs in tqdm(deg_results.items(), desc="  Clusters"):
        genes = degs["names"].head(100).tolist()
        if len(genes) < 5:
            pathway_results[cl] = pd.DataFrame()
            continue
        try:
            enr = gp.enrichr(
                gene_list=genes,
                gene_sets=GENE_SETS,
                organism="human",
                outdir=None,
                verbose=False,
            )
            if enr.results is None or enr.results.empty:
                pathway_results[cl] = pd.DataFrame()
                continue

            sig = enr.results[
                enr.results["Adjusted P-value"] < PVAL_CUTOFF
            ].sort_values("Combined Score", ascending=False)

            top = sig.head(20)
            pathway_results[cl] = top
            top.to_csv(PATHWAY_DIR / f"cluster_{cl}_pathways.csv", index=False)

        except Exception as e:
            print(f"  Cluster {cl}: error — {e}")
            pathway_results[cl] = pd.DataFrame()

    # Summary: how many sig pathways per cluster?
    for cl, df in pathway_results.items():
        n = len(df)
        top_name = df["Term"].iloc[0] if n > 0 else "—"
        print(f"  Cluster {cl:>2s}: {n:>3d} sig pathways | top → {top_name[:50]}")

    print(f"  Saved → results/pathways/cluster_*.csv")
    return pathway_results


# ─────────────────────────────────────────────────────────────────────────────
# 3. GPT-4o Naive Baseline
# ─────────────────────────────────────────────────────────────────────────────

NAIVE_SYSTEM = (
    "You are an expert computational biologist specializing in single-cell "
    "transcriptomics and lung cancer biology."
)

NAIVE_USER_TMPL = """\
Dataset: GSE131907 — human lung adenocarcinoma (LUAD), primary tumor (tLung)
Task   : Interpret the functional role of a cell cluster.

Top {n} differentially expressed genes for cluster {cl}:
{genes}

Provide a concise biological interpretation (~150 words):
1. Most likely cell type or functional state
2. Key active biological processes / pathways
3. Potential role in LUAD tumor microenvironment

Important: Do NOT invent gene functions. If uncertain, say so explicitly."""


def _cache_key(cl: str, genes: list) -> str:
    h = hashlib.md5(("_".join(genes[:10])).encode()).hexdigest()[:8]
    return f"gpt4o_naive_cl{cl}_{h}"


def _cache_load(key: str) -> str | None:
    p = CACHE_DIR / f"{key}.json"
    if p.exists():
        return json.loads(p.read_text())["response"]
    return None


def _cache_save(key: str, resp: str):
    p = CACHE_DIR / f"{key}.json"
    p.write_text(json.dumps({"response": resp}))


def run_gpt4o_naive(deg_results: dict) -> dict[str, str]:
    """
    Sends top-20 DEGs per cluster to GPT-4o with a plain text prompt.
    No knowledge grounding — this is Baseline B2 in Table 1.
    Responses are cached so re-runs cost nothing.
    """
    print("\n[3/3] GPT-4o Naive Baseline")

    if client is None:
        print("  Skipped — set OPENAI_API_KEY to enable.")
        return {cl: "SKIPPED (no API key)" for cl in deg_results}

    results = {}
    for cl in tqdm(sorted(deg_results.keys(), key=int), desc="  Clusters"):
        degs  = deg_results[cl]
        genes = degs["names"].head(N_DEGS_FOR_LLM).tolist()
        key   = _cache_key(cl, genes)

        cached = _cache_load(key)
        if cached:
            results[cl] = cached
            continue  # free

        prompt = NAIVE_USER_TMPL.format(
            n=len(genes), cl=cl, genes=", ".join(genes)
        )
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": NAIVE_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            text = resp.choices[0].message.content.strip()
            results[cl] = text
            _cache_save(key, text)
            time.sleep(0.8)   # gentle rate limiting for free tier
        except Exception as e:
            results[cl] = f"API_ERROR: {e}"
            print(f"  Cluster {cl}: {e}")

    # Print brief summaries
    for cl, txt in results.items():
        preview = txt[:120].replace("\n", " ")
        print(f"  Cluster {cl:>2s}: {preview}…")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Build summary table
# ─────────────────────────────────────────────────────────────────────────────

def build_summary_table(
    adata: sc.AnnData,
    deg_results: dict,
    pathway_results: dict,
    gpt_results: dict,
) -> pd.DataFrame:
    rows = []
    for cl in sorted(deg_results.keys(), key=int):
        degs     = deg_results[cl]
        pathways = pathway_results.get(cl, pd.DataFrame())
        gpt      = gpt_results.get(cl, "N/A")

        top_genes = degs["names"].head(10).tolist() if len(degs) else []
        top_paths = (
            pathways["Term"].head(TOP_N_PATHWAYS).tolist()
            if len(pathways) and "Term" in pathways.columns else []
        )

        rows.append({
            "cluster":           cl,
            "n_cells":           int((adata.obs["leiden"] == cl).sum()),
            "n_sig_degs":        len(degs),
            "top_10_degs":       "; ".join(top_genes),
            "top_pathways":      "; ".join(top_paths),
            "gpt4o_naive_brief": str(gpt)[:300],
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "cluster_summary_table.csv", index=False)
    print(f"\n  Saved → results/cluster_summary_table.csv")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_deg_dotplot(adata: sc.AnnData):
    """Top-5 DEGs per cluster — dot plot (size = pct expressed, colour = mean expr)."""
    print("  DEG dot plot…")
    try:
        sc.tl.rank_genes_groups(
            adata, groupby="leiden", method="wilcoxon", n_genes=5
        )
        sc.pl.rank_genes_groups_dotplot(
            adata, n_genes=5,
            save="_top5_degs_dotplot.png", show=False,
        )
    except Exception as e:
        print(f"  Warning: {e}")


def plot_pathway_heatmap(pathway_results: dict):
    """Combined Score heatmap — top-5 pathways per cluster."""
    print("  Pathway heatmap…")
    records = []
    for cl, df in pathway_results.items():
        if df.empty:
            continue
        for _, row in df.head(5).iterrows():
            term  = str(row.get("Term", "?"))[:45]
            score = float(row.get("Combined Score", 0))
            records.append({"cluster": cl, "pathway": term, "score": score})

    if not records:
        print("  No pathway data — skipping heatmap")
        return

    pivot = (
        pd.DataFrame(records)
        .pivot_table(index="pathway", columns="cluster", values="score", fill_value=0)
    )

    # Sort clusters numerically if possible
    try:
        pivot = pivot[sorted(pivot.columns, key=int)]
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.9), max(6, len(pivot) * 0.35)))
    sns.heatmap(
        pivot, annot=False, cmap="YlOrRd",
        linewidths=0.4, linecolor="white",
        ax=ax, cbar_kws={"label": "Enrichr Combined Score", "shrink": 0.6},
    )
    ax.set_title("Pathway enrichment by cluster (MSigDB Hallmark + KEGG)", pad=10)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"figure2_pathway_heatmap.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → figures/figure2_pathway_heatmap.png / .pdf")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Serialize results for Day 3 agents
# ─────────────────────────────────────────────────────────────────────────────

def save_baseline_json(
    deg_results: dict,
    pathway_results: dict,
    gpt_results: dict,
):
    """
    Saves all baseline results as JSON.
    Day 3 agents read this file as their input —
    they only need to add grounding + orchestration on top.
    """
    out = {"dataset": "GSE131907_tLung", "model": OPENAI_MODEL, "clusters": {}}

    for cl in sorted(deg_results.keys(), key=int):
        degs     = deg_results[cl]
        pathways = pathway_results.get(cl, pd.DataFrame())

        out["clusters"][cl] = {
            "top_degs":       degs["names"].head(TOP_N_DEGS).tolist() if len(degs) else [],
            "deg_table":      degs.head(TOP_N_DEGS).to_dict(orient="records") if len(degs) else [],
            "top_pathways":   (pathways["Term"].head(TOP_N_PATHWAYS).tolist()
                               if len(pathways) and "Term" in pathways.columns else []),
            "pathway_table":  (pathways.head(TOP_N_PATHWAYS)
                               .to_dict(orient="records") if len(pathways) else []),
            "gpt4o_naive":    gpt_results.get(cl, "N/A"),
        }

    json_path = RESULTS_DIR / "baseline_results.json"
    json_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"  Saved → results/baseline_results.json  ({json_path.stat().st_size // 1024} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Day 2 — DEG · Pathway · GPT-4o Naive Baseline")
    print(f"Input: {PROCESSED_FILE.name}")
    print("=" * 60)

    if not PROCESSED_FILE.exists():
        print(f"ERROR: {PROCESSED_FILE} not found.")
        print("Run day1_download_preprocess.py first.")
        sys.exit(1)

    adata = sc.read_h5ad(PROCESSED_FILE)
    print(f"  Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes  |  "
          f"{adata.obs['leiden'].nunique()} clusters")

    # Run pipeline
    deg_results     = run_deg_analysis(adata)
    pathway_results = run_pathway_enrichment(deg_results)
    gpt_results     = run_gpt4o_naive(deg_results)

    # Outputs
    summary = build_summary_table(adata, deg_results, pathway_results, gpt_results)

    print("\nGenerating figures…")
    plot_deg_dotplot(adata)
    plot_pathway_heatmap(pathway_results)
    save_baseline_json(deg_results, pathway_results, gpt_results)

    print("\n" + "=" * 60)
    print("Day 2 DONE")
    print("  results/all_degs.csv")
    print("  results/degs/cluster_*.csv")
    print("  results/pathways/cluster_*.csv")
    print("  results/cluster_summary_table.csv")
    print("  results/baseline_results.json  ← Day 3 agent input")
    print("  figures/figure2_pathway_heatmap.png")
    print("\nNext: python day3_agents_orchestrator.py")
    print("=" * 60)

    # Quick sanity print
    print("\nCluster summary preview:")
    print(summary[["cluster", "n_cells", "n_sig_degs",
                    "top_pathways"]].to_string(index=False))


if __name__ == "__main__":
    main()
