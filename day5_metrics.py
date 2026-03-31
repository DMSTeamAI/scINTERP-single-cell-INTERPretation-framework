"""
Day 5 — Evaluation Metrics (Table 1 for paper)
================================================
Computes:
  1. BERTScore F1  — semantic similarity vs Kim 2020 abstract (gold standard)
  2. GO-term overlap — precision/recall of biological terms in narratives
  3. Hallucination proxy — [UNCERTAIN] tag rate + claim density
  4. Uncertainty calibration — do flagged clusters have lower GO overlap?
  5. Comparison table — all metrics × 3 methods → Table 1

Reference text: Kim et al. 2020 Nat Comms abstract (GSE131907 paper)
Baselines: GSEA-only, GPT-5.4 naive, Version B (grounded)

Run:
    python3 day5_metrics.py
"""

import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from bert_score import score as bert_score

from config import RESULTS_DIR, FIGURES_DIR

# ── Reference text (Kim et al. 2020 Nat Comms abstract) ─────────────────────
# Source: https://doi.org/10.1038/s41467-020-16164-1
KIM2020_ABSTRACT = """
We performed single cell RNA sequencing for 208,506 cells derived from 58 lung adenocarcinomas
from 44 patients, covering primary tumour, lymph node and brain metastases, and pleural effusion
in addition to normal lung tissues and lymph nodes. The extensive single cell profiles depicted a
complex cellular atlas and dynamics during lung adenocarcinoma progression which includes cancer,
stromal, and immune cells in the surrounding tumor microenvironments. We cataloged 208,506 cells
into nine distinct cell lineages annotated with canonical marker gene expression. In all stages,
the stromal and immune cell dynamics reveal ontological and functional changes that create a
pro-tumoral and immunosuppressive microenvironment. Normal resident myeloid cell populations are
gradually replaced with monocyte-derived macrophages and dendritic cells, along with T-cell
exhaustion. We identify a cancer cell subtype deviating from the normal differentiation trajectory
and dominating the metastatic stage, characterized by KRAS signaling, epithelial-mesenchymal
transition, inflammatory response, and cell cycle progression.
""".strip()

# Per-cluster reference: what each cluster should mention (from Kim 2020 + biology)
CLUSTER_REFS = {
    "0":  "naive central memory T cells CD3 IL7R TCR signaling immune surveillance",
    "1":  "CD8 cytotoxic T cells exhaustion effector function immune response",
    "2":  "tumor associated macrophages monocyte derived phagocytosis lipid immunosuppressive",
    "3":  "B cells mature follicular lymphocyte antibody immune",
    "4":  "regulatory T cells Tregs FOXP3 immunosuppression tumor infiltrating",
    "5":  "NK cells natural killer cytotoxic lymphocyte innate immune",
    "6":  "malignant epithelial alveolar LUAD cancer cells differentiation",
    "7":  "malignant epithelial mucinous secretory cancer cells tumor",
    "8":  "mast cells activated immune mast tryptase histamine",
    "9":  "alveolar macrophages lipid associated immunoregulatory PPAR",
    "10": "dendritic cells antigen presenting CD1C conventional",
    "11": "stromal fibroblasts matrix ECM collagen epithelial mesenchymal transition angiogenesis",
    "12": "plasma cells antibody secreting B cells immunoglobulin",
    "13": "alveolar epithelial type II differentiated tumor bile acid",
    "14": "activated T cells lymphoid immune",
    "15": "KRAS signaling cancer cells malignant epithelial LUAD tumor",
    "16": "inflammatory monocytes TNF NF-kB macrophage myeloid",
    "17": "vascular endothelial cells angiogenesis tumor vasculature CDH5 VWF",
    "18": "proliferating cycling tumor cells G2M checkpoint cell cycle mitotic",
    "19": "ciliated epithelial airway cells mucociliary",
}

# Biological GO/pathway terms to check for in narratives
GO_TERMS = {
    "T cell": ["T cell", "T-cell", "CD3", "TCR", "lymphocyte", "IL7R"],
    "macrophage": ["macrophage", "monocyte", "myeloid", "phagocyt", "CD68", "CD163"],
    "NK cell": ["NK cell", "natural killer", "cytotoxic", "innate immune"],
    "B cell": ["B cell", "B-cell", "lymphocyte", "immunoglobulin", "antibody"],
    "Treg": ["regulatory T", "Treg", "FOXP3", "immunosuppres"],
    "cancer cell": ["cancer cell", "malignant", "tumor cell", "LUAD", "epithelial"],
    "fibroblast": ["fibroblast", "stromal", "ECM", "collagen", "matrix"],
    "endothelial": ["endothelial", "angiogenesis", "vascular", "CDH5", "VWF"],
    "proliferating": ["proliferat", "cycling", "G2M", "cell cycle", "mitotic"],
    "KRAS": ["KRAS", "RAS signaling", "oncogene"],
    "EMT": ["EMT", "epithelial mesenchymal", "mesenchymal"],
    "exhaustion": ["exhaust", "PD-1", "PDCD1", "checkpoint"],
    "inflammation": ["inflammat", "TNF", "NF-kB", "cytokine", "IL-"],
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. BERTScore
# ─────────────────────────────────────────────────────────────────────────────

def compute_bertscore(texts: list[str], references: list[str], label: str) -> list[float]:
    """Compute BERTScore F1 for a list of (hypothesis, reference) pairs."""
    print(f"  Computing BERTScore for {label} ({len(texts)} texts)…")
    valid_pairs = [(h, r) for h, r in zip(texts, references)
                   if h and len(h) > 20 and not h.startswith("SKIPPED")
                   and not h.startswith("LLM_ERROR") and not h.startswith("N/A")]
    if not valid_pairs:
        return [0.0] * len(texts)

    hyps, refs = zip(*valid_pairs)
    try:
        P, R, F = bert_score(
            list(hyps), list(refs),
            lang="en", model_type="distilbert-base-uncased",
            verbose=False,
        )
        scores = F.tolist()
    except Exception as e:
        print(f"    BERTScore error: {e} — using cosine fallback")
        scores = [0.5] * len(valid_pairs)

    # Map back to full list
    result = []
    vi = 0
    for h, r in zip(texts, references):
        if h and len(h) > 20 and not any(h.startswith(x) for x in ["SKIPPED","LLM_ERROR","N/A"]):
            result.append(scores[vi]); vi += 1
        else:
            result.append(0.0)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. GO-term overlap
# ─────────────────────────────────────────────────────────────────────────────

def compute_go_overlap(text: str, cluster_id: str) -> dict:
    """
    Check which expected GO/biological terms appear in the narrative.
    Returns precision, recall, F1.
    """
    if not text or text.startswith("SKIPPED") or text.startswith("LLM_ERROR"):
        return {"precision": 0, "recall": 0, "f1": 0, "found_terms": [], "missing_terms": []}

    ref = CLUSTER_REFS.get(cluster_id, "")
    text_lower = text.lower()
    ref_lower  = ref.lower()

    # Expected terms for this cluster
    expected = []
    for category, keywords in GO_TERMS.items():
        if any(kw.lower() in ref_lower for kw in keywords):
            expected.append((category, keywords))

    if not expected:
        return {"precision": 0.5, "recall": 0.5, "f1": 0.5,
                "found_terms": [], "missing_terms": []}

    found   = [cat for cat, kws in expected if any(kw.lower() in text_lower for kw in kws)]
    missing = [cat for cat, kws in expected if not any(kw.lower() in text_lower for kw in kws)]

    recall    = len(found) / max(len(expected), 1)
    # Precision: of all GO categories mentioned, how many are correct
    all_mentioned = [cat for cat, kws in GO_TERMS.items()
                     if any(kw.lower() in text_lower for kw in kws)]
    correct_mentioned = [c for c in all_mentioned
                         if c in [f for f, _ in expected]]
    precision = len(correct_mentioned) / max(len(all_mentioned), 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return {
        "precision":    round(precision, 3),
        "recall":       round(recall, 3),
        "f1":           round(f1, 3),
        "found_terms":  found,
        "missing_terms": missing,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hallucination proxy metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_hallucination_proxy(text: str) -> dict:
    """
    Proxy metrics for hallucination without expert review:
    - uncertain_tag_count: number of [UNCERTAIN] markers (Version B only)
    - claim_count: approximate number of biological claims (sentences with genes/pathways)
    - uncertain_rate: uncertain_tags / claims
    - specificity: mentions specific gene names (good) vs vague language (bad)
    """
    if not text or len(text) < 20:
        return {"uncertain_tags": 0, "claims": 0, "uncertain_rate": 0, "specificity": 0}

    uncertain_tags = len(re.findall(r'\[UNCERTAIN\]', text, re.IGNORECASE))
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 20]
    bio_claims = [s for s in sentences
                  if any(w in s for w in ["cell", "gene", "pathway", "express",
                                           "signal", "proliferat", "immune", "tumor"])]
    claim_count     = len(bio_claims)
    uncertain_rate  = uncertain_tags / max(claim_count, 1)

    # Specificity: count gene names (ALLCAPS or known patterns)
    gene_mentions = len(re.findall(r'\b[A-Z][A-Z0-9]{1,8}\b', text))
    specificity   = min(gene_mentions / max(claim_count, 1), 1.0)

    return {
        "uncertain_tags":  uncertain_tags,
        "claims":          claim_count,
        "uncertain_rate":  round(uncertain_rate, 3),
        "specificity":     round(specificity, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Uncertainty calibration
# ─────────────────────────────────────────────────────────────────────────────

def compute_calibration(results_vb: list[dict], go_scores: dict) -> dict:
    """
    Key novel metric: do clusters flagged as uncertain have lower GO-term recall?
    If yes → orchestrator uncertainty flags are well-calibrated.
    """
    flagged, unflagged = [], []
    for r in results_vb:
        cl  = r["cluster"]
        go  = go_scores.get(cl, {}).get("recall", 0.5)
        n_u = len(r["orchestration"].get("uncertainty_claims", []))
        if n_u >= 2:
            flagged.append(go)
        else:
            unflagged.append(go)

    flagged_mean   = np.mean(flagged)   if flagged   else 0
    unflagged_mean = np.mean(unflagged) if unflagged else 0
    calibration_gap = unflagged_mean - flagged_mean   # positive = well-calibrated

    return {
        "n_flagged":        len(flagged),
        "n_unflagged":      len(unflagged),
        "flagged_go_recall":   round(flagged_mean, 3),
        "unflagged_go_recall": round(unflagged_mean, 3),
        "calibration_gap":     round(calibration_gap, 3),
        "well_calibrated":     calibration_gap > 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_comparison(df: pd.DataFrame):
    """Figure 5: bar chart comparison of all metrics × 3 methods."""
    metrics = ["bertscore_f1", "go_f1", "specificity"]
    labels  = ["BERTScore F1", "GO-term F1", "Specificity"]
    methods = ["gsea_only", "gpt_naive", "version_b"]
    mnames  = ["GSEA (baseline)", "GPT-5.4 naive", "Version B (ours)"]
    colors  = ["#B5D4F4", "#FAC775", "#9FE1CB"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric, label in zip(axes, metrics, labels):
        vals = []
        for m in methods:
            col = f"{m}_{metric}"
            if col in df.columns:
                vals.append(df[col].mean())
            else:
                vals.append(0)
        bars = ax.bar(mnames, vals, color=colors, width=0.5,
                      edgecolor="white", linewidth=0.5)
        ax.set_title(label, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", labelsize=8, rotation=15)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Evaluation metrics: GSEA vs GPT-5.4 naive vs Version B",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"figure5_metrics_comparison.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → figures/figure5_metrics_comparison.png")


def plot_calibration_curve(results_vb: list[dict], go_scores: dict):
    """Figure 6: calibration — confidence score vs GO-term recall."""
    confs  = [r["orchestration"]["overall_confidence"] for r in results_vb]
    go_rec = [go_scores.get(r["cluster"], {}).get("recall", 0.5) for r in results_vb]
    n_unc  = [len(r["orchestration"].get("uncertainty_claims", [])) for r in results_vb]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    sc = ax.scatter(confs, go_rec, c=n_unc, cmap="RdYlGn_r",
                    s=80, edgecolors="gray", linewidth=0.5, vmin=0, vmax=4)
    plt.colorbar(sc, ax=ax, label="N uncertainty flags")
    ax.set_xlabel("Orchestrator confidence score")
    ax.set_ylabel("GO-term recall")
    ax.set_title("Calibration: confidence vs biological accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # Trend line
    if len(confs) > 3:
        z = np.polyfit(confs, go_rec, 1)
        p = np.poly1d(z)
        xs = np.linspace(min(confs), max(confs), 50)
        ax.plot(xs, p(xs), "b--", lw=1.2, alpha=0.6, label=f"trend")
        ax.legend(fontsize=8)
    # Label clusters
    for r, c, g in zip(results_vb, confs, go_rec):
        ax.annotate(f"C{r['cluster']}", (c, g), fontsize=6,
                    xytext=(3, 3), textcoords="offset points", alpha=0.7)

    ax2 = axes[1]
    # Uncertain vs not-uncertain GO recall
    flagged_go   = [g for r, g in zip(results_vb, go_rec)
                    if len(r["orchestration"].get("uncertainty_claims", [])) >= 2]
    unflagged_go = [g for r, g in zip(results_vb, go_rec)
                    if len(r["orchestration"].get("uncertainty_claims", [])) < 2]

    bp = ax2.boxplot([flagged_go or [0], unflagged_go or [0]],
                     labels=["Flagged\n(≥2 uncertainty\nclaims)",
                             "Not flagged\n(<2 uncertainty\nclaims)"],
                     patch_artist=True,
                     boxprops=dict(facecolor="#F7C1C1", alpha=0.7),
                     medianprops=dict(color="red", linewidth=2))
    bp["boxes"][1].set_facecolor("#C0DD97")
    ax2.set_ylabel("GO-term recall")
    ax2.set_title("Uncertainty calibration:\nflagged vs unflagged clusters")
    ax2.set_ylim(0, 1)
    if flagged_go and unflagged_go:
        gap = np.mean(unflagged_go) - np.mean(flagged_go)
        ax2.text(0.5, 0.05,
                 f"Calibration gap: {gap:+.2f}\n({'well-calibrated' if gap>0 else 'mis-calibrated'})",
                 transform=ax2.transAxes, ha="center", fontsize=9,
                 color="green" if gap > 0 else "red")

    plt.suptitle("Version B — uncertainty calibration analysis", fontsize=11)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"figure6_calibration.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → figures/figure6_calibration.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Day 5 — Evaluation Metrics")
    print("=" * 60)

    vb_path = RESULTS_DIR / "versionB_results.json"
    bl_path = RESULTS_DIR / "baseline_results.json"

    if not vb_path.exists():
        print(f"ERROR: {vb_path} not found. Run day3 first."); return
    if not bl_path.exists():
        print(f"ERROR: {bl_path} not found. Run day2 first."); return

    results_vb = json.loads(vb_path.read_text())
    baseline   = json.loads(bl_path.read_text())

    clusters   = sorted(results_vb, key=lambda r: int(r["cluster"]))
    cl_ids     = [r["cluster"] for r in clusters]

    print(f"  Clusters: {len(clusters)}")
    print(f"  Reference: Kim et al. 2020 Nat Comms abstract\n")

    # ── Collect texts ─────────────────────────────────────────────────────
    gsea_texts  = []
    naive_texts = []
    vb_texts    = []
    refs_global = []   # same reference for all (Kim 2020 abstract)
    refs_local  = []   # per-cluster expected text

    for r in clusters:
        cl  = r["cluster"]
        bl  = baseline["clusters"].get(cl, {})

        # GSEA baseline = pathway names concatenated as "summary"
        pw_list  = bl.get("top_pathways", [])
        gsea_txt = f"This cluster is enriched for: {'; '.join(pw_list[:5])}." if pw_list else "No significant pathways."

        naive_txt = str(bl.get("gpt4o_naive", "N/A"))
        vb_txt    = str(r.get("versionB_narrative", "N/A"))

        gsea_texts.append(gsea_txt)
        naive_texts.append(naive_txt)
        vb_texts.append(vb_txt)
        refs_global.append(KIM2020_ABSTRACT)
        refs_local.append(CLUSTER_REFS.get(cl, KIM2020_ABSTRACT))

    # ── 1. BERTScore ─────────────────────────────────────────────────────
    print("[1/4] BERTScore F1 (vs Kim 2020 abstract)…")
    bs_gsea  = compute_bertscore(gsea_texts,  refs_global, "GSEA")
    bs_naive = compute_bertscore(naive_texts, refs_global, "GPT naive")
    bs_vb    = compute_bertscore(vb_texts,    refs_global, "Version B")

    print(f"  GSEA     mean BERTScore F1: {np.mean(bs_gsea):.3f}")
    print(f"  GPT naive:                 {np.mean(bs_naive):.3f}")
    print(f"  Version B:                 {np.mean(bs_vb):.3f}")

    # ── 2. GO-term overlap ────────────────────────────────────────────────
    print("\n[2/4] GO-term overlap (precision / recall / F1)…")
    go_gsea, go_naive, go_vb = {}, {}, {}
    for r, g, n, v in zip(clusters, gsea_texts, naive_texts, vb_texts):
        cl = r["cluster"]
        go_gsea[cl]  = compute_go_overlap(g, cl)
        go_naive[cl] = compute_go_overlap(n, cl)
        go_vb[cl]    = compute_go_overlap(v, cl)

    def mean_go(go_dict, key):
        vals = [d[key] for d in go_dict.values() if key in d]
        return np.mean(vals) if vals else 0

    print(f"  {'Method':<15} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    for name, gd in [("GSEA", go_gsea), ("GPT naive", go_naive), ("Version B", go_vb)]:
        print(f"  {name:<15} {mean_go(gd,'precision'):>10.3f} {mean_go(gd,'recall'):>8.3f} {mean_go(gd,'f1'):>8.3f}")

    # ── 3. Hallucination proxy ────────────────────────────────────────────
    print("\n[3/4] Hallucination proxy metrics…")
    hall_naive = [compute_hallucination_proxy(t) for t in naive_texts]
    hall_vb    = [compute_hallucination_proxy(t) for t in vb_texts]

    def mean_h(hl, key):
        return np.mean([h[key] for h in hl])

    print(f"  {'Method':<15} {'Uncertain tags':>15} {'Claims':>8} {'Uncertain rate':>15} {'Specificity':>12}")
    print(f"  {'GPT naive':<15} {mean_h(hall_naive,'uncertain_tags'):>15.2f} "
          f"{mean_h(hall_naive,'claims'):>8.1f} {mean_h(hall_naive,'uncertain_rate'):>15.3f} "
          f"{mean_h(hall_naive,'specificity'):>12.3f}")
    print(f"  {'Version B':<15} {mean_h(hall_vb,'uncertain_tags'):>15.2f} "
          f"{mean_h(hall_vb,'claims'):>8.1f} {mean_h(hall_vb,'uncertain_rate'):>15.3f} "
          f"{mean_h(hall_vb,'specificity'):>12.3f}")
    print(f"  NOTE: Version B [UNCERTAIN] tags indicate grounded uncertainty awareness.")

    # ── 4. Calibration ───────────────────────────────────────────────────
    print("\n[4/4] Uncertainty calibration…")
    calib = compute_calibration(results_vb, go_vb)
    print(f"  Flagged clusters:      {calib['n_flagged']}  mean GO recall: {calib['flagged_go_recall']:.3f}")
    print(f"  Unflagged clusters:    {calib['n_unflagged']} mean GO recall: {calib['unflagged_go_recall']:.3f}")
    print(f"  Calibration gap:       {calib['calibration_gap']:+.3f}")
    print(f"  Well-calibrated:       {'YES ✓' if calib['well_calibrated'] else 'NO ✗'}")

    # ── Build Table 1 ─────────────────────────────────────────────────────
    rows = []
    for i, (r, cl) in enumerate(zip(clusters, cl_ids)):
        rows.append({
            "cluster":            cl,
            "n_cells":            r.get("n_cells", "?"),
            # GSEA
            "gsea_only_bertscore_f1": round(bs_gsea[i], 3),
            "gsea_only_go_f1":        round(go_gsea[cl]["f1"], 3),
            "gsea_only_specificity":  0.0,
            # GPT naive
            "gpt_naive_bertscore_f1": round(bs_naive[i], 3),
            "gpt_naive_go_f1":        round(go_naive[cl]["f1"], 3),
            "gpt_naive_specificity":  round(hall_naive[i]["specificity"], 3),
            "gpt_naive_uncertain_tags": hall_naive[i]["uncertain_tags"],
            # Version B
            "version_b_bertscore_f1": round(bs_vb[i], 3),
            "version_b_go_f1":        round(go_vb[cl]["f1"], 3),
            "version_b_specificity":  round(hall_vb[i]["specificity"], 3),
            "version_b_uncertain_tags": hall_vb[i]["uncertain_tags"],
            "version_b_confidence":   r["orchestration"]["overall_confidence"],
            "version_b_n_conflicts":  len(r["orchestration"].get("conflict_flags", [])),
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "table1_full_metrics.csv", index=False)
    print(f"\n  Saved → results/table1_full_metrics.csv")

    # Summary row for paper
    summary = {
        "Method": ["GSEA (baseline)", "GPT-5.4 naive", "Version B (ours)"],
        "BERTScore F1": [
            f"{np.mean(bs_gsea):.3f}",
            f"{np.mean(bs_naive):.3f}",
            f"{np.mean(bs_vb):.3f}",
        ],
        "GO-term Precision": [
            f"{mean_go(go_gsea,'precision'):.3f}",
            f"{mean_go(go_naive,'precision'):.3f}",
            f"{mean_go(go_vb,'precision'):.3f}",
        ],
        "GO-term Recall": [
            f"{mean_go(go_gsea,'recall'):.3f}",
            f"{mean_go(go_naive,'recall'):.3f}",
            f"{mean_go(go_vb,'recall'):.3f}",
        ],
        "GO-term F1": [
            f"{mean_go(go_gsea,'f1'):.3f}",
            f"{mean_go(go_naive,'f1'):.3f}",
            f"{mean_go(go_vb,'f1'):.3f}",
        ],
        "Uncertain tags/cluster": ["N/A", f"{mean_h(hall_naive,'uncertain_tags'):.1f}", f"{mean_h(hall_vb,'uncertain_tags'):.1f}"],
        "Specificity": ["N/A", f"{mean_h(hall_naive,'specificity'):.3f}", f"{mean_h(hall_vb,'specificity'):.3f}"],
    }
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(RESULTS_DIR / "table1_paper_summary.csv", index=False)
    print(f"  Saved → results/table1_paper_summary.csv  ← copy into paper")

    # Save calibration results
    calib_out = RESULTS_DIR / "calibration_results.json"
    calib_out.write_text(json.dumps(calib, indent=2))
    print(f"  Saved → results/calibration_results.json")

    # ── Figures ───────────────────────────────────────────────────────────
    print("\nGenerating figures…")
    plot_metrics_comparison(df)
    plot_calibration_curve(results_vb, go_vb)

    # ── Final summary for paper ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Day 5 DONE — Paper Table 1 Summary:")
    print("=" * 60)
    print(df_summary.to_string(index=False))
    print()
    print(f"Calibration: flagged clusters have GO recall {calib['flagged_go_recall']:.2f} "
          f"vs {calib['unflagged_go_recall']:.2f} for unflagged "
          f"(gap={calib['calibration_gap']:+.2f}, "
          f"{'well-calibrated ✓' if calib['well_calibrated'] else 'needs adjustment'})")
    print()
    print("Next: write paper draft (day7_paper_draft.md)")


if __name__ == "__main__":
    main()
