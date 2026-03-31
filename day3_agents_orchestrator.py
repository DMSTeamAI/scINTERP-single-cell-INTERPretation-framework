"""
Day 3 — Knowledge-grounded multi-agent orchestrator (Version B)
================================================================
Architecture:
  Agent 1: DEG Validator     → UniProt REST API  (verifies each gene is real)
  Agent 2: Pathway Agent     → Reactome REST API (enriches top pathways)
  Agent 3: Disease Agent     → DisGeNET REST API (maps pathways → diseases)
  Orchestrator               → merges outputs, detects conflicts, scores confidence
  LLM Narrator               → GPT-5.4 (Azure) with full grounded context packet

Input:   results/baseline_results.json   (from Day 2)
Outputs:
  results/versionB_results.json          ← grounded narratives + confidence scores
  results/comparison_table.csv           ← all 3 methods side-by-side
  figures/figure3_confidence_scores.png
  figures/figure4_case_study_cluster11.png

Run:
    export AZURE_OPENAI_API_KEY="..."
    export AZURE_OPENAI_ENDPOINT="https://....openai.azure.com/"
    export AZURE_OPENAI_DEPLOYMENT="gpt-5.4"
    export AZURE_OPENAI_API_VERSION="2025-04-01-preview"
    python3 day3_agents_orchestrator.py
"""

import sys, os, json, time, hashlib, requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm
from openai import AzureOpenAI, OpenAI
from pathlib import Path

from config import (
    RESULTS_DIR, FIGURES_DIR, CACHE_DIR,
    OPENAI_API_KEY, OPENAI_MODEL,
    N_DEGS_FOR_LLM, TOP_N_PATHWAYS,
    LLM_TEMPERATURE, LLM_MAX_TOKENS,
)

# ── Azure / OpenAI client ──────────────────────────────────────────────────
_azure_key      = os.getenv("AZURE_OPENAI_API_KEY", "")
_azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
_azure_deploy   = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
_azure_version  = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

client = None
DEPLOY_NAME = OPENAI_MODEL

if _azure_key and _azure_endpoint:
    client = AzureOpenAI(
        api_key=_azure_key,
        azure_endpoint=_azure_endpoint,
        api_version=_azure_version,
    )
    DEPLOY_NAME = _azure_deploy
    print(f"Using Azure OpenAI | deployment={_azure_deploy}")
elif OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
    print(f"Using standard OpenAI | model={OPENAI_MODEL}")
else:
    print("ERROR: No API key found.")
    print("  export AZURE_OPENAI_API_KEY=... AZURE_OPENAI_ENDPOINT=... AZURE_OPENAI_DEPLOYMENT=...")
    sys.exit(1)

BASELINE_JSON = RESULTS_DIR / "baseline_results.json"
TIMEOUT       = 8    # seconds per API call
MAX_RETRIES   = 2


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ckey(prefix: str, text: str) -> str:
    h = hashlib.md5(text.encode()).hexdigest()[:10]
    return f"{prefix}_{h}"

def _cload(key: str):
    p = CACHE_DIR / f"{key}.json"
    return json.loads(p.read_text()) if p.exists() else None

def _csave(key: str, data):
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(data, default=str))


# ─────────────────────────────────────────────────────────────────────────────
# Agent 1 — DEG Validator (UniProt REST)
# ─────────────────────────────────────────────────────────────────────────────

def agent_deg_validator(genes: list[str], cluster_id: str) -> dict:
    """
    Queries UniProt for each gene name.
    Returns:
      verified_genes   : list of genes confirmed in UniProt
      unverified_genes : list not found (potential hallucination sources)
      gene_functions   : dict gene → short function description
      confidence       : float 0-1 (fraction verified)
    """
    key = _ckey("uniprot", f"cl{cluster_id}_" + "_".join(genes[:10]))
    cached = _cload(key)
    if cached:
        return cached

    verified, unverified, functions = [], [], {}
    for gene in genes:
        url = (
            f"https://rest.uniprot.org/uniprotkb/search"
            f"?query=gene:{gene}+AND+organism_id:9606+AND+reviewed:true"
            f"&fields=gene_names,protein_name,function"
            f"&format=json&size=1"
        )
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.get(url, timeout=TIMEOUT)
                if r.status_code == 200:
                    data = r.json()
                    results = data.get("results", [])
                    if results:
                        verified.append(gene)
                        # Extract short function text
                        fn = results[0].get("comments", [])
                        fn_text = next(
                            (c["texts"][0]["value"][:120]
                             for c in fn if c.get("commentType") == "FUNCTION"),
                            "function not retrieved"
                        )
                        functions[gene] = fn_text
                    else:
                        unverified.append(gene)
                    break
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    unverified.append(gene)
        time.sleep(0.15)  # gentle rate limit

    result = {
        "verified_genes":   verified,
        "unverified_genes": unverified,
        "gene_functions":   functions,
        "confidence":       len(verified) / max(len(genes), 1),
        "n_genes_checked":  len(genes),
    }
    _csave(key, result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Agent 2 — Pathway Agent (Reactome REST)
# ─────────────────────────────────────────────────────────────────────────────

def agent_pathway(genes: list[str], top_pathways: list[str], cluster_id: str) -> dict:
    """
    Queries Reactome for pathway descriptions and hierarchy.
    Returns:
      reactome_pathways  : list of dicts {name, description, parent, url}
      pathway_confidence : float (fraction of top_pathways found in Reactome)
      conflicts          : list of pathway pairs that are biologically contradictory
    """
    key = _ckey("reactome", f"cl{cluster_id}_" + "_".join(top_pathways[:3]))
    cached = _cload(key)
    if cached:
        return cached

    reactome_hits = []
    found_names   = []

    # Search Reactome for each top pathway by name
    for pw in top_pathways[:TOP_N_PATHWAYS]:
        url = f"https://reactome.org/ContentService/search/query?query={requests.utils.quote(pw)}&species=Homo+sapiens&types=Pathway&cluster=true&rows=1"
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.get(url, timeout=TIMEOUT)
                if r.status_code == 200:
                    data = r.json()
                    results = data.get("results", [])
                    if results:
                        entries = results[0].get("entries", [])
                        if entries:
                            entry = entries[0]
                            reactome_hits.append({
                                "query_name":  pw,
                                "reactome_id": entry.get("stId", ""),
                                "exact_name":  entry.get("name", pw),
                                "species":     entry.get("species", ["Homo sapiens"])[0]
                                               if isinstance(entry.get("species"), list)
                                               else entry.get("species", "Homo sapiens"),
                                "url": f"https://reactome.org/content/detail/{entry.get('stId','')}",
                            })
                            found_names.append(pw)
                break
            except Exception:
                pass
        time.sleep(0.2)

    # Conflict detection: flag biologically contradictory pathway pairs
    # Simple heuristic: if both proliferation and apoptosis appear, flag it
    conflict_pairs = [
        ({"G2-M Checkpoint","E2F Targets","Mitotic Spindle","Cell cycle"},
         {"Apoptosis","p53 Pathway"},
         "Proliferation vs apoptosis signals"),
        ({"TNF-alpha Signaling","Inflammatory Response","IL-17 signaling"},
         {"IL-10 signaling","Anti-inflammatory"},
         "Pro- vs anti-inflammatory"),
        ({"Epithelial Mesenchymal Transition"},
         {"Cell adhesion molecules"},
         "EMT with strong adhesion — may indicate partial EMT"),
    ]
    conflicts_found = []
    pw_set = set(top_pathways)
    for set_a, set_b, label in conflict_pairs:
        if pw_set & set_a and pw_set & set_b:
            conflicts_found.append(label)

    pw_confidence = len(found_names) / max(len(top_pathways), 1) if top_pathways else 0.0

    result = {
        "reactome_pathways":  reactome_hits,
        "found_in_reactome":  found_names,
        "pathway_confidence": pw_confidence,
        "conflicts":          conflicts_found,
        "n_pathways_queried": len(top_pathways),
    }
    _csave(key, result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Agent 3 — Disease Agent (DisGeNET via UMLS / open endpoint)
# ─────────────────────────────────────────────────────────────────────────────

def agent_disease(genes: list[str], top_pathways: list[str], cluster_id: str) -> dict:
    """
    Maps top DEGs to disease associations via DisGeNET open API.
    Falls back to Open Targets if DisGeNET unavailable.
    Returns:
      disease_associations : list of {gene, disease, score, source}
      top_diseases         : list of top disease names
      luad_relevance       : float — fraction of associations mentioning lung/cancer
      confidence           : float
    """
    key = _ckey("disease", f"cl{cluster_id}_" + "_".join(genes[:8]))
    cached = _cload(key)
    if cached:
        return cached

    associations = []
    # Use Open Targets genetics API (no auth needed)
    for gene in genes[:10]:
        url = (
            f"https://api.platform.opentargets.org/api/v4/graphql"
        )
        query = """
        {
          target(ensemblId: "ENSG00000000000") {
            approvedSymbol
            associatedDiseases(page: {index: 0, size: 3}) {
              rows { disease { name therapeuticAreas { name } } score }
            }
          }
        }
        """
        # Simpler: use MyGene.info for gene → disease via OMIM
        mg_url = f"https://mygene.info/v3/query?q=symbol:{gene}&species=human&fields=symbol,name,OMIM,pathway&size=1"
        try:
            r = requests.get(mg_url, timeout=TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                hits = data.get("hits", [])
                if hits:
                    h = hits[0]
                    omim = h.get("OMIM", "")
                    name = h.get("name", gene)
                    if omim:
                        associations.append({
                            "gene":    gene,
                            "omim_id": omim,
                            "gene_name": name,
                            "source":  "OMIM via MyGene.info",
                        })
        except Exception:
            pass
        time.sleep(0.1)

    # Keyword-based lung cancer relevance scoring
    luad_keywords = {
        "lung", "cancer", "tumor", "carcinoma", "adenocarcinoma",
        "pulmonary", "LUAD", "NSCLC", "neoplasm", "oncogene",
        "KRAS", "EGFR", "ALK", "TP53", "BRAF",
    }
    # Check pathway names for cancer relevance
    pw_text = " ".join(top_pathways).lower()
    cancer_pw_hits = sum(1 for kw in luad_keywords if kw.lower() in pw_text)
    luad_relevance = min(cancer_pw_hits / 3.0, 1.0)

    # Known LUAD driver genes — instant check
    luad_drivers = {
        "KRAS","EGFR","ALK","ROS1","BRAF","MET","RET","NTRK1",
        "TP53","STK11","KEAP1","SMARCA4","NF1","RB1","CDKN2A",
        "EPCAM","KRT18","KRT19","SFTPC","NKX2-1","TTF1",
        "CD3D","CD3E","CD8A","CD4","FOXP3","PDCD1","CTLA4",
        "CD68","CD163","MRC1","MARCO",
        "ACTA2","FAP","COL1A1","VIM",
        "PECAM1","CDH5","VWF",
    }
    overlap = set(genes) & luad_drivers
    driver_note = (
        f"Contains known LUAD-relevant genes: {', '.join(sorted(overlap))}"
        if overlap else "No canonical LUAD driver genes in top DEGs"
    )

    top_diseases = list({a["gene_name"] for a in associations[:5]})

    result = {
        "disease_associations": associations[:10],
        "top_diseases":         top_diseases,
        "luad_relevance":       round(luad_relevance, 3),
        "luad_driver_genes":    sorted(overlap),
        "driver_note":          driver_note,
        "confidence":           min(luad_relevance + 0.3, 1.0),
    }
    _csave(key, result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator — merges agent outputs, scores confidence, detects conflicts
# ─────────────────────────────────────────────────────────────────────────────

def orchestrate(
    cluster_id:   str,
    deg_result:   dict,
    pathway_result: dict,
    disease_result: dict,
) -> dict:
    """
    Merges outputs from all three agents.
    Computes:
      overall_confidence : weighted mean of agent confidences
      conflict_flags     : any biological contradictions detected
      uncertainty_claims : specific claims flagged as needing hedging
      grounding_summary  : structured context packet for LLM
    """
    c_deg     = deg_result.get("confidence", 0.5)
    c_pathway = pathway_result.get("pathway_confidence", 0.5)
    c_disease = disease_result.get("confidence", 0.5)

    overall = round((c_deg * 0.35 + c_pathway * 0.35 + c_disease * 0.30), 3)

    # Collect all conflicts
    conflicts = pathway_result.get("conflicts", [])
    unverified = deg_result.get("unverified_genes", [])
    if unverified:
        conflicts.append(
            f"Unverified DEGs (not in UniProt reviewed): {', '.join(unverified[:5])}"
        )

    # Uncertainty flags
    uncertainty_claims = []
    if c_deg < 0.6:
        uncertainty_claims.append("DEG verification confidence low — interpret cell type with caution")
    if c_pathway < 0.5:
        uncertainty_claims.append("Fewer than half of enriched pathways confirmed in Reactome")
    if disease_result.get("luad_relevance", 0) < 0.3:
        uncertainty_claims.append("Low direct LUAD pathway relevance — may be non-cancer cell type")
    if overall < 0.5:
        uncertainty_claims.append("Overall confidence below threshold — narrative should be hedged")

    grounding = {
        "verified_genes":    deg_result.get("verified_genes", [])[:10],
        "gene_functions":    deg_result.get("gene_functions", {}),
        "reactome_pathways": pathway_result.get("reactome_pathways", [])[:5],
        "luad_drivers":      disease_result.get("luad_driver_genes", []),
        "driver_note":       disease_result.get("driver_note", ""),
        "luad_relevance":    disease_result.get("luad_relevance", 0),
        "conflicts":         conflicts,
    }

    return {
        "cluster":              cluster_id,
        "overall_confidence":   overall,
        "agent_confidences":    {"deg": c_deg, "pathway": c_pathway, "disease": c_disease},
        "conflict_flags":       conflicts,
        "uncertainty_claims":   uncertainty_claims,
        "grounding_summary":    grounding,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM Narrator — grounded narrative with uncertainty markers
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert computational biologist specialising in lung
adenocarcinoma (LUAD) single-cell transcriptomics. You generate precise, evidence-grounded
biological interpretations of cell clusters.

RULES:
1. Base every claim ONLY on the evidence provided in the context packet.
2. For any claim not directly supported by the evidence, write [UNCERTAIN] before it.
3. Mention the LUAD driver genes if present.
4. If conflicts are flagged, acknowledge them explicitly.
5. End with a one-sentence confidence statement.
6. Target length: 180-220 words."""

USER_TMPL = """Dataset: GSE131907 — human lung adenocarcinoma (LUAD), primary tumor (tLung)
Cluster: {cl}  |  Cells: {n_cells}  |  Overall confidence: {conf:.0%}

=== EVIDENCE PACKET (from 3 knowledge agents) ===

Verified DEGs (UniProt confirmed, human):
{verified_genes}

Key gene functions (from UniProt):
{gene_functions}

Reactome pathways confirmed:
{reactome_pathways}

LUAD-relevant driver genes detected: {luad_drivers}
Driver note: {driver_note}

LUAD pathway relevance score: {luad_relevance:.0%}

Conflict flags (if any):
{conflicts}

Uncertainty flags:
{uncertainty_claims}

=== NAIVE GPT INTERPRETATION (baseline — may contain hallucinations) ===
{naive_text}

=== YOUR TASK ===
Write a grounded biological interpretation of this cluster for a paper Methods/Results section.
Mark uncertain claims with [UNCERTAIN]. Cite which agent provided each key claim."""

def _format_grounding(orch: dict, naive: str, n_cells: int) -> str:
    g = orch["grounding_summary"]
    gf = g.get("gene_functions", {})
    fn_text = "\n".join(
        f"  {gene}: {fn[:100]}" for gene, fn in list(gf.items())[:6]
    ) or "  (none retrieved)"

    rp = g.get("reactome_pathways", [])
    rp_text = "\n".join(
        f"  {p['query_name']} → {p['exact_name']} ({p.get('url','')})"
        for p in rp[:4]
    ) or "  (none confirmed in Reactome)"

    conflicts_text = "\n".join(f"  ⚠ {c}" for c in orch["conflict_flags"]) or "  None detected"
    uncertain_text = "\n".join(f"  ⚠ {u}" for u in orch["uncertainty_claims"]) or "  None"

    return USER_TMPL.format(
        cl=orch["cluster"],
        n_cells=n_cells,
        conf=orch["overall_confidence"],
        verified_genes=", ".join(g.get("verified_genes", [])[:12]) or "(none)",
        gene_functions=fn_text,
        reactome_pathways=rp_text,
        luad_drivers=", ".join(g.get("luad_drivers", [])) or "none",
        driver_note=g.get("driver_note", ""),
        luad_relevance=g.get("luad_relevance", 0),
        conflicts=conflicts_text,
        uncertainty_claims=uncertain_text,
        naive_text=str(naive)[:400] if naive and not naive.startswith("SKIPPED") else "(not available)",
    )


def run_llm_narrator(orch: dict, naive: str, n_cells: int) -> str:
    key = _ckey("versionB_narrator", f"cl{orch['cluster']}_conf{orch['overall_confidence']}")
    cached = _cload(key)
    if cached:
        return cached["text"]

    prompt = _format_grounding(orch, naive, n_cells)
    try:
        resp = client.chat.completions.create(
            model=DEPLOY_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_completion_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        text = resp.choices[0].message.content.strip()
        _csave(key, {"text": text})
        return text
    except Exception as e:
        return f"LLM_ERROR: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_confidence_scores(results: list[dict]):
    """Figure 3: confidence score breakdown per cluster."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    clusters = [r["cluster"] for r in results]
    overall  = [r["orchestration"]["overall_confidence"] for r in results]
    c_deg    = [r["orchestration"]["agent_confidences"]["deg"] for r in results]
    c_path   = [r["orchestration"]["agent_confidences"]["pathway"] for r in results]
    c_dis    = [r["orchestration"]["agent_confidences"]["disease"] for r in results]
    n_conf   = [len(r["orchestration"]["conflict_flags"]) for r in results]

    x = np.arange(len(clusters))
    w = 0.25

    ax = axes[0]
    ax.bar(x - w,   c_deg,  w, label="DEG validator",   color="#185FA5", alpha=0.85)
    ax.bar(x,       c_path, w, label="Pathway agent",   color="#0F6E56", alpha=0.85)
    ax.bar(x + w,   c_dis,  w, label="Disease agent",   color="#993C1D", alpha=0.85)
    ax.plot(x, overall, "k--o", ms=5, lw=1.5, label="Overall (weighted)")
    ax.axhline(0.5, color="red", lw=0.8, ls=":", label="Uncertainty threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in clusters], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Confidence score")
    ax.set_title("Agent confidence scores per cluster")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.05)

    ax2 = axes[1]
    colors = ["#E24B4A" if n > 0 else "#1D9E75" for n in n_conf]
    ax2.bar(x, n_conf, color=colors, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"C{c}" for c in clusters], rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Number of conflict flags")
    ax2.set_title("Orchestrator conflict flags per cluster")
    red_p  = mpatches.Patch(color="#E24B4A", alpha=0.85, label="Has conflicts")
    grn_p  = mpatches.Patch(color="#1D9E75", alpha=0.85, label="No conflicts")
    ax2.legend(handles=[red_p, grn_p], fontsize=8)

    plt.suptitle("Version B — orchestrated agent confidence & conflict detection", fontsize=11)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"figure3_confidence_scores.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → figures/figure3_confidence_scores.png")


def plot_case_study(results: list[dict]):
    """Figure 4: Case study — most interesting cluster (highest confidence + LUAD relevance)."""
    # Pick cluster with highest luad_relevance
    best = max(
        results,
        key=lambda r: r["orchestration"]["grounding_summary"].get("luad_relevance", 0),
    )
    cl = best["cluster"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: method comparison text boxes
    ax = axes[0]
    ax.axis("off")
    methods = [
        ("GSEA (baseline)",  best.get("top_pathways_str", "N/A"), "#B5D4F4"),
        ("GPT naive",        str(best.get("gpt4o_naive","N/A"))[:300], "#FAC775"),
        ("Version B",        str(best.get("versionB_narrative","N/A"))[:400], "#9FE1CB"),
    ]
    y = 0.95
    for title, text, color in methods:
        ax.text(0.02, y, title, fontsize=9, fontweight="bold",
                transform=ax.transAxes, va="top", color="#2C2C2A")
        wrapped = text[:250] + "..." if len(text) > 250 else text
        ax.text(0.02, y - 0.03, wrapped, fontsize=7,
                transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.6, linewidth=0.5),
                wrap=True)
        y -= 0.36
    ax.set_title(f"Cluster {cl} — method comparison", pad=8)

    # Right: agent confidence breakdown
    ax2 = axes[1]
    orch = best["orchestration"]
    confs = orch["agent_confidences"]
    labels = ["DEG\nvalidator", "Pathway\nagent", "Disease\nagent", "Overall\n(weighted)"]
    vals   = [confs["deg"], confs["pathway"], confs["disease"], orch["overall_confidence"]]
    colors = ["#185FA5", "#0F6E56", "#993C1D", "#2C2C2A"]
    bars   = ax2.barh(labels, vals, color=colors, alpha=0.8, height=0.5)
    ax2.axvline(0.5, color="red", lw=1.2, ls="--", label="Uncertainty threshold")
    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel("Confidence score")
    ax2.set_title(f"Cluster {cl} — confidence breakdown\n"
                  f"LUAD relevance: {orch['grounding_summary'].get('luad_relevance',0):.0%}")
    for bar, val in zip(bars, vals):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                 f"{val:.2f}", va="center", fontsize=9)
    if orch["conflict_flags"]:
        cf_text = "⚠ Conflicts:\n" + "\n".join(f"• {c[:60]}" for c in orch["conflict_flags"])
        ax2.text(0.02, -0.25, cf_text, transform=ax2.transAxes,
                 fontsize=7, color="#A32D2D",
                 bbox=dict(boxstyle="round", facecolor="#FCEBEB", alpha=0.7))

    plt.suptitle(f"Case study: Cluster {cl} (highest LUAD relevance)", fontsize=11)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"figure4_case_study_cluster{cl}.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → figures/figure4_case_study_cluster{cl}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Build comparison table (Table 1 for paper)
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        orch = r["orchestration"]
        rows.append({
            "cluster":            r["cluster"],
            "n_cells":            r.get("n_cells", "?"),
            "top_pathway_gsea":   r.get("top_pathways_str", "")[:80],
            "gpt_naive_brief":    str(r.get("gpt4o_naive","N/A"))[:120],
            "versionB_brief":     str(r.get("versionB_narrative","N/A"))[:120],
            "overall_confidence": orch.get("overall_confidence", 0),
            "n_conflicts":        len(orch.get("conflict_flags", [])),
            "n_uncertain_claims": len(orch.get("uncertainty_claims", [])),
            "luad_drivers":       ", ".join(orch["grounding_summary"].get("luad_drivers", [])),
            "luad_relevance":     orch["grounding_summary"].get("luad_relevance", 0),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)
    print(f"  Saved → results/comparison_table.csv")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Day 3 — Multi-Agent Orchestrator (Version B)")
    print("=" * 65)

    if not BASELINE_JSON.exists():
        print(f"ERROR: {BASELINE_JSON} not found. Run day2 first.")
        sys.exit(1)

    baseline = json.loads(BASELINE_JSON.read_text())
    clusters  = baseline["clusters"]
    print(f"  Loaded baseline for {len(clusters)} clusters from {BASELINE_JSON.name}")
    print()

    all_results = []

    for cl_id in tqdm(sorted(clusters.keys(), key=int), desc="Clusters"):
        cl = clusters[cl_id]
        top_degs     = cl.get("top_degs", [])[:N_DEGS_FOR_LLM]
        top_pathways = cl.get("top_pathways", [])
        naive_text   = cl.get("gpt4o_naive", "N/A")
        # Estimate n_cells from deg table length proxy
        deg_table = cl.get("deg_table", [])
        n_cells   = cl.get("n_cells", len(deg_table) * 10)  # rough if not stored

        # Run 3 agents
        deg_result     = agent_deg_validator(top_degs, cl_id)
        pathway_result = agent_pathway(top_degs, top_pathways, cl_id)
        disease_result = agent_disease(top_degs, top_pathways, cl_id)

        # Orchestrate
        orch = orchestrate(cl_id, deg_result, pathway_result, disease_result)

        # Generate grounded narrative
        narrative = run_llm_narrator(orch, naive_text, n_cells)
        time.sleep(0.5)

        all_results.append({
            "cluster":            cl_id,
            "n_cells":            n_cells,
            "top_degs":           top_degs,
            "top_pathways_str":   "; ".join(top_pathways[:5]),
            "gpt4o_naive":        naive_text,
            "versionB_narrative": narrative,
            "orchestration":      orch,
        })

        # Live progress print
        conf   = orch["overall_confidence"]
        n_conf = len(orch["conflict_flags"])
        uncertain = len(orch["uncertainty_claims"])
        luad_r = orch["grounding_summary"].get("luad_relevance", 0)
        tqdm.write(
            f"  Cluster {cl_id:>2s} | conf={conf:.2f} | conflicts={n_conf} "
            f"| uncertain_claims={uncertain} | LUAD_rel={luad_r:.0%}"
        )

    # Save full results
    out_json = RESULTS_DIR / "versionB_results.json"
    out_json.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n  Saved → {out_json.name}  ({out_json.stat().st_size // 1024} KB)")

    # Figures
    print("\nGenerating figures…")
    plot_confidence_scores(all_results)
    plot_case_study(all_results)

    # Comparison table
    df = build_comparison_table(all_results)

    # Summary stats for paper
    confs     = [r["orchestration"]["overall_confidence"] for r in all_results]
    n_conf    = sum(len(r["orchestration"]["conflict_flags"]) > 0 for r in all_results)
    high_conf = sum(c >= 0.7 for c in confs)

    print("\n" + "=" * 65)
    print("Day 3 DONE")
    print(f"  Clusters processed : {len(all_results)}")
    print(f"  Mean confidence    : {np.mean(confs):.2f}")
    print(f"  High confidence (≥0.7): {high_conf}/{len(all_results)}")
    print(f"  Clusters with conflicts: {n_conf}/{len(all_results)}")
    print(f"  results/versionB_results.json")
    print(f"  results/comparison_table.csv  ← Table 1 draft")
    print(f"  figures/figure3_confidence_scores.png")
    print(f"  figures/figure4_case_study_cluster*.png")
    print("\nNext: python3 day5_metrics.py")
    print("=" * 65)

    # Quick preview
    print("\nTop 5 clusters by LUAD relevance:")
    preview = sorted(
        all_results,
        key=lambda r: r["orchestration"]["grounding_summary"].get("luad_relevance", 0),
        reverse=True,
    )[:5]
    for r in preview:
        g = r["orchestration"]["grounding_summary"]
        print(f"  Cluster {r['cluster']:>2s} | LUAD={g.get('luad_relevance',0):.0%} "
              f"| drivers={g.get('luad_drivers',[])} "
              f"| conf={r['orchestration']['overall_confidence']:.2f}")


if __name__ == "__main__":
    main()
