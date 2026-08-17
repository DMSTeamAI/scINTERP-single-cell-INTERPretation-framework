"""nb_09_orchestrator — integrate marker + pathway + disease evidence into a calibrated,
uncertainty-aware interpretation per cluster.

Reads the three agents' structured evidence, applies the EVIDENCE-HIERARCHY weights,
computes an integrated confidence, detects cross-agent AGREEMENT vs DISAGREEMENT
(e.g. cluster 10: marker says epithelial, pathway says EMT/mesenchymal), and decides
ABSTENTION when confidence is low. Builds the structured evidence packet the LLM
narrator (nb later) consumes — the LLM interprets records, it never generates evidence.

Reads:  agent_evidence/<DATASET>/{marker_expression,pathway,disease}_evidence.json
        DEG/<DATASET>/cluster_ground_truth.csv
Writes: agent_evidence/<DATASET>/orchestrated.json  (per-cluster confidence, flags, packet)

Runner:
    DATASET = "GSE131907"
    exec(open("/lakehouse/default/Files/scRNA/code/nb_09_orchestrator.py").read())
"""
assert "DATASET" in globals(), "Set DATASET before exec()"

import json, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/lakehouse/default/Files/scRNA")
EVID = ROOT/"agent_evidence"/DATASET
DEG  = ROOT/"DEG"/DATASET

marker  = json.load(open(EVID/"marker_expression_evidence.json"))
pathway = json.load(open(EVID/"pathway_evidence.json"))
disease = json.load(open(EVID/"disease_evidence.json"))
gt = pd.read_csv(DEG/"cluster_ground_truth.csv", index_col=0); gt.index = gt.index.astype(str)
CLUSTERS = list(marker.keys())
print(f"{DATASET}: orchestrating {len(CLUSTERS)} clusters")

# ---- evidence-hierarchy weights (per agreed design) ----
W = dict(sc_expression=1.00, curated_marker=0.80, tissue_expression=0.70,
         pathway_ora=0.80, pathway_gsea=0.80, genetic_association=0.80,
         disease_association=0.60, known_drug=0.40, functional_annotation=0.20)
UNCERTAINTY_THRESHOLD = 0.50

# Lineage map: coarse family each specific label belongs to (for cross-agent agreement).
LINEAGE = {
 "t cell":"lymphoid","nkt":"lymphoid","natural killer":"lymphoid","cd4":"lymphoid","cd8":"lymphoid",
 "regulatory t":"lymphoid","b cell":"lymphoid","plasma":"lymphoid","lymphocyte":"lymphoid",
 "monocyte":"myeloid","macrophage":"myeloid","dendritic":"myeloid","neutrophil":"myeloid",
 "mast":"myeloid","microglia":"myeloid","myeloid":"myeloid",
 "epithelial":"epithelial","cancer":"epithelial","ciliated":"epithelial","secretory":"epithelial",
 "alveolar":"epithelial","club":"epithelial","surfactant":"epithelial","basal":"epithelial",
 "fibroblast":"stromal","endothelial":"stromal","mesenchymal":"stromal","pericyte":"stromal",
 "smooth muscle":"stromal","emt":"stromal","stromal":"stromal",
}
# Priority-ordered patterns: compound/specific first so "epithelial mesenchymal
# transition" resolves to stromal (EMT) before the bare "epithelial" token matches.
_LINEAGE_PRIORITY = [
    ("epithelial mesenchymal transition","stromal"), ("mesenchymal","stromal"),
    ("emt","stromal"), ("fibroblast","stromal"), ("endothelial","stromal"),
    ("angiogenesis","stromal"), ("extracellular matrix","stromal"),
    ("collagen","stromal"), ("pericyte","stromal"), ("smooth muscle","stromal"),
    ("natural killer","lymphoid"), ("nkt","lymphoid"), ("t cell","lymphoid"),
    ("cd4","lymphoid"), ("cd8","lymphoid"), ("regulatory t","lymphoid"),
    ("b cell","lymphoid"), ("plasma","lymphoid"), ("lymphocyte","lymphoid"),
    ("alpha-beta t","lymphoid"), ("immunoregulatory","lymphoid"),
 ("b cell receptor","lymphoid"), ("bcr signaling","lymphoid"), ("immunoglobulin","lymphoid"),
 ("t cell activation","lymphoid"), ("t cell receptor","lymphoid"), ("interferon gamma","lymphoid"),
    ("neutrophil","myeloid"), ("macrophage","myeloid"), ("monocyte","myeloid"),
    ("dendritic","myeloid"), ("mast","myeloid"), ("microglia","myeloid"),
    ("complement","myeloid"), ("phagocy","myeloid"), ("myeloid","myeloid"),
    ("surfactant","epithelial"), ("ciliated","epithelial"), ("secretory","epithelial"),
    ("alveolar","epithelial"), ("club cell","epithelial"),
    ("cancer","epithelial"), ("carcinoma","epithelial"), ("epithelial","epithelial"),
]
def lineage_of(text):
    t = (text or "").lower()
    for k, fam in _LINEAGE_PRIORITY:
        if k in t: return fam
    return None

def marker_conf(m):
    """Confidence from the marker agent: enrichment support of top candidate, tempered by coverage."""
    cands = m.get("candidates") or []
    top = cands[0][1] if cands else 0.0
    return float(top) * (0.5 + 0.5*m.get("coverage", 0))

def pathway_lineage(p):
    """Infer lineage from the pathway agent's specific terms. Scan the top few terms and
    return the first that carries a lineage; the major-axis disagreement gate downstream
    prevents shared-term false positives (e.g. tight-junction) from causing spurious flags.
    Scanning deeper (not just top-1) lets a clear signal like 'B Cell Receptor Signaling'
    be seen even when a generic translation term ranks above it."""
    for rec in p.get("records", [])[:2]:            # top-2 only: dominant biology, not deep incidental terms
        fam = lineage_of(rec.get("object"))
        if fam: return fam
    return None

def orchestrate(cl):
    m, p, d = marker[cl], pathway[cl], disease[cl]
    flags = []

    # 1) per-agent confidence
    c_marker = marker_conf(m)
    c_path   = 1.0 if p.get("n_records",0) >= 3 else (0.5 if p.get("n_records",0) else 0.0)
    c_dis    = min(1.0, d.get("n_context_relevant",0)/3.0)   # disease is supportive, saturates fast

    # 2) cross-agent lineage agreement (marker vs pathway)
    m_lin = lineage_of(m.get("top_candidate"))
    p_lin = pathway_lineage(p)
    _AXIS = {"lymphoid":"immune","myeloid":"immune","epithelial":"nonimmune","stromal":"nonimmune"}
    agree = (m_lin is not None and m_lin == p_lin)
    major_conflict = (m_lin and p_lin and _AXIS.get(m_lin) != _AXIS.get(p_lin))
    if major_conflict:
        flags.append(f"AGENT_DISAGREEMENT:marker={m_lin}/pathway={p_lin}")
    elif m_lin and p_lin and m_lin != p_lin:
        flags.append(f"AGENT_MINOR_DIFF:marker={m_lin}/pathway={p_lin}")   # noted, not penalized
    if agree:
        flags.append(f"AGENT_AGREEMENT:{m_lin}")

    # 3) integrated confidence: weighted mean of agent confidences, boosted by agreement
    base = np.average([c_marker, c_path, c_dis], weights=[0.5, 0.35, 0.15])
    conf = base * (1.10 if agree else (0.80 if major_conflict else 1.0))
    conf = float(np.clip(conf, 0, 1))

    # 4) carry marker ambiguity / no-evidence notes up
    for n in m.get("notes", []):
        if n.startswith(("AMBIGUOUS_TOP","NO_CONFIDENT")): flags.append(f"MARKER_{n}")
    for n in p.get("notes", []):
        if n.startswith("CONFLICT"): flags.append(f"PATHWAY_{n}")

    abstain = conf < UNCERTAINTY_THRESHOLD
    if abstain: flags.append("ABSTAIN")

    # 5) resolved call: prefer marker unless disagreement — then defer to whichever
    #    lineage the pathway (functional) evidence supports, and flag for review.
    resolved = m.get("top_candidate")
    if major_conflict:
        resolved = f"{m.get('top_candidate')} (marker) vs {p_lin} (pathway) — REVIEW"

    return dict(cluster_id=cl,
                c_marker=round(c_marker,3), c_pathway=round(c_path,3), c_disease=round(c_dis,3),
                confidence=round(conf,3), abstain=abstain,
                marker_call=m.get("top_candidate"), pathway_lineage=p_lin,
                resolved_call=resolved, agreement=agree, flags=sorted(set(flags)),
                evidence_packet=dict(
                    marker=m.get("candidates"),
                    pathway_top=[r["object"] for r in p.get("records",[])[:5]],
                    disease_context=d.get("n_context_relevant",0),
                    drivers=d.get("drivers",[])))

results = {}
print("\ncl | conf | marker call            | path lin  | agree | flags")
print("-"*82)
for cl in CLUSTERS:
    r = orchestrate(cl); results[cl] = r
    author = gt.loc[cl,"majority_author_label"] if cl in gt.index else "?"
    keyflag = next((f for f in r["flags"] if "DISAGREE" in f or f=="ABSTAIN"), "")
    print(f"{cl:>2} | {r['confidence']:.2f} | {str(r['marker_call'])[:22]:22} | "
          f"{str(r['pathway_lineage'] or '-'):9} | {'Y' if r['agreement'] else '.':^5} | {keyflag}")

json.dump(results, open(EVID/"orchestrated.json","w"), indent=1, default=str)

n_abstain = sum(1 for r in results.values() if r["abstain"])
n_disagree = sum(1 for r in results.values() if any("DISAGREE" in f for f in r["flags"]))
n_agree = sum(1 for r in results.values() if r["agreement"])
mean_conf = np.mean([r["confidence"] for r in results.values()])
print("\n" + "="*60 + f"""
NOTEBOOK 09 COMPLETE
{"="*60}
Main function : Orchestrator — integrate 3 agents, confidence, abstention
Dataset       : {DATASET}
Mean confidence      : {mean_conf:.3f}
Cross-agent agreement: {n_agree}/{len(CLUSTERS)} clusters
Disagreement flags   : {n_disagree}  (marker vs pathway lineage)
Abstentions          : {n_abstain}  (confidence < {UNCERTAINTY_THRESHOLD})
Output       : orchestrated.json
Next notebook : nb_10_validation_benchmark""")
