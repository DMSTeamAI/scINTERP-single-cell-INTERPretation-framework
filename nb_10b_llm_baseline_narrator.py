"""nb_10b_llm_baseline_narrator - the two LLM-dependent pieces:
  (A) NAIVE-LLM BASELINE : feed each cluster's top DEGs to the model COLD (no agents,
      no evidence) and ask for a cell type. Scored the SAME way as the framework.
      -> direct rebuttal to reviewer "accuracy below naive GPT".
  (B) NARRATOR           : feed the orchestrator's STRUCTURED EVIDENCE per cluster and
      generate a grounded functional summary. The LLM interprets records; it never
      invents evidence. -> demonstrates "interpretable functional summarization".

CRITICAL: naive baseline and narrator use the SAME model (fair comparison).
Provider-agnostic: swap _call_llm() to Azure/OpenAI later with no other changes.

Key entry (safe, not stored):
    import getpass; ANTHROPIC_API_KEY = getpass.getpass("Paste Anthropic API key (hidden): ")
    DATASET = "GSE131907"
    exec(open("/lakehouse/default/Files/scRNA/code/nb_10b_llm_baseline_narrator.py").read())
"""
assert "DATASET" in globals(), "Set DATASET before exec()"
assert "ANTHROPIC_API_KEY" in globals() and ANTHROPIC_API_KEY.startswith("sk-ant-"), \
    "Set ANTHROPIC_API_KEY via getpass first (must start with sk-ant-)"

import json, time, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/lakehouse/default/Files/scRNA")
DEG  = ROOT/"DEG"/DATASET
EVID = ROOT/"agent_evidence"/DATASET
gt = pd.read_csv(DEG/"cluster_ground_truth.csv", index_col=0); gt.index = gt.index.astype(str)

MODEL = "claude-haiku-4-5-20251001"     # SAME model for baseline AND narrator
TISSUE = {"GSE131907":"lung (lung adenocarcinoma)", "GSE132465":"colon (colorectal cancer)", "GSE176078":"breast (breast cancer)"}[DATASET]

# ---------------- provider-agnostic LLM call ----------------
import requests
_URL = "https://api.anthropic.com/v1/messages"
_HDR = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}

def _call_llm(system, user, max_tokens=400, retries=4):
    # direct HTTPS (the anthropic SDK fails to connect inside this Fabric environment); API-default decoding as before
    body = {"model": MODEL, "max_tokens": max_tokens, "system": system, "messages": [{"role": "user", "content": user}]}
    for a in range(retries):
        try:
            r = requests.post(_URL, headers=_HDR, json=body, timeout=60)
            if r.status_code in (429, 500, 502, 503, 529): raise RuntimeError(f"HTTP {r.status_code}")
            if r.status_code != 200: print(f"  LLM call failed: HTTP {r.status_code} {r.text[:120]}"); return ""
            return "".join(b["text"] for b in r.json()["content"] if b.get("type") == "text").strip()
        except Exception as e:
            if a < retries-1: time.sleep(3*(a+1)); continue
            print(f"  LLM call failed: {str(e)[:80]}"); return ""
    return ""

# ---- lineage-equivalence scorer (same mapping as nb_10a) ----
_FAM = [("alveolar macrophage","myeloid"),("goblet","epithelial"),("ductal","epithelial"),
 ("secretory","epithelial"),("tuft","epithelial"),("enterocyte","epithelial"),
 ("colonocyte","epithelial"),("paneth","epithelial"),("enteroendocrine","epithelial"),
 ("ciliated","epithelial"),("alveolar","epithelial"),("club","epithelial"),
 ("surfactant","epithelial"),("luminal","epithelial"),("basal cell","epithelial"),
 ("lung cell","epithelial"),("lung epithel","epithelial"),("cancer","epithelial"),
 ("malignant","epithelial"),("carcinoma","epithelial"),("epitheli","epithelial"),
 ("mast","myeloid"),("plasmacytoid dendritic","myeloid"),("plasmacytoid","myeloid"),
 ("dendritic","myeloid"),("macrophage","myeloid"),("classical monocyte","myeloid"),
 ("monocyte","myeloid"),("neutrophil","myeloid"),("microglia","myeloid"),("myeloid","myeloid"),
 ("fibroblast","stromal"),("myofibroblast","stromal"),("endotheli","stromal"),
 ("mesenchym","stromal"),("pericyte","stromal"),("smooth muscle","stromal"),
 ("emt","stromal"),("stromal","stromal"),
 ("t/nk","lymphoid"),("nk/t","lymphoid"),("natural killer","lymphoid"),("nkt","lymphoid"),
 ("nk cell","lymphoid"),("plasma cell","lymphoid"),("plasma","lymphoid"),("cd4","lymphoid"),
 ("cd8","lymphoid"),("cytotoxic t","lymphoid"),("regulatory t","lymphoid"),("treg","lymphoid"),
 ("mature b","lymphoid"),("b cell","lymphoid"),("b lymph","lymphoid"),("t lymph","lymphoid"),
 ("t cell","lymphoid"),("lymphocyte","lymphoid"),("t-cell","lymphoid"),("b-cell","lymphoid"),("caf","stromal"),("pvl","stromal"),("perivascular","stromal")]
def family(label):
    t=(label or "").lower()
    for k,f in _FAM:
        if k in t: return f
    return None

tops = json.load(open(DEG/"degs_top.json"))
CLUSTERS = list(tops.keys())

# ================= (A) NAIVE-LLM BASELINE =================
print("="*70); print("A. NAIVE-LLM BASELINE (cold DEG -> cell type, no agents)"); print("="*70)
naive_sys = (f"You are a single-cell RNA-seq expert. Given a cluster's top marker genes from "
             f"human {TISSUE} tissue, name the MOST LIKELY single cell type. "
             f"Reply with ONLY the cell type name, nothing else.")
naive = {}
for cl in CLUSTERS:
    genes = ", ".join(tops[cl]["top_agents"][:20])
    ans = _call_llm(naive_sys, f"Top marker genes: {genes}\nCell type:", max_tokens=30)
    naive[cl] = ans
    author = gt.loc[cl,"majority_author_label"] if cl in gt.index else None
    ok = family(ans)==family(author)
    print(f"  cl {cl:>2}: {ans[:32]:32} | author {str(author)[:18]:18} | {'OK' if ok else 'x'}")
    time.sleep(0.3)
naive_acc = np.mean([family(naive[cl])==family(gt.loc[cl,"majority_author_label"])
                     for cl in CLUSTERS if cl in gt.index])
print(f"\nNaive-LLM accuracy (lineage) : {naive_acc:.3f}")

# ================= (B) NARRATOR =================
print("\n" + "="*70); print("B. NARRATOR (structured evidence -> grounded summary)"); print("="*70)
orch = json.load(open(EVID/"orchestrated.json"))
pathway = json.load(open(EVID/"pathway_evidence.json"))
disease = json.load(open(EVID/"disease_evidence.json"))
narr_sys = (
    "You write the functional interpretation of a single-cell cluster for a scientific paper, "
    "STRICTLY from the structured evidence provided. Hard rules:\n"
    "1. Cite ONLY pathway/disease names that appear VERBATIM in the evidence JSON. Never introduce "
    "a pathway, gene, disease, or database ID that is not in the input.\n"
    "2. If the top pathways are generic/housekeeping (translation, ribosomal, amino-acid sensing, "
    "EIF2AK4/GCN2, peptide chain elongation), SAY the pathway evidence is non-specific for this "
    "cluster and do NOT spin a biological story from them.\n"
    "3. Do NOT invent mechanistic explanations, marker genes, or disease IDs from your own knowledge.\n"
    "4. If the cell-type call conflicts with the pathway lineage or is flagged, report the "
    "disagreement plainly and note the call is uncertain.\n"
    "5. Write 2-3 factual sentences: (a) the call + confidence, (b) which SPECIFIC provided "
    "pathways/disease hits support or contradict it (or state none are specific), (c) any flagged "
    "uncertainty. No embellishment.")
narratives = {}
for cl in CLUSTERS:
    o = orch[cl]
    ev = {
      "cell_type_call": o.get("marker_call"),
      "confidence": o.get("confidence"),
      "abstained": o.get("abstain"),
      "cross_agent_flags": o.get("flags"),
      "top_pathways": [r["object"] for r in pathway.get(cl,{}).get("records",[])[:4]],
      "disease_evidence_objects": [r.get("object") for r in disease.get(cl,{}).get("records",[])[:6]
                                   if r.get("details",{}).get("context_relevant")],
      "disease_context_hit_count": disease.get(cl,{}).get("n_context_relevant",0),
      "note": "If top_pathways are translation/ribosomal/EIF2AK4 terms, they are NON-SPECIFIC.",
    }
    txt = _call_llm(narr_sys, f"Cluster {cl} evidence (JSON):\n{json.dumps(ev, indent=1)}\n\nWrite the interpretation:", max_tokens=250)
    narratives[cl] = txt
    print(f"\n--- Cluster {cl} (call: {ev['cell_type_call']}, conf {ev['confidence']}) ---")
    print("  " + txt.replace("\n"," "))
    time.sleep(0.3)

# ---- save + head-to-head ----
out = dict(model=MODEL, naive_baseline=naive, naive_accuracy=float(naive_acc),
           narratives=narratives)
json.dump(out, open(EVID/"llm_baseline_narrator.json","w"), indent=2, default=str)

# framework accuracy for the head-to-head (from nb_10a)
fw = json.load(open(EVID/"benchmark_10a.json"))["overall_accuracy"]
print("\n" + "="*60 + f"""
NOTEBOOK 10b COMPLETE
{"="*60}
Dataset                : {DATASET}   Model: {MODEL}
Naive-LLM accuracy     : {naive_acc:.3f}
scINTERP framework acc : {fw:.3f}   (from nb_10a)
Delta (framework-naive): {fw-naive_acc:+.3f}
Narratives generated   : {len(narratives)} clusters
Output                 : llm_baseline_narrator.json""")
