"""nb_07_pathway_agent — Pathway Agent (Reactome + GO BP + MSigDB Hallmark via Enrichr).

The AGENT runs the statistics; the LLM never does enrichment. Two complementary tests:
  ORA      : over-representation of the cluster's significant DEG list (top-N)
  prerank  : GSEA on the FULL ranked gene list (removes the arbitrary top-50 cutoff)
Every enriched term becomes a structured EvidenceRecord carrying FDR, overlap, and
contributing genes. Antagonistic-pathway CONFLICT detection runs here (testable later
via the synthetic-conflict harness). Includes a NEGATIVE CONTROL: random gene sets
should yield few/no significant terms.

Reads:  DEG/<DATASET>/degs_top.json, degs_ranked.parquet
Writes: agent_evidence/<DATASET>/pathway_evidence.json, pathway_negative_controls.json

Runner:
    DATASET = "GSE131907"
    exec(open("/lakehouse/default/Files/scRNA/code/nb_07_pathway_agent.py").read())
"""
assert "DATASET" in globals(), "Set DATASET before exec()"

import os, re, json, time, logging
from datetime import date
from pathlib import Path
import numpy as np, pandas as pd

log = logging.getLogger("nb07")
ROOT = Path("/lakehouse/default/Files/scRNA")
DEG_DIR  = ROOT/"DEG"/DATASET
EVID_DIR = ROOT/"agent_evidence"/DATASET
EVID_DIR.mkdir(parents=True, exist_ok=True)
TODAY = date.today().isoformat()

LIBRARIES = {
    "Reactome": "Reactome_Pathways_2024",
    "GO_BP":    "GO_Biological_Process_2023",
    "Hallmark": "MSigDB_Hallmark_2020",
}
RUN_GSEA  = False   # prerank GSEA is slow; ORA is the primary evidence. Turn on for final run.
FDR_ORA   = 0.05
FDR_GSEA  = 0.25
TOP_TERMS = 10

# Functionally antagonistic pathway pairs for conflict detection (extend freely)
ANTAGONISTIC = [
    ({"g2-m","e2f targets","mitotic","cell cycle"}, {"apoptosis","cell death"}),
    ({"tnf-alpha","inflammatory response","nf-kb"}, {"il-10","anti-inflammatory"}),
    ({"epithelial mesenchymal transition","emt"}, {"epithelial cell differentiation"}),
    ({"interferon","antiviral"}, {"immunosuppress"}),
]

tops   = json.load(open(DEG_DIR/"degs_top.json"))
ranked = pd.read_parquet(DEG_DIR/"degs_ranked.parquet")
CLUSTERS = list(tops.keys())
print(f"{DATASET}: {len(CLUSTERS)} clusters")

import warnings
warnings.filterwarnings("ignore", message="Duplicated values found in preranked stats")
import gseapy as gp

def run_ora(genes, retries=4):
    frames = []
    for label, lib in LIBRARIES.items():
        for attempt in range(retries):
            try:
                r = gp.enrichr(gene_list=list(genes), gene_sets=lib, outdir=None).results
                r["library"] = label
                frames.append(r)
                time.sleep(1.0)                     # throttle to avoid Enrichr 429
                break
            except Exception as e:
                if "429" in str(e) and attempt < retries-1:
                    wait = 5 * (attempt+1)
                    log.warning("ORA %s rate-limited, retry in %ss", lib, wait)
                    time.sleep(wait); continue
                log.warning("ORA %s failed: %s", lib, e); break
    if not frames: return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out[out["Adjusted P-value"] <= FDR_ORA]

def run_prerank(rnk):
    frames = []
    for label, lib in LIBRARIES.items():
        try:
            r = gp.prerank(rnk=rnk, gene_sets=lib, permutation_num=200,
                           outdir=None, seed=0, min_size=5, max_size=500).res2d
            r["library"] = label
            frames.append(r)
        except Exception as e:
            log.warning("prerank %s failed: %s", lib, e)
    if not frames: return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    fdr_col = "FDR q-val" if "FDR q-val" in out else ("fdr" if "fdr" in out else None)
    return out[out[fdr_col] <= FDR_GSEA] if fdr_col else out

def detect_conflicts(terms):
    low = [t.lower() for t in terms]
    hits = []
    for a, b in ANTAGONISTIC:
        ha = [t for t in low if any(k in t for k in a)]
        hb = [t for t in low if any(k in t for k in b)]
        if ha and hb:
            hits.append((ha[0], hb[0]))
    return hits

# Generic / low-information terms that win on raw significance but say little.
# We keep them in the evidence (they ARE enriched) but push them DOWN the ranking so
# the narrator sees specific biology first.
_GENERIC_PATTERNS = [
    # ribosomal / translation
    "peptide chain elongation","translation elongation","eukaryotic translation",
    "cytoplasmic translation","ribosom","rrna","srp-dependent","selenocysteine",
    "nonsense mediated decay","nonsense-mediated","nmd ","formation of a pool",
    "l13a-mediated","gtp hydrolysis","cap-dependent translation","influenza",
    "viral mrna translation","major pathway of rrna",
    # top-level catch-alls
    "immune system","adaptive immune system","innate immune system","metabolism of proteins",
    "gene expression","cellular responses to stimuli","generic transcription",
    "metabolism (r-hsa","disease","infectious disease","signal transduction",
    "post-translational protein","cytokine signaling in immune",
    # hallmark artifacts
    "allograft rejection","estrogen response","myc targets","g2m checkpoint proteins",
    "unfolded protein response","xenobiotic metabolism",
]
def _is_generic(term):
    t = term.lower()
    return any(p in t for p in _GENERIC_PATTERNS)

def _rank_key(rec):
    # specific terms first (0), generic later (1); then by FDR
    return (1 if _is_generic(rec["object"]) else 0, rec.get("fdr", 1.0) or 1.0)

def run_cluster(cl):
    records, notes = [], []
    degs = tops[cl]["top_ora"]
    ora = run_ora(degs)
    if ora.empty:
        notes.append("NO_SIGNIFICANT_ORA")
    else:
        for row in ora.nsmallest(TOP_TERMS, "Adjusted P-value").to_dict("records"):
            records.append(dict(
                database=f"Enrichr:{row['library']}", database_version="2020-2024 libs",
                retrieval_date=TODAY, evidence_type="pathway_ora",
                subject=cl, predicate="enriched_for", object=str(row["Term"]),
                pvalue=float(row.get("P-value", float("nan"))),
                fdr=float(row["Adjusted P-value"]),
                details={"overlap":str(row.get("Overlap","")),
                         "genes":str(row.get("Genes","")).split(";")[:15]}))
    # prerank on the full ranked list for this cluster
    sub = ranked[ranked.cluster == cl][["gene","scores"]].dropna()
    if RUN_GSEA and len(sub) > 50:
        rnk = sub.set_index("gene")["scores"].sort_values(ascending=False)
        gsea = run_prerank(rnk)
        fdr_col = "FDR q-val" if "FDR q-val" in gsea else ("fdr" if "fdr" in gsea else None)
        term_col = "Term" if "Term" in gsea else ("term" if "term" in gsea else None)
        if not gsea.empty and fdr_col and term_col:
            for row in gsea.nsmallest(TOP_TERMS, fdr_col).to_dict("records"):
                nes = row.get("NES", row.get("nes", float("nan")))
                try: nes = float(nes)
                except Exception: nes = None
                records.append(dict(
                    database=f"GSEA-prerank:{row.get('library','?')}",
                    database_version="gseapy", retrieval_date=TODAY,
                    evidence_type="pathway_gsea", subject=cl, predicate="enriched_for",
                    object=str(row[term_col]),
                    fdr=float(row[fdr_col]),
                    details={"NES":nes}))
    records.sort(key=_rank_key)   # specific biology first, generic/ribosomal last
    conflicts = detect_conflicts([r["object"] for r in records])
    for a,b in conflicts:
        notes.append(f"CONFLICT:{a[:30]}|{b[:30]}")
    return dict(agent="pathway", cluster_id=cl, n_records=len(records),
                n_ora=sum(1 for r in records if r["evidence_type"]=="pathway_ora"),
                n_gsea=sum(1 for r in records if r["evidence_type"]=="pathway_gsea"),
                conflicts=conflicts, notes=sorted(set(notes)), records=records)

outputs = {}
print("\ncluster | #ORA | #GSEA | top term")
print("-"*70)
for cl in CLUSTERS:
    out = run_cluster(cl)
    outputs[cl] = out
    top = out["records"][0]["object"][:40] if out["records"] else "(none)"
    print(f"{cl:>7} | {out['n_ora']:>4} | {out['n_gsea']:>5} | {top}")

json.dump(outputs, open(EVID_DIR/"pathway_evidence.json","w"), indent=1, default=str)

# ---------------- NEGATIVE CONTROL ----------------
print("\n" + "="*60 + "\nNEGATIVE CONTROL\n" + "="*60)
rng = np.random.default_rng(0)
universe = sorted(ranked.gene.unique())
real_terms = np.mean([outputs[cl]["n_ora"] for cl in CLUSTERS])
rand_counts = []
for _ in range(10):
    fake = list(rng.choice(universe, size=50, replace=False))
    rand_counts.append(len(run_ora(fake)))
rand_terms = float(np.mean(rand_counts))
sep = real_terms/rand_terms if rand_terms else float("inf")
print(f"significant ORA terms — real clusters: {real_terms:.1f}   random sets: {rand_terms:.1f}   "
      f"separation: {sep:.1f}x  (real should be much higher)")
json.dump({"real_mean_terms":float(real_terms),"random_mean_terms":rand_terms,
           "separation_x":float(sep)},
          open(EVID_DIR/"pathway_negative_controls.json","w"), indent=1)

total_conf = sum(len(outputs[cl]["conflicts"]) for cl in CLUSTERS)
mean_terms = np.mean([o["n_records"] for o in outputs.values()])
print("\n" + "="*60 + f"""
NOTEBOOK 07 COMPLETE
{"="*60}
Main function : Pathway Agent (ORA + prerank GSEA; Reactome/GO/Hallmark)
Dataset       : {DATASET}
Mean terms/cluster : {mean_terms:.1f}
Conflicts detected : {total_conf} (antagonistic-pathway pairs)
Neg-control sep    : {sep:.1f}x
Outputs       : pathway_evidence.json, pathway_negative_controls.json
Next notebook : nb_08_disease_agent""")
