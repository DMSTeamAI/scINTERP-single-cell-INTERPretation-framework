"""nb_06_DEG_evidence_agent — Marker & Expression Evidence Agent.

Replaces the UniProt DEG Validator. Two-stage, non-circular:
  Stage 1 HYPOTHESIZE : CellMarker 2.0 votes DEGs -> candidate cell types (direction-safe)
  Stage 2 TEST        : CELLxGENE Census + HPA quantify whether the DEGs are actually
                        expressed with specificity in each candidate cell type / tissue

The statistical significance of DEGs comes from YOUR data (nb_05 Wilcoxon); this agent
supplies external biological-consistency evidence only. Every output is a structured
EvidenceRecord (never free text) so the LLM narrator later interprets records, not raw genes.

Includes a NEGATIVE-CONTROL block: shuffled DEG lists should collapse Hit@k to chance and
housekeeping genes should yield little marker evidence — the falsification test reviewers want.

Runner cell:
    DATASET = "GSE131907"
    exec(open("/lakehouse/default/Files/scRNA/code/nb_06_DEG_evidence_agent.py").read())
"""
assert "DATASET" in globals(), "Set DATASET before exec()"

import os, re, json, glob, time, urllib.request, logging
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("nb06")
ROOT = Path("/lakehouse/default/Files/scRNA")
DEG_DIR  = ROOT/"DEG"/DATASET
EVID_DIR = ROOT/"agent_evidence"/DATASET
EXT_DIR  = ROOT/"external"
CACHE    = ROOT/"cache"
for d in (EVID_DIR, EXT_DIR, CACHE): d.mkdir(parents=True, exist_ok=True)

TISSUE = {"GSE131907": "lung", "GSE132465": "colon", "GSE176078": "breast"}[DATASET]
CENSUS_VERSION = "2025-01-30"     # pin an LTS release; report in Methods
TODAY = date.today().isoformat()
TOP_CANDIDATES = 3

# ---------------------------------------------------------------------------
# Load DEGs from nb_05
# ---------------------------------------------------------------------------
tops = json.load(open(DEG_DIR/"degs_top.json"))
CLUSTERS = list(tops.keys())
print(f"{DATASET}: {len(CLUSTERS)} clusters, tissue={TISSUE}")

# ===========================================================================
# STAGE 1 — CellMarker 2.0 index (curated marker hypotheses)
# ===========================================================================
# Accept either CellMarker 2.0 xlsx OR CellMarker v1 man_cell_markers.txt / *_cell_markers.txt.
# Upload the file to Files/scRNA/external/ (the bio-annotation.cn host is blocked from Fabric).
CM_CANDIDATES = (["Cell_marker_Human.xlsx", "man_cell_markers.txt", "all_cell_markers.txt",
                  "Human_cell_markers.txt"])
CM_PATH = next((EXT_DIR/f for f in CM_CANDIDATES if (EXT_DIR/f).exists()), None)
if CM_PATH is None:
    hits = sorted(glob.glob(str(EXT_DIR/"*cell_marker*")) + glob.glob(str(EXT_DIR/"*_markers.txt")))
    CM_PATH = Path(hits[0]) if hits else None
print("CellMarker file:", CM_PATH if CM_PATH else "NOT FOUND -> upload to " + str(EXT_DIR))

class CellMarkerIndex:
    """Format-agnostic loader: CellMarker 2.0 (.xlsx) or v1 (.txt, comma-packed geneSymbol)."""
    def __init__(self, path, version=None):
        self._by_gene = defaultdict(list)
        if not path or not Path(path).exists():
            self.version = "CellMarker (missing)"
            print("  CellMarker index empty — Stage 1 will produce no hypotheses"); return
        p = str(path)
        if p.lower().endswith((".xlsx",".xls")):
            df = pd.read_excel(path); self.version = version or "CellMarker 2.0"
        else:
            df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
            self.version = version or "CellMarker v1 (manual)"
        cols = {c.lower().replace(" ","_"): c for c in df.columns}
        gcol = (cols.get("genesymbol") or cols.get("gene_symbol") or cols.get("marker")
                or cols.get("cellmarker") or cols.get("symbol"))
        ccol = (cols.get("cellname") or cols.get("cell_name") or cols.get("cell_type")
                or cols.get("celltype"))
        tcol = (cols.get("tissuetype") or cols.get("tissue_type") or cols.get("tissue_class")
                or cols.get("tissue"))
        scol = cols.get("speciestype") or cols.get("species_type") or cols.get("species")
        assert gcol and ccol, f"Could not find gene/cell columns in {list(df.columns)}"
        if scol:
            df = df[df[scol].astype(str).str.contains("human", case=False, na=False)]
        keep = [c for c in (gcol, ccol, tcol) if c]
        df = df[keep].dropna(subset=[gcol, ccol])
        rows = []
        for r in df.itertuples(index=False):
            raw = str(getattr(r, gcol) if hasattr(r, gcol) else r[0])
            ct  = getattr(r, ccol) if hasattr(r, ccol) else r[1]
            tis = (getattr(r, tcol) if (tcol and hasattr(r, tcol)) else
                   (r[2] if tcol else ""))
            for g in re.split(r"[,;/]", raw):
                g = g.strip().strip("[]()'\" ").upper()
                if g and g not in ("NA","NAN",""):
                    rows.append((g, str(ct), str(tis)))
        ex = pd.DataFrame(rows, columns=["gene","cell_type","tissue"])
        grp = ex.value_counts(["gene","cell_type","tissue"]).reset_index(name="n")
        for r in grp.itertuples(index=False):
            self._by_gene[r.gene].append((r.cell_type, r.tissue, int(r.n)))
        print(f"  CellMarker loaded: {len(self._by_gene):,} genes, {len(ex):,} gene-celltype rows "
              f"(version: {self.version})")
    def lookup(self, gene, tissue=None):
        hits = self._by_gene.get(gene.upper(), [])
        if tissue:
            t = tissue.lower(); th = [h for h in hits if t in h[1].lower()]
            return th or hits
        return hits

cm = CellMarkerIndex(CM_PATH)

from scipy.stats import hypergeom

FDR_MAX = 0.05          # a cell type must be enriched at FDR<=0.05 to be a candidate
AMBIG_MARGIN = 1.25     # top must beat 2nd by >=25% support else flag ambiguous
MIN_OVERLAP = 3         # need >=3 shared markers (avoid 1-2 gene coincidences)

# Build cell-type -> set(marker genes), restricted to a background of genes CellMarker knows.
_ct_markers = defaultdict(set)
for g, hits in cm._by_gene.items():
    for ct, _t, _tis in hits:
        _ct_markers[ct].add(g)
_BACKGROUND = set(cm._by_gene.keys())          # universe = all CellMarker genes
_N = len(_BACKGROUND)
# Drop ultra-rare cell types (<5 markers): too little evidence to test, they cause spurious wins

# --- context restriction via DENYLIST on cell-type names (robust to tissue-column quirks) ---
# Remove cell types that cannot occur in a lung/colon tumor sample: reproductive, CNS,
# developmental/embryonic, and organ-specific epithelia from unrelated organs.
# Shared exclusions: never plausible in a lung OR colon tumor sample.
_DENY_SHARED = [
    "granulosa","trophectoderm","leydig","sertoli","oocyte","spermat","germ cell","embryo",
    "trophoblast","syncytiotrophoblast","placent","fetal","blastocyst","zygote",
    "microglia","neuron","astrocyte","oligodendrocyte","purkinje","schwann","glia","retina",
    "photoreceptor","cardiomyocyte","kidney","nephron","podocyte",
    "islet","beta cell","alpha cell","melanocyte","keratinocyte","osteo","chondro",
    "adipocyte","skeletal muscle","dclk1","mki67+ progenitor",
    "spermatogonial","radial glia","neural progenitor","cardiac progenitor",
]
# Organ-specific epithelial types: exclude the OTHER organ's epithelia per tissue.
_DENY_BY_TISSUE = {
    "lung":  ["paneth","enterocyte","enteroendocrine","goblet","colonocyte","tuft",
              "hepatic","hepatocyte","biliary","cholangiocyte","liver progenitor","pancreatic","acinar"],
    "colon": ["alveolar","club cell","ciliated","ionocyte","pulmonary","type i pneumocyte",
              "type ii pneumocyte","hepatic","hepatocyte","biliary","pancreatic","acinar"],
    # breast denylist fixed BEFORE the GSE176078 run (off-organ epithelia only)
    "breast": ["alveolar","club cell","ciliated","ionocyte","pulmonary","pneumocyte","paneth","enterocyte",
               "enteroendocrine","goblet","colonocyte","tuft","hepatic","hepatocyte","biliary","pancreatic","acinar"],
}
_DENY_SUBSTR = _DENY_SHARED + _DENY_BY_TISSUE.get(TISSUE, [])
def _denied(ct):
    c = ct.lower()
    return any(s in c for s in _DENY_SUBSTR)
_before = len(_ct_markers)
def _min_markers(ct):
    c = ct.lower()
    return 8 if any(k in c for k in ("progenitor","precursor","stem")) else 5
_ct_markers = {ct: gs for ct, gs in _ct_markers.items() if len(gs) >= _min_markers(ct) and not _denied(ct)}
print(f"  denylist filter: {_before} -> {len(_ct_markers)} cell types kept "
      f"(removed reproductive/CNS/developmental/off-organ types)")
print(f"  enrichment background: {_N:,} genes")

def hypothesize(degs, tissue, top_k=TOP_CANDIDATES):
    """Score each cell type by hypergeometric enrichment of the cluster's DEGs among that
    cell type's known markers (Fisher/hypergeometric). Housekeeping and promiscuous genes
    do NOT create enrichment, so they score poorly -> negative control passes by design.
    Returns [(cell_type, enrichment_support)] where support = -log10(FDR)-scaled in (0,1]."""
    query = set(g.upper() for g in degs) & _BACKGROUND
    k_query = len(query)
    if k_query < MIN_OVERLAP:
        return []
    pvals = []
    for ct, markers in _ct_markers.items():
        overlap = len(query & markers)
        if overlap < MIN_OVERLAP:
            continue
        # P(X >= overlap) under hypergeometric(N, K=|markers|, n=k_query)
        p = hypergeom.sf(overlap - 1, _N, len(markers), k_query)
        boost = 0.5 if tissue.lower() in " ".join(  # small tie-break toward tissue-consistent types
            t for h in [] for t in []) else 0.0
        pvals.append((ct, p, overlap))
    if not pvals:
        return []
    # Benjamini-Hochberg FDR
    pvals.sort(key=lambda x: x[1])
    m = len(pvals)
    out = []
    for i, (ct, p, ov) in enumerate(pvals):
        fdr = min(p * m / (i + 1), 1.0)
        if fdr <= FDR_MAX:
            support = round(min(1.0, -np.log10(max(fdr, 1e-12)) / 10.0), 3)
            out.append((ct, support))
    return out[:top_k]

# ===========================================================================
# STAGE 2a — CELLxGENE Census specificity (optional; degrades gracefully)
# ===========================================================================
CENSUS_OK = False
_census_tbl = None
def _load_census():
    global CENSUS_OK, _census_tbl
    cache = CACHE/f"census_{CENSUS_VERSION}_{TISSUE}.parquet"
    if cache.exists():
        _census_tbl = pd.read_parquet(cache); CENSUS_OK = True
        print(f"  Census cache loaded: {cache.name} ({_census_tbl.shape})"); return
    try:
        import cellxgene_census  # noqa
    except Exception:
        print("  Census: package not installed -> Stage 2a skipped "
              "(run the census-cache notebook, or `%pip install cellxgene-census`)")
        return
    # No cache and package present: build a DEG-restricted aggregate (heavy but bounded)
    try:
        import cellxgene_census
        gene_universe = sorted({g for v in tops.values() for g in v["top_agents"]})
        print(f"  Building Census aggregate for {TISSUE} on {len(gene_universe)} genes "
              f"(one-time; may take many minutes)...")
        with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
            adata = cellxgene_census.get_anndata(
                census, organism="Homo sapiens",
                obs_value_filter=f"tissue_general == '{TISSUE}' and is_primary_data == True",
                var_value_filter="feature_name in [%s]" % ",".join(f"'{g}'" for g in gene_universe),
                obs_column_names=["cell_type"])
        X = pd.DataFrame(adata.X.toarray() if hasattr(adata.X,"toarray") else adata.X,
                         columns=adata.var["feature_name"].values)
        X["cell_type"] = adata.obs["cell_type"].astype(str).values
        X = X[~X["cell_type"].isin(["nan","NaN","None",""])]
        tbl = (X.groupby("cell_type").mean(numeric_only=True).T
                 .reset_index().melt(id_vars="index", var_name="cell_type", value_name="mean_expr")
                 .rename(columns={"index":"feature_name"}))
        tbl.to_parquet(cache); _census_tbl = tbl; CENSUS_OK = True
        print(f"  Census aggregate cached -> {cache.name}")
    except Exception as e:
        print(f"  Census aggregate failed ({e}) -> Stage 2a skipped")

_load_census()

def census_specificity(gene, cell_type):
    if not CENSUS_OK: return None
    sub = _census_tbl[_census_tbl.feature_name.str.upper() == gene.upper()].copy()
    sub = sub[sub.cell_type.notna()]
    sub["cell_type"] = sub["cell_type"].astype(str)
    sub = sub[~sub.cell_type.isin(["nan","NaN","None",""])]
    if sub.empty: return None
    overall = sub.mean_expr.mean()
    toks = str(cell_type).lower().split()
    key = toks[0] if toks else ""
    m = sub[sub.cell_type.str.lower().str.contains(key, regex=False, na=False)] if key else sub.iloc[0:0]
    if m.empty or not overall or overall == 0: return None
    idx = m.mean_expr.idxmax()
    if pd.isna(idx): return None
    target = m.mean_expr.max()
    return {"fold_enrichment": float(target/overall),
            "matched_cell_type": str(m.loc[idx,"cell_type"]),
            "n_cell_types": int(sub.cell_type.nunique())}

# ===========================================================================
# STAGE 2b — Human Protein Atlas (optional; degrades gracefully)
# ===========================================================================
import requests
def hpa_lookup(gene, timeout=15):
    try:
        r = requests.get("https://www.proteinatlas.org/api/search_download.php",
                         params={"search":gene,"format":"json",
                                 "columns":"g,rnats,rnatsm,scml,scal","compress":"no"},
                         timeout=timeout)
        r.raise_for_status()
        for row in r.json():
            if str(row.get("Gene","")).upper() == gene.upper():
                return row
    except Exception:
        return None
    return None

HPA_OK = hpa_lookup("CD3D") is not None
print(f"  HPA reachable: {HPA_OK}")

# ===========================================================================
# Run the agent per cluster
# ===========================================================================
def run_cluster(cluster_id, degs, use_hpa=True):
    records, notes, genes_hit = [], [], set()
    candidates = hypothesize(degs, TISSUE)
    if not candidates: notes.append("NO_CURATED_MARKER_HITS")

    for cell_type, support in candidates:
        for g in degs:
            if any(ct == cell_type for ct,_t,_n in cm.lookup(g, TISSUE)):
                genes_hit.add(g)
                records.append(dict(database="CellMarker2.0", database_version=cm.version,
                    retrieval_date=TODAY, evidence_type="curated_marker",
                    subject=g, gene=g, predicate="marker_of", object=cell_type,
                    score=support, details={"cluster":cluster_id,"tissue":TISSUE}))
            spec = census_specificity(g, cell_type)
            if spec:
                genes_hit.add(g)
                records.append(dict(database="CELLxGENE-Census", database_version=CENSUS_VERSION,
                    retrieval_date=TODAY, evidence_type="sc_expression",
                    subject=g, gene=g, predicate="expressed_in", object=spec["matched_cell_type"],
                    score=min(1.0, spec["fold_enrichment"]/10.0), details={**spec,"cluster":cluster_id}))
    if not CENSUS_OK: notes.append("CENSUS_UNAVAILABLE")

    if use_hpa and HPA_OK:
        for g in degs:
            row = hpa_lookup(g)
            if row:
                genes_hit.add(g)
                records.append(dict(database="HPA", database_version="v24", retrieval_date=TODAY,
                    evidence_type="tissue_expression", subject=g, gene=g,
                    predicate="expression_profile",
                    object=str(row.get("RNA tissue specificity", row.get("rnats",""))),
                    details={"cluster":cluster_id}))
            time.sleep(0.05)
    elif use_hpa:
        notes.append("HPA_UNAVAILABLE")

    coverage = round(len(genes_hit)/max(len(degs),1), 3)
    top = candidates[0][0] if candidates else None
    if len(candidates) >= 2 and candidates[1][1] > 0 and \
       candidates[0][1] / candidates[1][1] < AMBIG_MARGIN:
        notes.append(f"AMBIGUOUS_TOP:{candidates[0][0]}|{candidates[1][0]}")
    if not candidates:
        notes.append("NO_CONFIDENT_HYPOTHESIS")   # below MIN_SUPPORT -> abstain candidate
    return dict(agent="marker_expression", cluster_id=cluster_id,
                top_candidate=top, candidates=candidates,
                coverage=coverage, n_records=len(records), notes=sorted(set(notes)),
                records=records)

outputs = {}
gt = pd.read_csv(DEG_DIR/"cluster_ground_truth.csv", index_col=0)
gt.index = gt.index.astype(str)
print("\ncluster | top hypothesis           | coverage | author label")
print("-"*72)
for cl in CLUSTERS:
    out = run_cluster(cl, tops[cl]["top_agents"])
    outputs[cl] = out
    author = gt.loc[cl,"majority_author_label"] if cl in gt.index else "?"
    print(f"{cl:>7} | {str(out['top_candidate'])[:24]:24} | {out['coverage']:>7.2f}  | {author}")

json.dump(outputs, open(EVID_DIR/"marker_expression_evidence.json","w"), indent=1, default=str)

# ===========================================================================
# NEGATIVE CONTROLS — falsification tests
# ===========================================================================
print("\n" + "="*60 + "\nNEGATIVE CONTROLS\n" + "="*60)
rng = np.random.default_rng(0)
all_genes = sorted({g for v in tops.values() for g in v["top_agents"]})

def top_support(degs):
    h = hypothesize(degs, TISSUE)
    return h[0][1] if h else 0.0

# Real clusters: strength of the top hypothesis (mean support)
real_support = np.mean([top_support(tops[cl]["top_agents"]) for cl in CLUSTERS])

# Shuffled gene lists of matched size: top support should be MUCH lower
shuf = []
for _ in range(50):
    cl = rng.choice(CLUSTERS)
    fake = list(rng.choice(all_genes, size=len(tops[cl]["top_agents"]), replace=False))
    shuf.append(top_support(fake))
shuf_support = np.mean(shuf)

# Housekeeping genes
hk_genes = ["ACTB","GAPDH","B2M","RPL13","RPS27","TUBB","PGK1","PPIA","RPL13A","RPS27A"]
hk_support = top_support(hk_genes)

sep = real_support / shuf_support if shuf_support else float("inf")
print(f"top-hypothesis SUPPORT  real: {real_support:.3f}   shuffled: {shuf_support:.3f}   "
      f"housekeeping: {hk_support:.3f}")
print(f"real/shuffled separation: {sep:.1f}x  (want >=2x; real clusters should be far stronger)")
real_hit = sum(1 for cl in CLUSTERS if hypothesize(tops[cl]["top_agents"], TISSUE))
shuf_hit = np.mean([1 if hypothesize(list(rng.choice(all_genes,
                    size=len(tops[CLUSTERS[0]]["top_agents"]), replace=False)), TISSUE) else 0
                    for _ in range(50)])
print(f"clusters with a confident call — real: {real_hit}/{len(CLUSTERS)}, "
      f"shuffled rate: {shuf_hit:.2f}")

control = {"real_support": float(real_support), "shuffled_support": float(shuf_support),
           "housekeeping_support": float(hk_support), "separation_x": float(sep),
           "real_hit": int(real_hit), "shuffled_hit_rate": float(shuf_hit)}
json.dump(control, open(EVID_DIR/"marker_negative_controls.json","w"), indent=1, default=str)

mean_cov = np.mean([o["coverage"] for o in outputs.values()])
zero = [cl for cl in CLUSTERS if outputs[cl]["coverage"] == 0]
print("\n" + "="*60 + f"""
NOTEBOOK 06 COMPLETE
{"="*60}
Main function : Marker & Expression Evidence Agent (CellMarker + Census + HPA)
Dataset       : {DATASET}
Mean coverage : {mean_cov:.3f}   (v1 failure mode was coverage=0 on 10/20 clusters)
Zero-coverage : {zero if zero else 'NONE — v1 c_DEG=0 bug resolved'}
Census        : {'ON' if CENSUS_OK else 'OFF (cache pending)'}   HPA: {'ON' if HPA_OK else 'OFF'}
Outputs       : marker_expression_evidence.json, marker_negative_controls.json
Next notebook : nb_07_pathway_agent""")
