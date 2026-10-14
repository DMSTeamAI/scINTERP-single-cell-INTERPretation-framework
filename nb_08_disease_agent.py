"""nb_08_disease_agent — Disease Agent (Open Targets GraphQL + GWAS Catalog).

Keeps two epistemically distinct things separate for the narrator:
  * "differentially expressed in OUR data"      (from nb_05 — the user's statistics)
  * "has PRIOR disease-association evidence"     (these records)
Open Targets overall association (DISEASE_ASSOCIATION) and its genetic_association
datatype (GENETIC_ASSOCIATION) are emitted as SEPARATE record types, plus GWAS Catalog
as an independent human-genetics layer. Driver-gene hits for the tumor context are flagged.

Negative control: housekeeping genes should surface little/no disease-specific association.

Reads:  DEG/<DATASET>/degs_top.json
Writes: agent_evidence/<DATASET>/disease_evidence.json, disease_negative_controls.json

Runner:
    DATASET = "GSE131907"
    exec(open("/lakehouse/default/Files/scRNA/code/nb_08_disease_agent.py").read())
"""
assert "DATASET" in globals(), "Set DATASET before exec()"

import os, json, time, logging
from datetime import date
from pathlib import Path
import numpy as np, pandas as pd, requests

log = logging.getLogger("nb08")
ROOT = Path("/lakehouse/default/Files/scRNA")
DEG_DIR  = ROOT/"DEG"/DATASET
EVID_DIR = ROOT/"agent_evidence"/DATASET
EVID_DIR.mkdir(parents=True, exist_ok=True)
TODAY = date.today().isoformat()

CONTEXT = {
 "GSE131907": dict(keywords=["lung adenocarcinoma","lung carcinoma","non-small cell lung","lung neoplasm"],
                   drivers={"EGFR","KRAS","TP53","STK11","BRAF","ALK","ROS1","MET","KEAP1","NKX2-1"}),
 "GSE132465": dict(keywords=["colorectal","colon carcinoma","colon adenocarcinoma","rectal"],
                   drivers={"APC","KRAS","TP53","SMAD4","BRAF","PIK3CA","FBXW7","CTNNB1","TCF7L2","MYC"}),
 # driver list and keywords fixed BEFORE the GSE176078 run
 "GSE176078": dict(keywords=["breast cancer","breast carcinoma","breast neoplasm","mammary"],
                   drivers={"TP53","PIK3CA","ERBB2","ESR1","GATA3","CDH1","PTEN","BRCA1","BRCA2","MAP3K1","AKT1","KMT2C"}),
}[DATASET]

OT_GRAPHQL = "https://api.platform.opentargets.org/api/v4/graphql"
MYGENE = "https://mygene.info/v3/query"
GWAS = "https://www.ebi.ac.uk/gwas/rest/api/associations/search/findByGene"

OT_QUERY = """
query td($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    approvedSymbol
    associatedDiseases(page:{index:0,size:8}) {
      rows { disease{ id name } score datatypeScores{ id score } }
    }
  }
}"""

_ens_cache = {}
def _get_json(url, params=None, json_body=None, timeout=30, retries=3):
    for attempt in range(retries):
        try:
            if json_body is not None:
                r = requests.post(url, json=json_body, timeout=timeout)
            else:
                r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 404:
                return None                      # 404 = gene has no records; not an error, don't retry
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if "404" in str(e):
                return None                      # belt-and-suspenders: never retry a 404
            if attempt < retries-1:
                time.sleep(2*(attempt+1)); continue
            log.warning("%s failed: %s", url.split("/")[2], str(e)[:80])
            return None

def symbol_to_ensembl(gene):
    if gene in _ens_cache: return _ens_cache[gene]
    j = _get_json(MYGENE, params={"q":f"symbol:{gene}","species":"human","fields":"ensembl.gene"})
    ens = None
    if j:
        hits = j.get("hits", [])
        if hits:
            e = hits[0].get("ensembl")
            if isinstance(e, list): e = e[0] if e else None
            ens = (e or {}).get("gene")
    _ens_cache[gene] = ens; return ens

def open_targets(gene, keywords):
    ens = symbol_to_ensembl(gene)
    if not ens: return []
    j = _get_json(OT_GRAPHQL, json_body={"query":OT_QUERY,"variables":{"ensemblId":ens}})
    if not j: return []
    target = (j.get("data") or {}).get("target") or {}
    recs, kw = [], [k.lower() for k in keywords]
    for row in (target.get("associatedDiseases") or {}).get("rows", []):
        dname = row["disease"]["name"]; rel = any(k in dname.lower() for k in kw)
        recs.append(dict(database="OpenTargets", database_version="v4",
            retrieval_date=TODAY, evidence_type="disease_association",
            subject=gene, gene=gene, predicate="associated_with", object=dname,
            score=float(row["score"]), source_id=row["disease"]["id"],
            details={"ensembl":ens,"context_relevant":rel}))
        for dt in row.get("datatypeScores", []):
            if dt["id"]=="genetic_association" and dt["score"]>0:
                recs.append(dict(database="OpenTargets", database_version="v4",
                    retrieval_date=TODAY, evidence_type="genetic_association",
                    subject=gene, gene=gene, predicate="genetically_associated_with",
                    object=dname, score=float(dt["score"]), source_id=row["disease"]["id"],
                    details={"ensembl":ens,"context_relevant":rel}))
    return recs

def gwas_catalog(gene, max_n=5):
    j = _get_json(GWAS, params={"geneName":gene,"size":max_n})
    if not j: return []
    assocs = (j.get("_embedded") or {}).get("associations", [])
    recs = []
    for a in assocs[:max_n]:
        traits = ", ".join(t.get("trait","") for t in a.get("efoTraits", [])) or "unspecified"
        recs.append(dict(database="GWAS-Catalog", database_version="REST v1",
            retrieval_date=TODAY, evidence_type="genetic_association",
            subject=gene, gene=gene, predicate="gwas_associated_with", object=traits,
            pvalue=a.get("pvalue"), details={}))
    return recs

def run_cluster(cl, degs):
    recs, notes, hit = [], [], set()
    for g in degs:
        r = open_targets(g, CONTEXT["keywords"]) + gwas_catalog(g)
        if r: hit.add(g); recs.extend(r)
        if g.upper() in CONTEXT["drivers"]: notes.append(f"DRIVER:{g.upper()}")
        time.sleep(0.1)
    ctx_hits = sum(1 for r in recs if r.get("details",{}).get("context_relevant"))
    coverage = round(len(hit)/max(len(degs),1),3)
    return dict(agent="disease", cluster_id=cl, coverage=coverage,
                n_records=len(recs), n_context_relevant=ctx_hits,
                drivers=sorted({n.split(":")[1] for n in notes if n.startswith("DRIVER")}),
                notes=sorted(set(notes)), records=recs)

tops = json.load(open(DEG_DIR/"degs_top.json"))
CLUSTERS = list(tops.keys())
print(f"{DATASET}: {len(CLUSTERS)} clusters, context={CONTEXT['keywords'][0]}")
print("\ncluster | cov | #rec | ctx | drivers")
print("-"*60)
outputs = {}
for cl in CLUSTERS:
    out = run_cluster(cl, tops[cl]["top_agents"][:10])   # top-10 to bound API calls
    outputs[cl] = out
    print(f"{cl:>7} | {out['coverage']:.2f} | {out['n_records']:>4} | {out['n_context_relevant']:>3} | "
          f"{','.join(out['drivers']) if out['drivers'] else '-'}")

json.dump(outputs, open(EVID_DIR/"disease_evidence.json","w"), indent=1, default=str)

# ---- negative control: housekeeping should have low context-relevant disease evidence ----
print("\n" + "="*60 + "\nNEGATIVE CONTROL\n" + "="*60)
hk = ["ACTB","GAPDH","B2M","RPL13","PPIA","TUBB"]
hk_out = run_cluster("housekeeping", hk)
real_ctx = np.mean([outputs[cl]["n_context_relevant"] for cl in CLUSTERS])
print(f"context-relevant disease records — real clusters: {real_ctx:.1f}   "
      f"housekeeping: {hk_out['n_context_relevant']}")
json.dump({"real_mean_context":float(real_ctx),
           "housekeeping_context":hk_out["n_context_relevant"]},
          open(EVID_DIR/"disease_negative_controls.json","w"), indent=1)

driver_clusters = [cl for cl in CLUSTERS if outputs[cl]["drivers"]]
print("\n" + "="*60 + f"""
NOTEBOOK 08 COMPLETE
{"="*60}
Main function : Disease Agent (Open Targets + GWAS Catalog)
Dataset       : {DATASET}
Clusters with driver-gene evidence : {len(driver_clusters)} {driver_clusters}
Mean context-relevant records/cluster : {real_ctx:.1f}
Outputs       : disease_evidence.json, disease_negative_controls.json
Next notebook : nb_09_orchestrator""")
