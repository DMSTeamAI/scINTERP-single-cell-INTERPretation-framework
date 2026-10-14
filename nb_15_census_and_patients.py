"""nb_15_census_and_patients - CSB Reports revision (R3: leakage, batch/patient effects)

Part A  Census overlap check (leakage): which CELLxGENE Census datasets/collections look like the three evaluation datasets?
        nb_06 uses a Census tissue aggregate as ONE evidence source; if the Census contains the evaluation data, say so.
        Matches by publication DOI (verify the DOIs against the papers) and by tissue/cancer keywords in the collection name.
Part B  Per-cluster patient/sample composition: cells, number of patients, dominant-patient fraction, effective number of
        patients (inverse Simpson), joined to scINTERP correctness. No batch correction was applied in the pipeline, so this
        quantifies how pooled each cluster is.

Runner (Part B needs no internet; Part A needs the cellxgene-census package and internet):
    DATASETS = ["GSE131907", "GSE132465", "GSE176078"]
    RUN_CENSUS = True            # set False to skip Part A
    exec(open("/lakehouse/default/Files/scRNA/code/nb_15_census_and_patients.py").read())
Optional: PATIENT_COL = {"GSE131907": "Sample", ...} to override the automatically detected patient column.
Writes: ROOT/reanalysis/census_overlap.csv, ROOT/reanalysis/patient_composition_<DS>.csv, patient_composition_summary.csv
"""
import json, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

DATASETS = globals().get("DATASETS", ["GSE131907", "GSE132465", "GSE176078"])
RUN_CENSUS = globals().get("RUN_CENSUS", True)
PATIENT_COL = dict(globals().get("PATIENT_COL", {}))
CENSUS_VERSION = globals().get("CENSUS_VERSION", "2025-01-30")
ROOT = Path("/lakehouse/default/Files/scRNA"); OUT = ROOT/"reanalysis"; OUT.mkdir(exist_ok=True)

# ====================================================================== Part A
if RUN_CENSUS:
    print("="*72 + f"\nA. CENSUS OVERLAP CHECK (Census {CENSUS_VERSION})\n" + "="*72)
    # DOIs of the source publications - VERIFY against the papers before citing
    DOIS = {"GSE131907 (Kim et al. 2020, LUAD)": "10.1038/s41467-020-16164-1",
            "GSE132465 (Lee et al. 2020, CRC)": "10.1038/s41588-020-0636-z",
            "GSE176078 (Wu et al. 2021, breast)": "10.1038/s41588-021-00911-1"}
    KEYWORDS = "breast|lung adenocarcinoma|luad|colorectal|colon|crc|tumou?r|cancer|carcinoma"
    try:
        try:
            import cellxgene_census
        except Exception:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cellxgene-census"])
            import cellxgene_census
        with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
            ds = census["census_info"]["datasets"].read().concat().to_pandas()
        print("datasets table columns:", list(ds.columns))
        col = lambda *c: next((x for x in c if x in ds.columns), None)
        c_doi, c_name, c_title, c_n, c_id = col("collection_doi"), col("collection_name"), col("dataset_title"), col("dataset_total_cell_count"), col("dataset_id")
        show = [x for x in [c_name, c_doi, c_title, c_n, c_id] if x]
        rows = []
        for label, doi in DOIS.items():
            m = ds[ds[c_doi].astype(str).str.contains(re.escape(doi), case=False, na=False)] if c_doi else ds.iloc[0:0]
            print(f"\n{label}  DOI {doi}:  {'MATCH in Census: ' + str(len(m)) + ' dataset(s)' if len(m) else 'no DOI match'}")
            if len(m): print(m[show].to_string(index=False))
            rows.append(dict(evaluation_dataset=label, doi=doi, n_census_datasets_by_doi=len(m),
                             census_dataset_ids=";".join(m[c_id].astype(str)) if len(m) and c_id else ""))
        # keyword view (tumour-related collections) so a human can spot unmatched copies of the same data
        kw = ds[ds[c_name].astype(str).str.contains(KEYWORDS, case=False, na=False) | ds[c_title].astype(str).str.contains(KEYWORDS, case=False, na=False)] if c_name and c_title else ds.iloc[0:0]
        print(f"\nTumour/tissue-keyword collections in Census: {len(kw)} datasets (full list in census_overlap_keyword_hits.csv)")
        print(kw[show].drop_duplicates(subset=[c_name]).head(40).to_string(index=False))
        kw[show].to_csv(OUT/"census_overlap_keyword_hits.csv", index=False)
        pd.DataFrame(rows).to_csv(OUT/"census_overlap.csv", index=False)
        print("\nINTERPRETATION: a DOI match means the evaluation dataset (or part of it) is in the Census. The Census tissue aggregate then")
        print("contains expression of the cells being annotated. State this in the Methods/Limitations; if it matters, rebuild the cache")
        print("excluding the matching dataset_id values (nb_census_cache) and re-run nb_06 -> nb_09 -> nb_10a.")
    except Exception as e:
        print("Census check failed:", str(e)[:300])
        print("Run it where cellxgene-census can reach the Census (the same session that built the cache), or set RUN_CENSUS = False.")

# ====================================================================== Part B
print("\n" + "="*72 + "\nB. PER-CLUSTER PATIENT / SAMPLE COMPOSITION\n" + "="*72)
import anndata as _anndata

# same lineage map as nb_12, extracted so scoring is identical
_src = open(ROOT/"code"/"nb_12_reanalysis.py").read()
_ns = {}; exec(_src[_src.index("_FAM = ["):_src.index("COARSE2")], _ns); family = _ns["family"]

def pick_col(obs_cols):
    for pat in ["patient", "donor", "subject", "orig.ident", "sample"]:
        for c in obs_cols:
            if pat in c.lower() and not c.lower().startswith("sample_origin"): return c
    return None

summ = []
for DATASET in DATASETS:
    print(f"\n--- {DATASET} ---")
    ad = _anndata.read_h5ad(ROOT/"h5ad"/f"{DATASET}_clustered.h5ad")
    print("obs columns:", list(ad.obs.columns))
    pc = PATIENT_COL.get(DATASET) or pick_col(list(ad.obs.columns))
    if pc is None or pc not in ad.obs.columns:
        print("  no patient/sample column found - set PATIENT_COL = {'%s': '<column>'} and re-run" % DATASET); continue
    obs = ad.obs[["leiden", pc]].copy(); obs["leiden"] = obs["leiden"].astype(str); obs[pc] = obs[pc].astype(str)
    print(f"  patient column used: '{pc}' ({obs[pc].nunique()} levels, {len(obs)} cells)")
    gt = pd.read_csv(ROOT/"DEG"/DATASET/"cluster_ground_truth.csv", index_col=0); gt.index = gt.index.astype(str)
    bench = pd.DataFrame(json.load(open(ROOT/"agent_evidence"/DATASET/"benchmark_10a.json"))["per_cluster"]); bench["cluster"] = bench["cluster"].astype(str)
    bm = bench.set_index("cluster")
    rows = []
    for cl, g in obs.groupby("leiden"):
        p = g[pc].value_counts(normalize=True)
        rows.append(dict(cluster=cl, n_cells=len(g), n_patients=int((g[pc].value_counts() > 0).sum()),
                         top_patient=p.index[0], top_patient_frac=round(float(p.iloc[0]), 3),
                         eff_n_patients=round(float(1/np.sum(p.values**2)), 2),
                         author=gt.loc[cl, "majority_author_label"] if cl in gt.index else None,
                         purity=round(float(gt.loc[cl, "purity"]), 3) if cl in gt.index and "purity" in gt.columns else np.nan,
                         scinterp_call=bm.loc[cl, "call"] if cl in bm.index else None,
                         correct=bool(bm.loc[cl, "correct"]) if cl in bm.index else None,
                         conf=round(float(bm.loc[cl, "conf"]), 3) if cl in bm.index else np.nan))
    R = pd.DataFrame(rows).sort_values("cluster", key=lambda s: s.astype(int))
    R.to_csv(OUT/f"patient_composition_{DATASET}.csv", index=False)
    print(R.to_string(index=False))
    dom = R[R.top_patient_frac > 0.5]
    err = R[R.correct == False]
    print(f"\n  clusters dominated by one patient (>50% of cells): {len(dom)}/{len(R)}  -> {list(dom.cluster)}")
    print(f"  median top-patient fraction {R.top_patient_frac.median():.2f}; median effective number of patients {R.eff_n_patients.median():.1f}")
    print(f"  scINTERP error clusters {list(err.cluster)}: top-patient fraction {list(err.top_patient_frac)}, effective patients {list(err.eff_n_patients)}")
    summ.append(dict(dataset=DATASET, patient_col=pc, n_levels=obs[pc].nunique(), n_clusters=len(R), n_dominated_gt50=len(dom),
                     median_top_frac=R.top_patient_frac.median(), median_eff_patients=R.eff_n_patients.median(),
                     error_clusters=";".join(err.cluster), error_top_fracs=";".join(map(str, err.top_patient_frac))))
if summ:
    S = pd.DataFrame(summ); S.to_csv(OUT/"patient_composition_summary.csv", index=False)
    print("\n" + "="*72 + "\nSUMMARY\n" + "="*72); print(S.round(2).to_string(index=False))
    print("\nReport in the Supplement: the per-cluster table, and in the Limitations that no batch/patient correction was applied.")
print("\nNOTEBOOK 15 COMPLETE")
