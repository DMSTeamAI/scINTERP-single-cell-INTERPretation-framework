"""nb_16_census_leaveout - CSB Reports revision (R3: leakage). Re-run the Census-dependent steps with the evaluation
dataset's own Census copy REMOVED, and compare with the main results.

Why: nb_15 found that the Wu et al. 2021 breast atlas (GSE176078) is in the CELLxGENE Census (DOI match, 13 dataset records
incl. the full 100,064-cell object). nb_06 (marker/expression agent) and nb_10c (SingleR-style baseline) use a Census tissue
aggregate (mean expression per cell type), so for breast the reference contained the cells being annotated.

What it does (about 30 min for breast):
  1. backs up agent_evidence/<DS>/ and the Census cache (nothing is lost; restored in a finally-block),
  2. rebuilds the Census aggregate WITHOUT every Census dataset whose collection DOI matches the evaluation paper,
  3. re-runs nb_06 -> nb_09 -> nb_10a -> nb_10c on that cache (nb_07/nb_08 do not use the Census and are unchanged),
  4. saves the leave-out results to agent_evidence/<DS>_noCensusOverlap/ and RESTORES the main results,
  5. prints main vs leave-out: accuracy, AUROC, mean confidence, per-cluster call changes, SingleR-style accuracy.

Runner:
    DATASET = "GSE176078"
    exec(open("/lakehouse/default/Files/scRNA/code/nb_16_census_leaveout.py").read())
If the kernel dies mid-run, FIRST restore, then re-run:
    DATASET = "GSE176078"; RESTORE_ONLY = True
    exec(open("/lakehouse/default/Files/scRNA/code/nb_16_census_leaveout.py").read())
Needs: cellxgene-census, internet, nb_06/09/10a/10c in the code folder. No Claude key needed.
"""
import json, shutil, io, contextlib, time, filecmp
from pathlib import Path
import numpy as np, pandas as pd

DATASET = globals().get("DATASET", "GSE176078")
RESTORE_ONLY = globals().get("RESTORE_ONLY", False)
DOIS = {"GSE176078": "10.1038/s41588-021-00911-1"}          # verify against the paper; extend if nb_15 finds other matches
DOI = globals().get("EXCLUDE_DOI", DOIS.get(DATASET))
assert DOI, f"no DOI to exclude for {DATASET}; set EXCLUDE_DOI = '...' before exec()"
TISSUE = {"GSE131907": "lung", "GSE132465": "colon", "GSE176078": "breast"}[DATASET]
CENSUS_VERSION = "2025-01-30"
ROOT = Path("/lakehouse/default/Files/scRNA"); CODE = ROOT/"code"; CACHE = ROOT/"cache"
EV = ROOT/"agent_evidence"; EVID = EV/DATASET; BK = EV/f"{DATASET}__main_backup"; LO = EV/f"{DATASET}_noCensusOverlap"
cache = CACHE/f"census_{CENSUS_VERSION}_{TISSUE}.parquet"; cache_bk = CACHE/f"census_{CENSUS_VERSION}_{TISSUE}.MAIN_BACKUP.parquet"

def restore():
    if BK.exists():
        if EVID.exists(): shutil.rmtree(EVID)
        shutil.copytree(BK, EVID); shutil.rmtree(BK)
        print(f"restored main results -> {EVID}")
    if cache_bk.exists():
        if cache.exists(): cache.unlink()
        cache_bk.rename(cache); print(f"restored main Census cache -> {cache.name}")

if RESTORE_ONLY:
    restore(); print("restore finished; re-run without RESTORE_ONLY if you still need the leave-out analysis")
else:
    assert not BK.exists() and not cache_bk.exists(), \
        "a backup from an interrupted run exists: run once with RESTORE_ONLY = True first (see the docstring)"
    assert cache.exists(), f"main Census cache missing: {cache}"
    assert (EVID/"orchestrated.json").exists(), "main results missing - run nb_06..nb_10a first"
    t0 = time.time()
    shutil.copytree(EVID, BK); shutil.copy(cache, cache_bk)
    print(f"backed up main results and cache ({time.time()-t0:.0f}s)")
    try:
        # ---------------- 2. leave-out Census aggregate ----------------
        import cellxgene_census
        tops = json.load(open(ROOT/"DEG"/DATASET/"degs_top.json"))
        genes = sorted({g for v in tops.values() for g in v["top_agents"]})
        with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
            dsets = census["census_info"]["datasets"].read().concat().to_pandas()
            drop_ids = set(dsets[dsets["collection_doi"].astype(str).str.contains(DOI, regex=False, na=False)]["dataset_id"])
            print(f"Census datasets with DOI {DOI}: {len(drop_ids)} (will be excluded)")
            adata = cellxgene_census.get_anndata(
                census, organism="Homo sapiens",
                obs_value_filter=f"tissue_general == '{TISSUE}' and is_primary_data == True",
                var_value_filter="feature_name in [%s]" % ",".join(f"'{g}'" for g in genes),
                obs_column_names=["cell_type", "dataset_id"])
        n_all = adata.n_obs
        keep = ~adata.obs["dataset_id"].isin(drop_ids).values
        adata = adata[keep]
        print(f"Census {TISSUE} cells: {n_all:,} -> {adata.n_obs:,} after excluding the evaluation paper ({n_all-adata.n_obs:,} removed)")
        X = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, columns=adata.var["feature_name"].values)
        X["cell_type"] = adata.obs["cell_type"].astype(str).values
        X = X[~X["cell_type"].isin(["nan", "NaN", "None", ""])]
        tbl = (X.groupby("cell_type", observed=True).mean(numeric_only=True).T.reset_index()
                 .melt(id_vars="index", var_name="cell_type", value_name="mean_expr").rename(columns={"index": "feature_name"}))
        tbl.to_parquet(cache)            # standard name, so nb_06 / nb_10c read it; the main cache is safe in cache_bk
        print(f"leave-out cache: {tbl.cell_type.nunique()} cell types (main had "
              f"{pd.read_parquet(cache_bk).cell_type.nunique()})  [{time.time()-t0:.0f}s]")

        # ---------------- 3. re-run Census-dependent steps ----------------
        for nb in ["nb_06_DEG_evidence_agent.py", "nb_09_orchestrator.py", "nb_10a_validation_benchmark.py", "nb_10c_singler_baseline.py"]:
            t1 = time.time(); buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    exec(open(CODE/nb).read(), {"DATASET": DATASET, "__name__": "__main__"})
            except Exception as e:
                print(f"  {nb} FAILED: {type(e).__name__}: {str(e)[:200]}"); print("\n".join(buf.getvalue().splitlines()[-15:])); raise
            print(f"  {nb} done ({time.time()-t1:.0f}s)")
        # ---------------- 4. keep leave-out results ----------------
        if LO.exists(): shutil.rmtree(LO)
        shutil.copytree(EVID, LO); print(f"leave-out results saved -> {LO}")
    finally:
        restore()

    # ---------------- 5. compare ----------------
    def J(d, f):
        p = d/f
        return json.load(open(p)) if p.exists() else None
    main_b, lo_b = J(EVID, "benchmark_10a.json"), J(LO, "benchmark_10a.json")
    main_o, lo_o = J(EVID, "orchestrated.json"), J(LO, "orchestrated.json")
    main_s, lo_s = J(EVID, "singler_baseline.json"), J(LO, "singler_baseline.json")
    print("\n" + "="*72 + f"\nMAIN vs LEAVE-OUT (Census without {DOI})  -  {DATASET}\n" + "="*72)
    rows = []
    for k in ["overall_accuracy", "auroc", "hit_at_1", "hit_at_3"]:
        rows.append((k, main_b[k], lo_b[k]))
    rows.append(("mean_confidence", np.mean([v["confidence"] for v in main_o.values()]), np.mean([v["confidence"] for v in lo_o.values()])))
    if main_s and lo_s: rows.append(("SingleR-style accuracy", main_s["accuracy"], lo_s["accuracy"]))
    print(pd.DataFrame(rows, columns=["metric", "main", "leave_out"]).round(3).to_string(index=False))
    ch = []
    for cl in main_o:
        a, b = main_o[cl]["marker_call"], lo_o[cl]["marker_call"]
        ch.append(dict(cluster=cl, main_call=a, leaveout_call=b, changed=a != b,
                       main_conf=round(main_o[cl]["confidence"], 3), leaveout_conf=round(lo_o[cl]["confidence"], 3)))
    C = pd.DataFrame(ch); C.to_csv(ROOT/"reanalysis"/f"census_leaveout_{DATASET}.csv", index=False)
    print(f"\nclusters whose marker call changed: {int(C.changed.sum())}/{len(C)}")
    print(C[C.changed].to_string(index=False) if C.changed.any() else "  (none)")
    mc = np.array([main_o[c]["confidence"] for c in main_o]); lc = np.array([lo_o[c]["confidence"] for c in main_o])
    print(f"\nconfidence: Spearman main vs leave-out = {pd.Series(mc).corr(pd.Series(lc), method='spearman'):.3f}; "
          f"max |delta| = {np.abs(mc-lc).max():.3f}")
    print("\nMain results are untouched (restored). Report both in the paper: main result + 'Census copy of the evaluation paper removed' result.")
    # sanity: restored files equal the leave-out-free originals
    print("restored orchestrated.json present:", (EVID/"orchestrated.json").exists())
print("\nNOTEBOOK 16 COMPLETE")
