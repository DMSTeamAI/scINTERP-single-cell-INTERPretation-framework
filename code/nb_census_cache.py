"""nb_census_cache — one-time CELLxGENE Census aggregate builder.

Run ONCE per tissue in a session that has cellxgene-census available
(either via Full-mode env, or by putting `%pip install -q cellxgene-census`
in a cell ABOVE the exec and letting the kernel restart first).

It writes  Files/scRNA/cache/census_<VERSION>_<tissue>.parquet
which nb_06 then just READS — so nb_06 needs no census dependency at all.

Runner:
    DATASET = "GSE131907"     # or "GSE132465"
    exec(open("/lakehouse/default/Files/scRNA/code/nb_census_cache.py").read())
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

assert "DATASET" in globals(), "Set DATASET before exec()"
ROOT = Path("/lakehouse/default/Files/scRNA")
DEG_DIR = ROOT/"DEG"/DATASET
CACHE = ROOT/"cache"; CACHE.mkdir(parents=True, exist_ok=True)
TISSUE = {"GSE131907": "lung", "GSE132465": "colon"}[DATASET]
CENSUS_VERSION = "2025-01-30"
out = CACHE/f"census_{CENSUS_VERSION}_{TISSUE}.parquet"

if out.exists():
    print(f"cache already exists: {out.name} — delete it to rebuild"); 
else:
    import cellxgene_census
    tops = json.load(open(DEG_DIR/"degs_top.json"))
    genes = sorted({g for v in tops.values() for g in v["top_agents"]})
    print(f"Building Census aggregate: tissue={TISSUE}, {len(genes)} genes, version={CENSUS_VERSION}")
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        adata = cellxgene_census.get_anndata(
            census, organism="Homo sapiens",
            obs_value_filter=f"tissue_general == '{TISSUE}' and is_primary_data == True",
            var_value_filter="feature_name in [%s]" % ",".join(f"'{g}'" for g in genes),
            obs_column_names=["cell_type"])
    X = pd.DataFrame(adata.X.toarray() if hasattr(adata.X,"toarray") else adata.X,
                     columns=adata.var["feature_name"].values)
    X["cell_type"] = adata.obs["cell_type"].astype(str).values
    X = X[~X["cell_type"].isin(["nan","NaN","None",""])]
    tbl = (X.groupby("cell_type", observed=True).mean(numeric_only=True).T
             .reset_index().melt(id_vars="index", var_name="cell_type", value_name="mean_expr")
             .rename(columns={"index":"feature_name"}))
    tbl.to_parquet(out)
    print(f"cached -> {out}  ({tbl.shape[0]:,} rows, {tbl.cell_type.nunique()} cell types)")

print("DONE — nb_06 will now load this cache and report Census: ON")
