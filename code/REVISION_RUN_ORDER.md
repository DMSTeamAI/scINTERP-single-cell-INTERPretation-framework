# scINTERP -> CSB Reports revision: what to run on Fabric, in order

Upload `scINTERP_revision_code.zip` to Lakehouse Files/scRNA/code/ and unzip (cell below). It overwrites only the
files listed; nothing in your results folders is touched.

```python
import zipfile; zipfile.ZipFile("/lakehouse/default/Files/scRNA/code/scINTERP_revision_code.zip").extractall("/lakehouse/default/Files/scRNA/code/")
```

Changed existing files: nb_03/04/05 (breast config), nb_06 (breast tissue + denylist), nb_08 (breast keywords/drivers),
nb_09 (PATHWAY_FILE/OUT_FILE overrides; logic unchanged), nb_10a/10b (extra lineage-map entries appended last; breast tissue),
nb_11 (breast label), nb_census_cache (breast).
NOTE: re-running nb_10a/10b is NOT needed for lung/colon: the appended map entries (t-cell, b-cell, caf, pvl, perivascular)
match no existing lung/colon label. Verify once by re-running nb_12 and checking accuracy is still 0.972 pooled.

## Step 1 - no API key, no new pipeline run (about 5 min)
```python
DATASETS = ["GSE131907", "GSE132465"]
exec(open("/lakehouse/default/Files/scRNA/code/nb_12_reanalysis.py").read())
```
Outputs in Files/scRNA/reanalysis/. Then a domain expert fills `curated_equivalent_Y_N` in finegrained_template.csv,
saves it as finegrained_curated.csv in the same folder, and nb_12 is re-run for fine-grained accuracy.

## Step 2 - LLM robustness (Claude; ~30-40 min for both datasets, low cost)
Cell 1 (key, hidden; the output shows nothing, that is normal):
```python
import getpass; ANTHROPIC_API_KEY = getpass.getpass("Claude key: ")
```
Cell 2 (quick test first, about 1 minute; you should see "nb_13 starting...", "client ready", then A/B/C lines):
```python
DATASETS = ["GSE131907"]; N_RUNS_A = 1; N_RUNS_T0 = 0; N_RUNS_B = 1; N_RUNS_C = 1
exec(open("/lakehouse/default/Files/scRNA/code/nb_13_llm_robustness.py").read())
```
Cell 3 (full run): same exec line with no overrides (clear DATASETS/N_RUNS_* first: `for v in ("DATASETS","N_RUNS_A","N_RUNS_T0","N_RUNS_B","N_RUNS_C"): globals().pop(v, None)`).
Never paste the key into a saved cell or commit it. Use exec(...), not runpy.

## Step 3 - ORA universe / cutoff robustness (needs internet for Enrichr libraries; 15-30 min per dataset)
```python
DATASETS = ["GSE131907", "GSE132465"]; RUN_GSEA_CFG = False     # True adds prerank GSEA (slow)
exec(open("/lakehouse/default/Files/scRNA/code/nb_14_ora_robustness.py").read())
```

## Step 4 - third dataset GSE176078 (breast). Freeze the config BEFORE looking at any result.
1. Download GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz (GEO supplementary) into Files/scRNA/raw/GSE176078/ (upload via browser if Fabric cannot reach GEO).
2. For DATASET = "GSE176078" run in order: nb_02c_build_AnnData_GSE176078 -> nb_03 -> nb_04 -> nb_05 -> nb_census_cache -> nb_06 -> nb_07 -> nb_08 -> nb_09 -> nb_10a -> nb_10b (-> nb_10c SingleR, nb_10d if wanted).
   - nb_02c prints matrix orientation and metadata columns: check them; if the author column is not `celltype_major`, set AUTHOR_COL.
   - Report the number of Leiden clusters you actually get; do NOT retune the resolution.
3. Then re-run Steps 1-3 with DATASETS = ["GSE131907","GSE132465","GSE176078"].
Pre-registered choices (documented here before running): breast denylist in nb_06, driver genes and keywords in nb_08,
author-label level = celltype_major, identical QC/clustering/DE parameters.
