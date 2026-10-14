# scINTERP: knowledge-grounded interpretation of single-cell clusters

scINTERP annotates cell clusters from single-cell RNA-seq with a cell type, a calibrated confidence score and a short plain-language explanation. Three deterministic evidence modules produce the call, and a language model only writes the explanation.

This repository holds the code, cached evidence and result tables for the revised manuscript submitted to *CSB Reports* (manuscript CSBJ-D-26-00878, transferred from *Computational and Structural Biotechnology Journal*).

> Status: revision in progress. The revised code, results and figures are on the branch `csbreports-revision`. The original submission files are at the top level of `main`.

## How it works

```
cluster marker genes (Scanpy DEG)
        |
        +--> Marker module   CellMarker 2.0, hypergeometric test      --> cell-type call (top-1)
        +--> Pathway module  Enrichr over-representation analysis     --> supporting evidence
        +--> Disease module  Open Targets                             --> supporting evidence
                |
                v
   Orchestrator (deterministic): weights 0.50 / 0.35 / 0.15; x1.10 when modules agree,
   x0.80 when they conflict; abstains below 0.50
                |
                v
   Narrator (LLM, Claude Haiku 4.5, claude-haiku-4-5-20251001): writes the explanation
   from the structured evidence packet only. It does not change the call or the confidence.
```

The cell-type call is the top-ranked marker-module result. Confidence and conflict flags come from the orchestrator, not from the LLM.

## Datasets

All three are public GEO series. No data are redistributed here.

| Short name | GEO series | Tissue |
| --- | --- | --- |
| LUAD | GSE131907 | lung adenocarcinoma |
| CRC | GSE132465 | colorectal cancer |
| BRCA | GSE176078 | breast cancer |

## Headline results (58 clusters, lineage level)

| Method | Accuracy |
| --- | --- |
| scINTERP | 0.931 (54/58; 95% CI 0.836 to 0.973) |
| CellTypist | 0.966 |
| Naive LLM (gene list only) | 0.966 |
| SingleR-style correlation (Census reference) | 0.879 |

Per dataset: LUAD 1.000, CRC 0.944, BRCA 0.864. Differences against the baselines are not significant (exact McNemar p 0.453 to 0.688). Confidence is moderately informative (AUROC 0.752, 95% CI 0.567 to 0.955; ECE 0.055). The 21 clusters with confidence of at least 0.95 were all correct. The manuscript and supplement report the full set of statistics.

## Repository layout (branch `csbreports-revision`)

| Path | Contents |
| --- | --- |
| `code/` | Pipeline notebooks as Python scripts (`nb_02c` to `nb_20`), `llm_client.py`, `REVISION_RUN_ORDER.md` |
| `results/reanalysis/` | Per-cluster results for all methods, bootstrap and McNemar tables, weight-grid sensitivity, Census overlap and leave-out, patient composition |
| `results/agent_evidence/` | Cached evidence records, orchestrator output, negative controls, LLM robustness runs |
| `results/DEG/` | Differential-expression outputs per dataset |
| `figures/` | Manuscript and supplementary figures (PNG and PDF) |

Large files (`.h5ad`, raw matrices, archives) are not stored here. Rebuild them with the scripts below.

## Running the pipeline

The code was run in Microsoft Fabric notebooks with Python and Scanpy 1.9.6. The scripts read and write under a Lakehouse folder (`Files/scRNA/`). Edit the `ROOT` path at the top of a script to run it elsewhere.

Main steps, in order (full instructions in `code/REVISION_RUN_ORDER.md`):

1. Build, QC and cluster: `nb_02*`, `nb_03_QC`, `nb_04_scanpy_clustering`
2. Differential expression: `nb_05_DEG_analysis`
3. Evidence modules: `nb_06_DEG_evidence_agent` (marker), `nb_07_pathway_agent`, `nb_08_disease_agent`
4. Orchestrator and benchmark: `nb_09_orchestrator`, `nb_10a_validation_benchmark`, `nb_10b_llm_baseline_narrator`
5. Reanalysis and robustness for the revision: `nb_12_reanalysis`, `nb_13_llm_robustness`, `nb_14_ora_robustness`, `nb_15_census_and_patients`, `nb_16_census_leaveout`
6. Expert scoring: `nb_17_expert_sheets`, `nb_17b_expert_scoring`
7. Figures: `nb_11_figures_report`, `nb_18_figures_v2`

Steps 1 to 4 and the benchmark tables need no API key. The narrator, the naive-LLM baseline and the robustness runs need an Anthropic API key.

## API keys and secrets

No key is stored in this repository. The notebooks read the Anthropic key from a hidden prompt at run time. Do not commit keys; `secrets.template.yaml` shows the expected shape only. The LLM calls use the Anthropic Messages API (version 2023-06-01) with prompts and outputs saved for every run.

## Knowledge sources

| Source | Used for |
| --- | --- |
| CellMarker 2.0 | marker module |
| Enrichr libraries (Reactome, GO, MSigDB Hallmark) | pathway module |
| Open Targets Platform | disease module |
| CELLxGENE Census | reference for the correlation baseline and the leakage checks |

The GWAS Catalog is not used in any reported result. A call to its retired REST endpoint remains in `code/nb_08_disease_agent.py` (the endpoint now returns HTTP 410) and contributes no evidence.

## Reproducibility notes

- The deterministic modules give identical output on identical inputs and cached evidence.
- LLM output varies between runs. The paper reports means and standard deviations over repeated runs, including temperature-0 runs.
- Database versions and retrieval dates are stored in the evidence records. The Open Targets release is not stored in the cached answers.

## Citation

Author list and full reference: to be added (manuscript under revision at *CSB Reports*).
Archived release: DOI to be added after the Zenodo deposit.

## Licence and contact

Licence: to be confirmed by the authors.
Contact: Jinlian Wang, UT Austin (corresponding author: Hongfang Liu).
