# scINTERP v2 — Knowledge-Grounded Multi-Agent scRNA-seq Interpretation

Reproducible pipeline for the CSBJ special-issue revision of *"Knowledge Grounded
Multi-Agent Orchestration for Interpretable Functional Summarization of Single-Cell
Transcriptomics"* (ICIBM 2026, Paper 12).

This branch contains the redesigned v2 pipeline: three parallel evidence agents
(marker / pathway / disease) that run statistics themselves, an orchestrator that
integrates their structured evidence with confidence and cross-agent verification,
and an LLM used **only** as a grounded narrator (never to generate evidence).

## Pipeline (run in order)

Each notebook is a plain `.py` runner. On Microsoft Fabric, set `DATASET` and
`exec()` the file; locally, run as a script after setting `DATASET`.

| Stage | File | Purpose |
|-------|------|---------|
| Build | `nb_02_build_AnnData.py` / `nb_02_build_AnnData_GSE132465.py` | GEO counts -> AnnData, tumor subset |
| QC | `nb_03_QC.py` | filtering, 10k subsample (seed=0) |
| Cluster | `nb_04_scanpy_clustering.py` | HVG -> PCA -> Leiden (res 0.5) -> UMAP; ground-truth map |
| DEG | `nb_05_DEG_analysis.py` | Wilcoxon rank_genes_groups; top-N + full ranking |
| Cache | `nb_census_cache.py` | one-time CELLxGENE Census aggregate (pinned 2025-01-30) |
| Agent 1 | `nb_06_DEG_evidence_agent.py` | Marker & Expression: CellMarker 2.0 + Census + HPA, hypergeometric enrichment |
| Agent 2 | `nb_07_pathway_agent.py` | Pathway: ORA + prerank GSEA over Reactome / GO BP / MSigDB Hallmark |
| Agent 3 | `nb_08_disease_agent.py` | Disease: Open Targets + GWAS Catalog |
| Orchestrate | `nb_09_orchestrator.py` | evidence-hierarchy integration, confidence, cross-agent agreement/disagreement, abstention |
| Benchmark | `nb_10a_validation_benchmark.py` | accuracy, calibration, trajectory metrics, CellTypist baseline, ablation |
| LLM | `nb_10b_llm_baseline_narrator.py` | naive-LLM baseline + grounded narrator (same model) |
| Pool | `nb_pool_calibration.py` | combine both datasets for final calibration |

## Datasets
- **GSE131907** — lung adenocarcinoma (Kim et al. 2020), tumor subset
- **GSE132465** — colorectal cancer (Lee et al. 2020, SMC cohort), tumor subset

Both: tumor subset -> QC -> 10,000-cell subsample -> 18 Leiden clusters.

## Knowledge bases
CellMarker 2.0; CELLxGENE Census (LTS 2025-01-30); Human Protein Atlas; Reactome;
GO Biological Process; MSigDB Hallmark; Open Targets; GWAS Catalog. Agents run all
enrichment statistics; the LLM never performs enrichment or generates evidence.

## Results (pooled, 36 clusters across both datasets)
- Cell-type accuracy (lineage-equivalence): **0.972 (35/36)**
  - Lung 18/18 (1.000), Colon 17/18 (0.944)
- CellTypist baseline: 0.944 on both (framework matches on colon, exceeds on lung)
- Calibration AUROC: 0.700, 95% CI [0.543, 0.843]; accuracy = 1.000 at confidence >= 0.95
- Each agent passes an independent negative control; cross-agent conflict detection
  fires on real data; leave-one-agent-out ablation confirms each agent contributes.

The naive-LLM baseline is stochastic (0.89-1.00 across runs) and is reported as
mean +/- SD. On these well-annotated datasets, scINTERP, CellTypist, and a naive LLM
are statistically indistinguishable on accuracy; scINTERP's contribution is grounded,
calibrated, reproducible, auditable interpretation — not raw accuracy.

## Reproducibility notes
- Subsampling and clustering use fixed seeds (seed=0).
- Census version pinned to 2025-01-30 and cached; agents read the cache, so runtime
  needs no live Census dependency.
- The lineage-equivalence mapping used for scoring is in `code/nb_10a_validation_benchmark.py`
  (`_FAMILY_PRIORITY`) and documented in `docs/lineage_equivalence.md`.

## LLM configuration
`nb_10b` uses Claude (default model `claude-haiku-4-5`) for both the naive baseline
and the narrator — the SAME model for a fair comparison. The API key is read at
runtime from an `ANTHROPIC_API_KEY` variable and is **never** stored in code. Provide
it via environment variable or a runtime prompt; never commit it. The client is
provider-agnostic (swap to Azure OpenAI in one place).
