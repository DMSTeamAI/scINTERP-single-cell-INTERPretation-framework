"""
Central configuration for lung cancer scRNA-seq multi-agent interpretation.
Edit OPENAI_API_KEY and paths before running.
"""
import os
from pathlib import Path

# ── API Keys ─────────────────────────────────────────────────────────────────
# Set via environment: export OPENAI_API_KEY="sk-..."
# Or paste directly (do not commit to git)
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL    = "gpt-4o"          # or "gpt-4.5-preview" if you have access

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
DATA_DIR     = BASE_DIR / "data"
RESULTS_DIR  = BASE_DIR / "results"
FIGURES_DIR  = BASE_DIR / "figures"
CACHE_DIR    = BASE_DIR / "cache"   # stores LLM responses so you don't re-pay

for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, CACHE_DIR,
          RESULTS_DIR / "degs", RESULTS_DIR / "pathways"]:
    d.mkdir(parents=True, exist_ok=True)

# ── GEO download ─────────────────────────────────────────────────────────────
GEO_ACCESSION = "GSE131907"
GEO_BASE_URL  = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131907/suppl"
)
COUNTS_FILENAME = "GSE131907_Lung_normalized_filtered_gene_counts_matrix.txt.gz"
ANNOT_FILENAME  = "GSE131907_Lung_cell_annotation.csv.gz"
COUNTS_FILE     = DATA_DIR / COUNTS_FILENAME
ANNOT_FILE      = DATA_DIR / ANNOT_FILENAME

# ── Subset settings ───────────────────────────────────────────────────────────
# Use only primary tumor lung cells to keep local compute fast
TARGET_TISSUE = "tLung"
MAX_CELLS     = 10_000   # subsample; increase if you have ≥32 GB RAM

# ── Scanpy QC thresholds ──────────────────────────────────────────────────────
MIN_GENES    = 200
MAX_GENES    = 5_000
MAX_MITO_PCT = 15       # % mitochondrial reads
MIN_CELLS    = 3

# ── Processing ────────────────────────────────────────────────────────────────
N_HVG              = 2_000
N_PCS              = 30
N_NEIGHBORS        = 15
LEIDEN_RESOLUTION  = 0.5

# ── DEG ───────────────────────────────────────────────────────────────────────
TOP_N_DEGS     = 50    # saved to CSV
N_DEGS_FOR_LLM = 20   # sent to GPT-4o
MIN_LOGFC      = 0.5
MIN_PCT        = 0.1

# ── Pathway enrichment ────────────────────────────────────────────────────────
# gseapy will download gene set libraries automatically on first run
GENE_SETS      = ["MSigDB_Hallmark_2020", "KEGG_2021_Human"]
PVAL_CUTOFF    = 0.05
TOP_N_PATHWAYS = 5

# ── LLM generation ────────────────────────────────────────────────────────────
LLM_TEMPERATURE = 0      # deterministic for reproducibility
LLM_MAX_TOKENS  = 600
