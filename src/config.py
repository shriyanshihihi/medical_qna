"""Central settings. Interview tip: every number here is a decision you can be asked about."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "sample_corpus"
QUESTIONS_FILE = ROOT / "data" / "sample_questions.json"
INDEX_DIR = ROOT / "outputs" / "index"

# --- Chunking ---------------------------------------------------------------
# CV says ~500 tokens per chunk. Our demo corpus is tiny, so we use small chunks.
# For the real MedRAG corpus run:  python main.py build --chunk-size 500 --overlap 50
CHUNK_SIZE = 60      # measured in WORDS here (approximation of tokens)
CHUNK_OVERLAP = 10   # overlap so a sentence cut at a boundary still appears whole once

# --- Embeddings -------------------------------------------------------------
ST_MODEL_MINILM = "sentence-transformers/all-MiniLM-L6-v2"                       # 384-dim, fast
ST_MODEL_BIOBERT = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"     # 768-dim, biomedical
DEFAULT_BACKEND = "auto"   # auto = use sentence-transformers if it can load, else offline LSA

# --- Retrieval --------------------------------------------------------------
TOP_K = 3
