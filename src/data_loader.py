"""Load raw text documents (one .txt file = one 'textbook')."""
from pathlib import Path


def load_documents(corpus_dir: Path):
    """Return a list of {'source': filename, 'text': full text}. Raises a clear error if empty."""
    corpus_dir = Path(corpus_dir)
    docs = []
    for p in sorted(corpus_dir.glob("*.txt")):
        text = p.read_text(encoding="utf-8").strip()
        if text:
            docs.append({"source": p.stem, "text": text})
    if not docs:
        raise FileNotFoundError(f"No .txt files found in {corpus_dir}")
    return docs
