"""Step 3: vector store with FAISS.

IndexFlatIP = exact (brute-force) inner-product search. With normalised vectors this equals cosine similarity.
Exact search is fine up to ~1M vectors; for 3 lakh x 384-dim it is ~460 MB and still OK.
For bigger corpora use IndexIVFFlat / IndexHNSWFlat (approximate, faster, slightly less accurate).
"""
import json
import pickle
from pathlib import Path

import faiss
import numpy as np


class VectorIndex:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)
        self.chunks = []

    def add(self, vectors, chunks):
        self.index.add(np.ascontiguousarray(vectors, dtype="float32"))
        self.chunks.extend(chunks)

    def search(self, qvec, k):
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(np.ascontiguousarray(qvec, dtype="float32"), k)
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def save(self, folder: Path):
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / "faiss.index"))
        (folder / "chunks.json").write_text(json.dumps(self.chunks, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path):
        folder = Path(folder)
        if not (folder / "faiss.index").exists():
            raise FileNotFoundError(f"No index found in {folder}")
        obj = cls.__new__(cls)
        obj.index = faiss.read_index(str(folder / "faiss.index"))
        obj.chunks = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        return obj
