"""Glue: build the index once, then answer questions (Step 4 = retrieval lives here)."""
import time
from pathlib import Path

from . import config
from .chunking import chunk_documents
from .data_loader import load_documents
from .embeddings import Embedder
from .index import VectorIndex


def build_index(corpus_dir, chunk_size, overlap, backend, model_name, out_dir):
    docs = load_documents(corpus_dir)
    chunks = chunk_documents(docs, chunk_size, overlap)
    emb = Embedder(backend, model_name).fit([c["text"] for c in chunks])
    vecs = emb.encode([c["text"] for c in chunks])
    idx = VectorIndex(vecs.shape[1]); idx.add(vecs, chunks)
    idx.save(out_dir)
    return docs, chunks, emb, idx


class RAG:
    """Holds embedder + index + generator. Used by CLI and evaluation."""
    def __init__(self, embedder, index, generator, top_k=config.TOP_K):
        self.embedder, self.index, self.generator, self.top_k = embedder, index, generator, top_k

    def retrieve(self, question):
        qv = self.embedder.encode([question])           # (1, dim), normalised
        return self.index.search(qv, self.top_k)        # [(chunk, cosine score), ...]

    def ask(self, question):
        t0 = time.perf_counter()
        retrieved = self.retrieve(question)
        t1 = time.perf_counter()
        answer = self.generator.answer(question, retrieved)
        t2 = time.perf_counter()
        return {"answer": answer, "retrieved": retrieved,
                "retrieval_ms": (t1 - t0) * 1000, "generation_ms": (t2 - t1) * 1000}
