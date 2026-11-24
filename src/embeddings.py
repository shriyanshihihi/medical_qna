"""Step 2: turn text into dense vectors (embeddings).

Two backends:
  * 'st'   -> Hugging Face Sentence-Transformers (MiniLM or a BioBERT-based model). This is what the CV describes.
  * 'lsa'  -> OFFLINE FALLBACK (TF-IDF + SVD). Only so the demo runs without internet. NOT a CV technology;
              it is weaker than a neural model and exists purely for testing the pipeline.

All vectors are L2-normalised, so inner product == cosine similarity.
"""
import numpy as np


class Embedder:
    def __init__(self, backend="auto", model_name=None):
        self.model_name = model_name
        self.backend = backend
        self._st = None
        self._tfidf = None
        self._svd = None
        if backend in ("auto", "st"):
            try:
                from sentence_transformers import SentenceTransformer
                self._st = SentenceTransformer(model_name)
                self.backend = "st"
            except Exception as e:  # no package / no internet / bad model name
                if backend == "st":
                    raise RuntimeError(f"Could not load sentence-transformers model '{model_name}': {e}")
                print(f"[embeddings] sentence-transformers unavailable ({type(e).__name__}); using offline LSA fallback.")
                self.backend = "lsa"

    # ---- fitting is only needed for the LSA fallback ----
    def fit(self, texts):
        if self.backend == "lsa":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            self._tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True)
            X = self._tfidf.fit_transform(texts)
            k = max(2, min(64, X.shape[0] - 1, X.shape[1] - 1))
            self._svd = TruncatedSVD(n_components=k, random_state=0).fit(X)
        return self

    def encode(self, texts, batch_size=64):
        if self.backend == "st":
            v = self._st.encode(list(texts), batch_size=batch_size, show_progress_bar=False)
        else:
            v = self._svd.transform(self._tfidf.transform(list(texts)))
        v = np.asarray(v, dtype="float32")
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.clip(norms, 1e-12, None)   # L2 normalise (guard against zero vectors)
