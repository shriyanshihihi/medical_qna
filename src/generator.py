"""Step 5: produce the final answer from the retrieved chunks.

Two generators so latency/quality can be compared (this mirrors the CV's benchmark idea):
  * ExtractiveGenerator : local, no model download. Picks the most question-relevant sentences from the chunks.
  * OpenAIGenerator     : GPT-3.5 via API (optional, needs OPENAI_API_KEY + internet).

IMPORTANT (interview honesty): MiniLM is an EMBEDDING model - it cannot write text. In a RAG system MiniLM does the
RETRIEVAL part. 'MiniLM vs GPT-3.5' can only be a comparison of whole pipelines
(MiniLM-retrieval + local answer  vs  GPT-3.5 answer), not model against model.
"""
import os
import re

PROMPT = (
    "You are a careful medical information assistant. Answer ONLY using the context below. "
    "If the context does not contain the answer, say you do not know. Be concise.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)
_STOP = set("the a an of and or to in is are what which how does do for with on as by be it its that this from at".split())


def _tokens(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP]


class ExtractiveGenerator:
    name = "extractive-local"

    def answer(self, question, retrieved, n_sentences=2):
        q = set(_tokens(question))
        sents = []
        for chunk, score in retrieved:
            for s in re.split(r"(?<=[.!?])\s+", chunk["text"]):
                if len(s.split()) >= 6:   # skip short heading-like fragments
                    overlap = len(q & set(_tokens(s)))
                    sents.append((overlap + 0.5 * score, s.strip()))
        if not sents:
            return "I do not know based on the available documents."
        sents.sort(key=lambda x: -x[0])
        return " ".join(s for _, s in sents[:n_sentences])


class OpenAIGenerator:
    name = "gpt-3.5-turbo"

    def __init__(self, model="gpt-3.5-turbo"):
        from openai import OpenAI          # imported lazily so the demo works without the package
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client, self.model = OpenAI(), model

    def answer(self, question, retrieved):
        context = "\n---\n".join(c["text"] for c, _ in retrieved)
        r = self.client.chat.completions.create(
            model=self.model, temperature=0,
            messages=[{"role": "user", "content": PROMPT.format(context=context, question=question)}])
        return r.choices[0].message.content.strip()
