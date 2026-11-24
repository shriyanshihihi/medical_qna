"""Entry point.   python main.py demo        (build index + sample questions + evaluation)
                  python main.py build       (only build the index)
                  python main.py ask "question"
                  python main.py eval
"""
import argparse
import sys

from src import config
from src.embeddings import Embedder
from src.evaluate import evaluate
from src.generator import ExtractiveGenerator, OpenAIGenerator
from src.index import VectorIndex
from src.pipeline import RAG, build_index

MODELS = {"minilm": config.ST_MODEL_MINILM, "biobert": config.ST_MODEL_BIOBERT}


def load_rag(args, generator=None):
    idx = VectorIndex.load(config.INDEX_DIR)
    emb = Embedder(args.backend, MODELS[args.model]).fit([c["text"] for c in idx.chunks])
    return RAG(emb, idx, generator or ExtractiveGenerator())


def show(res):
    print(f"\nANSWER: {res['answer']}\n")
    for c, s in res["retrieved"]:
        print(f"  [cos={s:.3f}] ({c['source']} #{c['id']}) {c['text'][:90]}...")
    print(f"\n  retrieval {res['retrieval_ms']:.1f} ms | generation {res['generation_ms']:.1f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["demo", "build", "ask", "eval"])
    ap.add_argument("question", nargs="?")
    ap.add_argument("--backend", default=config.DEFAULT_BACKEND, choices=["auto", "st", "lsa"])
    ap.add_argument("--model", default="minilm", choices=list(MODELS))
    ap.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    ap.add_argument("--corpus", default=str(config.CORPUS_DIR))
    args = ap.parse_args()

    try:
        if args.command in ("build", "demo"):
            docs, chunks, emb, _ = build_index(args.corpus, args.chunk_size, args.overlap,
                                               args.backend, MODELS[args.model], config.INDEX_DIR)
            print(f"Built index: {len(docs)} documents -> {len(chunks)} chunks | embedding backend = {emb.backend}")
        if args.command == "ask":
            if not args.question:
                sys.exit("Give a question: python main.py ask \"What is the first-line drug for type 2 diabetes?\"")
            show(load_rag(args).ask(args.question))
        if args.command == "demo":
            rag = load_rag(args)
            for q in ["What is the first-line drug for type 2 diabetes?",
                      "What are the symptoms of iron deficiency anemia?"]:
                print("\nQ:", q); show(rag.ask(q))
        if args.command in ("eval", "demo"):
            rag = load_rag(args)
            results = [evaluate(rag, config.QUESTIONS_FILE)]
            try:
                results.append(evaluate(RAG(rag.embedder, rag.index, OpenAIGenerator()), config.QUESTIONS_FILE))
            except Exception as e:
                print(f"\n(GPT-3.5 comparison skipped: {type(e).__name__}: {e})")
            print("\nEVALUATION on sample_questions.json (toy synthetic set, NOT a benchmark):")
            for r in results: print("  ", r)
    except FileNotFoundError as e:
        sys.exit(f"File problem: {e}\nHint: run  python main.py build  first.")


if __name__ == "__main__":
    main()
