"""Step 6: measure quality (ROUGE-L) and speed (latency) for each generator.

ROUGE-L = F-measure based on the Longest Common Subsequence between generated answer and reference answer.
The numbers produced here come from a TINY synthetic question set; they only show the method works.
They are NOT evidence for the CV's '5x faster / <3% drop' claim.
"""
import json
import statistics

from rouge_score import rouge_scorer


def evaluate(rag, questions_file):
    qs = json.loads(open(questions_file, encoding="utf-8").read())
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rl, lat, hit = [], [], 0
    for q in qs:
        out = rag.ask(q["question"])
        rl.append(scorer.score(q["reference"], out["answer"])["rougeL"].fmeasure)
        lat.append(out["retrieval_ms"] + out["generation_ms"])
        # retrieval hit@k: did any retrieved chunk come from the expected source?
        hit += any(c["source"] == q["source"] for c, _ in out["retrieved"])
    return {"generator": rag.generator.name, "n": len(qs),
            "rougeL_mean": round(statistics.mean(rl), 3),
            "latency_ms_mean": round(statistics.mean(lat), 2),
            "retrieval_hit_at_k": round(hit / len(qs), 2)}
