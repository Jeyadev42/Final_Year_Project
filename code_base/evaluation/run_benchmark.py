import json
import time
import os

# import your evaluation functions
from evaluation_utils import (
    call_system_answer,
    evaluate_answer,
)
from evaluation_utils import ollama_embed as embedding_model  # embedding model alias

# If needed, import or define your local LLM runner
def run_local_llm(prompt: str) -> str:
    """
    Minimal safe LLM wrapper for benchmark.
    Does not require UI.
    """
    import requests
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma3:4b", "prompt": prompt},
            timeout=30
        )
        return r.json().get("response", "")
    except Exception:
        return ""


# -----------------------------
# Configuration
# -----------------------------
GOLDEN_FILE = "../golden_dataset.jsonl"
RESULT_DIR = "results"
OUTPUT_FILE = "results/benchmark_output.jsonl"
SUMMARY_FILE = "results/summary.json"


# -----------------------------
# Ensure folders exist
# -----------------------------
os.makedirs(RESULT_DIR, exist_ok=True)


# -----------------------------
# Benchmark Runner
# -----------------------------
def run_benchmark():
    print("=== Starting Benchmark ===")
    print(f"Loading dataset: {GOLDEN_FILE}")

    results = []
    total = 0

    with open(GOLDEN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            qid = record["id"]
            query = record["query"]
            gold = record["gold_answer"]

            total += 1
            print(f"\n[{total}] Processing {qid}: {query}")

            # 1 — get system answer
            system_answer = call_system_answer(query)

            # 2 — produce evaluation using the SAME function as app.py
            metrics = evaluate_answer(
                query=query,
                final_answer=system_answer,
                local_answer="",              # No local model output in benchmark
                web_answer="",                # No web answer in benchmark
                summaries=[gold],             # Gold acts as reference summary
                sources=[],                   # No sources fetched in benchmark
                stability=0.5,                # Neutral stability for benchmark
                embedding_model=embedding_model,
                llm_fn=run_local_llm,
            )

            # 3 — store result
            result = {
                "id": qid,
                "query": query,
                "gold_answer": gold,
                "system_answer": system_answer,
                "metrics": metrics,
                "timestamp": time.time()
            }

            # Save line-by-line
            with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
                out.write(json.dumps(result) + "\n")

            results.append(result)

    # -----------------------------
    # Compute summary stats
    # -----------------------------
    print("\n=== Benchmark Completed ===")

    avg = {
        "avg_relevance": round(sum(r["metrics"]["relevance"] for r in results) / total, 4),
        "avg_grounding": round(sum(r["metrics"]["grounding"]["score"] for r in results) / total, 4),
        "avg_completeness": round(sum(r["metrics"]["completeness"]["score"] for r in results) / total, 4),
        "avg_contradiction": round(sum(r["metrics"]["contradiction"]["score"] for r in results) / total, 4),
        "avg_reliability": round(sum(r["metrics"]["reliability"]["score"] for r in results) / total, 4),
        "avg_overall": round(sum(r["metrics"]["overall_score"] for r in results) / total, 4),
        "total_items": total
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as sf:
        json.dump(avg, sf, indent=2)

    print(f"\nSummary saved to: {SUMMARY_FILE}")
    print(f"Detailed results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_benchmark()
