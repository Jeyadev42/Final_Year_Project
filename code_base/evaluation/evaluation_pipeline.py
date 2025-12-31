import json
import os
import time
import sys
from typing import List, Dict, Any

import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np

from evaluation_utils import evaluate_answer
from llm_clients import build_eval_llm, run_local_llm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from vector_store import VectorStore


# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

CONFIG_PATH = "../config.json"  # adjust if running from another folder
VECTOR_DB_PATH = "../cache.db"

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError("config.json not found. Please create it with GOOGLE_API_KEY and GOOGLE_CX.")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

GOOGLE_API_KEY = cfg["GOOGLE_API_KEY"]
GOOGLE_CX = cfg["GOOGLE_CX"]
eval_llm_fn = build_eval_llm(cfg)

GOLDEN_FILE = "../golden_dataset.jsonl"  # path relative to this script
RESULT_DIR = "results"
OUTPUT_FILE = os.path.join(RESULT_DIR, "eval_pipeline_output.jsonl")
SUMMARY_FILE = os.path.join(RESULT_DIR, "eval_summary.json")


os.makedirs(RESULT_DIR, exist_ok=True)


# -------------------------------------------------------
# EMBEDDINGS (LOCAL, SAME AS APP)
# -------------------------------------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def academic_rewrite(text: str) -> str:
    prompt = (
        "Rewrite the following answer in a formal academic tone. "
        "Improve clarity and remove informal phrasing. "
        "Do not add new content. "
        "Do not add conversational phrases or follow-up questions. "
        "Return only the rewritten explanation.\n\n"
        f"{text}"
    )
    return run_local_llm(prompt)


# -------------------------------------------------------
# DOMAIN CLASSIFICATION FOR SEARCH (SAME AS APP)
# -------------------------------------------------------

def classify_query_domain(query: str) -> str:
    """
    Classify the domain of the user's query so that web search becomes relevant.
    """
    prompt = f"""
Determine the intended domain of the following query.

Query: "{query}"

Choose only one domain from the following list and respond with only that domain name:

- machine_learning
- data_science
- computer_science_general
- compliance_regulation
- medicine
- finance
- ecommerce
- general_knowledge
- law
- statistics
- mathematics
- other

Do not add explanations. Respond with one domain name only.
    """
    response = run_local_llm(prompt).strip().lower()

    valid_domains = {
        "machine_learning",
        "data_science",
        "computer_science_general",
        "compliance_regulation",
        "medicine",
        "finance",
        "ecommerce",
        "general_knowledge",
        "law",
        "statistics",
        "mathematics",
        "other",
    }
    if response not in valid_domains:
        return "general_knowledge"
    return response


def expand_query_with_domain(original_query: str, domain: str) -> str:
    """
    Expand Google search query based on detected domain.
    """
    if domain == "machine_learning":
        return f"{original_query} machine learning classifier model supervised learning examples"
    if domain == "data_science":
        return f"{original_query} data science classification feature engineering"
    if domain == "statistics":
        return f"{original_query} statistical classification discriminant analysis"
    if domain == "law":
        return f"{original_query} legal definition explanation"
    if domain == "compliance_regulation":
        return f"{original_query} compliance framework guidelines"
    if domain == "medicine":
        return f"{original_query} medical diagnosis classification diagnostic model"
    if domain == "finance":
        return f"{original_query} fraud detection credit risk classification model"
    if domain == "ecommerce":
        return f"{original_query} recommendation system product classifier"

    # general fallback
    return original_query


# -------------------------------------------------------
# GOOGLE SEARCH (NO CACHE)
# -------------------------------------------------------

def google_search(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    domain = classify_query_domain(query)
    expanded_query = expand_query_with_domain(query, domain)

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CX,
        "q": expanded_query,
        "num": max_results,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
    except Exception:
        return []

    if resp.status_code != 200:
        return []

    data = resp.json()
    items = data.get("items", [])

    results = []
    for item in items:
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }
        )

    return results


# -------------------------------------------------------
# FETCH WEBPAGE CONTENT (NO CACHE)
# -------------------------------------------------------

PAYWALL_MARKERS = [
    "subscribe to read",
    "subscription required",
    "sign in to continue",
    "paywall",
    "members-only",
]


def is_paywalled(text: str) -> bool:
    lower = text.lower()
    return any(k in lower for k in PAYWALL_MARKERS)


def fetch_page(url: str) -> str:
    try:
        resp = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
    except Exception:
        return ""

    if resp.status_code != 200:
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = " ".join(soup.stripped_strings)
    text = " ".join(text.split())

    if not text or len(text) < 200:
        return ""

    if is_paywalled(text):
        return ""

    return text


# -------------------------------------------------------
# SUMMARIZATION PER PAGE (NO CACHE, SAME PROMPT)
# -------------------------------------------------------

def summarize_page(url: str, text: str, user_query: str) -> str:
    """
    Summaries must be structured, rich, and aligned to the question.
    """
    prompt = f"""
You are summarizing a webpage excerpt to support answering the user’s question.

User question:
{user_query}

Webpage text:
{text}

Your task:
- Extract only information that is relevant for answering the question.
- If the webpage contains indirect or partial information, infer meaning where logically possible.
- Provide a structured summary in 4 bullet points:
  - Key definition(s)
  - Mechanism or how it works
  - Examples mentioned or implied
  - Any domain-specific context

Do not include unrelated content. Do not mention that this is a summary.
Return only the bullet points.
    """
    summary = run_local_llm(prompt)
    return summary


# -------------------------------------------------------
# WEB ANSWER SYNTHESIS (SAME AS APP)
# -------------------------------------------------------

def synthesize_web_answer(
    user_query: str,
    summaries: List[str],
    rag_chunks: List[dict],
) -> str:
    combined = "\n".join(summaries)
    context = "\n\n".join(
        f"Source: {c['url']}\n{c['content']}" for c in rag_chunks
    )

    prompt = f"""
You are writing an academically structured explanation based on multiple summarized sources.

User query:
{user_query}

Relevant extracted summaries:
{combined}

Retrieved context chunks:
{context}

Your task:
1. Use all relevant information from the summaries.
2. Use retrieved context chunks as grounding evidence.
3. Add missing conceptual fundamentals using your own knowledge.
4. Provide a complete academic explanation.
5. Include:
   - definition
   - mechanism or working principle
   - general use cases
   - examples (if relevant)
6. Focus on the intended technical meaning of the query.
7. Do not include follow-up questions or conversational remarks.
8. Return only the explanation.

Write a complete academic answer below.
    """
    return run_local_llm(prompt)


# -------------------------------------------------------
# PIPELINE: SEARCH → FETCH → RANK → SUMMARIZE → ANSWER
# (NO STREAMLIT, NO CACHE)
# -------------------------------------------------------

def run_pipeline(query: str, max_results: int = 10) -> Dict[str, Any]:
    # 1. Search
    search_results = google_search(query, max_results=max_results)
    if not search_results:
        return {
            "search_results": [],
            "used_sources": [],
            "summaries": [],
            "local_answer": "",
            "web_answer": "",
            "final_answer": "",
        }

    # 2. Fetch content
    pages = []
    for item in search_results:
        url = item["url"]
        text = fetch_page(url)
        pages.append(
            {
                "title": item["title"],
                "url": url,
                "snippet": item["snippet"],
                "text": text,
            }
        )

    # Keep only pages with usable text for ranking
    usable_pages = [p for p in pages if p["text"]]

    # If nothing usable, fallback to local-only answer
    if not usable_pages:
        local_prompt = (
            "You are to provide a factual, academic explanation to the user query. "
            "Produce only the answer. "
            "Do not include follow-up questions, suggestions, or conversational remarks. "
            "Return only the explanation.\n\n"
            f"User query: {query}"
        )
        local_answer = run_local_llm(local_prompt)
        return {
            "search_results": search_results,
            "used_sources": [],
            "summaries": [],
            "local_answer": local_answer,
            "web_answer": "",
            "final_answer": local_answer,
        }

    # 3. Embedding-based ranking
    q_emb = embedding_model.encode(query)
    for p in usable_pages:
        doc_emb = embedding_model.encode(p["text"][:3000])
        p["score"] = cosine_sim(q_emb, doc_emb)

    usable_pages.sort(key=lambda x: x["score"], reverse=True)
    top_k = usable_pages[:5]

    # 4. Summaries for top_k
    summaries = []
    for p in top_k:
        s = summarize_page(p["url"], p["text"], query)
        summaries.append(
            {
                "url": p["url"],
                "title": p["title"],
                "summary": s,
            }
        )

    # 5. RAG vector store context
    vector_store = VectorStore(VECTOR_DB_PATH, embedding_model)
    vector_store.add_pages(top_k)
    rag_chunks = vector_store.query(
        query,
        top_k=6,
        urls=[p["url"] for p in top_k],
    )

    # 6. Local answer (no web)
    local_prompt = (
        "You are to provide a factual, academic explanation to the user query. "
        "Produce only the answer. "
        "Do not include follow-up questions, suggestions, or conversational remarks. "
        "Return only the explanation.\n\n"
        f"User query: {query}"
    )
    local_answer = run_local_llm(local_prompt)

    # 7. Web-based synthesized answer
    web_answer = synthesize_web_answer(
        query,
        [s["summary"] for s in summaries],
        rag_chunks,
    )

    # 8. Fused final answer (local + web)
    fused_prompt = f"""
You have two answers to the same question.

Question:
{query}

Answer A (from your own knowledge):
{local_answer}

Answer B (from web-based information):
{web_answer}

Your task:
- Merge them into a single, coherent, academic-style answer.
- Correct any mistakes.
- Prefer facts that are explicitly supported by Answer B when they conflict.
- Include fundamental conceptual details, even if they are not mentioned in the web-based answer.
- Do not mention Answer A or Answer B.
- Do not add follow-up questions or conversational phrases.
- Return only the final explanation.
    """
    final_answer = run_local_llm(fused_prompt)

    return {
        "search_results": search_results,
        "used_sources": top_k,
        "summaries": summaries,
        "rag_chunks": rag_chunks,
        "local_answer": local_answer,
        "web_answer": web_answer,
        "final_answer": final_answer,
    }


# -------------------------------------------------------
# BENCHMARK OVER GOLDEN DATASET
# -------------------------------------------------------

def run_eval_pipeline():
    print("=== Starting Evaluation Pipeline ===")
    print(f"Loading dataset: {GOLDEN_FILE}")

    if not os.path.exists(GOLDEN_FILE):
        raise FileNotFoundError(f"{GOLDEN_FILE} not found.")

    results = []
    total = 0

    # Clear previous output if exists
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    with open(GOLDEN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line.strip())
            qid = record["id"]
            query = record["query"]
            gold = record["gold_answer"]

            total += 1
            print(f"\n[{total}] Processing {qid}: {query}")

            # 1) Run the full pipeline (no Streamlit)
            pipeline_result = run_pipeline(query, max_results=10)

            # 2) Evaluate the final answer with the same metrics as app.py
            final_answer = pipeline_result["final_answer"]
            local_answer = pipeline_result["local_answer"]
            web_answer = pipeline_result["web_answer"]
            summaries = [s["summary"] for s in pipeline_result["summaries"]]
            used_sources = pipeline_result["used_sources"]

            eval_result = evaluate_answer(
                query=query,
                final_answer=final_answer,
                local_answer=local_answer,
                web_answer=web_answer,
                summaries=summaries,
                sources=used_sources,
                stability=None,               # no historical stability in offline eval
                embedding_model=embedding_model,
                llm_fn=eval_llm_fn,
            )

            result = {
                "id": qid,
                "query": query,
                "gold_answer": gold,
                "system_answer": final_answer,
                "local_answer": local_answer,
                "web_answer": web_answer,
                "metrics": eval_result,
                "used_sources": [
                    {"title": s["title"], "url": s["url"], "score": s.get("score", 0.0)}
                    for s in used_sources
                ],
                "timestamp": time.time(),
            }

            with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps(result) + "\n")

            results.append(result)

    # ---------------------------------------------------
    # Aggregate summary over all questions
    # ---------------------------------------------------
    print("\n=== Evaluation Completed ===")

    if total == 0:
        print("No records found in golden dataset.")
        return

    avg = {
        "avg_relevance": round(sum(r["metrics"]["relevance"] for r in results) / total, 4),
        "avg_grounding": round(sum(r["metrics"]["grounding"]["score"] for r in results) / total, 4),
        "avg_completeness": round(sum(r["metrics"]["completeness"]["score"] for r in results) / total, 4),
        "avg_contradiction": round(sum(r["metrics"]["contradiction"]["score"] for r in results) / total, 4),
        "avg_reliability": round(sum(r["metrics"]["reliability"]["score"] for r in results) / total, 4),
        "avg_overall": round(sum(r["metrics"]["overall_score"] for r in results) / total, 4),
        "total_items": total,
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as sf:
        json.dump(avg, sf, indent=2)

    print(f"\nSummary saved to: {SUMMARY_FILE}")
    print(f"Detailed results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_eval_pipeline()
