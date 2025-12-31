import json
import re
from typing import List, Dict, Any, Callable, Optional

import numpy as np
import requests


# ===============================================================
# 0) LOCAL EMBEDDING (replaces embedding_model.encode)
# ===============================================================

def ollama_embed(text: str, model: str = "nomic-embed-text") -> np.ndarray:
    """
    Local embedding using Ollama (no HF dependency).
    """
    try:
        r = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=20
        )
        vec = r.json().get("embedding", [])
        return np.array(vec, dtype=float)
    except Exception:
        return np.zeros(768)


# ===============================================================
# 1) COSINE SIMILARITY (shared)
# ===============================================================

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ===============================================================
# 2) ORIGINAL APP FUNCTIONS (unchanged except embedding_model replaced)
# ===============================================================

def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 0]


def compute_relevance_score(
    query: str,
    final_answer: str,
    embedding_model=None
) -> float:
    if not query or not final_answer:
        return 0.0

    # Replace embedding_model.encode with ollama embedding
    q_emb = ollama_embed(query)
    a_emb = ollama_embed(final_answer)

    return _cosine_sim(q_emb, a_emb)


def compute_grounding_score(
    final_answer: str,
    summaries: List[str],
    embedding_model=None,
    threshold: float = 0.6
) -> Dict[str, Any]:

    sentences = _split_sentences(final_answer)
    if not sentences or not summaries:
        return {"score": 0.0, "per_sentence": []}

    sum_embs = [ollama_embed(s) for s in summaries]

    per_sentence = []
    grounded = 0

    for sent in sentences:
        s_emb = ollama_embed(sent)
        sims = [_cosine_sim(s_emb, se) for se in sum_embs]
        max_sim = max(sims) if sims else 0.0
        per_sentence.append({"sentence": sent, "max_sim": max_sim})
        if max_sim >= threshold:
            grounded += 1

    score = grounded / len(sentences)
    return {"score": score, "per_sentence": per_sentence}


def compute_completeness_score(
    query: str,
    final_answer: str,
    llm_fn: Callable[[str], str]
) -> Dict[str, Any]:

    prompt = f"""
You are evaluating whether an answer is structurally complete.

Question:
{query}

Answer:
{final_answer}

Does the answer include:
- definition
- mechanism
- use_cases
- examples

Return JSON only:
{{
  "definition": 1 or 0,
  "mechanism": 1 or 0,
  "use_cases": 1 or 0,
  "examples": 1 or 0
}}
"""
    try:
        raw = llm_fn(prompt)
        data = json.loads(raw)

        flags = {
            "definition": int(bool(data.get("definition", 0))),
            "mechanism": int(bool(data.get("mechanism", 0))),
            "use_cases": int(bool(data.get("use_cases", 0))),
            "examples": int(bool(data.get("examples", 0))),
        }

        score = sum(flags.values()) / 4.0
        return {"score": score, "flags": flags}

    except Exception:
        return {
            "score": 0.5,
            "flags": {
                "definition": 0,
                "mechanism": 0,
                "use_cases": 0,
                "examples": 0,
            }
        }


def compute_source_reliability(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not sources:
        return {"score": 0.0, "details": []}

    def domain_score(url: str):
        url = url.lower()
        if ".gov" in url:
            return 1.0
        if ".edu" in url or "ac." in url:
            return 0.95
        if "wikipedia.org" in url:
            return 0.8
        if any(x in url for x in ["scikit-learn", "pytorch.org", "tensorflow.org"]):
            return 0.9
        if any(x in url for x in ["medium.com", "towardsdatascience"]):
            return 0.6
        return 0.5

    details = []
    scores = []

    for s in sources:
        url = s.get("url", "")
        text = s.get("text", "") or ""

        dom = domain_score(url)
        length = len(text)

        if length <= 0:
            length_factor = 0.4
        elif length < 1000:
            length_factor = 0.6
        elif length < 5000:
            length_factor = 0.8
        else:
            length_factor = 1.0

        combined = 0.7 * dom + 0.3 * length_factor

        scores.append(combined)
        details.append({
            "url": url,
            "domain_score": dom,
            "length": length,
            "length_factor": length_factor,
            "combined": combined
        })

    avg = sum(scores) / len(scores)
    return {"score": avg, "details": details}


def compute_contradiction_score(
    final_answer: str,
    summaries: List[str],
    llm_fn: Callable[[str], str]
) -> Dict[str, Any]:

    if not final_answer or not summaries:
        return {"score": 0.0, "raw": ""}

    joined = "\n".join(summaries)
    prompt = f"""
Check whether the final answer contradicts web evidence.

Evidence:
{joined}

Final answer:
{final_answer}

Return a number 0-1 only.
"""
    try:
        raw = llm_fn(prompt).strip()
        m = re.search(r"(0(\.\d+)?)|(1(\.0+)?)", raw)
        val = float(m.group(0)) if m else 0.5
        val = max(0.0, min(1.0, val))
        return {"score": val, "raw": raw}
    except Exception:
        return {"score": 0.5, "raw": ""}


# ===============================================================
# 3) BACKEND API CALL (for benchmark)
# ===============================================================

def call_system_answer(query: str) -> str:
    """
    Benchmark runner calls this route.
    """
    try:
        r = requests.post(
            "http://localhost:8080/api/answer",
            json={"query": query},
            timeout=60
        )
        return r.json().get("answer", "").strip()

    except Exception:
        return "ERROR: System did not return a valid response."


# ===============================================================
# 4) UNIFIED evaluate_answer (app-compatible + benchmark-compatible)
# ===============================================================

def evaluate_answer(
    query: str,
    final_answer: str,
    local_answer: str,
    web_answer: str,
    summaries: List[str],
    sources: List[Dict[str, Any]],
    stability: Optional[float],
    embedding_model,
    llm_fn: Callable[[str], str],
    gold_answer: Optional[str] = None,
) -> Dict[str, Any]:
    """
    If gold_answer is None → UI mode
    If gold_answer is provided → Benchmark mode
    """

    # -----------------------------------
    # BENCHMARK MODE
    # -----------------------------------
    if gold_answer is not None:

        # use local embedding
        rel = compute_relevance_score(query, final_answer)
        grd = compute_grounding_score(final_answer, summaries)
        comp = compute_completeness_score(query, final_answer, llm_fn)
        reli = compute_source_reliability(sources)
        contra = compute_contradiction_score(final_answer, summaries, llm_fn)

        overall = (
            0.25 * rel +
            0.20 * grd["score"] +
            0.20 * comp["score"] +
            0.15 * reli["score"] +
            0.10 * (1 - contra["score"]) +
            0.10 * (stability if stability is not None else 0.5)
        )

        return {
            "relevance": rel,
            "grounding": grd,
            "completeness": comp,
            "reliability": reli,
            "contradiction": contra,
            "stability": stability,
            "overall_score": overall
        }

    # -----------------------------------
    # UI MODE (no gold answer)
    # -----------------------------------
    rel = compute_relevance_score(query, final_answer)
    grd = compute_grounding_score(final_answer, summaries)
    comp = compute_completeness_score(query, final_answer, llm_fn)
    reli = compute_source_reliability(sources)
    contra = compute_contradiction_score(final_answer, summaries, llm_fn)

    stability_val = stability if stability is not None else 0.5

    overall = (
        0.25 * rel +
        0.20 * grd["score"] +
        0.20 * comp["score"] +
        0.15 * reli["score"] +
        0.10 * (1 - contra["score"]) +
        0.10 * stability_val
    )

    return {
        "relevance": rel,
        "grounding": grd,
        "completeness": comp,
        "reliability": reli,
        "contradiction": contra,
        "stability": stability_val,
        "overall_score": overall
    }
