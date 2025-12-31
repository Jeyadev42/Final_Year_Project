import streamlit as st
import json
import requests
import sqlite3
import time
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import hashlib
from evaluation.evaluation_utils import evaluate_answer
# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

CONFIG_PATH = "config.json"
DB_CACHE_PATH = "cache.db"
DB_EVAL_PATH = "evaluation.db"

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError("config.json not found. Please create it with GOOGLE_API_KEY and GOOGLE_CX.")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

GOOGLE_API_KEY = cfg["GOOGLE_API_KEY"]
GOOGLE_CX = cfg["GOOGLE_CX"]

# -------------------------------------------------------
# DB INITIALIZATION (CACHE + EVAL)
# -------------------------------------------------------

def init_cache_db():
    conn = sqlite3.connect(DB_CACHE_PATH)
    cur = conn.cursor()

    # search_cache: query -> results (JSON)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS search_cache (
            query TEXT PRIMARY KEY,
            results TEXT
        )
        """
    )

    # page_cache: url -> content
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS page_cache (
            url TEXT PRIMARY KEY,
            content TEXT
        )
        """
    )

    # summary_cache: key -> summary
    # key is a hash(text + query) to allow multiple summaries per URL per query
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS summary_cache (
            url TEXT PRIMARY KEY,
            summary TEXT
        )
        """
    )

    conn.commit()
    conn.close()


def init_eval_db():
    conn = sqlite3.connect(DB_EVAL_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluation_scores (
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            query TEXT,
            relevance_score INTEGER
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluation_pairwise (
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            query TEXT,
            winner TEXT
        )
        """
    )

    # For stability: store last final answer per query
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS previous_answers (
            query TEXT PRIMARY KEY,
            answer TEXT
        )
        """
    )

    conn.commit()
    conn.close()


init_cache_db()
init_eval_db()

# -------------------------------------------------------
# EMBEDDINGS (LOCAL)
# -------------------------------------------------------

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = get_embedding_model()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# -------------------------------------------------------
# LOCAL LLM VIA OLLAMA (gemma3:4b)
# -------------------------------------------------------

def run_local_llm(prompt: str) -> str:
    """
    Run a local Ollama LLM (gemma3:4b).
    """
    try:
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "gemma3:4b",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,
        )
        response_json = resp.json()
        return response_json["message"]["content"].strip()
    except Exception as e:
        return f"(Local LLM error: {e})"


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
# CACHE HELPERS
# -------------------------------------------------------

def search_cache_get(query: str):
    conn = sqlite3.connect(DB_CACHE_PATH)
    cur = conn.cursor()
    cur.execute("SELECT results FROM search_cache WHERE query = ?", (query,))
    row = cur.fetchone()
    conn.close()
    if row:
        try:
            return json.loads(row[0])
        except Exception:
            return None
    return None


def search_cache_set(query: str, results: list):
    conn = sqlite3.connect(DB_CACHE_PATH)
    cur = conn.cursor()
    cur.execute(
        "REPLACE INTO search_cache (query, results) VALUES (?, ?)",
        (query, json.dumps(results)),
    )
    conn.commit()
    conn.close()


def page_cache_get(url: str):
    conn = sqlite3.connect(DB_CACHE_PATH)
    cur = conn.cursor()
    cur.execute("SELECT content FROM page_cache WHERE url = ?", (url,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def page_cache_set(url: str, content: str):
    conn = sqlite3.connect(DB_CACHE_PATH)
    cur = conn.cursor()
    cur.execute(
        "REPLACE INTO page_cache (url, content) VALUES (?, ?)",
        (url, content),
    )
    conn.commit()
    conn.close()


def summary_cache_get(key: str):
    conn = sqlite3.connect(DB_CACHE_PATH)
    cur = conn.cursor()
    cur.execute("SELECT summary FROM summary_cache WHERE url = ?", (key,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def summary_cache_set(key: str, summary: str):
    conn = sqlite3.connect(DB_CACHE_PATH)
    cur = conn.cursor()
    cur.execute(
        "REPLACE INTO summary_cache (url, summary) VALUES (?, ?)",
        (key, summary),
    )
    conn.commit()
    conn.close()


# -------------------------------------------------------
# DOMAIN CLASSIFICATION FOR SEARCH
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
# GOOGLE SEARCH
# -------------------------------------------------------

def google_search(query: str, max_results: int = 10):
    cached = search_cache_get(query)
    if cached:
        return cached

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

    search_cache_set(query, results)
    return results


# -------------------------------------------------------
# FETCH WEBPAGE CONTENT
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
    cached = page_cache_get(url)
    if cached:
        return cached

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

    page_cache_set(url, text)
    return text


# -------------------------------------------------------
# SUMMARIZATION PER PAGE (IMPROVED)
# -------------------------------------------------------

def summarize_page(url: str, text: str, user_query: str) -> str:
    """
    Summaries must be structured, rich, and aligned to the question.
    """
    key = hashlib.md5((url + user_query).encode("utf-8")).hexdigest()
    cached = summary_cache_get(key)
    if cached:
        return cached

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
    summary_cache_set(key, summary)
    return summary


# -------------------------------------------------------
# WEB ANSWER SYNTHESIS (IMPROVED)
# -------------------------------------------------------

def synthesize_web_answer(user_query: str, summaries: list[str]) -> str:
    combined = "\n".join(summaries)

    prompt = f"""
You are writing an academically structured explanation based on multiple summarized sources.

User query:
{user_query}

Relevant extracted summaries:
{combined}

Your task:
1. Use all relevant information from the summaries.
2. Add missing conceptual fundamentals using your own knowledge.
3. Provide a complete academic explanation.
4. Include:
   - definition
   - mechanism or working principle
   - general use cases
   - examples (if relevant)
5. Focus on the intended technical meaning of the query.
6. Do not include follow-up questions or conversational remarks.
7. Return only the explanation.

Write a complete academic answer below.
    """
    return run_local_llm(prompt)


# -------------------------------------------------------
# PIPELINE: SEARCH → FETCH → RANK → SUMMARIZE → ANSWER
# -------------------------------------------------------

def run_pipeline(query: str, max_results: int = 10):
    # 1. Search
    search_results = google_search(query, max_results=max_results)
    if not search_results:
        return None

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

    if not usable_pages:
        return {
            "search_results": search_results,
            "used_sources": [],
            "summaries": [],
            "local_answer": "",
            "web_answer": "",
            "final_answer": "",
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

    # 5. Local answer (no web)
    local_prompt = (
        "You are to provide a factual, academic explanation to the user query. "
        "Produce only the answer. "
        "Do not include follow-up questions, suggestions, or conversational remarks. "
        "Return only the explanation.\n\n"
        f"User query: {query}"
    )
    local_answer = run_local_llm(local_prompt)

    # 6. Web-based synthesized answer
    web_answer = synthesize_web_answer(query, [s["summary"] for s in summaries])

    # 7. Fused final answer (local + web)
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
        "local_answer": local_answer,
        "web_answer": web_answer,
        "final_answer": final_answer,
    }


# -------------------------------------------------------
# STABILITY (COMPARE WITH PREVIOUS ANSWER)
# -------------------------------------------------------

def compute_stability(query: str, new_answer: str):
    conn = sqlite3.connect(DB_EVAL_PATH)
    cur = conn.cursor()
    cur.execute("SELECT answer FROM previous_answers WHERE query = ?", (query,))
    row = cur.fetchone()

    prev_answer = row[0] if row else None

    # Update stored answer to new one
    cur.execute(
        "REPLACE INTO previous_answers (query, answer) VALUES (?, ?)",
        (query, new_answer),
    )
    conn.commit()
    conn.close()

    if not prev_answer:
        return None

    emb_prev = embedding_model.encode(prev_answer)
    emb_new = embedding_model.encode(new_answer)
    return cosine_sim(emb_prev, emb_new)


# -------------------------------------------------------
# STREAMLIT UI (PERPLEXITY-STYLE)
# -------------------------------------------------------

st.set_page_config(page_title="Local Web-Augmented AI", layout="wide")

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""
if "last_stability" not in st.session_state:
    st.session_state["last_stability"] = None

st.title("Local Web-Augmented AI")

left_col, center_col, right_col = st.columns([1, 2, 1])

# ---------------- LEFT PANEL ----------------
with left_col:
    st.subheader("Query and Settings")

    query = st.text_area(
        "Enter your query:",
        value=st.session_state.get("last_query", ""),
        height=120,
        key="query_input",
    )

    academic_toggle = st.checkbox(
        "Rewrite final answer in academic tone", value=False
    )
    debug_toggle = st.checkbox("Show debug search info", value=False)

    run_clicked = st.button("Run search and answer")
    

# Run pipeline when button clicked
if run_clicked and query.strip():
    with st.spinner("Searching the web and generating answer..."):
        result = run_pipeline(query, max_results=10)

    st.session_state["last_query"] = query

    if result is None:
        st.session_state["last_result"] = None
        st.session_state["last_stability"] = None
    else:
        # Compute stability before academic rewrite
        if result["final_answer"]:
            stability = compute_stability(query, result["final_answer"])
        else:
            stability = None

        # Optional academic rewrite
        if academic_toggle and result["final_answer"]:
            rewritten = academic_rewrite(result["final_answer"])
            result["final_answer"] = rewritten
            # Evaluate the final answer
        eval_result = evaluate_answer(
            query=query,
            final_answer=result["final_answer"],
            local_answer=result["local_answer"],
            web_answer=result["web_answer"],
            summaries=[s["summary"] for s in result["summaries"]],
            sources=result["used_sources"],
            stability=stability,
            embedding_model=embedding_model,
            llm_fn=run_local_llm,
        )

        st.session_state["last_result"] = result
        st.session_state["last_stability"] = stability
        st.session_state["eval_result"] = eval_result

# ---------------- CENTER PANEL ----------------
with center_col:
    st.subheader("Answer")

    res = st.session_state.get("last_result", None)
    if res and res["final_answer"]:
        st.markdown("### Final Answer")
        st.write(res["final_answer"])

        with st.expander("Web-based synthesized answer"):
            st.write(res["web_answer"])

        with st.expander("Local model-only answer"):
            st.write(res["local_answer"])
    elif st.session_state.get("last_query", "").strip():
        st.info("No answer could be generated from the retrieved web pages.")
    else:
        st.info("Enter a query on the left and click 'Run search and answer'.")

# ---------------- RIGHT PANEL ----------------
with right_col:
    st.subheader("Sources and Summaries")

    res = st.session_state.get("last_result", None)
    if res and res["used_sources"]:
        st.markdown("#### Top sources used")

        for p in res["used_sources"]:
            st.write(f"- [{p['title']}]({p['url']}) (score: {p.get('score', 0):.3f})")

        with st.expander("Summaries used for answering"):
            for s in res["summaries"]:
                st.markdown(f"**[{s['title']}]({s['url']})**")
                st.write(s["summary"])
                st.markdown("---")

        if debug_toggle:
            with st.expander("All search results (debug)"):
                for item in res["search_results"]:
                    st.markdown(f"- **{item['title']}**  \n{item['url']}")
    elif st.session_state.get("last_query", "").strip():
        st.info("No usable sources were extracted.")
    else:
        st.info("Sources will appear here after you run a query.")

# ---------------- LEFT PANEL (EVAL + STABILITY) ----------------
with left_col:
    st.markdown("---")
    st.subheader("Evaluation")

    res = st.session_state.get("last_result", None)
    last_q = st.session_state.get("last_query", "")

    # Stability display
    eval_result = st.session_state.get("eval_result", None)

    if res and res["final_answer"] and eval_result:
        st.markdown("### Automatic metrics")

        metrics_table = f"""
        <table style="width:100%; border:1px solid #444; border-collapse: collapse;">
        <tr style="border-bottom:1px solid #444;">
            <th style="text-align:left; padding:6px; border-right:1px solid #444;">Metric</th>
            <th style="text-align:right; padding:6px;">Score</th>
        </tr>
        <tr>
            <td style="padding:6px; border-right:1px solid #444;">Relevance</td>
            <td style="padding:6px; text-align:right;">{eval_result['relevance']:.3f}</td>
        </tr>
        <tr>
            <td style="padding:6px; border-right:1px solid #444;">Grounding</td>
            <td style="padding:6px; text-align:right;">{eval_result['grounding']['score']:.3f}</td>
        </tr>
        <tr>
            <td style="padding:6px; border-right:1px solid #444;">Completeness</td>
            <td style="padding:6px; text-align:right;">{eval_result['completeness']['score']:.3f}</td>
        </tr>
        <tr>
            <td style="padding:6px; border-right:1px solid #444;">Source reliability</td>
            <td style="padding:6px; text-align:right;">{eval_result['reliability']['score']:.3f}</td>
        </tr>
        <tr>
            <td style="padding:6px; border-right:1px solid #444;">Contradiction<br>(0=no conflict, 1=high)</td>
            <td style="padding:6px; text-align:right;">{eval_result['contradiction']['score']:.3f}</td>
        </tr>
        <tr style="border-top:1px solid #444;">
            <td style="padding:6px; border-right:1px solid #444;"><b>Overall score</b></td>
            <td style="padding:6px; text-align:right;"><b>{eval_result['overall_score']:.3f}</b></td>
        </tr>
        </table>
        """

        st.markdown(metrics_table, unsafe_allow_html=True)

    # if res and res["final_answer"] and stability is not None:
    #     st.metric("Stability (0–1)", f"{stability:.3f}")
    # elif res and res["final_answer"]:
    #     st.write("Stability: first run for this query.")

    if res and res["final_answer"]:
        with st.form("eval_form"):
            relevance = st.slider(
                "Relevance score (0 = bad, 10 = excellent)", 0, 10, 7
            )
            pair_choice = st.radio(
                "Which answer is better?",
                ["Local Answer", "Web-based Answer", "Both Similar", "Neither Good"],
                index=1,
            )
            submitted = st.form_submit_button("Submit evaluation")

            if submitted:
                conn = sqlite3.connect(DB_EVAL_PATH)
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO evaluation_scores (query, relevance_score) VALUES (?, ?)",
                    (last_q, relevance),
                )
                cur.execute(
                    "INSERT INTO evaluation_pairwise (query, winner) VALUES (?, ?)",
                    (last_q, pair_choice),
                )
                conn.commit()
                conn.close()
                st.success("Evaluation saved.")
    else:
        st.info("Run a query to enable evaluation.")
