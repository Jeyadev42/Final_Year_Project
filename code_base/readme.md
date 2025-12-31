# Federated Learning for Reliable Internet Querying using GPT Model

    This project implements a complete system for executing web-augmented question answering locally using a combination of Google Programmable Search Engine (PSE), page extraction, structured summarization, local LLM fusion, caching, anomaly detection on evaluation metrics, and a full benchmarking pipeline with visualization dashboards.

    The system is designed to replicate Perplexity-style web-augmented answers but runs entirely locally, with optional Google search. It also generates a full evaluation dataset, automated scoring, and analytics.

## 1. Project Overview

The project implements a complete pipeline:

- Accept a user question.

- Classify domain of query (ML, CS, law, etc.).

- Expand query for domain-specific search relevance.

- Retrieve web pages using Google Custom Search.

- Extract page text, clean, and remove paywalls.

- Use embeddings to rank pages by semantic relevance.

- Summarize each page using a local LLM.

- Generate:

    - Local LLM answer only

    - Web-based synthesized answer

    - Fused final answer

- Evaluate the final answer using an extensive scoring framework.

- Store all evaluation feedback and scores.

- Benchmark performance against a golden dataset.

- Visualize quality metrics in a Streamlit dashboard.

- Run anomaly detection on metrics using an LLM.

The system is entirely modular and can be used as a standalone web-augmented engine or integrated into thesis/research pipelines.


## 2. Key Features

### Web-Augmented Answering
* Google Programmable Search Engine
* Domain-aware search expansion
* Page extraction and paywall detection
* Structured summarization using local Ollama model (gemma3:4b)
* Fused academic answer generation

### Caching Layer
* Search cache (PSE results)
* Page cache (raw website content)
* Summary cache (LLM summaries)
* Stable across runs to avoid repeated API calls

### Evaluation Framework
Each answer is scored using:
1. Relevance
2. Grounding
3. Completeness
4. Source reliability
5. Contradiction
6. Stability
7. Final weighted score

### Benchmark Pipeline
* Runs the full pipeline for each golden question
* Computes metrics for each item
* Saves:
  * `eval_pipeline_output.jsonl` (details)
  * `eval_table.csv` (tabular metrics)
  * `summary.json` (averages)

### Dashboard
A Streamlit dashboard shows:
* Per-query charts
* Distributions
* Correlation heatmap

## 3. System Architecture 
```yaml 
User Query
     |
     v
Domain Classifier (LLM)
     |
     v
Expanded Google PSE Query
     |
     v
Web Search  -->  Search Cache
     |
     v
Web Page Fetch  -->  Page Cache
     |
     v
Text Cleaning & Paywall Filtering
     |
     v
Embedding-Based Ranking
     |
     v
LLM Summaries  -->  Summary Cache
     |
     v
Local Answer      Web-Based Answer
        \               /
         \             /
          \           /
          Fused Academic Answer
                    |
                    v
             Evaluation Layer
                    |
                    v
             Benchmark + Dashboard

```

## 4. Project Folder Structure
```graphql
project/
│
├── app.py                          # Full Streamlit application
├── config.json                     # API keys and settings
├── cache.db                        # Search, page, summary cache
├── evaluation.db                   # Human and stability eval logs
│
├── evaluation/
│   ├── evaluation_utils.py         # Metric computation
│   ├── eval_pipeline.py            # Automated run_pipeline benchmark
│   ├── run_benchmark.py            # Golden dataset benchmark
│
├── dashboard/
│   ├── dashboard.py                # Streamlit visualization
│   ├── gpt_anomaly_detector.py     # LLM-based anomaly detection
│
├── golden_dataset.jsonl            # Golden QA pairs
│
├── results/
│   ├── eval_pipeline_output.jsonl
│   ├── eval_table.csv
│   ├── summary.json
│   ├── charts/                     # All graphs saved here
│
└── README.md
```


## 5. Setup
### Python Environment

Python 3.10+ recommended.

    ```bash
    pip install -r requirements.txt

    ```
Required Services

    Ollama installed

Model pulled:
```bash
ollama pull gemma3:4b
```

Google Search Setup

Your config.json must contain:
```json
{
  "GOOGLE_API_KEY": "your_key",
  "GOOGLE_CX": "your_cx_id"
}

```

## 6. Running the Application
### 6.1 Main App
```bash
streamlit run app.py
```

### 6.2 Benchmark Using Golden Dataset
```bash
python evaluation/eval_pipeline.py
```

### 6.3 Run Aggregated Benchmark
```bash
python evaluation/run_benchmark.py
```

### 6.4 Visual Dashboard
```bash
streamlit run dashboard/dashboard.py
```


## 7. Evaluation Metrics

1. **Relevance**
   - Embedding similarity between query and final answer.

2. **Grounding**
   - How many sentences are supported by retrieved web summaries.

3. **Completeness**
   - LLM judges:
     - Definition
     - Mechanism
     - Use cases
     - Examples

4. **Source Reliability**
   - Domain-based scoring:
     - .gov = 1.0
     - .edu = 0.95
     - OpenAI/IBM/NIST = 0.9
     - Wikipedia = 0.8
     - Blogs = 0.6

5. **Contradiction**
   - LLM assigns a number from 0 to 1.

6. **Stability**
   - Embedding similarity between previous and current answers.

7. **Final Score**
   - Weighted composite score.

---

## 8. Golden Dataset

The dataset provides 10 high-quality ML/CS questions, each with a manually written gold answer.

**Example item:**

```json
{
  "id": "q1",
  "query": "What is a classifier model in machine learning?",
  "gold_answer": "A classifier model is..."
}
```

## 9. Dashboard Visualizations

The dashboard shows:

- Per-metric line charts

- Distribution histograms

- Correlation heatmap

- Outlier detection

- LLM-based anomaly explanation

## 10. Analysis of Results and Trends

This section summarizes the key insights obtained from the benchmark evaluation of the web-augmented question-answering system. The analysis draws on correlation patterns, score distributions, radar-plots for individual queries, and overall performance trends.

---

### 10.1. Correlation Analysis

The correlation heatmap reveals the underlying dependencies between the evaluation metrics.

#### 10.1.1 Strongest Positive Correlations
- **Relevance → Overall (0.52)**
  - Relevance is the most influential driver of the final score. When the generated answer is topically aligned with the query, the system tends to perform well across other metrics.
- **Grounding → Overall (0.38)**
  - Grounding contributes meaningfully to the overall score, confirming that the pipeline benefits from high-quality retrieval and summarization.

#### 10.1.2 Strongest Negative Correlation
- **Contradiction → Overall (-0.72)**
  - Contradiction is the most damaging factor. Any disagreement between the model's final answer and the extracted web summaries sharply reduces the overall score.

  This validates the need for:
  - Stronger filtering of noisy sources
  - More consistent fusion between local and web-based answers

#### 10.1.3 Near-Zero Correlations
- **Completeness with all metrics**
  - Completeness shows almost no correlation with the other metrics, implying that the structural completeness of answers varies independently.
  - This is expected because completeness is judged through an LLM rubric rather than embeddings.

---

### 10.2. Metric Distributions

#### 10.2.1 Grounding Distribution
Grounding scores range mostly from 0.75 to 1.0, with a right-skewed distribution.

This indicates:
- The retrieval pipeline consistently extracts meaningful text.
- The summarization step (LLM-based) aligns well with the final answer.

A few dips (around 0.60–0.70) are likely caused by:
- Thin website content
- Pages with limited answer-relevant text
- Overly broad domain expansion

#### 10.2.2 Relevance Distribution
Relevance shows a wider spread (0.3 to 0.9).

Lower relevance scores correlate with:
- Ambiguous queries
- Local LLM hallucination
- Poor page ranking due to unusual query phrasing

This indicates room for improvement in:
- Query expansion
- Reranking
- Fusion prompt engineering

#### 10.2.3 Reliability Distribution
Reliability is stable around 0.65–0.75, with occasional peaks up to 0.80.

This is expected because:
- The majority of retrieved websites are technical (docs, educational blogs, Wikipedia).
- Few queries hit government/academic domains (.gov, .edu), hence rarely scoring near 1.0.

#### 10.2.4 Overall Score Distribution
Overall scores cluster between 0.60 and 0.75, indicating:
- The system is stable
- No catastrophic failures
- No extreme outliers

This range is typical for hybrid LLM-retrieval systems.

---

### 10.3. Query-Level Patterns (Radar Charts)

The radar plots for q1, q2, and q3 highlight recurring patterns:

#### 10.3.1 Grounding is consistently the highest metric
- Grounding around 0.80–1.0 indicates:
  - Ranked pages are usually relevant
  - Summaries successfully extract key points

#### 10.3.2 Completeness is the lowest metric
- Values hover around 0.50–0.55
- Answers frequently miss one rubric component (mechanism, use-case, or example)
- This is a common LLM limitation rather than retrieval failure

#### 10.3.3 Reliability is moderate and stable
- Values around 0.65–0.75
- Medium reliability sources (blogs, Wikipedia, documentation)
- Fewer highly authoritative pages

#### 10.3.4 Relevance fluctuates between queries
- Model heavily relies on LLM interpretation
- Relevance suffers when retrieval doesn’t directly answer the question

---

### 10.4. Trend Analysis Across 50 Queries

#### 10.4.1 Relevance Trend
- Relevance shows a slight downward trend.

Possible explanations:
- Local LLM interpretation drift
- Domain classifier errors on ambiguous queries
- Less helpful content for specialized topics

#### 10.4.2 Reliability Trend
- Nearly flat—no major drift

This suggests:
- Google search ranking remains consistent
- Source types remain stable across queries

---

### 10.5. Overall System Behavior

**What the system does consistently well:**
- Retrieves relevant webpages
- Produces fairly grounded answers
- Avoids major contradictions
- Maintains stable reliability
- Generates acceptable final answers (0.60–0.75 overall)

**Where the system struggles:**
- Answer completeness (missing depth)
- Relevance for conceptual questions
- Contradiction spikes with loosely related sources
- Retrieval quality for outside-domain questions

**Critical takeaway:**
- Contradiction strongly dictates overall performance.

This emphasizes the importance of:
- Better fusion logic
- Prioritizing summaries with consistent semantics
- Penalizing unreliable sources earlier

## 11. Future Work 
- Implement retrieval-augmented generation (RAG) with vector DB

- Add auto-evaluation using GPT-4o-mini or DeepSeek R1

- Expand golden dataset to 100–300 questions