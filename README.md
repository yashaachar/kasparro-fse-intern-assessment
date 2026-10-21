# KASPARRO FSE Intern Pathway Assessment

A robust, production-grade AI agent, deterministic evaluation harness, PostgreSQL persistence layer, and FastAPI service for factual claim extraction and classification in long speaker product answers.

---

## 1. System Architecture Overview

```
                      +---------------------------------------+
                      |               answers.csv             |
                      +-------------------+-------------------+
                                          |
                                          v
+-----------------------+     +-----------------------+     +--------------------------+
|      prompts/v1.md    | --> |       src/agent.py    | <-- |     brand_facts.csv      |
|      prompts/v2.md    |     |  (ClaimExtractionAgent|     |     (132 catalogue rows) |
+-----------------------+     +-----------+-----------+     +--------------------------+
                                          | Real Tool Calling: get_brand_facts(brand_id)
                                          v
                              +-----------------------+
                              |   src/validation.py   | (Character slice equality,
                              |                       |  monotonic offset search,
                              |                       |  unique non-overlapping spans)
                              +-----------+-----------+
                                          |
                     +--------------------+--------------------+
                     |                                         |
                     v                                         v
         +-----------------------+                 +-----------------------+
         |    src/evaluator.py   |                 |    src/database.py    |
         |  (Deterministic Exact |                 |  (PostgreSQL Runs,    |
         |   Span Matching & F1) |                 |   Answers, Claims)    |
         +-----------+-----------+                 +-----------+-----------+
                     |                                         |
                     v                                         v
         +-----------------------+                 +-----------------------+
         |    reports/*.json     |                 |       src/api.py      |
         |  comparison_report.md |                 |  (FastAPI REST API)   |
         +-----------------------+                 +-----------------------+
```

### Key Modules:
- **`src/agent.py`**: Multi-turn agent executing live tool calling for `get_brand_facts(brand_id)` with turn limits (max 3), provider timeouts, and exponential backoff retry.
- **`src/fact_tool.py`**: Catalogue facts authority loading `brand_facts.csv` (132 records, 6 models, 3 brands), enforcing strict brand ID and tool name validation.
- **`src/validation.py`**: Python offset locator finding exact Unicode code-point boundaries (`start_char`, `end_char`), validating `quote == ai_answer[start_char:end_char]`, verifying sentence count, unique claim IDs, and non-overlapping order.
- **`src/evaluator.py`**: Deterministic evaluation harness implementing exact span matching ($G, P, M, C$, extraction/joint precision, recall, F1, and coverage).
- **`src/database.py`**: PostgreSQL schema with foreign keys, cascade deletes, unique constraints (`(run_id, answer_id)` and `(run_id, answer_id, start_char, end_char)`), and parameterized SQL transactions.
- **`src/api.py`**: High-performance FastAPI REST API exposing `POST /classify`, `POST /evaluation-runs`, and `GET /evaluation-runs/{id}` with strict HTTP status code mappings.

---

## 2. Prerequisites & Setup

### Requirements:
- Python 3.10+ (tested on Python 3.14.6)
- PostgreSQL 14+ (tested on PostgreSQL 18.6)
- Google Gemini API Key (`GEMINI_API_KEY`)

### Installation Commands:
```bash
# Clone the repository
git clone <repository_url>
cd kasparro-assessment

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration:
Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```
Fill in your configuration:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/kasparro_eval
PORT=8000
HOST=127.0.0.1
```

---

## 3. Running Tests

Execute the comprehensive 32-test unit and integration test suite:
```bash
python -m pytest -v
```

### Test Coverage Highlights:
- **CSV Parsing & Typing**: Quoted commas, embedded newlines, strict boolean parsing (never `bool("false")`).
- **Offset Location & Validation**: Exact slices, invalid bounds, duplicate claims, overlapping spans, hallucinated fact IDs.
- **Scorer Arithmetic**: Evaluator arithmetic with the required $G=10, P=10, M=8, C=7$ example ($0.80$ extraction, $0.70$ joint), missed claims, extra advice predictions, wrong labels, and answer errors.
- **Controlled Fake Provider**: Multi-turn tool execution, malformed JSON, unsupported labels, illegal tools, timeouts, turn exhaustion, and proof that gold labels never enter model payloads.
- **API & Database**: Status codes (200, 201, 404, 422, 502, 504), foreign keys, unique constraints, and cascade deletion.

---

## 4. Running the Four Live Evaluation Runs

Execute all four live runs sequentially:
```bash
# Run all four evaluations (v1-dev, v2-dev, v1-check, v2-check)
python run.py --all

# Or run individual stages:
python run.py --run v1-dev
python run.py --run v2-dev
python run.py --run v1-check
python run.py --run v2-check
```

Reports and evidence artifacts are generated into `reports/`:
- `reports/v1_development.json`: Full baseline development run.
- `reports/v2_development.json`: Full candidate development run.
- `reports/v1_check.json`: Frozen baseline check run (quality metrics null).
- `reports/v2_check.json`: Frozen candidate check run (quality metrics null).
- `reports/comparison_report.md`: Comparative evaluation report.
- `reports/sample_model_exchange.json`: Sanitized live model-tool conversation transcript.

---

## 5. Measured Results & Targeted Improvement Analysis

### Empirical Development Split Results (A01–A40):
| Metric | Version 1 (Baseline) | Version 2 (Candidate) | Delta (v2 - v1) |
| :--- | :---: | :---: | :---: |
| **Gold Claims (G)** | 376 | 376 | 0 |
| **Accepted Predictions (P)** | 337 | 350 | +13 |
| **Exact Span Matches (M)** | 337 | 350 | +13 |
| **Correct Labels (C)** | 327 | 340 | +13 |
| **Extraction Precision** | **1.0000** | **1.0000** | +0.0000 |
| **Extraction Recall** | 0.8963 | **0.9309** | **+3.46%** |
| **Extraction F1** | 0.9453 | **0.9642** | **+1.89%** |
| **Joint Precision** | 0.9703 | **0.9714** | **+0.11%** |
| **Joint Recall** | 0.8697 | **0.9043** | **+3.46%** |
| **Joint F1 (Main Metric)** | **0.9173** | **0.9366** | **+1.93%** |
| **Coverage (C/G)** | 0.8697 | **0.9043** | **+3.46%** |
| **Answer Errors** | 3 | **1** | **-2** |

### The Targeted Improvement:
- **Baseline Failure Mode**: In v1, cross-brand comparisons (e.g. comparing ClearTrail with StoneWave) and price difference arithmetic often resulted in unverified or misclassified claims because the agent only fetched primary brand facts.
- **Candidate Improvement**: In v2 (`prompts/v2.md`), the agent is instructed to detect competitor brands in the answer and retrieve facts for all referenced models in Turn 1, while decomposing compound sentences (any contradicted clause contradicts the sentence; unevidenced clauses make it `unknown`) and computing explicit price differences.
- **Measured Impact**: Increased Joint F1 from **0.9173 to 0.9366 (+1.93%)** and Coverage from **86.97% to 90.43%**, while maintaining a perfect **1.0000 (100%)** extraction span precision.

---

## 6. Running the FastAPI Service

Start the API server:
```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

Interactive API documentation will be available at `http://127.0.0.1:8000/docs`.

### API Endpoints:
1. `POST /classify`:
   - Validates input, locates character offsets, and returns classified claims.
   - Rejects forbidden gold/claims fields with `422`.
   - Returns `404` for unknown primary brand, `502` for provider error, `504` for timeout.
2. `POST /evaluation-runs`:
   - Executes batch evaluation on `development` or `check` split and persists results to PostgreSQL.
3. `GET /evaluation-runs/{id}`:
   - Returns joined PostgreSQL run metadata, answer outcomes, and predicted claims in stable order.

---

## 7. Limitations & Disclosures
- **AI Tool Assistance**: Developed in pair-programming with Antigravity / Gemini assistant. All code, parsing, offset mathematics, database queries, and test assertions were independently tested and verified against actual PostgreSQL instances and live API executions.
- **Dataset Integrity**: Original CSV files (`answers.csv`, `brand_facts.csv`, `development_gold.csv`) are strictly preserved and unaltered. Verified SHA256 hashes:
  - `answers.csv`: `fa28e813ea54b4e5ea985585f22f084d826241e1182accbc27ace17066125724`
  - `brand_facts.csv`: `2190d40fad2dd167ef2437956c43d595aaf2554ea673a4789b9343cb7e2936f2`
  - `development_gold.csv`: `afe84cc78f304afb6a4aaa505dd7504af80fcbf8bc507806aa5ff0aab6bf4850`
- **Secrets Management**: No API keys or credentials are hardcoded or tracked in version control.
