# KASPARRO FSE Intern Pathway Assessment - Evaluation Report

## Executive Summary
This report presents the empirical findings from four live evaluation runs comparing **Version 1 (Baseline)** and **Version 2 (Candidate)** across both the **Development Split (A01–A40, 376 reference gold claims)** and the **Final-Check Split (A41–A50)**.

### Target Improvement Rationale
- **Version 1 Baseline**: Core single-sentence factual claim extraction with primary brand retrieval via `get_brand_facts(brand_id)`.
- **Targeted Failure Diagnosis**: In v1, cross-brand comparison sentences (e.g., comparing ClearTrail with StoneWave or CloudNote) and compound clauses with price arithmetic/unit conversions frequently failed because the agent only fetched primary brand facts and lacked explicit rules for linked clause refutation.
- **Version 2 Targeted Change**: Explicit cross-brand competitor retrieval in Turn 1, strict clause-level truth decomposition (any contradicted clause makes the whole sentence `contradicted`; otherwise any unevidenced clause makes it `unknown`), and arithmetic resolution for price comparisons.

---

## 1. Development Split Results (A01–A40)

| Metric | Version 1 (Baseline) | Version 2 (Candidate) | Delta (v2 - v1) |
| :--- | :---: | :---: | :---: |
| **Gold Claims (G)** | 376 | 376 | 0 |
| **Accepted Predictions (P)** | 337 | 350 | +13 |
| **Exact Span Matches (M)** | 337 | 350 | +13 |
| **Correct Labels (C)** | 327 | 340 | +13 |
| **Extraction Precision** | 1.0 | 1.0 | +0.0000 |
| **Extraction Recall** | 0.8963 | 0.9309 | +0.0346 |
| **Extraction F1** | 0.9453 | 0.9642 | +0.0189 |
| **Joint Precision** | 0.9703 | 0.9714 | +0.0011 |
| **Joint Recall** | 0.8697 | 0.9043 | +0.0346 |
| **Joint F1 (Main Metric)** | **0.9173** | **0.9366** | **+0.0193** |
| **Coverage (C/G)** | 0.8697 | 0.9043 | +0.0346 |
| **Answer Errors** | 3 | 1 | -2 |

---

## 2. Final-Check Split Summary (A41–A50)
*Per assessment rules, ground truth is held private by the hiring team. Quality metrics (G, M, C, precision, recall, F1, coverage) remain strictly null.*

| Metric | Version 1 (Frozen) | Version 2 (Frozen) |
| :--- | :---: | :---: |
| **Answers Scheduled** | 10 | 10 |
| **Accepted Predictions (P)** | 89 | 86 |
| **Answer Errors** | 0 | 1 |
| **Duration (seconds)** | 70.94s | 70.45s |
| **Quality Scores** | *NULL (Evaluation pending review)* | *NULL (Evaluation pending review)* |

---

## 3. Analysis of Targeted Improvement & Regressions
- **Extraction Span Consistency**: Both versions maintained high span extraction fidelity, accurately locating exact Unicode offsets without relying on model-provided character counting.
- **Classification Performance**: v2 demonstrates improved accuracy on cross-brand comparison and price differential claims due to multi-brand retrieval and explicit arithmetic evaluation.
- **Regressions / Boundary Conditions**: On certain nuanced policy scopes (e.g. warranty repair turnaround vs parts availability), unevidenced specifications occasionally required strict fallback to `unknown`.

## 4. Run Metadata & Integrity
- **Model**: gemini-3.5-flash-lite
- **Dataset Version**: long-brand-answers-50-v4
- **Fact Version**: speaker-catalogue-v4
- **Evaluator Version**: sentence-spans-and-labels-v4
- **File Hashes**:
  - `answers.csv`: `fa28e813ea54b4e5ea985585f22f084d826241e1182accbc27ace17066125724`
  - `brand_facts.csv`: `2190d40fad2dd167ef2437956c43d595aaf2554ea673a4789b9343cb7e2936f2`
  - `development_gold.csv`: `afe84cc78f304afb6a4aaa505dd7504af80fcbf8bc507806aa5ff0aab6bf4850`
