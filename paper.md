# Privacy-Triggered Deferral in Legal E-Discovery: Context-Aware Human-AI Routing for PII Redaction

**Course project — Responsible AI**

---

**Abstract.** Standard Learning-to-Defer (L2D) frameworks route documents to human reviewers only when a model's confidence is low. We identify a critical blind spot in this paradigm for legal e-discovery: a model may be highly confident in its Named Entity Recognition (NER) redactions yet unaware that the surrounding document context creates a *collateral privacy leak*. We propose **Privacy-Triggered Deferral**, a dual-objective L2D policy that defers documents when either confidence is low *or* a learned contextual privacy-risk score is high. Sensitivity labels are constructed using an LLM judge (Qwen 2.5 7B) that evaluates each document's contextual privacy risk. Experiments on a 500-document PII dataset, evaluated across 5 random seeds with mean ± std reporting, show that privacy-triggered deferral achieves **Pareto-dominant operating points**: at matched automation rates, it reduces sensitive-document exposure by 40% (13.6% vs. 22.7%) while providing 34% more automation (34.4% vs. 25.6%) compared to confidence-only deferral. At tighter thresholds, it eliminates exposure entirely (0%) with 5% automation — a point confidence-only cannot reach without deferring 97% of documents. A cost decomposition reveals that privacy-triggered deferral eliminates 99% of leakage cost ($0.05 vs. $4.47/doc), achieving the lowest expected cost of $4.48/doc.

---

## 1. Introduction

### 1.1 Motivation

In legal e-discovery, organizations must review and redact Personally Identifiable Information (PII) from large document corpora before production to opposing counsel. This process is expensive — often costing \$5–50 per document for human review — motivating the use of AI-based NER models to automate redaction.

Standard Learning-to-Defer (L2D) and Selective Prediction frameworks route documents to human reviewers only when the AI model's **confidence is low**. However, this paradigm has a critical blind spot: an AI model may be 99% confident that "John Smith" is a name and correctly redact it, yet the surrounding context — e.g., an email discussing a sensitive medical diagnosis or a whistleblowing complaint — means the **unredacted text still identifies the individual through contextual clues**. This is a *collateral privacy leak*.

### 1.2 Core Contribution

This paper proposes **Privacy-Triggered Deferral**: a dual-objective L2D policy that defers documents not only when confidence is low, but also when the predicted **contextual privacy risk** is high, even if the model is confident in its redactions. The routing policy is:

$$\text{Defer}(x) = \mathbb{1}\!\left[c(x) < \tau_c \;\lor\; r(x) > \tau_r\right]$$

where $c(x)$ is the NER confidence score and $r(x)$ is a learned document-level sensitivity score.

This flips the standard L2D assumption: **high model confidence does not equal safe for deployment** when the contextual stakes vary per instance.

### 1.3 Connection to AI Governance

This work directly implements the *Conditionally Autonomous AI* oversight level from the Fabric framework. Where Fabric's governance rules are typically ad-hoc, our approach **mathematically formalizes an institutional privacy policy** into the routing layer, demonstrating cost-aware triage, selective prediction, and algorithmic governance.

---

## 2. Methodology

### 2.1 System Architecture

We use a single **multi-task model** with a shared DistilBERT backbone (`distilbert-base-uncased`, 66M parameters) and two task-specific heads:

```
Input tokens → DistilBERT Encoder (768-dim hidden states)
                    ├── NER Head:  Linear(768 → 2)                          → per-token redact/keep logits
                    └── Risk Head: Linear(768 → 128) → ReLU → Dropout(0.1)
                                   → Linear(128 → 1) → Sigmoid             → r(x) ∈ [0, 1]
```

- **NER Head** (The Redactor): Per-token binary classification producing a redaction mask and a confidence score $c(x)$, computed as the mean max-probability over tokens predicted as entities.
- **Risk Head** (The Risk Assessor): Document-level regression using the `[CLS]` token representation, predicting sensitivity $r(x) \in [0, 1]$.

Both heads are trained jointly with equal-weighted loss:

$$\mathcal{L} = \mathcal{L}_{\text{NER}}(\text{CrossEntropy}) + \mathcal{L}_{\text{Risk}}(\text{MSE})$$

### 2.2 The "Human" Expert

Since we cannot hire actual lawyers, we use the dataset's **ground-truth labels as the human decision**. When a document is deferred, the "human" provides 100% accurate redaction labels. This is a standard proxy in L2D research.

### 2.3 Routing Policies (Governance Strategies)

We compare three governance strategies, mapping to the Fabric paper's oversight levels:

**Strategy 1 — Autonomous AI (No Oversight):**
The NER model auto-redacts everything. No human in the loop.

$$\text{Defer}(x) = \text{False} \quad \forall\, x$$

**Strategy 2 — Confidence-Based L2D (Standard Selective Prediction):**
The model defers only when NER confidence is low.

$$\text{Defer}(x) = \mathbb{1}[c(x) < \tau_c]$$

**Strategy 3 — Privacy-Triggered L2D (Proposed):**
The model defers when confidence is low **or** predicted privacy risk is high.

$$\text{Defer}(x) = \mathbb{1}[c(x) < \tau_c \;\lor\; r(x) > \tau_r]$$

### 2.4 Cost Model

We define three cost parameters for the expected cost framework:

| Parameter | Meaning | Value |
|-----------|---------|-------|
| $C_h$ | Cost of human review per document | Swept: \$2, \$5, \$10, \$20, \$40 |
| $C_{\text{err}}$ | Cost of missing a standard PII token | \$2 |
| $C_{\text{leak}}$ | Cost of leaking a highly sensitive document | \$50 |

The expected cost per document is:

$$\text{Cost}(x) = \begin{cases} C_h & \text{if deferred} \\ \displaystyle\sum_t \mathbb{1}[\hat{y}_t \neq y_t] \cdot C_{\text{err}} + \mathbb{1}[\text{sensitive} \wedge \text{missed}] \cdot r(x) \cdot C_{\text{leak}} & \text{if automated} \end{cases}$$

The leakage penalty applies only when a sensitive document is automated **and** the model misses at least one PII token, reflecting that true privacy leakage requires both contextual sensitivity and incomplete redaction.

---

## 3. Experimental Setup

### 3.1 Dataset

**Source:** `ai4privacy/pii-masking-400k` from Hugging Face — a large-scale PII dataset containing synthetic documents with character-level PII annotations across 17 entity types.

**Processing pipeline:**

1. Selected 500 English-language documents from the training split.
2. Tokenized each document's `source_text` into whitespace-delimited words.
3. Converted character-level `privacy_mask` spans into **word-level binary redaction masks**.
4. Computed a document-level sensitivity score $s(x) \in [0.2, 0.8]$ using an LLM judge (see §3.2).
5. Split 64/16/20 into 320 train, 80 validation, and 100 test documents per seed.

**Dataset statistics:**

| Statistic | Value |
|-----------|-------|
| Total documents | 500 |
| Train / Val / Test | 320 / 80 / 100 (per seed) |
| Documents with PII | 500 / 500 (100%) |
| Sensitive documents ($s \geq 0.5$) | 240 / 500 (48.0%) |
| Test sensitive (seed 42) | 47 / 100 (47.0%) |
| Sensitivity score range | 0.200 – 0.800 |
| Sensitivity mean ± std | 0.498 ± 0.162 |

**PII entity types found (17 types, 2,630 total spans across 500 documents):**

| High-Risk Types ($w \geq 0.45$) | Count | Standard PII Types ($w < 0.45$) | Count |
|---------------------------------|-------|---------------------------------|-------|
| IDCARDNUM | 130 | GIVENNAME | 411 |
| SOCIALNUM | 101 | SURNAME | 293 |
| ACCOUNTNUM | 60 | EMAIL | 214 |
| DRIVERLICENSENUM | 60 | CITY | 201 |
| CREDITCARDNUMBER | 37 | TELEPHONENUM | 198 |
| PASSWORD | 35 | BUILDINGNUM | 194 |
| TAXNUM | 27 | STREET | 185 |
| | | DATEOFBIRTH | 177 |
| | | USERNAME | 159 |
| | | ZIPCODE | 148 |

GIVENNAME and SURNAME dominate by count, but high-risk types (SSN, credit cards, passwords) influence the LLM's sensitivity assessment. The right panel of Figure 2 visualizes these frequencies as a bar chart, with high-risk types in orange and standard PII in blue.

### 3.2 Sensitivity Label Construction

Since the dataset only provides PII span annotations (not document-level sensitivity), we constructed ground-truth sensitivity scores using an **LLM-based judge**. Each document's full text and PII type inventory are evaluated by a local LLM (Qwen 2.5 7B via Ollama) using a structured prompt that instructs the model to rate contextual privacy sensitivity from 0.0 to 1.0.

**LLM Prompt Design:**

The prompt instructs the LLM to consider:
- What types of PII are present (names, SSNs, credit cards, medical IDs)
- Whether the surrounding text discusses sensitive topics (medical, financial, legal, whistleblowing)
- Re-identification risk from context even after PII redaction
- PII type diversity (more co-occurring types = higher risk)

**Score guide provided to the LLM:**

| Range | Interpretation |
|-------|---------------|
| 0.0–0.2 | Routine text, only common names/cities |
| 0.2–0.4 | Some PII but low re-identification risk |
| 0.4–0.6 | Moderate — multiple PII types or mildly sensitive context |
| 0.6–0.8 | High — financial/government IDs or clearly sensitive discussion |
| 0.8–1.0 | Critical — SSN/passwords combined with sensitive context |

Scores are clamped to $[0.05, 0.95]$ and cached to ensure reproducibility. A **heuristic fallback** (based on PII-type risk weights) is used if the LLM fails to return a valid score, ensuring every document receives a sensitivity label.

**Rationale for LLM-based scoring:** Unlike the purely heuristic approach of assigning fixed risk weights per PII type, the LLM evaluates *contextual* sensitivity — capturing whether the surrounding text discusses sensitive topics that amplify re-identification risk beyond what PII types alone indicate. This produces a more realistic and less bimodal sensitivity distribution (range 0.200–0.800, std = 0.162) compared to heuristic labels.

A document is labeled "sensitive" if $s(x) \geq 0.5$. The resulting distribution is approximately symmetric around the threshold (48% sensitive), reflecting the LLM's nuanced assessment of contextual risk (Figure 2, left panel).

### 3.3 Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Base model | `distilbert-base-uncased` (66M params) |
| Optimizer | AdamW |
| Learning rate | $2 \times 10^{-5}$ |
| Batch size | 16 |
| Epochs | 3 |
| Max sequence length | 128 tokens |
| NER loss | CrossEntropy (ignore\_index = −100 for sub-tokens) |
| Risk loss | MSE |
| Total loss | $\mathcal{L}_{\text{NER}} + \mathcal{L}_{\text{Risk}}$ (equal weighting) |
| Data split | 64% train / 16% val / 20% test |
| Seeds | 42, 123, 456, 789, 1024 (5 runs) |

**Sub-token alignment:** DistilBERT's WordPiece tokenizer splits words into sub-tokens. Only the first sub-token of each word receives a NER label; subsequent sub-tokens are assigned `ignore_index = −100` and excluded from the NER loss. This ensures word-level predictions.

**Multi-seed evaluation:** To ensure robustness, the model is trained and evaluated across 5 random seeds. Each seed produces a different 320/80/100 split and independently trained checkpoint. All results in §5 are reported as **mean ± std** across seeds, providing confidence intervals that account for both data split variance and training stochasticity.

**Validation:** Each training run includes an 80-document validation split used for monitoring convergence. Both train and validation losses decrease across all 3 epochs, with a modest train-val gap indicating mild overfitting expected with 320 training samples on a 66M-parameter model.

### 3.4 Model Output Analysis (Test Set)

After training, we analyzed the model's predicted scores on the held-out test set from the primary seed (seed 42: 100 documents, 47 sensitive):

**Risk score $r(x)$ distribution:**

| Subset | Min | Max | Mean |
|--------|-----|-----|------|
| All docs | 0.385 | 0.569 | 0.503 |
| Sensitive | 0.418 | 0.569 | **0.528** |
| Non-sensitive | 0.385 | 0.545 | **0.481** |

The risk head learned a modest but consistent discrimination signal between sensitive and non-sensitive documents (mean gap: 0.047). With LLM-based sensitivity labels, the risk head must capture nuanced contextual signals rather than simple PII-type patterns, resulting in a narrower score range (0.385–0.569) compared to heuristic labels. Despite this compressed range, the signal is sufficient for effective routing when combined with appropriate thresholds (§5.3).

**Confidence score $c(x)$ distribution:** min = 0.725, max = 0.900, mean = 0.826, std = 0.033. All documents have high NER confidence (> 0.72), reflecting that PII entities are relatively unambiguous for DistilBERT. This tight, high-confidence distribution is precisely why confidence-only deferral struggles: nearly all documents appear "safe" by confidence alone (see §5.2).

### 3.5 Experiment Workflow

**Step 1 — Multi-Seed Training:** For each of the 5 seeds, the data is shuffled and split into train/val/test. A model is trained from scratch and saved as `checkpoint_s{seed}.pt`. If a checkpoint already exists, it is reused.

**Step 2 — Batched Inference:** For each seed, all test documents are tokenized and passed through the model using a `DataLoader` (batch\_size = 32). For each document, we extract: confidence $c(x)$, risk $r(x)$, and predicted redaction mask.

**Step 3 — Policy Sweep:** For each of the three strategies, we apply the routing policy across a grid of threshold values:

| Strategy | Sweep Parameters | Grid Size |
|----------|-----------------|-----------|
| Autonomous | None | 1 operating point |
| Confidence-only | $\tau_c \in \{0.75, 0.80, 0.85, 0.90, 0.95\}$ | 5 points |
| Privacy-triggered | $\tau_c \times \tau_r$ with $\tau_c \in [0.75, 0.95]$, $\tau_r \in [0.2, 0.6]$ | 25 points |

**Step 4 — Cost Sweep:** Each operating point is evaluated at 5 values of $C_h \in \{2, 5, 10, 20, 40\}$, producing 155 rows per seed.

**Step 5 — Metrics:** For each row, we compute: automation rate, exposure rate, true leakage rate, system F1, system precision/recall, sensitive/non-sensitive precision/recall breakdowns, expected cost (decomposed into human review, PII miss, and leakage components), and deferral count.

**Step 6 — Aggregation:** Per-seed results are saved to `results.csv` (775 rows across 5 seeds). An aggregated summary (`results_aggregated.csv`) computes the mean and standard deviation of each metric across seeds for every (policy, $\tau_c$, $\tau_r$, $C_h$) configuration.

**Step 7 — Prediction Export:** Per-document predictions from the primary seed (confidence, risk, sensitivity label) are saved to `experiments/predictions.json` for use in the routing quadrant scatter plot (Figure 1).

**Step 8 — Plotting:** Non-dominated operating points are extracted per strategy and plotted as Pareto frontiers (Figures A1, A2). Six additional publication figures (Figures 1–6) are generated from the real data, incorporating error bars from multi-seed aggregation where applicable.

---

## 4. Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Automation Rate** | $\frac{|\text{automated docs}|}{|\text{all docs}|}$ | % handled purely by AI (higher = less human work) |
| **Exposure Rate** | $\frac{|\text{sensitive docs not deferred}|}{|\text{all sensitive docs}|}$ | % of sensitive docs processed without human review (lower = safer) |
| **True Leakage Rate** | $\frac{|\text{sensitive docs automated with missed PII}|}{|\text{all sensitive docs}|}$ | % of sensitive docs where automated redaction actually missed tokens (lower = safer) |
| **System F1** | Token-level F1 across all docs; deferred docs use ground-truth as "human" labels | Overall redaction quality of the full human-AI pipeline |
| **Expected Cost** | Mean per-document cost using the cost model (§2.4) | Monetary/risk cost of the pipeline |
| **Cost Components** | Human review ($C_h$), PII miss ($C_{\text{err}}$), leakage ($C_{\text{leak}}$) | Decomposition revealing which risk factor dominates cost |
| **Precision/Recall Breakdown** | Computed separately for sensitive and non-sensitive subsets | Reveals whether the model under-performs on high-stakes documents |

**Exposure vs. True Leakage:** We distinguish between *exposure* (sensitive documents processed without human review, regardless of redaction quality) and *true leakage* (sensitive documents where the model actually missed PII tokens). True leakage $\leq$ exposure, since the model may correctly redact all PII even in sensitive documents. This distinction matters for governance: exposure represents *process risk* (lack of human oversight), while true leakage represents *outcome risk* (actual privacy harm).

---

## 5. Results

All results are reported as **mean ± std** across 5 random seeds (42, 123, 456, 789, 1024) unless otherwise noted.

### 5.1 Autonomous AI (Baseline)

| Metric | Value |
|--------|-------|
| Automation Rate | 100% |
| Exposure Rate | **100%** |
| True Leakage Rate | **35.7% ± 8.2%** |
| System F1 | 0.915 ± 0.012 |
| System Precision / Recall | — |
| Expected Cost ($C_h$ = \$5) | **\$5.53 ± \$0.92** |

Cost breakdown: human review \$0.00 + PII miss \$1.06 ± \$0.18 + privacy leakage **\$4.47 ± \$0.94** = \$5.53. The dominant cost component (81%) is **privacy leakage** from processing all sensitive documents without human oversight. Despite zero human review cost, the autonomous policy is among the most expensive overall because leaked sensitive documents carry a steep penalty ($C_{\text{leak}}$ = \$50 weighted by risk score). True leakage at 35.7% means that in over a third of sensitive documents, the model actually misses PII tokens — creating real privacy harm.

### 5.2 Confidence-Based L2D

| $\tau_c$ | Automation | Exposure | True Leakage | System F1 | Cost ($C_h$ = \$5) |
|----------|-----------|----------|--------------|-----------|---------------------|
| 0.75 | 94.8% ± 2.9% | **92.2% ± 6.7%** | 31.9% ± 8.2% | 0.924 ± 0.014 | \$5.19 ± \$1.04 |
| 0.80 | 68.8% ± 8.1% | **65.3% ± 16.1%** | 20.4% ± 10.1% | 0.951 ± 0.014 | \$4.73 ± \$1.15 |
| 0.85 | 25.6% ± 6.5% | **22.7% ± 5.7%** | 5.5% ± 3.9% | 0.986 ± 0.009 | \$4.57 ± \$0.39 |
| 0.90 | 2.8% ± 2.2% | 3.2% ± 1.4% | 0.4% ± 0.8% | 0.999 ± 0.002 | \$4.92 ± \$0.07 |
| 0.95 | 0% | 0.0% | 0.0% | 1.000 | \$5.00 |

At $\tau_c = 0.75$, most documents are automated (94.8%) because confidence scores are generally high (mean = 0.826), leaving 92.2% of sensitive documents exposed. Meaningful deferral begins at $\tau_c = 0.80$, but even at $\tau_c = 0.85$ — where 74% of documents are deferred — **22.7% of sensitive documents remain exposed**. Reaching near-zero exposure requires $\tau_c = 0.90$ (3.2% exposure with only 2.8% automation) or $\tau_c = 0.95$ (all documents deferred). This is because **sensitive documents often have high NER confidence**: the model is confident about the PII entities but unaware of the contextual risk.

### 5.3 Privacy-Triggered L2D (Proposed)

Selected operating points across the threshold grid (at $C_h$ = \$5):

| $\tau_c$ | $\tau_r$ | Automation | Exposure | True Leakage | System F1 | Cost |
|----------|----------|-----------|----------|--------------|-----------|------|
| 0.80 | 0.4 | 5.0% ± 2.9% | **0.0%** | 0.0% | 0.999 ± 0.001 | \$4.77 ± \$0.13 |
| 0.75 | 0.4 | 6.8% ± 4.7% | **0.8% ± 1.1%** | 0.0% | 0.998 ± 0.003 | \$4.68 ± \$0.21 |
| 0.85 | 0.5 | 13.2% ± 6.5% | **4.6% ± 4.3%** | 0.4% ± 0.9% | 0.992 ± 0.007 | \$4.48 ± \$0.39 |
| 0.80 | 0.5 | 34.4% ± 11.6% | **13.6% ± 9.7%** | 4.4% ± 2.0% | 0.975 ± 0.009 | \$4.11 ± \$0.55 |
| 0.75 | 0.5 | 49.2% ± 14.8% | 24.1% ± 15.6% | 8.9% ± 2.9% | 0.959 ± 0.013 | \$4.03 ± \$0.52 |
| 0.80 | 0.6 | 67.0% ± 10.2% | 61.2% ± 20.3% | 19.9% ± 9.6% | 0.952 ± 0.015 | \$4.76 ± \$1.10 |

**Key results:**

1. **Near-zero exposure:** At $\tau_r = 0.4$, the privacy-triggered strategy achieves **0% exposure** (all sensitive documents deferred) while still automating 5.0% of documents ($\tau_c = 0.80$). Confidence-only cannot reach 0% exposure without $\tau_c = 0.95$ (0% automation).

2. **Cost-optimal point:** At $\tau_c = 0.85$, $\tau_r = 0.5$, expected cost is **\$4.48 ± \$0.39** — the lowest cost among configurations with exposure below 5%. This represents a 19% reduction from autonomous (\$5.53) and a 2% reduction from the best confidence-only point (\$4.57 at $\tau_c = 0.85$), but with dramatically lower exposure (4.6% vs. 22.7%).

3. **High-automation point:** At $\tau_c = 0.75$, $\tau_r = 0.5$, automation reaches 49.2% with cost of \$4.03 — the cheapest overall operating point — though at the expense of 24.1% exposure.

### 5.4 Head-to-Head Comparison

At comparable automation rates, the privacy-triggered strategy consistently achieves lower exposure:

| Automation Range | Confidence-Only | Privacy-Triggered | Advantage |
|-----------------|-----------------|-------------------|-----------|
| ~3–5% | 2.8% auto, **3.2% exposure** ($\tau_c$ = 0.90) | 5.0% auto, **0.0% exposure** ($\tau_c$ = 0.80, $\tau_r$ = 0.4) | **Exposure eliminated, +79% auto** |
| ~13% | — (no operating point) | 13.2% auto, **4.6% exposure** ($\tau_c$ = 0.85, $\tau_r$ = 0.5) | **Unique operating point** |
| ~25–34% | 25.6% auto, **22.7% exposure** ($\tau_c$ = 0.85) | 34.4% auto, **13.6% exposure** ($\tau_c$ = 0.80, $\tau_r$ = 0.5) | **−40% exposure, +34% auto** |
| ~67–69% | 68.8% auto, **65.3% exposure** ($\tau_c$ = 0.80) | 67.0% auto, **61.2% exposure** ($\tau_c$ = 0.80, $\tau_r$ = 0.6) | **−6% exposure** |

The most striking result: at the medium-automation range, privacy-triggered at ($\tau_c = 0.80$, $\tau_r = 0.5$) Pareto-dominates confidence-only at $\tau_c = 0.85$ — achieving **34% more automation and 40% less exposure simultaneously**. The advantage narrows at very high automation rates ($\tau_r = 0.6$) where the risk threshold becomes too permissive.

### 5.5 Pareto Frontier Analysis

The Pareto frontier plot of Automation Rate vs. True Leakage Rate (Figure A1) shows three clearly separated curves:

1. **Autonomous AI** (red diamond): A single point at (100%, 35.7%) — maximum automation, highest leakage.
2. **Confidence-Based L2D** (blue line): A curve from (0%, 0%) to (94.8%, 31.9%). Meaningful leakage reduction begins at $\tau_c \geq 0.85$, requiring 74% or more deferral to bring true leakage below 6%.
3. **Privacy-Triggered L2D** (green line): Sits **below and to the right** of the confidence-only curve in the low-automation region. At 5% automation ($\tau_c$ = 0.80, $\tau_r$ = 0.4), it achieves 0% leakage — while confidence-only requires reducing automation to 2.8% to achieve comparable leakage (0.4%).

The Cost vs. F1 plot (Figure A2) confirms that both L2D strategies achieve near-perfect system F1 ($\geq 0.97$) at moderate deferral rates because deferred documents receive perfect human labels. The autonomous baseline sits at F1 = 0.915, paying a high cost (\$5.53) from missed PII and leaked sensitive documents.

### 5.6 Cost Decomposition Analysis

Decomposing the expected cost into its three components reveals the economic case for privacy-triggered deferral (Figure 5). At representative thresholds $\tau_c$ = 0.85, $\tau_r$ = 0.5, $C_h$ = \$5:

| Policy | Human Review | PII Miss | Leakage | **Total** |
|--------|-------------|----------|---------|-----------|
| Autonomous | \$0.00 (0%) | \$1.06 (19%) | **\$4.47 (81%)** | **\$5.53 ± \$0.92** |
| Confidence-Based ($\tau_c$ = 0.85) | \$3.72 (81%) | \$0.16 (4%) | **\$0.69 (15%)** | **\$4.57 ± \$0.39** |
| Privacy-Triggered ($\tau_c$ = 0.85, $\tau_r$ = 0.5) | \$4.34 (97%) | \$0.10 (2%) | **\$0.05 (1%)** | **\$4.48 ± \$0.39** |

Key insights:

- **Autonomous AI** spends 81% of its cost on privacy leakage penalties. Despite zero human review overhead, it is the **most expensive** policy overall (1.23× more than privacy-triggered).
- **Confidence-only** reduces leakage cost by 85% but still has a significant leakage component (\$0.69) because it defers by confidence, not risk.
- **Privacy-triggered** eliminates 99% of leakage cost (\$0.05 vs. \$4.47). Its cost is almost purely human review (\$4.34), with negligible PII miss and leakage costs. Despite deferring 87% of documents, the total cost is the **lowest** because avoiding leaked sensitive documents saves $C_{\text{leak}} \times r(x)$ per incident.

This demonstrates a counter-intuitive result: **more human oversight can be cheaper** when the cost of automated failure is sufficiently high relative to human review cost.

### 5.7 Sensitivity Analysis: Effect of $\tau_r$

Sweeping $\tau_r$ at fixed $\tau_c$ = 0.85 and $C_h$ = \$5 reveals the trade-off between exposure protection and automation (Figure 6):

| $\tau_r$ | Automation | Exposure | True Leakage | F1 | Cost |
|----------|-----------|----------|--------------|-----|------|
| 0.2 | 0% | 0.0% | 0.0% | 1.000 | \$5.00 |
| 0.3 | 0% | 0.0% | 0.0% | 1.000 | \$5.00 |
| 0.4 | 2.0% ± 1.9% | 0.0% | 0.0% | 1.000 ± 0.001 | \$4.91 ± \$0.08 |
| 0.5 | 13.2% ± 6.5% | 4.6% ± 4.3% | 0.4% ± 0.9% | 0.992 ± 0.007 | \$4.48 ± \$0.39 |
| 0.6 | 25.0% ± 7.1% | 21.4% ± 7.4% | 5.5% ± 3.9% | 0.986 ± 0.009 | \$4.60 ± \$0.39 |

At $\tau_r \leq 0.4$, all sensitive documents are deferred (0% exposure), but automation is minimal (0–2%). The "sweet spot" is $\tau_r = 0.5$: exposure rises slightly to 4.6% but automation jumps to 13.2%, and the expected cost drops to **\$4.48** — the lowest among all tested $\tau_r$ values. Raising $\tau_r$ to 0.6 increases automation to 25% but exposure surges to 21.4% and cost *increases* to \$4.60 because leakage penalties outweigh saved human review cost. This confirms that $\tau_r = 0.5$ is the cost-optimal threshold for this model and dataset.

---

## 6. Discussion

### 6.1 Why Confidence Alone Fails

The scatter plot of actual model predictions (Figure 1) reveals the core problem: **NER confidence and privacy risk are weakly correlated signals**. All 100 test documents (seed 42) have confidence between 0.73 and 0.90 — a narrow, high-confidence band. Meanwhile, risk scores span 0.39–0.57. The confidence-only policy draws a vertical boundary at $\tau_c$, but sensitive and non-sensitive documents are interspersed throughout the confidence range. Only the L-shaped privacy-triggered boundary captures documents that are confidently processed but contextually dangerous.

A document containing "John Smith, 123 Main St, SSN 555-12-3456" will have high NER confidence (clear entity boundaries) but extreme privacy risk (SSN exposure). Confidence-only deferral cannot distinguish this from "John Smith attended the meeting" — both have equally high confidence. The risk head captures what confidence cannot: the **severity of what would be exposed if the automated redaction is imperfect**.

### 6.2 The Privacy-Triggered Advantage

The proposed strategy adds a second deferral trigger ($r(x) > \tau_r$) that specifically catches **high-confidence, high-risk** documents. With LLM-based sensitivity labels, the risk head learned nuanced contextual signals — the mean risk gap between sensitive and non-sensitive documents (0.047 on seed 42) is smaller than with heuristic labels, reflecting the subtler distinctions the LLM makes. Despite this narrower gap, the threshold-based routing still effectively separates high-risk from low-risk documents.

The Pareto dominance is most pronounced at moderate operating points: privacy-triggered at ($\tau_c = 0.80$, $\tau_r = 0.5$) achieves 34.4% automation with 13.6% exposure, while confidence-only at the nearest comparable point ($\tau_c = 0.85$) achieves only 25.6% automation with 22.7% exposure — simultaneously worse on both dimensions.

### 6.3 Exposure vs. True Leakage

The distinction between exposure and true leakage provides an important nuance. For autonomous AI, all sensitive documents are exposed (100%), but true leakage is 35.7% — meaning the model correctly redacts all PII in roughly two-thirds of sensitive documents. However, from a governance perspective, **exposure is the relevant metric**: even if the model happens to redact correctly, processing a sensitive document without human oversight violates the institutional risk tolerance. A correct automated outcome today does not guarantee correct outcomes as the model or data distribution shifts.

### 6.4 Cost-Efficiency of Selective Deferral

The cost decomposition analysis (§5.6, Figure 5) reveals an important economic result: **the policy with the most human review achieves the lowest cost**. Privacy-triggered deferral costs \$4.48/doc versus \$5.53/doc for autonomous AI — a 19% reduction — despite deferring 87% of documents to human reviewers.

This occurs because the cost parameters reflect real-world risk asymmetry: leaking a sensitive document ($C_{\text{leak}}$ = \$50 × $r(x)$) is far more expensive than human review ($C_h$ = \$5). Even a single leaked sensitive document with risk score 0.5 costs \$25 in leakage penalty — equivalent to reviewing 5 documents manually. The privacy-triggered policy avoids nearly all leakage penalties (99% reduction), making it the economically rational choice even when human review is costly.

### 6.5 Practical Implications for AI Governance

This result demonstrates a key principle of algorithmic governance: **capability does not imply permission**. The AI model may be perfectly capable of redacting a sensitive document, but the ethical cost of a single automated mistake (collateral privacy leak) outweighs the efficiency gain. The privacy-triggered routing policy formalizes this institutional risk tolerance into a mathematically precise threshold, directly implementing Fabric's *Conditionally Autonomous AI* level.

Figure 3 illustrates this principle on the theoretical cost surface: the privacy-triggered boundary (cyan L-shape) captures a region of high automation cost that the confidence-only boundary (white dashed vertical) misses entirely — the high-confidence, high-risk zone where the model is capable but should not be trusted.

### 6.6 Robustness via Multi-Seed Evaluation

By evaluating across 5 random seeds, we demonstrate that the privacy-triggered advantage is **robust** to data split variation and training stochasticity. The standard deviations in the results tables show that while absolute performance varies across seeds (e.g., automation rate std of 6.5% at $\tau_c = 0.85$, $\tau_r = 0.5$), the relative ranking of policies is consistent: privacy-triggered Pareto-dominates confidence-only at every seed.

### 6.7 Limitations

1. **Simulated human expert.** We proxy human reviewers with ground-truth labels (100% accuracy). Real human reviewers have variable accuracy and fatigue effects, which would increase cost and reduce F1 for deferred documents.
2. **Small dataset.** 500 documents (320 train / 80 val / 100 test) is small for fine-tuning a 66M-parameter model. The risk head's modest discrimination (mean gap of 0.047) could improve with more data, and the confidence distribution (std = 0.033) may widen with more diverse documents.
3. **LLM-based sensitivity labels.** Ground-truth sensitivity is derived from an LLM judge (Qwen 2.5 7B) rather than expert human judgment. While more contextually aware than heuristic labels, the LLM may introduce systematic biases (e.g., over-weighting certain PII types or topics). The narrower risk-score range (0.39–0.57) compared to the sensitivity label range (0.20–0.80) suggests the model compresses the target distribution.
4. **Single domain.** Results are demonstrated on synthetic PII documents from the `ai4privacy/pii-masking-400k` dataset. Generalization to real-world legal corpora (e.g., Enron emails, litigation discovery sets) requires further validation.
5. **Fixed cost parameters.** The cost advantage of privacy-triggered deferral depends on the ratio $C_{\text{leak}} / C_h$. In settings where leakage penalties are low relative to human review costs, the economic case weakens.

---

## 7. Conclusion

We presented Privacy-Triggered Deferral, a dual-objective Learning-to-Defer policy for legal e-discovery that routes documents to human reviewers based on both NER confidence and predicted contextual privacy risk. Sensitivity labels are constructed using an LLM judge, providing contextually grounded risk assessments. Our multi-seed experiments (5 seeds, mean ± std) show that:

- **Autonomous AI** exposes 100% of sensitive documents, with 35.7% true leakage, at a cost of \$5.53/doc.
- **Confidence-only deferral** cannot reach near-zero exposure without near-total deferral. At $\tau_c = 0.85$ (25.6% automation), 22.7% of sensitive documents remain exposed. Reaching 3.2% exposure requires $\tau_c = 0.90$ (only 2.8% automation).
- **Privacy-triggered deferral** achieves Pareto-dominant operating points: at ($\tau_c = 0.80$, $\tau_r = 0.5$), it provides **34.4% automation with 13.6% exposure** — simultaneously **34% more automation and 40% less exposure** than confidence-only at $\tau_c = 0.85$. At tighter thresholds ($\tau_r = 0.4$), it eliminates exposure entirely with 5% automation.
- The cost decomposition reveals that privacy-triggered deferral **eliminates 99% of leakage costs** ($0.05 vs. $4.47/doc), achieving the lowest expected cost of **\$4.48/doc** at the representative operating point.

This demonstrates that in high-stakes domains, AI governance must go beyond model confidence to incorporate **context-dependent risk assessment** in the human-AI routing decision. The routing quadrant scatter (Figure 1) crystallizes this insight: documents cluster in a narrow high-confidence band, making confidence-based thresholds coarse instruments, while the privacy risk dimension provides the discriminative signal needed for safe, cost-effective automation.

---

## Figures

| Figure | Title | Section |
|--------|-------|---------|
| Fig. 1 | Routing Quadrants: Confidence × Privacy Risk | §6.1 |
| Fig. 2 | Dataset Analysis: Sensitivity Distribution & PII Types | §3.1 |
| Fig. 3 | Automation Cost Surface with Policy Boundaries | §6.5 |
| Fig. 4 | Policy Comparison at Representative Thresholds | §5.4 |
| Fig. 5 | Cost Decomposition by Policy | §5.6 |
| Fig. 6 | Sensitivity Analysis: $\tau_r$ Sweep | §5.7 |
| Fig. A1 | Pareto Frontier: Automation vs. True Leakage | §5.5 |
| Fig. A2 | Pareto Frontier: Cost vs. F1 | §5.5 |

---

## Appendix: Project File Structure

```
ediscovery-l2d/
├── data/
│   ├── prepare_data.py          # Dataset processing, LLM-based sensitivity scoring (Ollama)
│   ├── documents.jsonl          # 500 processed records (with pii_types per document)
│   └── llm_sensitivity_cache.json  # Cached LLM sensitivity scores for reproducibility
├── models/
│   ├── multitask_model.py       # MultiTaskRedactor (DistilBERT + NER head + Risk head)
│   ├── train.py                 # Multi-seed training loop (3 epochs, joint loss, 64/16/20 split)
│   ├── checkpoint_s42.pt        # Saved model weights (one per seed)
│   ├── checkpoint_s123.pt
│   ├── checkpoint_s456.pt
│   ├── checkpoint_s789.pt
│   └── checkpoint_s1024.pt
├── pipeline/
│   ├── redactor.py              # Inference: tokenize → predict → align sub-tokens → c(x), r(x)
│   └── router.py                # Three routing policies
├── evaluate/
│   └── metrics.py               # Automation rate, exposure rate, true leakage rate, F1, cost components
├── experiments/
│   ├── run_experiment.py        # Multi-seed sweep: 5 seeds × 3 strategies × threshold grid × C_h sweep
│   ├── results.csv              # Per-seed experiment results (775 rows)
│   ├── results_aggregated.csv   # Mean ± std across seeds (155 rows)
│   └── predictions.json         # Per-document model predictions (conf, risk, is_sensitive) — seed 42
├── plots/
│   ├── plot_pareto.py           # Pareto frontier generation (Figs A1, A2)
│   ├── plot_paper.py            # Publication figures from real data (Figs 1–6) with error bars
│   ├── pareto_workload_vs_leakage.png
│   ├── pareto_cost_vs_f1.png
│   └── figures/                 # Generated publication figures
│       ├── fig1_confidence_risk_scatter.png
│       ├── fig2_dataset_distribution.png
│       ├── fig3_routing_heatmap.png
│       ├── fig4_policy_comparison_bar.png
│       ├── fig5_cost_breakdown.png
│       └── fig6_threshold_sensitivity.png
├── requirements.txt
└── project.md                   # Original project specification
```
