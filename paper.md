# Privacy-Triggered Deferral in Legal E-Discovery: Context-Aware Human-AI Routing for PII Redaction

---

## 1. Introduction

### 1.1 Motivation

In legal e-discovery, organizations must review and redact Personally Identifiable Information (PII) from large document corpora before production to opposing counsel. This process is expensive—often costing \$5–50 per document for human review—motivating the use of AI-based NER (Named Entity Recognition) models to automate redaction.

Standard Learning-to-Defer (L2D) and Selective Prediction frameworks route documents to human reviewers only when the AI model's **confidence is low**. However, this paradigm has a critical blind spot: an AI model may be 99% confident that "John Smith" is a name and correctly redact it, yet the surrounding context—e.g., an email discussing a sensitive medical diagnosis or a whistleblowing complaint—means the **unredacted text still identifies the individual through contextual clues**. This is a *collateral privacy leak*.

### 1.2 Core Contribution

This paper proposes **Privacy-Triggered Deferral**: a dual-objective L2D policy that defers documents not only when confidence is low, but also when the predicted **contextual privacy risk** is high, even if the model is confident in its redactions. The policy is:

$$\text{Defer}(x) = \mathbb{1}[c(x) < \tau_c \;\lor\; r(x) > \tau_r]$$

where $c(x)$ is the NER confidence score and $r(x)$ is a learned document-level sensitivity score.

This flips the standard L2D assumption: **high model confidence does not equal safe for deployment** when the contextual stakes vary per instance.

### 1.3 Connection to AI Governance

This work directly implements the *Conditionally Autonomous AI* oversight level from the Fabric framework. Where Fabric's governance rules are typically ad-hoc, our approach **mathematically formalizes an institutional privacy policy** into the routing layer, demonstrating cost-aware triage, selective prediction, and algorithmic governance.

---

## 2. Methodology

### 2.1 System Architecture

We use a single **multi-task model** with a shared DistilBERT backbone (`distilbert-base-uncased`, 66M parameters) and two task-specific heads:

**Architecture:**
```
Input tokens → DistilBERT Encoder (768-dim hidden states)
                    ├── NER Head: Linear(768 → 2)          → per-token redact/keep logits
                    └── Risk Head: Linear(768 → 128) → ReLU → Dropout(0.1) → Linear(128 → 1) → Sigmoid → r(x) ∈ [0,1]
```

- **NER Head** (The Redactor): Per-token binary classification producing a redaction mask and a confidence score $c(x)$ computed as the mean max-probability over tokens predicted as entities.
- **Risk Head** (The Risk Assessor): Document-level regression using the `[CLS]` token representation, predicting sensitivity $r(x) \in [0, 1]$.

Both heads are trained jointly with equal-weighted loss:

$$\mathcal{L} = \mathcal{L}_{\text{NER}}(\text{CrossEntropy}) + \mathcal{L}_{\text{Risk}}(\text{MSE})$$

### 2.2 The "Human" Expert

Since we cannot hire actual lawyers, we use the dataset's **ground-truth labels as the human decision**. When a document is deferred, the "human" provides 100% accurate redaction labels. This is a standard proxy in L2D research.

### 2.3 Routing Policies (Governance Strategies)

We compare three governance strategies, mapping to the Fabric paper's oversight levels:

**Strategy 1 — Autonomous AI (No Oversight):**
The NER model auto-redacts everything. No human in the loop.
$$\text{Defer}(x) = \text{False} \quad \forall x$$

**Strategy 2 — Confidence-Based L2D (Standard Selective Prediction):**
The model defers only when NER confidence is low.
$$\text{Defer}(x) = \mathbb{1}[c(x) < \tau_c]$$

**Strategy 3 — Privacy-Triggered L2D (Proposed):**
The model defers when confidence is low **OR** predicted privacy risk is high.
$$\text{Defer}(x) = \mathbb{1}[c(x) < \tau_c \;\lor\; r(x) > \tau_r]$$

### 2.4 Cost Model

We define three cost parameters for the expected cost framework:

| Parameter | Meaning | Value |
|-----------|---------|-------|
| $C_h$ | Cost of human review per document | Swept: \$2, \$5, \$10, \$20, \$40 |
| $C_{\text{err}}$ | Cost of missing a standard PII token | \$2 |
| $C_{\text{leak}}$ | Cost of leaking a highly sensitive document | \$50 |

The expected cost per document is:
$$\text{Cost}(x) = \begin{cases} C_h & \text{if deferred} \\ \sum_t \mathbb{1}[\hat{y}_t \neq y_t] \cdot C_{\text{err}} + \mathbb{1}[\text{sensitive}] \cdot r(x) \cdot C_{\text{leak}} & \text{if automated} \end{cases}$$

---

## 3. Experimental Setup

### 3.1 Dataset

**Source:** `ai4privacy/pii-masking-400k` from Hugging Face — a large-scale PII dataset containing synthetic documents with character-level PII annotations across 17 entity types.

**Processing pipeline:**
1. Selected 500 English-language documents from the training split.
2. Tokenized each document's `source_text` into whitespace-delimited words.
3. Converted character-level `privacy_mask` spans into **word-level binary redaction masks** by mapping each character offset to its enclosing word index.
4. Computed a **document-level sensitivity score** $s(x) \in [0.05, 0.95]$ based on the PII types present (see §3.2).
5. Split 80/20 into 400 train and 100 test documents.

**Dataset statistics:**

| Statistic | Value |
|-----------|-------|
| Total documents | 500 |
| Train / Test | 400 / 100 |
| Documents with PII | 500 / 500 (100%) |
| Sensitive documents ($s \geq 0.5$) | 259 / 500 (51.8%) |
| Test sensitive | 37 / 100 (37%) |
| Sensitivity score range | 0.053 – 0.950 |
| Sensitivity mean ± std | 0.545 ± 0.314 |

*Refer to `plots/figures/fig2_dataset_distribution.png` (Fig. 2)*

**PII entity types found (17 types, 2,630 total spans across 500 docs):**

| High-Risk Types ($w \geq 0.45$) | Count | Low-Risk Types ($w < 0.45$) | Count |
|----------------|-------|----------------|-------|
| SOCIALNUM | 101 | GIVENNAME | 411 |
| CREDITCARDNUMBER | 37 | SURNAME | 293 |
| PASSWORD | 35 | EMAIL | 214 |
| ACCOUNTNUM | 60 | CITY | 201 |
| TAXNUM | 27 | TELEPHONENUM | 198 |
| DRIVERLICENSENUM | 60 | BUILDINGNUM | 194 |
| IDCARDNUM | 130 | STREET | 185 |
| | | DATEOFBIRTH | 177 |
| | | USERNAME | 159 |
| | | ZIPCODE | 148 |

The right panel of Fig. 2 visualizes these frequencies as a bar chart, with high-risk types in orange and standard PII in blue. GIVENNAME and SURNAME dominate by count, but the high-risk types (SSN, credit cards, passwords) drive the sensitivity scoring.

### 3.2 Sensitivity Label Construction

Since the dataset only provides PII span annotations (not document-level sensitivity), we constructed a ground-truth sensitivity score for each document using a **PII-type risk model**. Each of the 17 PII types was assigned a risk weight reflecting the severity of exposure:

$$s(x) = \max_{l \in \text{PII\_types}(x)} w_l \;\cdot\; \left(0.7 + 0.3 \cdot \min\!\left(\frac{|\text{unique\_types}(x)|}{5}, 1\right)\right)$$

where $w_l$ is the per-type risk weight (e.g., SOCIALNUM=0.95, GIVENNAME=0.05). The diversity term boosts the score when multiple PII types are present, reflecting higher **re-identification risk** when name + address + SSN appear together versus name alone.

**Risk weight rationale:**
- **High risk ($w > 0.7$):** Social security numbers, credit cards, passwords, bank accounts — direct financial/identity theft vectors.
- **Medium risk ($0.4 < w < 0.7$):** Driver's licenses, ID cards — government identifiers.
- **Low risk ($w < 0.3$):** Names, emails, addresses, phone numbers — commonly available information.

A document is labeled "sensitive" if $s(x) \geq 0.5$. The resulting distribution is bimodal: documents containing only names/addresses cluster around 0.05–0.25, while documents with financial or government PII cluster around 0.65–0.95.

### 3.3 Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Base model | `distilbert-base-uncased` (66M params) |
| Optimizer | AdamW |
| Learning rate | 2 × 10⁻⁵ |
| Batch size | 16 |
| Epochs | 3 |
| Max sequence length | 128 tokens |
| NER loss | CrossEntropy (ignore_index=-100 for sub-tokens) |
| Risk loss | MSE |
| Total loss | $\mathcal{L}_{\text{NER}} + \mathcal{L}_{\text{Risk}}$ (equal weighting) |

**Sub-token alignment:** DistilBERT's WordPiece tokenizer splits words into sub-tokens. Only the first sub-token of each word receives a NER label; subsequent sub-tokens are assigned `ignore_index=-100` and excluded from the NER loss. This ensures word-level predictions.

**Training results:**

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 0.4825 | 0.3452 |
| 2 | 0.2569 | 0.2767 |
| 3 | 0.1581 | 0.2290 |

Both losses decrease monotonically. The train-val gap at epoch 3 (0.07) indicates mild overfitting, expected with 400 training samples on a 66M-parameter model.

### 3.4 Model Output Analysis (Test Set)

After training, we analyzed the model's predicted scores on the held-out test set (100 documents, 37 sensitive):

**Risk score $r(x)$ distribution:**

| Subset | Min | Max | Mean |
|--------|-----|-----|------|
| All docs | 0.217 | 0.809 | 0.593 |
| Sensitive | 0.544 | 0.809 | **0.740** |
| Non-sensitive | 0.217 | 0.801 | **0.506** |

The risk head successfully learned to **discriminate** sensitive from non-sensitive documents (mean gap: 0.234), confirming that the PII-type-based sensitivity labels provide a learnable signal. Sensitive documents cluster in the upper risk range (0.54–0.81), while non-sensitive documents spread across the lower range (0.22–0.80) with a substantially lower mean.

**Confidence score $c(x)$ distribution:** min=0.762, max=0.941, mean=0.855, std=0.042. All documents have high NER confidence (>0.76), reflecting that PII entities are relatively unambiguous for DistilBERT. This tight, high-confidence distribution is precisely why confidence-only deferral struggles: nearly all documents appear "safe" by confidence alone (see §5.2).

### 3.5 Experiment Workflow

**Step 1 — Batched Inference:** All 100 test documents are tokenized in a single batch and passed through the model using a `DataLoader` (batch_size=32). For each document, we extract: confidence $c(x)$, risk $r(x)$, and predicted redaction mask.

**Step 2 — Policy Sweep:** For each of the three strategies, we apply the routing policy across a grid of threshold values:

| Strategy | Sweep parameters | Grid size |
|----------|-----------------|-----------|
| Autonomous | None | 1 operating point |
| Confidence-only | $\tau_c \in \{0.75, 0.80, 0.85, 0.90, 0.95\}$ | 5 points |
| Privacy-triggered | $\tau_c \times \tau_r$ with $\tau_c \in \{0.75\text{–}0.95\}$, $\tau_r \in \{0.2\text{–}0.6\}$ | 25 points |

**Step 3 — Cost Sweep:** Each operating point is evaluated at 5 values of $C_h \in \{2, 5, 10, 20, 40\}$, producing 155 total rows in `results.csv`.

**Step 4 — Metrics:** For each row, we compute: automation rate, leakage rate, system F1, system precision/recall, sensitive/non-sensitive precision/recall breakdowns, expected cost (decomposed into human review, PII miss, and leakage components), and deferral count.

**Step 5 — Prediction Export:** Per-document predictions (confidence, risk, sensitivity label) are saved to `experiments/predictions.json` for use in the routing quadrant scatter plot (Fig. 1).

**Step 6 — Pareto Plotting:** Non-dominated operating points are extracted per strategy and plotted as Pareto frontiers (Figs. A1, A2). Six additional publication figures (Figs. 1–6) are generated from the real data.

---

## 4. Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Automation Rate** | $\frac{|\text{automated docs}|}{|\text{all docs}|}$ | % handled purely by AI (higher = less human work) |
| **Leakage Rate** | $\frac{|\text{sensitive docs NOT deferred}|}{|\text{all sensitive docs}|}$ | % of sensitive docs that slipped through without human review (lower = safer) |
| **System F1** | Token-level F1 across all docs (deferred docs use ground-truth as "human" labels) | Overall redaction quality of the full human-AI pipeline |
| **Expected Cost** | Mean per-document cost using the cost model (§2.4) | Monetary/risk cost of the pipeline |
| **Cost Components** | Human review ($C_h$), PII miss ($C_{\text{err}}$), leakage ($C_{\text{leak}}$) | Decomposition revealing which risk factor dominates cost |
| **Precision/Recall Breakdown** | Computed separately for sensitive and non-sensitive document subsets | Reveals whether the model under-performs on high-stakes documents |

---

## 5. Results

### 5.1 Autonomous AI (Baseline)

| Metric | Value |
|--------|-------|
| Automation Rate | 100% |
| Leakage Rate | **100%** |
| System F1 | 0.883 |
| System Precision / Recall | 0.886 / 0.880 |
| Expected Cost ($C_h$=5) | **\$15.40** |

Cost breakdown: human review \$0.00 + PII miss \$1.72 + privacy leakage **\$13.68** = \$15.40. The dominant cost component (89%) is **privacy leakage** from processing all 37 sensitive documents without human oversight. Despite zero human review cost, the autonomous policy is the most expensive overall because leaked sensitive documents carry a steep penalty ($C_{\text{leak}}=\$50$ weighted by risk score).

### 5.2 Confidence-Based L2D

| $\tau_c$ | Automation | Leakage | System F1 | Cost ($C_h$=5) |
|----------|-----------|---------|-----------|-----------------|
| 0.75 | 100% | **100.0%** | 0.883 | \$15.40 |
| 0.80 | 89% | **89.2%** | 0.894 | \$14.37 |
| 0.85 | 51% | **40.5%** | 0.945 | \$8.78 |
| 0.90 | 19% | 16.2% | 0.977 | \$6.74 |
| 0.95 | 0% | 0.0% | 1.000 | \$5.00 |

At $\tau_c=0.75$, no documents are deferred because all test documents have confidence $\geq 0.762$, making this threshold equivalent to the autonomous baseline. Meaningful deferral only begins at $\tau_c=0.80$. Even at $\tau_c=0.85$, where 49% of documents are deferred, **40.5% of sensitive documents still leak**. Reaching zero leakage requires $\tau_c=0.95$, which defers 100% of documents — defeating the purpose of automation entirely. This is because **sensitive documents often have high NER confidence** (the model is confident about the PII entities but unaware of the contextual risk).

### 5.3 Privacy-Triggered L2D (Proposed)

Selected operating points across the threshold grid (at $C_h$=\$5):

| $\tau_c$ | $\tau_r$ | Automation | Leakage | System F1 | Cost |
|----------|----------|-----------|---------|-----------|------|
| 0.90 | 0.5 | 9% | **0.0%** | 0.998 | \$4.55 |
| 0.85 | 0.5 | 23% | **0.0%** | 0.988 | \$3.93 |
| 0.75 | 0.5 | 30% | **0.0%** | 0.984 | \$3.68 |
| 0.80 | 0.6 | 37% | 10.8% | 0.976 | \$4.58 |
| 0.75 | 0.6 | 41% | 13.5% | 0.974 | \$4.70 |

**Key result:** With $\tau_r \leq 0.5$, the privacy-triggered strategy achieves **zero leakage** across all tested $\tau_c$ values while still automating up to 30% of documents (at $\tau_c=0.75, \tau_r=0.5$). This is impossible with confidence-only deferral at any threshold — confidence-only can only reach zero leakage by deferring 100% of documents ($\tau_c=0.95$).

The cost-optimal zero-leakage point is $\tau_c=0.75, \tau_r=0.5$ at **\$3.68 per document** — a 76% reduction from the autonomous baseline's \$15.40. Relaxing the risk threshold to $\tau_r=0.6$ permits some leakage (10–13%) but increases automation to 37–41%.

### 5.4 Head-to-Head Comparison

At comparable automation rates, the privacy-triggered strategy consistently achieves lower leakage:

| Automation Rate | Confidence-Only | Privacy-Triggered | Leakage Reduction |
|----------------|-----------------|-------------------|-------------------|
| ~20% | 19% auto, **16.2% leak** ($\tau_c$=0.90) | 20% auto, **0.0% leak** ($\tau_c$=0.85, $\tau_r$=0.4) | **∞** (eliminated) |
| ~30% | 19% auto*, **16.2% leak** | 30% auto, **0.0% leak** ($\tau_c$=0.75, $\tau_r$=0.5) | **∞** (with +11% auto) |
| ~41–51% | 51% auto, **40.5% leak** ($\tau_c$=0.85) | 41% auto, **13.5% leak** ($\tau_c$=0.75, $\tau_r$=0.6) | **3.0×** |

*\*Confidence-only has no operating point near 30% automation; nearest is 19% ($\tau_c$=0.90) or 51% ($\tau_c$=0.85).*

The most striking result: at ~20% automation, confidence-only leaks 16.2% of sensitive documents while privacy-triggered leaks **zero**. Privacy-triggered Pareto-dominates across the entire operating range.

### 5.5 Pareto Frontier Analysis

*Refer to `plots/pareto_workload_vs_leakage.png` (Fig. A1)*

The Pareto frontier plot (Automation Rate vs. Leakage Rate) shows three clearly separated curves:

1. **Autonomous AI** (red diamond): A single point at (100%, 100%) — maximum automation, maximum leakage.
2. **Confidence-Based L2D** (blue line): A curve from (0%, 0%) to (100%, 100%). Meaningful leakage reduction only begins at $\tau_c \geq 0.85$, requiring 49% or more deferral to bring leakage below 41%.
3. **Privacy-Triggered L2D** (green line): Sits **below and to the right** of the confidence-only curve. At 30% automation ($\tau_c$=0.75, $\tau_r$=0.5), it achieves 0% leakage — a Pareto-dominant operating point that confidence-only cannot reach at any threshold short of 100% deferral.

*Refer to `plots/pareto_cost_vs_f1.png` (Fig. A2)*

The Cost vs. F1 plot confirms that both L2D strategies achieve near-perfect system F1 (≥0.98) because deferred documents receive perfect human labels. The autonomous baseline sits at F1=0.883, paying a high cost (\$15.40) from missed PII and leaked sensitive documents.

### 5.6 Cost Decomposition Analysis

*Refer to `plots/figures/fig5_cost_breakdown.png` (Fig. 5)*

Decomposing the expected cost into its three components reveals the economic case for privacy-triggered deferral (at representative thresholds $\tau_c$=0.85, $\tau_r$=0.5, $C_h$=\$5):

| Policy | Human Review | PII Miss | Leakage | **Total** |
|--------|-------------|----------|---------|-----------|
| Autonomous | \$0.00 (0%) | \$1.72 (11%) | **\$13.68 (89%)** | **\$15.40** |
| Confidence-Based ($\tau_c$=0.85) | \$2.45 (28%) | \$0.94 (11%) | **\$5.39 (61%)** | **\$8.78** |
| Privacy-Triggered ($\tau_c$=0.85, $\tau_r$=0.5) | \$3.85 (98%) | \$0.08 (2%) | **\$0.00 (0%)** | **\$3.93** |

Key insights:
- **Autonomous AI** spends 89% of its cost on privacy leakage penalties. Despite zero human review overhead, it is the **most expensive** policy overall (3.9× more than privacy-triggered).
- **Confidence-only** reduces leakage cost by 61% but still has a dominant leakage component (\$5.39) because it defers by confidence, not risk.
- **Privacy-triggered** eliminates leakage cost entirely. Its cost is almost purely human review (\$3.85), with negligible PII miss cost (\$0.08). Despite deferring 77% of documents, the total cost is the **lowest** because avoiding a single leaked sensitive document saves \$50 × risk_score.

This demonstrates a counter-intuitive result: **more human oversight can be cheaper** when the cost of automated failure is sufficiently high relative to human review cost.

### 5.7 Sensitivity Analysis: Effect of $\tau_r$

*Refer to `plots/figures/fig6_threshold_sensitivity.png` (Fig. 6)*

Sweeping $\tau_r$ at fixed $\tau_c=0.85$ and $C_h=\$5$ reveals the tradeoff between leakage protection and automation:

| $\tau_r$ | Automation | Leakage | F1 | Cost |
|----------|-----------|---------|-----|------|
| 0.2 | 0% | 0.0% | 1.000 | \$5.00 |
| 0.3 | 16% | 0.0% | 0.992 | \$4.22 |
| 0.4 | 20% | 0.0% | 0.990 | \$4.06 |
| 0.5 | 23% | 0.0% | 0.988 | \$3.93 |
| 0.6 | 28% | 5.4% | 0.982 | \$4.39 |

At $\tau_r=0.5$, a "sweet spot" emerges: zero leakage is maintained with 23% automation and the lowest cost (\$3.93). Raising $\tau_r$ to 0.6 crosses the leakage threshold (5.4% of sensitive documents now leak) and paradoxically *increases* cost (\$4.39) because the leakage penalty outweighs the saved human review cost. This confirms that $\tau_r=0.5$ is the cost-optimal zero-leakage threshold for this model and dataset.

---

## 6. Discussion

### 6.1 Why Confidence Alone Fails

*Refer to `plots/figures/fig1_confidence_risk_scatter.png` (Fig. 1)*

The scatter plot of actual model predictions (Fig. 1) reveals the core problem: **NER confidence and privacy risk are orthogonal signals**. All 100 test documents have confidence between 0.76 and 0.94 — a narrow, high-confidence band. Meanwhile, risk scores span a wide range (0.22–0.81). The confidence-only policy draws a vertical boundary at $\tau_c$, but this cannot separate the 28 high-risk documents (orange triangles in the upper-right "novel" quadrant) from the 23 low-risk documents (green circles in the lower-right). Only the L-shaped privacy-triggered boundary captures documents that are confidently processed but contextually dangerous.

A document containing "John Smith, 123 Main St, SSN 555-12-3456" will have high NER confidence (clear entity boundaries) but extreme privacy risk (SSN exposure). Confidence-only deferral cannot distinguish this from "John Smith attended the meeting" — both have equally high confidence. The risk head captures what confidence cannot: the **severity of what would be exposed if the automated redaction is imperfect**.

### 6.2 The Privacy-Triggered Advantage

The proposed strategy adds a second deferral trigger ($r(x) > \tau_r$) that specifically catches **high-confidence, high-risk** documents. The risk head learned to assign higher scores to documents containing financial PII (SOCIALNUM, CREDITCARDNUMBER, ACCOUNTNUM) versus those with only names and addresses (mean risk gap: 0.234 between sensitive and non-sensitive documents; §3.4).

This targeted deferral explains the zero-leakage result: at $\tau_r=0.5$, all 37 sensitive test documents are routed to human review. The 23% of documents that proceed automatically are precisely the low-risk documents where mistakes have minimal consequences — even if the model misses a PII token in a document containing only names and zip codes, the privacy cost is far lower than missing one in a document containing social security numbers.

### 6.3 Cost-Efficiency of Selective Deferral

The cost decomposition analysis (§5.6, Fig. 5) reveals a counter-intuitive economic result: **the policy with the most human review is the cheapest**. Privacy-triggered deferral costs \$3.93/doc versus \$15.40/doc for autonomous AI — a 3.9× reduction — despite deferring 77% of documents to human reviewers.

This occurs because the cost parameters reflect real-world risk asymmetry: leaking a sensitive document ($C_{\text{leak}}=\$50 \times r(x)$) is far more expensive than human review ($C_h=\$5$). A single leaked sensitive document with risk score 0.7 costs \$35 in leakage penalty — equivalent to reviewing 7 documents manually. The privacy-triggered policy avoids all leakage penalties, making it the economically rational choice even when human review is costly.

### 6.4 Practical Implications for AI Governance

This result demonstrates a key principle of algorithmic governance: **capability does not imply permission**. The AI model may be perfectly capable of redacting a sensitive document, but the ethical cost of a single automated mistake on that document (collateral privacy leak) outweighs the efficiency gain. The privacy-triggered routing policy formalizes this institutional risk tolerance into a mathematically precise threshold, directly implementing Fabric's *Conditionally Autonomous AI* level.

*Refer to `plots/figures/fig3_routing_heatmap.png` (Fig. 3)*

Fig. 3 illustrates this principle on the theoretical cost surface: the privacy-triggered boundary (cyan L-shape) captures a region of high automation cost that the confidence-only boundary (white dashed vertical) misses entirely — the high-confidence, high-risk zone where the model is capable but should not be trusted.

### 6.5 Limitations

1. **Simulated human expert.** We proxy human reviewers with ground-truth labels (100% accuracy). Real human reviewers have variable accuracy and fatigue effects, which would increase the cost and reduce the F1 of deferred documents.
2. **Small dataset.** 500 documents (400 train / 100 test) is small for fine-tuning a 66M-parameter model. The risk head's discrimination (mean gap of 0.234 between sensitive and non-sensitive) could improve with more data, and the tight confidence distribution (std=0.042) may widen with more diverse documents.
3. **Heuristic sensitivity labels.** Ground-truth sensitivity is derived from PII-type risk weights rather than expert judgment or LLM annotation. The bimodal distribution (see Fig. 2, left panel) may not capture nuanced contextual sensitivity (e.g., two documents both containing names but one discussing a medical diagnosis).
4. **Single domain.** Results are demonstrated on synthetic PII documents from the ai4privacy/pii-masking-400k dataset. Generalization to real-world legal corpora (e.g., Enron emails, litigation discovery sets) requires further validation.
5. **Fixed cost parameters.** The cost advantage of privacy-triggered deferral depends on the ratio $C_{\text{leak}} / C_h$. In settings where leakage penalties are low relative to human review costs, the economic case weakens.

---

## 7. Conclusion

We presented Privacy-Triggered Deferral, a dual-objective Learning-to-Defer policy for legal e-discovery that routes documents to human reviewers based on both NER confidence and predicted contextual privacy risk. Our experiments show that:

- **Autonomous AI** leaks 100% of sensitive documents at a cost of \$15.40/doc.
- **Confidence-only deferral** cannot reach zero leakage without deferring 100% of documents ($\tau_c=0.95$). Even at $\tau_c=0.85$ (49% deferral), 40.5% of sensitive documents leak.
- **Privacy-triggered deferral** achieves **0% leakage while automating 30%** of documents ($\tau_c=0.75, \tau_r=0.5$) at a cost of **\$3.68/doc** — a 76% cost reduction and a Pareto-dominant operating point that confidence-only cannot reach.
- The cost decomposition reveals that **more human oversight can be cheaper**: privacy leakage penalties dominate the autonomous cost structure (89%), making selective deferral economically rational.

This demonstrates that in high-stakes domains, AI governance must go beyond model confidence to incorporate **context-dependent risk assessment** in the human-AI routing decision. The routing quadrant scatter (Fig. 1) crystallizes this insight: nearly all documents cluster in the high-confidence region, making confidence-based thresholds ineffective, while the privacy risk dimension provides the discriminative signal needed for safe automation.

### Figures

| Figure | Title | Section | File |
|--------|-------|---------|------|
| Fig. 1 | Routing Quadrants: Confidence × Privacy Risk | §6.1 | `plots/figures/fig1_confidence_risk_scatter.png` |
| Fig. 2 | Dataset Analysis: Sensitivity Distribution & PII Types | §3.1 | `plots/figures/fig2_dataset_distribution.png` |
| Fig. 3 | Automation Cost Surface with Policy Boundaries | §6.4 | `plots/figures/fig3_routing_heatmap.png` |
| Fig. 4 | Policy Comparison at Representative Thresholds | §5.4 | `plots/figures/fig4_policy_comparison_bar.png` |
| Fig. 5 | Cost Decomposition by Policy | §5.6 | `plots/figures/fig5_cost_breakdown.png` |
| Fig. 6 | Sensitivity Analysis: τ_r Sweep | §5.7 | `plots/figures/fig6_threshold_sensitivity.png` |
| Fig. A1 | Pareto Frontier: Automation vs. Leakage | §5.5 | `plots/pareto_workload_vs_leakage.png` |
| Fig. A2 | Pareto Frontier: Cost vs. F1 | §5.5 | `plots/pareto_cost_vs_f1.png` |

---

## Appendix: Project File Structure

```
ediscovery-l2d/
├── data/
│   ├── prepare_data.py          # Dataset download, PII mask extraction, sensitivity scoring
│   └── documents.jsonl          # 500 processed records (with pii_types per document)
├── models/
│   ├── multitask_model.py       # MultiTaskRedactor (DistilBERT + NER head + Risk head)
│   ├── train.py                 # Training loop (3 epochs, joint loss)
│   └── checkpoint.pt            # Saved model weights
├── pipeline/
│   ├── redactor.py              # Inference: tokenize → predict → align sub-tokens → c(x), r(x)
│   └── router.py                # Three routing policies (autonomous, confidence-only, privacy-triggered)
├── evaluate/
│   └── metrics.py               # Automation rate, leakage rate, F1, P/R, cost components
├── experiments/
│   ├── run_experiment.py        # Full sweep: 3 strategies × threshold grid × C_h sweep → 155 rows
│   ├── results.csv              # All experiment results (with human_cost, error_cost, leak_cost)
│   └── predictions.json         # Per-document model predictions (conf, risk, is_sensitive)
├── plots/
│   ├── plot_pareto.py           # Pareto frontier generation (Figs A1, A2)
│   ├── plot_paper.py            # Publication figures from real data (Figs 1–6)
│   ├── figures/                 # Generated publication figures
│   │   ├── fig1_confidence_risk_scatter.png
│   │   ├── fig2_dataset_distribution.png
│   │   ├── fig3_routing_heatmap.png
│   │   ├── fig4_policy_comparison_bar.png
│   │   ├── fig5_cost_breakdown.png
│   │   └── fig6_threshold_sensitivity.png
│   ├── pareto_workload_vs_leakage.png
│   └── pareto_cost_vs_f1.png
├── requirements.txt
└── project.md                   # Original project specification
```
