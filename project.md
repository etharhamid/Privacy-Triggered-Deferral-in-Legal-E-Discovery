# Project: Privacy-Triggered Deferral in Legal E-Discovery

## 1. High-Level Overview

**Core idea**

Learning-to-Defer (L2D) typically triggers when AI confidence is *low*. This project flips the paradigm: the AI defers when confidence is *high*, but the secondary "privacy risk" is too extreme to automate.

**Human-AI interaction model**

- **What the human does:** Reviews documents flagged for complex privacy concerns.
- **What the AI does:** Redacts standard PII, but calculates a "collateral privacy score" for intertwined identities.
- **How they interact:** AI routes to the human specifically to make ethical privacy judgments.

**Technical approach**

- **Model:** NLP Named Entity Recognition (NER) model paired with a secondary classifier predicting sensitivity.
- **Extension:** Formulate a dual-objective L2D policy: $f(\text{confidence},\, \text{privacy\_score})$. Compare against standard confidence-only deferral.

**Risk and evaluation**

- **Risks:** Automated over-redaction and collateral privacy leaks.
- **Metrics:** PII leakage rate, human workload, and deferral appropriateness.

**Why it matters**

Demonstrates that algorithmic governance must sometimes restrict perfectly capable models because the ethical cost of an automated decision is too high.

---

## 2. Fabric Repository Profile

*This section maps the project directly to the qualitative governance frameworks discussed in the "Fabric" reference paper (Appendix B format).*

**Entry 21 — Privacy-Triggered E-Discovery Redactor**

**AI Workflow Diagram:**

- **Input:** Raw Legal Documents / Emails
- **Process:** Inputs are passed to an **AI System** containing two sub-models: an NER Redactor (calculates confidence) and a Contextual Risk Assessor (calculates privacy severity).
- **Decision Point:** A safety check evaluates — *Is NER Confidence high AND Contextual Risk low?*
- **Branch 1 (True):** The system acts autonomously. Output flows directly to the **Final Outcome** (Auto-Redacted Document).
- **Branch 2 (False):** The system routes to a **Human Check** (Reviewing Lawyer). The lawyer can **Modify** or **Accept** the redactions. Their decision flows to the **Final Outcome** (Human-Redacted Document).

| Field | Value |
|-------|-------|
| **Sector** | Private |
| **Domain** | Law / Cross-Domain |
| **Human Oversight Level** | Conditionally Autonomous AI |

**Task:** The AI system processes bulk legal documents during e-discovery to redact PII, routing contextually sensitive or uncertain documents to a human lawyer for manual review.

**Intent:** Drastically reduce the administrative cost and time of legal document review while ensuring that highly sensitive contexts (e.g., whistleblowing, medical disclosures) are not inadvertently leaked due to over-reliance on simple name-redaction.

**Risks:** The AI could confidently redact specific names but leave enough contextual clues for the individual to be identified (collateral privacy leak). Alternatively, over-reliance on the AI could expose trade secrets or legally privileged information.

**Institutional Oversight:**

- Attorney-client privilege compliance (regulation)
- Data protection laws: GDPR / CCPA (regulation)
- Dual-objective cost-matrix thresholds set by the firm's partners (organizational policy)
- E-discovery software procurement guidelines (industry standard)

**Workflow explanation:**

1. *Input:* An administrator or legal team submits a batch of raw, unredacted documents (e.g., corporate emails) to the AI system.
2. *Process:* The AI evaluates each document on two fronts — it predicts redaction masks with a statistical *confidence score*, and it generates a *contextual privacy risk score*. These scores are fed into a routing policy.
3. *Output:* If the AI's redaction confidence is high AND the contextual risk is below the firm's threshold, the document is automatically redacted and finalized. Otherwise, the document is escalated to a human lawyer who reviews, modifies if necessary, and approves the final redacted document.

---

## 3. Comprehensive Implementation Guide

*This section provides the dataset, models, mathematical formulation, and experimental design needed to write the 4-page NeurIPS workshop paper.*

### 3.1 Scenario and Motivation

In standard Learning-to-Defer (L2D) or Selective Prediction, a model defers to a human when it is *uncertain* (low confidence). However, in legal e-discovery, an AI might be 99.9% confident that a word is a person's name and correctly redact it, yet the surrounding context — e.g., an email discussing a sensitive medical diagnosis, a whistleblowing complaint, or an affair — means the **unredacted text still identifies the individual through contextual clues**. This is a *collateral privacy leak*.

Here, the AI should not just auto-redact the name — it must defer the entire document to a human lawyer to assess *collateral privacy risks* (e.g., does the unredacted text still identify the person?).

### 3.2 Dataset Setup

Do not collect your own data; use an established NLP privacy dataset.

- **Primary Recommendation:** The **Enron Email Dataset** annotated for PII, or the **Text Anonymization Benchmark (TAB)**.
- **Task:** Token-level classification (NER) to find entities that must be redacted (Names, Phones, Medical Conditions, Financials).
- **Creating the Sensitivity Label:** Since standard datasets only label PII, a document-level *sensitivity* score is needed. Use an LLM (e.g., GPT-4o-mini or Claude 3.5 Haiku) to pre-process the training set and score each document from 0 to 1 for "Contextual Sensitivity" (`1` = discusses illegal activity, medical history, or highly personal grievances; `0` = meeting scheduling). This becomes the ground-truth risk score.

### 3.3 System Architecture and Models

Two models (or a single multi-task model) act as the AI agent:

1. **The Redactor (NER Model):** A lightweight model (e.g., fine-tuned `DistilRoBERTa-base-NER`) that predicts token-level redaction masks with a confidence score $c(x)$.
2. **The Risk Assessor:** A sequence classification head on the same model that predicts the document-level sensitivity score $r(x)$.

**The "Human":** Since actual lawyers cannot be hired, the dataset's ground-truth labels serve as the "Human Decision." When a document is deferred, the "Human" provides 100% accuracy. This is a standard proxy in L2D research.

### 3.4 The Collaboration Algorithm

Design a **Cost-Aware Routing Policy** with the following cost parameters:

| Parameter | Meaning |
|-----------|---------|
| $C_h$ | Cost of human review (e.g., \$5 per document) |
| $C_{\text{err}}$ | Cost of missing a standard PII redaction |
| $C_{\text{leak}}$ | Cost of releasing a highly sensitive document (exponentially higher than $C_{\text{err}}$) |

**Baseline Strategy — Confidence-Only Selective Prediction:**

The AI auto-redacts the document and defers only if average token confidence $c(x)$ is below threshold $\tau_c$:

$$\text{Defer}(x) = \mathbb{1}[c(x) < \tau_c]$$

**Proposed Strategy — Privacy-Triggered Routing:**

The AI evaluates both redaction confidence $c(x)$ and predicted sensitivity $r(x)$:

$$\text{Defer}(x) = \mathbb{1}[c(x) < \tau_c \;\lor\; r(x) > \tau_r]$$

**Advanced Formulation — Joint Optimization:**

Instead of hard thresholds, define an expected cost function for automating document $x$:

$$\mathbb{E}[\text{Cost}_{\text{auto}}] = (1 - c(x)) \cdot C_{\text{err}} + r(x) \cdot c(x) \cdot C_{\text{leak}}$$

Defer if $\mathbb{E}[\text{Cost}_{\text{auto}}] > C_h$.

*(This explicitly demonstrates course concepts: cost-aware triage and decision-theoretic routing.)*

### 3.5 Experimental Design

Run the dataset through three governance strategies, mapping to the *Fabric* paper's oversight levels:

| Strategy | Description |
|----------|-------------|
| **Autonomous AI** | The NER model auto-redacts everything. No human in the loop. |
| **Confidence-Based L2D** | Model defers only when NER confidence is low. |
| **Privacy-Triggered L2D (Proposed)** | Model defers when confidence is low OR predicted privacy risk is high. |

Sweep the cost threshold ($C_h$) to generate a Pareto frontier (trade-off curves) for each strategy.

### 3.6 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Coverage / Automation Rate** | % of documents handled purely by AI (lower human workload = better) |
| **Catastrophic Leakage Rate** | % of *highly sensitive* documents that slipped through without human review (target: near 0%) |
| **System F1 Score** | Overall accuracy of the final redacted corpus (AI + Human combined) |
| **Expected Cost** | Total monetary/risk cost of the pipeline based on the defined cost matrix |

### 3.7 Connection to Course Concepts and Reference Paper

- **Fabric Connection:** Directly implements Fabric's *Conditionally Autonomous AI* level with *Institutional Oversight*. Where Fabric's governance rules are typically ad-hoc, this paper mathematically formalizes an "Institutional Privacy Policy" into the routing layer.
- **Course Material:** Demonstrates selective prediction, cost-aware triage, and algorithmic governance.
- **Novelty:** Flips the standard L2D assumption — "high model confidence does not equal safe for deployment" when contextual stakes vary per instance.

### 3.8 Suggested Paper Outline (4 Pages)

| Section | Content | Length |
|---------|---------|--------|
| **1. Introduction** | E-discovery context, risk of high-confidence privacy leaks, framing of context-aware human-AI routing | 0.75 pages |
| **2. Methodology** | Dual-objective L2D framework, Expected Cost equation, proxying human experts using ground-truth data | 1.25 pages |
| **3. Experimental Setup** | Dataset (e.g., Enron), models (NER + Risk Assessor), baseline policies | 0.75 pages |
| **4. Results and Discussion** | Pareto curves (Workload vs. Leakage Rate), privacy-triggered deferral preventing catastrophic failures, connection to AI governance and Fabric | 1.25 pages |
