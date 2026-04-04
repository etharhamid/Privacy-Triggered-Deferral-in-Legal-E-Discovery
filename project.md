
\f0\fs24 \cf0 # Project Idea: Privacy-Triggered Deferral in Legal E-Discovery\
\
## 1. High-Level Overview\
\
**Core idea**  \
Learning-to-defer (L2D) typically triggers when AI confidence is *low*. This project flips the paradigm: the AI defers when confidence is *high*, but the secondary "privacy risk" is too extreme to automate.\
\
**Human-AI interaction model**  \
* **What the human does:** Reviews documents flagged for complex privacy concerns.\
* **What the AI does:** Redacts standard PII, but calculates a "collateral privacy score" for intertwined identities. \
* **How they interact:** AI routes to the human specifically to make ethical privacy judgments.\
\
**Technical approach**  \
* **Model:** NLP Named Entity Recognition (NER) model paired with a secondary classifier predicting sensitivity. \
* **Extension:** Formulate a dual-objective L2D policy: $f(\\text\{confidence\}, \\text\{privacy\\_score\})$. Compare against standard confidence-only deferral.\
\
**Risk / evaluation**  \
* **Risks:** Automated over-redaction and collateral privacy leaks.\
* **Metrics:** PII leakage rate, human workload, and deferral appropriateness.\
\
**Why it matters**  \
Demonstrates that algorithmic governance must sometimes restrict perfectly capable models because the ethical cost of an automated decision is too high.\
\
---\
\
## 2. Fabric Repository Profile (Reference Paper Mapping)\
\
*This section maps the project directly to the qualitative governance frameworks discussed in the "Fabric" reference paper (Appendix B format).*\
\
**21. Privacy-Triggered E-Discovery Redactor**\
\
**AI Workflow Diagram Description (using Fabric\'92s legend):**\
*   **Input:** Raw Legal Documents / Emails\
*   **Process:** The inputs are passed to an **AI System** containing two sub-models: an NER Redactor (calculates confidence) and a Contextual Risk Assessor (calculates privacy severity).\
*   **Decision Point:** A safety check evaluates the equation: *Is NER Confidence high AND Contextual Risk low?*\
*   **Branch 1 (True):** The system acts autonomously. The output flows directly to the **Final Outcome** (Auto-Redacted Document).\
*   **Branch 2 (False):** The system routes to a **Human Check** (Reviewing Lawyer). The lawyer can **Modify** or **Accept** the redactions. Their decision flows to the **Final Outcome** (Human-Redacted Document).\
\
**Sector:** Private  \
**Domain:** Law / Cross-Domain  \
**Task:** The AI system processes bulk legal documents during e-discovery to redact Personally Identifiable Information (PII), routing contextually sensitive or uncertain documents to a human lawyer for manual review.  \
**Intent:** The aims of the AI system are to drastically reduce the administrative cost and time of legal document review while ensuring that highly sensitive contexts (e.g., whistleblowing, medical disclosures) are not inadvertently leaked due to over-reliance on simple name-redaction.  \
**Risks:** The AI could confidently redact specific names but leave enough contextual clues for the individual to be identified (collateral privacy leak). Alternatively, over-reliance on the AI could lead to the exposure of trade secrets or legally privileged information.   \
**Human Oversight Level:** Conditionally Autonomous AI  \
**Institutional Oversight Examples:** Attorney-client privilege compliance (regulation), Data protection laws such as GDPR/CCPA (regulation), Dual-objective cost-matrix thresholds set by the firm's partners (organization policy), E-discovery software procurement guidelines (industry standard).  \
\
**Explanation of AI Workflow:**\
*   *Input:* An administrator or legal team submits a batch of raw, unredacted documents (e.g., corporate emails) to the AI system.\
*   *Process:* The AI system evaluates the document on two fronts: it predicts redaction masks with a statistical *confidence score*, and it evaluates the overall document text to generate a *contextual privacy risk score*. These scores are fed into a routing policy (Decision Point).\
*   *Output:* If the AI's redaction confidence is high AND the contextual risk is below the firm's threshold, the document is automatically redacted and finalized (autonomous action). If the AI is uncertain, OR if it is highly confident but the document is deemed highly sensitive, it escalates the document to a human lawyer. The lawyer reviews, modifies if necessary, and approves the final redacted document.\
\
---\
\
## 3. Comprehensive Implementation Guide\
\
*This section provides the dataset, models, mathematical formulation, and experimental design needed to write the 4-page NeurIPS workshop paper.*\
\
### 3.1 The Scenario and Motivation\
In standard Learning-to-Defer (L2D) or Selective Prediction, a model defers to a human when it is *uncertain* (low confidence). However, in legal e-discovery (reviewing documents before releasing them to opposing counsel or the public), an AI might be 99.9% confident that a word is a person's name, but the surrounding context (e.g., an email discussing a sensitive medical diagnosis, a whistleblowing complaint, or an affair) makes the document a **high privacy risk**. \
\
Here, the AI shouldn't just auto-redact the name\'97it must defer the entire document to a human lawyer to assess *collateral privacy risks* (e.g., does the unredacted text still identify the person?).\
\
### 3.2 Dataset Setup\
Do not collect your own data; use an established NLP privacy dataset. \
* **Primary Recommendation:** The **Enron Email Dataset** annotated for PII, or the **Text Anonymization Benchmark (TAB)**. \
* **Task:** Token-level classification (Named Entity Recognition - NER) to find entities that must be redacted (Names, Phones, Medical Conditions, Financials).\
* **Creating the "Sensitivity" Label:** Since standard datasets only label PII, you need a document-level *sensitivity* score. Use an LLM (like GPT-4o-mini or Claude 3.5 Haiku) to pre-process the training set and score each document from 0 to 1 for "Contextual Sensitivity" (e.g., `1` = discusses illegal activity, medical history, or highly personal grievances; `0` = meeting scheduling). This becomes your ground truth risk score.\
\
### 3.3 System Architecture & Models\
You will train/use two models (or a single multi-task model) to act as the AI agent:\
1. **The Redactor (NER Model):** A lightweight model (e.g., fine-tuned `DistilRoBERTa-base-NER`) that predicts token-level redaction masks with a confidence score $c(x)$.\
2. **The Risk Assessor:** A sequence classification head on the same model that predicts the document-level sensitivity score $r(x)$.\
\
**The "Human":** Since you cannot hire actual lawyers, use the dataset's ground-truth labels as the "Human Decision." When a document is deferred, the "Human" provides 100% accuracy.\
\
### 3.4 The Collaboration Algorithm (Technical Approach)\
You need to design a **Cost-Aware Routing Policy**. Define the following costs:\
* $C_h$: Cost of human review (e.g., $5 per document).\
* $C_\{err\}$: Cost of missing a standard PII redaction.\
* $C_\{leak\}$: Cost of releasing a highly sensitive document (this should be exponentially higher than $C_\{err\}$).\
\
**Baseline Strategy (Confidence-Only Selective Prediction):**\
The AI auto-redacts the document. It only defers to the human if the average token confidence $c(x)$ is below a threshold $\\tau_c$.\
* *Policy:* `If c(x) < \\tau_c: Defer()`\
\
**Proposed Strategy (Privacy-Triggered Routing):**\
The AI evaluates both its confidence in the redaction $c(x)$ AND the document's predicted sensitivity $r(x)$. \
* *Policy:* `If c(x) < \\tau_c OR r(x) > \\tau_r: Defer()`\
\
**Advanced Formulation (Joint Optimization):**\
Instead of hard thresholds, define an expected cost function for automating document $x$:\
\
$$ \\mathbb\{E\}[\\text\{Cost\}_\{auto\}] = (1 - c(x)) \\cdot C_\{err\} + r(x) \\cdot c(x) \\cdot C_\{leak\} $$\
\
* *Policy:* `Defer if \\mathbb\{E\}[\\text\{Cost\}_\{auto\}] > C_h`\
\
*(This explicitly demonstrates course concepts: cost-aware triage and decision-theoretic routing).*\
\
### 3.5 Experimental Design\
Run the dataset through your system under three different governance strategies (mapping back to the *Fabric* paper's oversight levels):\
\
1. **Autonomous AI:** The NER model auto-redacts everything. No human in the loop.\
2. **Confidence-Based L2D:** Model defers only when NER confidence is low.\
3. **Privacy-Triggered L2D (Yours):** Model defers when confidence is low OR predicted privacy risk is high.\
\
Sweep the cost threshold ($C_h$) to generate a Pareto frontier (trade-off curves) for each strategy.\
\
### 3.6 Evaluation Metrics (What to Plot)\
To prove your method works, measure and plot:\
* **Coverage / Automation Rate:** What % of documents are handled purely by AI? (Lower human workload = better).\
* **Catastrophic Leakage Rate:** What % of *highly sensitive* documents slipped through with only automated redactions? (Should be near 0 in your proposed strategy).\
* **System F1 Score:** The overall accuracy of the final redacted corpus (AI + Human combined).\
* **Expected Cost:** The total monetary/risk cost of the pipeline based on your defined cost matrix.\
\
### 3.7 Connection to Course Concepts & Reference Paper\
* **Fabric Connection:** You are directly implementing Fabric's *Conditionally Autonomous AI* level with *Institutional Oversight*. In the Fabric paper, rules are often ad-hoc. Your paper mathematically formalizes an "Institutional Privacy Policy" into the routing layer.\
* **Course Material:** You are demonstrating selective prediction, cost-aware triage, and algorithmic governance. \
* **Novelty:** It flips the standard L2D assumption. You are showing that "high model confidence does not equal safe for deployment" when the contextual stakes vary per instance.\
\
### 3.8 Suggested Paper Outline (4 Pages)\
* **Section 1: Introduction (0.75 pages):** Introduce e-discovery, the risk of high-confidence privacy leaks, and frame the paper around context-aware human-AI routing.\
* **Section 2: Methodology (1.25 pages):** Define the dual-objective L2D framework. Formulate the Expected Cost equation. Explain how you proxy human experts using ground-truth data.\
* **Section 3: Experimental Setup (0.75 pages):** Detail the dataset (e.g., Enron), the models (NER + Risk Assessor), and the baseline policies.\
* **Section 4: Results & Discussion (1.25 pages):** Show the Pareto curves (Workload vs. Leakage Rate). Discuss how privacy-triggered deferral prevents catastrophic failures while maintaining high automation rates for trivial documents. Tie back to AI governance and the *Fabric* paper.}