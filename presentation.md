# Privacy-Triggered Deferral in Legal E-Discovery

**Ethar Hamid** | African Institute for Mathematical Sciences (AIMS)

---

## Slide 1 — Problem

- Legal e-discovery requires redacting PII from documents before production to opposing counsel
- Manual review costs **$5–$50/doc** → motivates AI (NER) pipelines
- Standard Learning-to-Defer: route to humans only when model **confidence is low**
- **Blind spot:** NER confidence clusters in a narrow band (std = 0.033) — sensitive and non-sensitive documents have nearly identical mean confidence (0.825 vs 0.827)
- A model can be **highly confident yet the document is high-risk** (e.g., SSNs + multiple identifiers)
- **Core thesis:** Confidence alone is an insufficient routing signal in high-stakes domains

---

## Slide 2 — Method: Privacy-Triggered Deferral

**Key idea:** Add a second routing trigger — predicted privacy risk — alongside confidence.

**Routing policies compared:**

| Policy | Rule |
|---|---|
| Autonomous AI | Automate everything |
| Confidence-only L2D | Defer if confidence < τ_c |
| **Privacy-Triggered L2D** (ours) | **Defer if confidence < τ_c  OR  risk > τ_r** |

**Architecture:**
- Shared **DistilBERT** backbone (66M params) with two heads:
  - **NER head** → per-token redact/keep + confidence c(x)
  - **Risk head** → document-level risk score r(x) ∈ [0, 1]
- Joint training: L = L_NER + L_Risk

**Sensitivity labels:** Generated offline by **Qwen-2.5 7B** (local LLM); only DistilBERT used at inference.

---

## Slide 3 — Experimental Setup

- **Dataset:** 500 synthetic English documents (ai4privacy/pii-masking-400k), 17 PII entity types
- **Split:** 64/16/20 train/val/test; all results = mean ± std over **5 seeds**
- **Cost model:** C_human = $5/doc, C_error = $2/token, C_leak = $50/doc
- **Threshold sweeps:** τ_c ∈ {0.75 … 0.95}, τ_r ∈ {0.2 … 0.6}
- **Key metrics:**
  - Automation rate
  - Sensitive-document exposure (% sensitive docs auto-processed)
  - True leakage rate (% sensitive docs auto-processed AND incompletely redacted)
  - System F1, Expected cost per doc

---

## Slide 4 — Key Results

At **τ_c = 0.85**, comparing confidence-only vs privacy-triggered (τ_r = 0.5):

| Metric | Confidence-only | Privacy-Triggered | Change |
|---|---|---|---|
| Sensitive exposure | 22.7% | **4.6%** | −80% |
| True leakage | 5.5% | **0.4%** | **~14× reduction** |
| System F1 | 0.986 | **0.992** | +0.006 |
| Cost/doc | $4.57 | **$4.48** | comparable |

- At tighter thresholds (τ_c = 0.75, τ_r = 0.4): **true leakage = 0.0%**
- Autonomous baseline: 35.7% leakage, $5.53/doc — worst on both axes

---

## Slide 5 — Insights: When Collaboration Helps / Fails

**When the risk trigger helps most:**
- Documents with high-confidence NER but high contextual risk (e.g., SSN + multiple co-occurring identifiers)
- These sit in the "high-confidence, high-risk" quadrant that confidence-only structurally cannot catch

**When it struggles:**
- Small training set (320 docs) → risk head discrimination is modest (gap = 0.047)
- Meaningful seed-to-seed variance (exposure 4.6% ± 4.3%, leakage 0.4% ± 0.9%)
- Sensitivity labels come from a 7B LLM, not human experts

**Important trade-off:**
- Autonomous → automation bias (trusting AI blindly)
- Confidence-only at τ_c = 0.95 → under-reliance (sends everything to humans)
- Privacy-Triggered → **calibrated middle ground**: model over-flags rather than under-flags (correct failure mode for safety)

---

## Slide 6 — Conclusion

1. **NER confidence is structurally insufficient** for routing in PII redaction — the narrow confidence band (std = 0.033) cannot distinguish risk levels
2. **Privacy-Triggered Deferral** adds a risk axis, reducing true leakage **~14×** (5.5% → 0.4%) at comparable cost
3. **Core insight: capability does not imply permission** — routing policies in high-stakes domains should encode risk signals beyond model confidence
4. **Limitations:** synthetic data, oracle-human assumption, LLM-generated labels — validation on real legal corpora needed

---
