# Privacy-Triggered Deferral in Legal E-Discovery

**Risk-Aware Routing for PII Redaction**

Standard Learning-to-Defer (L2D) systems route documents to human reviewers only when model confidence is low. This project shows that confidence alone is insufficient in privacy-sensitive domains: NER confidence clusters in a narrow band (std = 0.033), providing almost no signal about document-level privacy risk. We propose **Privacy-Triggered Deferral**, a dual-objective routing policy that defers when confidence is low **or** contextual privacy risk is high:

$$\text{Defer}(x) = \mathbb{1}[c(x) < \tau_c \;\lor\; r(x) > \tau_r]$$

## Key Results

Mean ± std over 5 seeds, at τ_c = 0.85, C_h = $5/doc:

| Policy | Auto. (%) | Exposure (%) | Leakage (%) | F1 | Cost/doc |
|--------|-----------|-------------|-------------|------|----------|
| Autonomous AI | 100 | 100 | 35.7 ± 8.2 | .915 | $5.53 |
| Confidence-only (τ_c=0.85) | 25.6 ± 6.5 | 22.7 ± 5.7 | 5.5 ± 3.9 | .986 | $4.57 |
| **Privacy-Triggered** (τ_c=0.85, τ_r=0.5) | **13.2 ± 6.5** | **4.6 ± 4.3** | **0.4 ± 0.9** | **.992** | **$4.48** |

Privacy-Triggered Deferral reduces true leakage by **~14×** (5.5% → 0.4%) and sensitive-document exposure from 22.7% to 4.6% at comparable cost. At tighter thresholds (τ_c=0.75, τ_r=0.4), true leakage reaches **0.0%**.

## Architecture

A single multi-task DistilBERT model (66M params) with two heads:

```
Input tokens --> DistilBERT Encoder (768-dim)
                    |-- NER Head: Linear(768->2)       --> per-token redact/keep + confidence c(x)
                    |-- Risk Head: Linear(768->128->1)  --> document-level risk r(x) ∈ [0,1]
```

Sensitivity labels are generated offline by a local **Qwen-2.5 7B** LLM; only DistilBERT is used at inference.

## Project Structure

```
ediscovery-l2d/
├── data/
│   ├── prepare_data.py              # Download dataset, LLM sensitivity annotation
│   ├── documents.jsonl              # 500 processed documents with sensitivity scores
│   └── llm_sensitivity_cache.json   # Cached LLM sensitivity ratings
├── models/
│   ├── multitask_model.py           # MultiTaskRedactor model definition
│   └── train.py                     # Multi-seed training (joint NER + risk loss)
├── pipeline/
│   ├── redactor.py                  # Single-document inference
│   └── router.py                    # Three routing policies
├── evaluate/
│   └── metrics.py                   # Automation rate, leakage, exposure, F1, cost
├── experiments/
│   ├── run_experiment.py            # Full parameter sweep across seeds
│   ├── results.csv                  # Per-seed experiment results
│   ├── results_aggregated.csv       # Aggregated mean ± std results
│   └── predictions.json             # Per-document model predictions
├── plots/
│   ├── plot_pareto.py               # Pareto frontier plots
│   ├── plot_paper.py                # Publication figures (Figs 1–6)
│   └── figures/                     # Generated figures
├── latex.txt                        # LaTeX source (NeurIPS workshop format)
├── presentation.md                  # 7-minute presentation slides
└── requirements.txt
```

## Setup & Reproduction

**Requirements:** Python 3.10+, ~4 GB disk (for dataset + model).

```bash
# 1. Clone and install
git clone https://github.com/etharhamid/Privacy-Triggered-Deferral-in-Legal-E-Discovery.git
cd Privacy-Triggered-Deferral-in-Legal-E-Discovery
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Prepare data (downloads ai4privacy/pii-masking-400k, ~2 min)
python data/prepare_data.py

# 3. Train model (~5 min on CPU, ~1 min on GPU per seed)
python models/train.py

# 4. Run experiments (generates results.csv + predictions.json)
python experiments/run_experiment.py

# 5. Generate figures
python -m plots.plot_pareto
python -m plots.plot_paper
```

> **Note:** Model checkpoints (`models/checkpoint_s*.pt`, ~254 MB each) are excluded from the repo. Run step 3 to regenerate them.

## Dataset

**Source:** [ai4privacy/pii-masking-400k](https://huggingface.co/datasets/ai4privacy/pii-masking-400k) — synthetic documents with character-level PII annotations across 17 entity types.

- 500 documents, split 64/16/20 (train/val/test) per seed
- ~47 of 100 test documents are sensitive per seed
- 240/500 (48%) labeled sensitive (LLM score ≥ 0.5)

## Three Routing Policies

| Policy | Rule | Oversight Level |
|--------|------|-----------------|
| **Autonomous AI** | Automate all | Full Autonomy |
| **Confidence-only L2D** | Defer if c(x) < τ_c | Standard Selective Prediction |
| **Privacy-Triggered L2D** | Defer if c(x) < τ_c **or** r(x) > τ_r | Conditionally Autonomous AI |

## Citation

```bibtex
@misc{hamid2025privacy,
  title={Privacy-Triggered Deferral in Legal E-Discovery: Risk-Aware Routing for PII Redaction},
  author={Ethar Hamid},
  year={2025},
  url={https://github.com/etharhamid/Privacy-Triggered-Deferral-in-Legal-E-Discovery}
}
```

## License

This project is for academic/research purposes.
