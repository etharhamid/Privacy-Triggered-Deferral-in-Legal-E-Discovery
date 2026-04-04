"""
run_experiment.py
Loads the trained model, runs all policies over the test set
with parameter sweeps, and saves results to experiments/results.csv
"""

import json, sys
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.multitask_model import MultiTaskRedactor, get_tokenizer
from pipeline.router import POLICIES
from evaluate.metrics import compute_metrics

DATA_FILE = Path("data/documents.jsonl")
CKPT_FILE = Path("models/checkpoint.pt")
OUT_CSV   = Path("experiments/results.csv")
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# Cost parameters
C_H    = 5.0
C_ERR  = 2.0
C_LEAK = 50.0

# Threshold sweeps
BATCH_SIZE = 32
TAU_C_VALUES = [0.75, 0.80, 0.85, 0.90, 0.95]
TAU_R_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6]
CH_VALUES    = [2.0, 5.0, 10.0, 20.0, 40.0]


def load_data(split="test"):
    records = [json.loads(l) for l in open(DATA_FILE)]
    n       = len(records)
    return records[int(0.8 * n):]   # held-out test set


def batch_predict(model, tokenizer, records, batch_size=BATCH_SIZE, max_len=128):
    """Batched inference over all records using DataLoader."""
    all_tokens = [rec["tokens"] for rec in records]
    enc = tokenizer(
        all_tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
    )
    word_ids_list = [enc.word_ids(batch_index=i) for i in range(len(records))]

    dataset = TensorDataset(enc["input_ids"], enc["attention_mask"])
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_ner_probs, all_risk_scores = [], []
    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(loader, desc="Batched inference"):
            ner_logits, risk = model(
                input_ids.to(DEVICE), attention_mask.to(DEVICE)
            )
            all_ner_probs.append(F.softmax(ner_logits, dim=-1).cpu())
            all_risk_scores.append(risk.cpu())

    all_ner_probs   = torch.cat(all_ner_probs, dim=0)
    all_risk_scores = torch.cat(all_risk_scores, dim=0)

    predictions = []
    for i, rec in enumerate(records):
        probs     = all_ner_probs[i]
        pred_ids  = probs.argmax(dim=-1).tolist()
        max_probs = probs.max(dim=-1).values.tolist()

        pred_mask, seen = [], set()
        for wid, pred in zip(word_ids_list[i], pred_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            pred_mask.append(pred)

        entity_probs = [p for pid, p in zip(pred_ids, max_probs) if pid == 1]
        conf = (sum(entity_probs) / len(entity_probs)) if entity_probs else (
            sum(max_probs) / len(max_probs)
        )
        predictions.append({
            "conf":         float(conf),
            "risk":         float(all_risk_scores[i].item()),
            "pred_mask":    pred_mask,
            "true_mask":    rec["redaction_mask"],
            "is_sensitive": rec["is_sensitive"],
        })
    return predictions


def evaluate_decisions(predictions, policy_fn, policy_kwargs):
    """Apply a routing policy to all predictions (C_h-independent)."""
    results = []
    for p in predictions:
        decision = policy_fn(conf=p["conf"], risk=p["risk"], **policy_kwargs)
        results.append({
            "deferred":     decision.defer,
            "reason":       decision.reason,
            "is_sensitive": p["is_sensitive"],
            "pred_mask":    p["pred_mask"],
            "true_mask":    p["true_mask"],
            "risk":         p["risk"],
        })
    return results


def main():
    print("Loading data and model...")
    records   = load_data()
    tokenizer = get_tokenizer()
    model     = MultiTaskRedactor().to(DEVICE)

    if CKPT_FILE.exists():
        model.load_state_dict(torch.load(CKPT_FILE, map_location=DEVICE))
        print(f"Loaded checkpoint from {CKPT_FILE}")
    else:
        print("WARNING: No checkpoint found — using random weights (run train.py first)")

    # Pre-compute predictions for all test docs (batched for speed)
    print("Running inference on test set...")
    predictions = batch_predict(model, tokenizer, records)

    # Save per-document predictions for plotting (fig1 scatter)
    pred_file = Path("experiments/predictions.json")
    pred_records = [{"conf": p["conf"], "risk": p["risk"],
                     "is_sensitive": p["is_sensitive"]}
                    for p in predictions]
    with open(pred_file, "w") as f:
        json.dump(pred_records, f)
    print(f"Saved {len(pred_records)} predictions → {pred_file}")

    rows = []

    # --- Strategy 1: Autonomous AI (no deferral, single operating point) ---
    decisions = evaluate_decisions(predictions, POLICIES["autonomous"], {})
    for C_h in CH_VALUES:
        m = compute_metrics(decisions, C_h=C_h, C_err=C_ERR, C_leak=C_LEAK)
        rows.append({"policy": "autonomous", "tau_c": None, "tau_r": None,
                     "C_h": C_h, **m})

    # --- Strategy 2: Confidence-Based L2D (sweep tau_c × C_h) ---
    for tau_c in TAU_C_VALUES:
        decisions = evaluate_decisions(
            predictions, POLICIES["confidence_only"], {"tau_c": tau_c})
        for C_h in CH_VALUES:
            m = compute_metrics(decisions, C_h=C_h, C_err=C_ERR, C_leak=C_LEAK)
            rows.append({"policy": "confidence_only", "tau_c": tau_c,
                         "tau_r": None, "C_h": C_h, **m})

    # --- Strategy 3: Privacy-Triggered L2D (sweep tau_c × tau_r × C_h) ---
    for tau_c in TAU_C_VALUES:
        for tau_r in TAU_R_VALUES:
            decisions = evaluate_decisions(
                predictions, POLICIES["privacy_triggered"],
                {"tau_c": tau_c, "tau_r": tau_r})
            for C_h in CH_VALUES:
                m = compute_metrics(decisions, C_h=C_h, C_err=C_ERR, C_leak=C_LEAK)
                rows.append({"policy": "privacy_triggered", "tau_c": tau_c,
                             "tau_r": tau_r, "C_h": C_h, **m})

    OUT_CSV.parent.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nResults saved → {OUT_CSV}  ({len(df)} rows)")
    summary = (df.groupby("policy")[["automation_rate", "leakage_rate",
                                     "system_f1", "expected_cost"]]
               .agg(["min", "max"]))
    print(summary)


if __name__ == "__main__":
    main()