"""
run_experiment.py
Runs multi-seed experiments: for each seed, trains a model on a
different random split, evaluates all policies, and saves per-seed
and aggregated results to experiments/.
"""

import json, sys, random, argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.multitask_model import MultiTaskRedactor, get_tokenizer
from models.train import split_data, train as train_model
from pipeline.router import POLICIES
from evaluate.metrics import compute_metrics

DATA_FILE = Path("data/documents.jsonl")
CKPT_DIR  = Path("models")
OUT_CSV   = Path("experiments/results.csv")
AGG_CSV   = Path("experiments/results_aggregated.csv")
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

C_ERR  = 2.0
C_LEAK = 50.0

BATCH_SIZE   = 32
TAU_C_VALUES = [0.75, 0.80, 0.85, 0.90, 0.95]
TAU_R_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6]
CH_VALUES    = [2.0, 5.0, 10.0, 20.0, 40.0]
SEEDS        = [42, 123, 456, 789, 1024]


def batch_predict(model, tokenizer, records, batch_size=BATCH_SIZE, max_len=128):
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
        for input_ids, attention_mask in tqdm(loader, desc="  inference"):
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


def sweep_one_seed(seed: int, records: list[dict], tokenizer):
    """Train on one seed's split and evaluate all policies."""
    _, _, test_r = split_data(records, seed)
    n_sens = sum(r["is_sensitive"] for r in test_r)
    print(f"\n{'='*60}")
    print(f"SEED {seed}: {len(test_r)} test docs, {n_sens} sensitive")
    print(f"{'='*60}")

    ckpt = CKPT_DIR / f"checkpoint_s{seed}.pt"
    if not ckpt.exists():
        print(f"  Training model for seed {seed}...")
        train_model(seed=seed)

    model = MultiTaskRedactor().to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print(f"  Loaded {ckpt}")

    predictions = batch_predict(model, tokenizer, test_r)

    # Save predictions for first seed (used for scatter plot)
    if seed == SEEDS[0]:
        pred_file = Path("experiments/predictions.json")
        pred_records = [{"conf": p["conf"], "risk": p["risk"],
                         "is_sensitive": p["is_sensitive"]}
                        for p in predictions]
        with open(pred_file, "w") as f:
            json.dump(pred_records, f)
        print(f"  Saved {len(pred_records)} predictions → {pred_file}")

    confs = np.array([p["conf"] for p in predictions])
    risks = np.array([p["risk"] for p in predictions])
    rho, pval = spearmanr(confs, risks)
    print(f"  Spearman ρ(conf, risk) = {rho:.3f} (p={pval:.4f})")

    rows = []

    decisions = evaluate_decisions(predictions, POLICIES["autonomous"], {})
    for C_h in CH_VALUES:
        m = compute_metrics(decisions, C_h=C_h, C_err=C_ERR, C_leak=C_LEAK)
        rows.append({"seed": seed, "policy": "autonomous",
                     "tau_c": None, "tau_r": None, "C_h": C_h, **m})

    for tau_c in TAU_C_VALUES:
        decisions = evaluate_decisions(
            predictions, POLICIES["confidence_only"], {"tau_c": tau_c})
        for C_h in CH_VALUES:
            m = compute_metrics(decisions, C_h=C_h, C_err=C_ERR, C_leak=C_LEAK)
            rows.append({"seed": seed, "policy": "confidence_only",
                         "tau_c": tau_c, "tau_r": None, "C_h": C_h, **m})

    for tau_c in TAU_C_VALUES:
        for tau_r in TAU_R_VALUES:
            decisions = evaluate_decisions(
                predictions, POLICIES["privacy_triggered"],
                {"tau_c": tau_c, "tau_r": tau_r})
            for C_h in CH_VALUES:
                m = compute_metrics(decisions, C_h=C_h, C_err=C_ERR, C_leak=C_LEAK)
                rows.append({"seed": seed, "policy": "privacy_triggered",
                             "tau_c": tau_c, "tau_r": tau_r,
                             "C_h": C_h, **m})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    records   = [json.loads(l) for l in open(DATA_FILE)]
    tokenizer = get_tokenizer()
    print(f"Loaded {len(records)} documents, running seeds: {args.seeds}")

    all_rows = []
    for seed in args.seeds:
        rows = sweep_one_seed(seed, records, tokenizer)
        all_rows.extend(rows)

    OUT_CSV.parent.mkdir(exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nPer-seed results → {OUT_CSV}  ({len(df)} rows)")

    # Aggregate across seeds: mean ± std for each (policy, tau_c, tau_r, C_h)
    group_cols = ["policy", "tau_c", "tau_r", "C_h"]
    metric_cols = ["automation_rate", "exposure_rate", "leakage_rate",
                   "system_f1", "expected_cost", "human_cost", "error_cost",
                   "leak_cost"]
    agg = df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(AGG_CSV, index=False)
    print(f"Aggregated results → {AGG_CSV}  ({len(agg)} rows)")

    # Summary at representative operating point
    rep = agg[(agg["C_h"] == 5.0)]
    for policy in ["autonomous", "confidence_only", "privacy_triggered"]:
        if policy == "autonomous":
            row = rep[rep["policy"] == policy].iloc[0]
        elif policy == "confidence_only":
            row = rep[(rep["policy"] == policy) & (rep["tau_c"] == 0.85)].iloc[0]
        else:
            row = rep[(rep["policy"] == policy) &
                      (rep["tau_c"] == 0.85) & (rep["tau_r"] == 0.5)].iloc[0]
        print(f"\n{policy}:")
        print(f"  Auto:    {row['automation_rate_mean']:.3f} ± {row['automation_rate_std']:.3f}")
        print(f"  Leak:    {row['leakage_rate_mean']:.3f} ± {row['leakage_rate_std']:.3f}")
        print(f"  F1:      {row['system_f1_mean']:.3f} ± {row['system_f1_std']:.3f}")
        print(f"  Cost:    ${row['expected_cost_mean']:.2f} ± ${row['expected_cost_std']:.2f}")


if __name__ == "__main__":
    main()