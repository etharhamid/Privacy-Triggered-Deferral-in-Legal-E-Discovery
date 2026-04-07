"""
metrics.py
Computes: automation rate, leakage rate, system F1, expected cost,
and precision/recall breakdown for sensitive vs non-sensitive documents.
"""

from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np


def _token_pr_f1(results: list[dict]) -> dict:
    """Token-level precision, recall, F1 for a list of result dicts.
    Deferred documents are treated as human-labelled (perfect)."""
    all_true, all_pred = [], []
    for r in results:
        t = r["true_mask"]
        p = r["true_mask"] if r["deferred"] else r["pred_mask"]
        min_len = min(len(t), len(p))
        all_true.extend(t[:min_len])
        all_pred.extend(p[:min_len])
    if not all_true:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "precision": precision_score(all_true, all_pred, zero_division=0),
        "recall":    recall_score(all_true, all_pred, zero_division=0),
        "f1":        f1_score(all_true, all_pred, zero_division=0),
    }


def compute_metrics(results: list[dict], C_h=5.0, C_err=2.0, C_leak=50.0) -> dict:
    """
    Each result dict contains:
        deferred      (bool)
        is_sensitive  (int)  ground-truth 0/1
        pred_mask     (list[int]) AI prediction
        true_mask     (list[int]) ground-truth
    """
    n = len(results)
    automated  = [r for r in results if not r["deferred"]]
    deferred   = [r for r in results if r["deferred"]]

    # Automation rate
    automation_rate = len(automated) / n

    # Sensitive-exposure rate: sensitive docs that were NOT deferred
    sensitive_docs    = [r for r in results if r["is_sensitive"]]
    nonsensitive_docs = [r for r in results if not r["is_sensitive"]]
    exposed           = [r for r in sensitive_docs if not r["deferred"]]
    exposure_rate     = len(exposed) / max(len(sensitive_docs), 1)

    # True leakage rate: sensitive docs automated AND redaction missed ≥1 token
    leaked = [r for r in exposed
              if sum(1 for ti, pi in zip(r["true_mask"], r["pred_mask"])
                     if ti == 1 and pi == 0) > 0]
    leakage_rate = len(leaked) / max(len(sensitive_docs), 1)

    # Overall token-level precision / recall / F1
    overall = _token_pr_f1(results)

    # Breakdown by document sensitivity
    sens   = _token_pr_f1(sensitive_docs)
    nosens = _token_pr_f1(nonsensitive_docs)

    # Expected cost — broken down by component
    human_cost_total = 0.0
    error_cost_total = 0.0
    leak_cost_total  = 0.0
    for r in results:
        if r["deferred"]:
            human_cost_total += C_h
        else:
            t, p   = r["true_mask"], r["pred_mask"]
            missed = sum(1 for ti, pi in zip(t, p) if ti == 1 and pi == 0)
            error_cost_total += missed * C_err
            if r["is_sensitive"] and missed > 0:
                leak_cost_total += r["risk"] * C_leak

    return {
        "automation_rate": automation_rate,
        "exposure_rate":   exposure_rate,
        "leakage_rate":    leakage_rate,
        "system_f1":       overall["f1"],
        "system_precision": overall["precision"],
        "system_recall":    overall["recall"],
        "sens_precision":  sens["precision"],
        "sens_recall":     sens["recall"],
        "sens_f1":         sens["f1"],
        "nsens_precision": nosens["precision"],
        "nsens_recall":    nosens["recall"],
        "nsens_f1":        nosens["f1"],
        "expected_cost":   (human_cost_total + error_cost_total + leak_cost_total) / n,
        "human_cost":      human_cost_total / n,
        "error_cost":      error_cost_total / n,
        "leak_cost":       leak_cost_total / n,
        "n_deferred":      len(deferred),
        "n_total":         n,
    }