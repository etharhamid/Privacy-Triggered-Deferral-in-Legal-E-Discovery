"""
prepare_data.py
Downloads the ai4privacy/pii-masking-400k dataset and builds
word-level tokens + binary PII masks from its character-level
privacy_mask spans.  Each document also gets a keyword-based
sensitivity score.
Saves: data/documents.jsonl
"""

import json
from pathlib import Path
from datasets import load_dataset

OUTPUT_FILE = Path("data/documents.jsonl")
N_DOCS = 500
TRAIN_RATIO = 0.8

PII_TYPE_RISK = {
    "SOCIALNUM":         0.95,
    "CREDITCARDNUMBER":  0.90,
    "PASSWORD":          0.85,
    "ACCOUNTNUM":        0.80,
    "TAXNUM":            0.75,
    "DRIVERLICENSENUM":  0.55,
    "IDCARDNUM":         0.45,
    "DATEOFBIRTH":       0.30,
    "TELEPHONENUM":      0.20,
    "EMAIL":             0.15,
    "USERNAME":          0.12,
    "ZIPCODE":           0.10,
    "STREET":            0.08,
    "BUILDINGNUM":       0.06,
    "CITY":              0.05,
    "SURNAME":           0.05,
    "GIVENNAME":         0.05,
}


def get_sensitivity_score(privacy_mask: list[dict], num_tokens: int) -> float:
    """Score document sensitivity from the most dangerous PII type present,
    boosted by PII-type diversity (more types → easier re-identification)."""
    if not privacy_mask:
        return 0.05

    labels = [span["label"] for span in privacy_mask]
    max_type_risk   = max(PII_TYPE_RISK.get(l, 0.05) for l in labels)
    unique_labels   = set(labels)
    diversity_boost = min(len(unique_labels) / 5.0, 1.0)

    score = max_type_risk * (0.7 + 0.3 * diversity_boost)
    return round(min(max(score, 0.05), 0.95), 4)


def build_word_mask(source_text: str, privacy_mask: list[dict]) -> tuple[list[str], list[int]]:
    """Convert character-level PII spans into word-level binary mask.

    Returns (tokens, mask) where mask[i]=1 if word i overlaps any PII span.
    """
    tokens = source_text.split()
    mask   = [0] * len(tokens)

    char_to_word = {}
    pos = 0
    for wi, tok in enumerate(tokens):
        for ci in range(len(tok)):
            char_to_word[pos + ci] = wi
        pos += len(tok) + 1

    for span in privacy_mask:
        for c in range(span["start"], span["end"]):
            wi = char_to_word.get(c)
            if wi is not None:
                mask[wi] = 1

    return tokens, mask


def assign_split(i: int, n: int) -> str:
    return "train" if i < int(n * TRAIN_RATIO) else "test"


def main():
    print("Loading dataset...")
    ds = load_dataset("ai4privacy/pii-masking-400k", split="train")
    print(f"Dataset columns: {ds.column_names}")

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    records = []

    n = min(N_DOCS, len(ds))
    print(f"Processing {n} documents...")

    for i, example in enumerate(ds.select(range(n))):
        tokens, redaction_mask = build_word_mask(
            example["source_text"], example["privacy_mask"])

        text        = " ".join(tokens)
        has_pii     = int(any(m == 1 for m in redaction_mask))
        sensitivity = get_sensitivity_score(example["privacy_mask"], len(tokens))
        pii_types   = [span["label"] for span in example["privacy_mask"]]

        records.append({
            "id":             i,
            "split":          assign_split(i, n),
            "tokens":         tokens,
            "redaction_mask": redaction_mask,
            "has_pii":        has_pii,
            "text":           text,
            "sensitivity":    sensitivity,
            "is_sensitive":   int(sensitivity >= 0.5),
            "pii_types":      pii_types,
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n}] processed")

    with open(OUTPUT_FILE, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    sensitive_count = sum(r["is_sensitive"] for r in records)
    pii_count       = sum(r["has_pii"]       for r in records)
    print(f"\nSaved {len(records)} records → {OUTPUT_FILE}")
    print(f"  Has PII:     {pii_count}/{n} ({100*pii_count/n:.1f}%)")
    print(f"  Sensitive:   {sensitive_count}/{n} ({100*sensitive_count/n:.1f}%)")
    print(f"  Train/Test:  {int(n*TRAIN_RATIO)}/{n - int(n*TRAIN_RATIO)}")


if __name__ == "__main__":
    main()