"""
redactor.py
Applies a token-level binary mask to produce a redacted string.
Also computes per-document confidence score c(x) and routes
documents through the model.
"""

import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast
from models.multitask_model import MultiTaskRedactor, get_tokenizer

REDACT_TOKEN = "[REDACTED]"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def apply_mask(tokens: list[str], mask: list[int]) -> str:
    """Replace tokens where mask==1 with [REDACTED]."""
    return " ".join(REDACT_TOKEN if m else t for t, m in zip(tokens, mask))


def predict(
    model: MultiTaskRedactor,
    tokenizer: DistilBertTokenizerFast,
    tokens: list[str],
    max_len: int = 128,
) -> tuple[float, float, list[int]]:
    """
    Returns:
        conf   (float): mean max-prob over predicted entity tokens (proxy for c(x))
        risk   (float): document-level sensitivity score r(x) in [0,1]
        pred_mask (list[int]): predicted binary redaction mask
    """
    model.eval()
    with torch.no_grad():
        enc = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        ner_logits, risk_score = model(input_ids, attention_mask)

        probs      = F.softmax(ner_logits[0], dim=-1)   # (T, 2)
        pred_ids   = probs.argmax(dim=-1).cpu().tolist() # (T,)
        max_probs  = probs.max(dim=-1).values.cpu().tolist()

    # Align sub-tokens back to original tokens
    word_ids   = enc.word_ids(batch_index=0)
    pred_mask  = []
    seen       = set()
    for wid, pred, p in zip(word_ids, pred_ids, max_probs):
        if wid is None or wid in seen:
            continue
        seen.add(wid)
        pred_mask.append(pred)

    # c(x): mean confidence on tokens predicted as entities (or overall mean)
    entity_probs = [p for pid, p in zip(pred_ids, max_probs) if pid == 1]
    conf = float(sum(entity_probs) / len(entity_probs)) if entity_probs else float(
        sum(max_probs) / len(max_probs)
    )
    risk = float(risk_score.item())

    return conf, risk, pred_mask