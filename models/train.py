"""
train.py
Trains the MultiTaskRedactor on documents.jsonl.
Saves checkpoint to models/checkpoint.pt
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from models.multitask_model import MultiTaskRedactor, get_tokenizer

DATA_FILE   = Path("data/documents.jsonl")
CKPT_FILE   = Path("models/checkpoint.pt")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS      = 3
BATCH_SIZE  = 16
LR          = 2e-5
MAX_LEN     = 128


class RedactionDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.records   = records
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec    = self.records[idx]
        tokens = rec["tokens"]
        mask   = rec["redaction_mask"]       # per-token binary labels
        sens   = rec["sensitivity"]          # float 0-1

        # Tokenize — word_ids lets us align sub-tokens to original tokens
        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )

        # Align labels: first sub-token of each word gets label; others get -100 (ignored)
        word_ids   = enc.word_ids(batch_index=0)
        ner_labels = []
        seen_words = set()
        for wid in word_ids:
            if wid is None:
                ner_labels.append(-100)
            elif wid in seen_words:
                ner_labels.append(-100)
            else:
                seen_words.add(wid)
                ner_labels.append(mask[wid] if wid < len(mask) else 0)

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "ner_labels":     torch.tensor(ner_labels, dtype=torch.long),
            "risk_score":     torch.tensor(sens, dtype=torch.float),
        }


def train():
    records = [json.loads(l) for l in open(DATA_FILE)]
    split   = int(0.8 * len(records))
    train_r, val_r = records[:split], records[split:]

    tokenizer  = get_tokenizer()
    train_ds   = RedactionDataset(train_r, tokenizer)
    val_ds     = RedactionDataset(val_r,   tokenizer)
    train_dl   = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl     = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model = MultiTaskRedactor().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    ner_loss_fn  = nn.CrossEntropyLoss(ignore_index=-100)
    risk_loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            ner_labels     = batch["ner_labels"].to(DEVICE)
            risk_targets   = batch["risk_score"].to(DEVICE)

            ner_logits, risk_pred = model(input_ids, attention_mask)

            loss_ner  = ner_loss_fn(ner_logits.view(-1, 2), ner_labels.view(-1))
            loss_risk = risk_loss_fn(risk_pred, risk_targets)
            loss      = loss_ner + loss_risk           # equal weighting

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"  Train loss: {total_loss/len(train_dl):.4f}")

        # Quick validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dl:
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                ner_labels     = batch["ner_labels"].to(DEVICE)
                risk_targets   = batch["risk_score"].to(DEVICE)
                ner_logits, risk_pred = model(input_ids, attention_mask)
                val_loss += (ner_loss_fn(ner_logits.view(-1, 2), ner_labels.view(-1))
                             + risk_loss_fn(risk_pred, risk_targets)).item()
        print(f"  Val   loss: {val_loss/len(val_dl):.4f}")

    CKPT_FILE.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), CKPT_FILE)
    print(f"Saved checkpoint → {CKPT_FILE}")


if __name__ == "__main__":
    train()