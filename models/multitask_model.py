"""
multitask_model.py
Shared DistilBERT backbone with:
  - NER head:  token-level binary classifier (redact / keep)
  - Risk head: document-level sensitivity score (0–1)
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast


class MultiTaskRedactor(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", num_ner_labels: int = 2):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size  # 768

        # NER head: per-token binary classification
        self.ner_head = nn.Linear(hidden, num_ner_labels)

        # Risk head: document-level regression (sigmoid output)
        self.risk_head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_hidden = outputs.last_hidden_state        # (B, T, H)
        cls_hidden   = token_hidden[:, 0, :]            # (B, H)  — [CLS] token

        ner_logits   = self.ner_head(token_hidden)      # (B, T, 2)
        risk_score   = self.risk_head(cls_hidden)       # (B, 1)

        return ner_logits, risk_score.squeeze(-1)       # (B, T, 2), (B,)


def get_tokenizer(model_name: str = "distilbert-base-uncased"):
    return DistilBertTokenizerFast.from_pretrained(model_name)