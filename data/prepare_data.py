"""
prepare_data.py
Downloads the ai4privacy/pii-masking-400k dataset and builds
word-level tokens + binary PII masks from its character-level
privacy_mask spans.  Sensitivity scores are annotated by a local
LLM (Ollama) that reads each document's full text and PII context.
Saves: data/documents.jsonl
"""

import json, re, time, urllib.request, urllib.error
from pathlib import Path

OUTPUT_FILE  = Path("data/documents.jsonl")
LLM_CACHE    = Path("data/llm_sensitivity_cache.json")
N_DOCS       = 500
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

SENSITIVITY_PROMPT = (
    "You are a legal privacy expert. Rate this document's contextual privacy "
    "sensitivity from 0.0 to 1.0.\n\n"
    "Consider ALL of the following:\n"
    "- What types of PII are present (names, SSNs, credit cards, medical IDs)?\n"
    "- Does the surrounding text discuss sensitive topics (medical, financial, "
    "legal, whistleblowing, sexual, criminal)?\n"
    "- Could someone be re-identified from context even after PII redaction?\n"
    "- How many different PII types co-occur (more types = higher risk)?\n\n"
    "Score guide:\n"
    "  0.0–0.2: routine text, only common names/cities\n"
    "  0.2–0.4: some PII but low re-identification risk\n"
    "  0.4–0.6: moderate — multiple PII types or mildly sensitive context\n"
    "  0.6–0.8: high — financial/gov IDs or clearly sensitive discussion\n"
    "  0.8–1.0: critical — SSN/passwords combined with sensitive context\n\n"
    "PII types found: {pii_types}\n\n"
    "Document:\n{text}\n\n"
    "Reply with ONLY a single decimal number between 0.0 and 1.0."
)


def _call_ollama(prompt: str, retries: int = 2) -> str | None:
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 10, "temperature": 0},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())["response"].strip()
        except (urllib.error.URLError, TimeoutError, KeyError):
            if attempt < retries:
                time.sleep(2)
    return None


def _parse_score(raw: str | None) -> float | None:
    if raw is None:
        return None
    m = re.search(r"(0(?:\.\d+)?|1(?:\.0+)?)", raw)
    if m:
        return round(min(max(float(m.group(1)), 0.05), 0.95), 4)
    return None


def score_with_llm(text: str, pii_types: list[str]) -> float | None:
    truncated = " ".join(text.split()[:400])
    unique_types = sorted(set(pii_types)) if pii_types else ["NONE"]
    prompt = SENSITIVITY_PROMPT.format(
        text=truncated, pii_types=", ".join(unique_types),
    )
    raw = _call_ollama(prompt)
    return _parse_score(raw)


def build_word_mask(source_text: str, privacy_mask: list[dict]) -> tuple[list[str], list[int]]:
    """Convert character-level PII spans into word-level binary mask."""
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


def rescore_existing():
    """Re-score an existing documents.jsonl with LLM-based sensitivity."""
    print("Re-scoring existing documents with LLM...")

    records = [json.loads(l) for l in open(OUTPUT_FILE)]
    print(f"Loaded {len(records)} documents from {OUTPUT_FILE}")

    llm_cache: dict[str, float] = {}
    if LLM_CACHE.exists():
        llm_cache = json.loads(LLM_CACHE.read_text())
        print(f"Loaded {len(llm_cache)} cached LLM scores")

    llm_ok, llm_cached, llm_fail = 0, 0, 0

    for i, rec in enumerate(records):
        cache_key = str(rec["id"])

        if cache_key in llm_cache:
            sensitivity = llm_cache[cache_key]
            llm_cached += 1
        else:
            llm_score = score_with_llm(rec["text"], rec.get("pii_types", []))
            if llm_score is not None:
                sensitivity = llm_score
                llm_cache[cache_key] = sensitivity
                llm_ok += 1
            else:
                labels = rec.get("pii_types", [])
                PII_RISK = {"SOCIALNUM": 0.95, "CREDITCARDNUMBER": 0.90,
                            "PASSWORD": 0.85, "ACCOUNTNUM": 0.80, "TAXNUM": 0.75,
                            "DRIVERLICENSENUM": 0.55, "IDCARDNUM": 0.45}
                max_r = max((PII_RISK.get(l, 0.05) for l in labels), default=0.05)
                sensitivity = round(max_r * 0.7, 4)
                llm_cache[cache_key] = sensitivity
                llm_fail += 1

        rec["sensitivity"]  = sensitivity
        rec["is_sensitive"] = int(sensitivity >= 0.5)
        if "split" in rec:
            del rec["split"]

        if (i + 1) % 25 == 0:
            LLM_CACHE.write_text(json.dumps(llm_cache))
            print(f"  [{i+1}/{len(records)}] scored "
                  f"(llm={llm_ok} cached={llm_cached} fallback={llm_fail})")

    LLM_CACHE.write_text(json.dumps(llm_cache))

    with open(OUTPUT_FILE, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    n = len(records)
    sensitive_count = sum(r["is_sensitive"] for r in records)
    pii_count       = sum(r["has_pii"]       for r in records)
    print(f"\nSaved {n} records → {OUTPUT_FILE}")
    print(f"  Has PII:       {pii_count}/{n} ({100*pii_count/n:.1f}%)")
    print(f"  Sensitive:     {sensitive_count}/{n} ({100*sensitive_count/n:.1f}%)")
    print(f"  LLM scored:    {llm_ok}")
    print(f"  LLM cached:    {llm_cached}")
    print(f"  LLM fallback:  {llm_fail}")


def main():
    if OUTPUT_FILE.exists():
        rescore_existing()
    else:
        print("ERROR: No existing documents.jsonl found.")
        print("Please run the original prepare_data.py first to download data.")


if __name__ == "__main__":
    main()