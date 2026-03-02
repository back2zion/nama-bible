#!/usr/bin/env python3
"""Evaluate saved checkpoint on Nama test set."""

import json
import random
import warnings

warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("accelerate.utils.modeling").setLevel(logging.ERROR)

import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ── Config ───────────────────────────────────────────────────────────────────

from pathlib import Path
_BASE = Path(__file__).resolve().parent.parent
CHECKPOINT = str(_BASE / "output" / "nllb-nama-pilot" / "checkpoint-540")
DATA_PATH = str(_BASE / "nama_eng_parallel.json")
NEW_LANG_CODE = "nmx_Latn"
PROXY_LANG_CODE = "tpi_Latn"
MAX_LENGTH = 128
SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10


def load_test_data():
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    aligned = [d for d in data if d.get("aligned")]
    random.seed(SEED)
    shuffled = aligned.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    test_start = int(n * TRAIN_RATIO) + int(n * VAL_RATIO)
    test = shuffled[test_start:]
    print(f"Test set: {len(test)} verses")
    return test


def translate(model, tokenizer, test_data, forced_bos_id):
    tokenizer.src_lang = "eng_Latn"
    predictions = []
    model.eval()
    for item in test_data:
        inputs = tokenizer(
            item["english"], return_tensors="pt", max_length=MAX_LENGTH, truncation=True
        ).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                max_new_tokens=MAX_LENGTH,
                repetition_penalty=1.2,
            )
        pred = tokenizer.decode(generated[0], skip_special_tokens=True)
        predictions.append(pred)
    return predictions


def evaluate(predictions, references):
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    return bleu.score, chrf.score


def main():
    test_data = load_test_data()
    references = [item["nama"] for item in test_data]

    print(f"\nLoading checkpoint: {CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT, tie_word_embeddings=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model on {device}")

    # Fine-tuned evaluation
    print("\n" + "=" * 60)
    print("FINE-TUNED (multilingual + Nama)")
    print("=" * 60)
    forced_bos = tokenizer.convert_tokens_to_ids(NEW_LANG_CODE)
    preds = translate(model, tokenizer, test_data, forced_bos)
    bleu, chrf = evaluate(preds, references)
    print(f"  BLEU:  {bleu:.2f}")
    print(f"  chrF:  {chrf:.2f}")

    # Sample translations
    print("\n" + "=" * 60)
    print("SAMPLE TRANSLATIONS (10 verses)")
    print("=" * 60)
    for item, pred in zip(test_data[:10], preds[:10]):
        print(f"\n--- [{item['ref']}] ---")
        print(f"  ENG:   {item['english']}")
        print(f"  NAMA:  {item['nama']}")
        print(f"  PRED:  {pred}")

    # Save results
    results = {
        "checkpoint": CHECKPOINT,
        "test_size": len(test_data),
        "bleu": bleu,
        "chrf": chrf,
        "predictions": [
            {"ref": t["ref"], "english": t["english"], "nama": t["nama"], "predicted": p}
            for t, p in zip(test_data, preds)
        ],
    }
    out_path = str(_BASE / "evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
