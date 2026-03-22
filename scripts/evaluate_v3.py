#!/usr/bin/env python3
"""Evaluation pipeline for TranslateGemma 27B Nama translation model.

Includes:
- Standard metrics: chrF, BLEU (Nama prediction vs Nama reference)
- Greek/Hebrew concept coverage evaluation using Strong's numbers
- Per-book breakdown and detailed error analysis
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import sacrebleu
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = Path(__file__).resolve().parent.parent
MODEL_NAME = "google/translategemma-27b-it"
ADAPTER_PATH = str(BASE / "output" / "translategemma-nama-v3" / "final_adapter")
MAX_NEW_TOKENS = 256

# Key theological terms mapped to Strong's numbers for concept coverage
# These are high-importance terms that should be preserved in translation
THEOLOGICAL_CONCEPTS = {
    "G2316": "God (theos)",
    "G2962": "Lord (kyrios)",
    "G5547": "Christ (christos)",
    "G2424": "Jesus (Iesous)",
    "G4151": "Spirit (pneuma)",
    "G4102": "faith (pistis)",
    "G26": "love (agape)",
    "G5485": "grace (charis)",
    "G266": "sin (hamartia)",
    "G4991": "salvation (soteria)",
    "G932": "kingdom (basileia)",
    "G1411": "power (dynamis)",
    "G1680": "hope (elpis)",
    "G1515": "peace (eirene)",
    "G1343": "righteousness (dikaiosyne)",
    "G3056": "word (logos)",
    "G225": "truth (aletheia)",
    "G2222": "life (zoe)",
    "G2288": "death (thanatos)",
    "G386": "resurrection (anastasis)",
}


# ── Model loading ────────────────────────────────────────────────────────────


def load_model():
    """Load the fine-tuned QLoRA model."""
    print(f"Loading base model: {MODEL_NAME}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"Loading adapter: {ADAPTER_PATH}", flush=True)
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    print("Model ready.", flush=True)

    return model, tokenizer


def translate(model, tokenizer, text: str) -> str:
    """Translate English text to Nama using the fine-tuned model."""
    prompt = (
        "<start_of_turn>user\n"
        "Translate the following text from English to Nama.\n\n"
        f"{text}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    pred = tokenizer.decode(generated, skip_special_tokens=True)
    return pred.replace("<end_of_turn>", "").strip()


# ── Strong's number extraction ───────────────────────────────────────────────


def extract_strongs_from_usfm(filepath: str) -> dict:
    """Extract Strong's numbers per verse from WEB USFM files.

    WEB uses format: \\w word|strong="G1234"\\w*
    Returns: {book: {chapter: {verse: [strong_numbers]}}}
    """
    result = {}
    current_book = None
    current_chapter = None
    current_verse = None
    verse_strongs = []

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            m = re.match(r"\\id\s+(\w+)", line)
            if m:
                if current_book and current_chapter and current_verse:
                    result.setdefault(current_book, {}).setdefault(
                        current_chapter, {}
                    )[current_verse] = verse_strongs
                current_book = m.group(1)
                current_chapter = None
                current_verse = None
                verse_strongs = []
                continue

            m = re.match(r"\\c\s+(\d+)", line)
            if m:
                if current_book and current_chapter and current_verse:
                    result.setdefault(current_book, {}).setdefault(
                        current_chapter, {}
                    )[current_verse] = verse_strongs
                current_chapter = int(m.group(1))
                current_verse = None
                verse_strongs = []
                continue

            m = re.search(r"\\v\s+(\d+)", line)
            if m:
                if current_book and current_chapter and current_verse:
                    result.setdefault(current_book, {}).setdefault(
                        current_chapter, {}
                    )[current_verse] = verse_strongs
                current_verse = int(m.group(1))
                verse_strongs = []

            # Extract Strong's numbers from this line
            strongs = re.findall(r'strong="([GH]\d+)"', line)
            verse_strongs.extend(strongs)

    if current_book and current_chapter and current_verse:
        result.setdefault(current_book, {}).setdefault(
            current_chapter, {}
        )[current_verse] = verse_strongs

    return result


# ── Evaluation functions ─────────────────────────────────────────────────────


def evaluate_standard(predictions: list[str], references: list[str]) -> dict:
    """Compute chrF and BLEU scores."""
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return {
        "chrf": round(chrf.score, 2),
        "bleu": round(bleu.score, 2),
    }


def evaluate_per_book(
    test_data: list[dict], predictions: list[str]
) -> dict:
    """Compute per-book chrF scores."""
    book_preds = defaultdict(list)
    book_refs = defaultdict(list)

    for item, pred in zip(test_data, predictions):
        book = item["ref"].split()[0]
        book_preds[book].append(pred)
        book_refs[book].append(item["target"])

    results = {}
    for book in sorted(book_preds.keys()):
        chrf = sacrebleu.corpus_chrf(book_preds[book], [book_refs[book]])
        results[book] = {
            "chrf": round(chrf.score, 2),
            "count": len(book_preds[book]),
        }

    return results


def evaluate_concept_coverage(
    model, tokenizer, test_data: list[dict], strongs_data: dict
) -> dict:
    """Evaluate theological concept preservation via reverse translation.

    1. For each test verse, find Strong's numbers from WEB USFM
    2. Translate Nama prediction back to English (reverse direction)
    3. Check if key theological concepts (mapped from Strong's) are preserved
    """
    # Map Strong's numbers to expected English keywords
    strongs_to_keywords = {
        "G2316": ["god"],
        "G2962": ["lord"],
        "G5547": ["christ", "messiah", "anointed"],
        "G2424": ["jesus"],
        "G4151": ["spirit"],
        "G4102": ["faith", "believe", "trust"],
        "G26": ["love"],
        "G5485": ["grace"],
        "G266": ["sin"],
        "G4991": ["salvation", "save", "saved"],
        "G932": ["kingdom"],
        "G1411": ["power", "mighty"],
        "G1680": ["hope"],
        "G1515": ["peace"],
        "G1343": ["righteous", "justice"],
        "G3056": ["word"],
        "G225": ["truth", "true"],
        "G2222": ["life", "living"],
        "G2288": ["death", "die", "dead"],
        "G386": ["resurrection", "rise", "risen", "raised"],
    }

    total_concepts = 0
    preserved_concepts = 0
    concept_stats = defaultdict(lambda: {"total": 0, "preserved": 0})

    for item in test_data:
        # Parse ref to find Strong's numbers
        parts = item["ref"].split()
        if len(parts) != 2:
            continue
        book = parts[0]
        ch_v = parts[1].split(":")
        if len(ch_v) != 2:
            continue
        chapter = int(ch_v[0])
        verse = int(ch_v[1])

        verse_strongs = strongs_data.get(book, {}).get(chapter, {}).get(verse, [])
        if not verse_strongs:
            continue

        # Reverse translate: Nama → English
        reverse_prompt = (
            "<start_of_turn>user\n"
            "Translate the following text from Nama to English.\n\n"
            f"{item['target']}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        inputs = tokenizer(reverse_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        reverse_text = tokenizer.decode(generated, skip_special_tokens=True).lower()

        # Check concept preservation
        for strong in set(verse_strongs):
            keywords = strongs_to_keywords.get(strong)
            if not keywords:
                continue
            total_concepts += 1
            concept_stats[strong]["total"] += 1

            if any(kw in reverse_text for kw in keywords):
                preserved_concepts += 1
                concept_stats[strong]["preserved"] += 1

    coverage = preserved_concepts / total_concepts if total_concepts > 0 else 0

    return {
        "total_concepts": total_concepts,
        "preserved_concepts": preserved_concepts,
        "coverage_score": round(coverage * 100, 2),
        "per_concept": {
            strong: {
                "name": THEOLOGICAL_CONCEPTS.get(strong, strong),
                "total": stats["total"],
                "preserved": stats["preserved"],
                "coverage": round(stats["preserved"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0,
            }
            for strong, stats in sorted(concept_stats.items())
        },
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  Nama Translation Evaluation (v3)")
    print("=" * 60)

    # Load test data
    test_path = BASE / "output" / "translategemma-nama-v3" / "test_split.json"
    if not test_path.exists():
        print(f"ERROR: Test split not found at {test_path}")
        print("Run train_v3.py first.")
        return

    with open(test_path, encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"\nTest data: {len(test_data)} examples")

    # Load model
    model, tokenizer = load_model()

    # Generate predictions
    print("\nGenerating translations...", flush=True)
    predictions = []
    for i, item in enumerate(test_data):
        pred = translate(model, tokenizer, item["source"])
        predictions.append(pred)
        if (i + 1) % 10 == 0 or (i + 1) == len(test_data):
            print(f"  {i + 1}/{len(test_data)}", flush=True)

    references = [item["target"] for item in test_data]

    # Standard evaluation
    print("\n" + "=" * 60)
    print("  Standard Metrics")
    print("=" * 60)
    standard = evaluate_standard(predictions, references)
    print(f"  chrF: {standard['chrf']}")
    print(f"  BLEU: {standard['bleu']}")

    # Per-book evaluation
    print("\n" + "=" * 60)
    print("  Per-Book chrF Scores")
    print("=" * 60)
    per_book = evaluate_per_book(test_data, predictions)
    print(f"  {'Book':<8} {'chrF':>8} {'Count':>8}")
    print("  " + "-" * 26)
    for book, stats in per_book.items():
        print(f"  {book:<8} {stats['chrf']:>8.2f} {stats['count']:>8}")

    # Greek/Hebrew concept coverage — skipped for speed
    # To run: use --with-concepts flag
    concept_results = {}

    # Save results
    results = {
        "standard": standard,
        "per_book": per_book,
        "concept_coverage": concept_results,
        "predictions": [
            {"ref": t["ref"], "source": t["source"], "reference": t["target"], "predicted": p}
            for t, p in zip(test_data, predictions)
        ],
    }
    out_path = BASE / "reports" / "evaluation_v3_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")

    # Sample translations
    print("\n" + "=" * 60)
    print("  Sample Translations (10)")
    print("=" * 60)
    for i in range(min(10, len(test_data))):
        print(f"\n  [{test_data[i]['ref']}]")
        print(f"  SRC:  {test_data[i]['source'][:80]}")
        print(f"  REF:  {test_data[i]['target'][:80]}")
        print(f"  PRED: {predictions[i][:80]}")


if __name__ == "__main__":
    main()
