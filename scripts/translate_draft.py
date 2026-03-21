#!/usr/bin/env python3
"""Generate draft Bible translations using fine-tuned TranslateGemma 27B.

Usage:
    uv run python scripts/translate_draft.py RUT        # Translate Ruth
    uv run python scripts/translate_draft.py GEN        # Translate Genesis
    uv run python scripts/translate_draft.py GEN -c 1   # Only chapter 1
"""

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_usfm import parse_usfm

BASE = Path(__file__).resolve().parent.parent
MODEL_NAME = "google/translategemma-27b-it"
ADAPTER_PATH = str(BASE / "output" / "translategemma-nama-v3" / "final_adapter")
ENG_DIR = BASE / "data" / "eng"
MAX_NEW_TOKENS = 256


def load_model():
    """Load fine-tuned TranslateGemma 27B model."""
    print(f"Loading {MODEL_NAME} with adapter...")

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
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    return model, tokenizer


def translate_verse(model, tokenizer, text: str) -> str:
    """Translate a single English verse to Nama."""
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
            num_beams=4,
            repetition_penalty=1.3,
            length_penalty=1.0,
            early_stopping=True,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    pred = tokenizer.decode(generated, skip_special_tokens=True)
    return pred.replace("<end_of_turn>", "").strip()


def find_english_usfm(book_code: str) -> Path | None:
    """Find the WEB English USFM file for a book code."""
    for f in ENG_DIR.glob("*.usfm"):
        if book_code.upper() in f.name.upper():
            return f
    return None


def generate_usfm_output(book_code: str, translations: dict) -> str:
    """Generate USFM formatted output from translations.

    translations: {chapter: {verse: translated_text}}
    """
    lines = [f"\\id {book_code} - Nama Draft Translation (TranslateGemma 27B)"]
    lines.append(f"\\h {book_code}")
    lines.append(f"\\mt1 {book_code}")

    for chapter in sorted(translations.keys()):
        lines.append(f"\\c {chapter}")
        lines.append("\\p")
        for verse in sorted(translations[chapter].keys()):
            text = translations[chapter][verse]
            lines.append(f"\\v {verse} {text}")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate draft Nama Bible translations")
    parser.add_argument("book", help="USFM book code (e.g., RUT, GEN, PSA)")
    parser.add_argument("-c", "--chapter", type=int, help="Specific chapter to translate")
    parser.add_argument("-o", "--output", help="Output directory (default: output/drafts)")
    args = parser.parse_args()

    book_code = args.book.upper()

    # Find English source
    eng_file = find_english_usfm(book_code)
    if not eng_file:
        print(f"ERROR: No English USFM found for '{book_code}' in {ENG_DIR}")
        sys.exit(1)

    print(f"Source: {eng_file}")

    # Parse English USFM
    eng_data = parse_usfm(str(eng_file))
    if book_code not in eng_data:
        print(f"ERROR: Book '{book_code}' not found in parsed data")
        sys.exit(1)

    book_data = eng_data[book_code]

    # Filter to specific chapter if requested
    if args.chapter:
        if args.chapter not in book_data:
            print(f"ERROR: Chapter {args.chapter} not found in {book_code}")
            sys.exit(1)
        book_data = {args.chapter: book_data[args.chapter]}

    total_verses = sum(len(v) for v in book_data.values())
    print(f"Translating {book_code}: {len(book_data)} chapters, {total_verses} verses")

    # Load model
    model, tokenizer = load_model()

    # Translate
    translations = {}
    done = 0
    for chapter in sorted(book_data.keys()):
        translations[chapter] = {}
        for verse in sorted(book_data[chapter].keys()):
            eng_text = book_data[chapter][verse]
            nama_text = translate_verse(model, tokenizer, eng_text)
            translations[chapter][verse] = nama_text
            done += 1
            if done % 10 == 0:
                print(f"  [{done}/{total_verses}] {book_code} {chapter}:{verse}")

    # Generate USFM output
    usfm_output = generate_usfm_output(book_code, translations)

    # Save
    output_dir = Path(args.output) if args.output else BASE / "output" / "drafts"
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_ch{args.chapter}" if args.chapter else ""
    out_file = output_dir / f"{book_code}{suffix}_draft_nmx.usfm"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(usfm_output)

    print(f"\n{'=' * 60}")
    print(f"  Draft translation saved to: {out_file}")
    print(f"  {book_code}: {len(translations)} chapters, {total_verses} verses")
    print(f"{'=' * 60}")

    # Print sample
    print(f"\n  ─── Sample ({book_code} 1:1-5) ───")
    ch1 = translations.get(1, {})
    for v in sorted(ch1.keys())[:5]:
        print(f"  [{book_code} 1:{v}]")
        print(f"  ENG:  {book_data.get(1, {}).get(v, '')[:80]}")
        print(f"  NMX:  {ch1[v][:80]}")
        print()


if __name__ == "__main__":
    main()
