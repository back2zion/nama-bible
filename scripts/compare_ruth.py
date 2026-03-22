"""
Compare model-generated Ruth draft with human translation.
1. Parse both USFM files into verse-aligned data
2. Build English-Nama parallel pairs from human translation
3. Compute chrF/BLEU between model output and human reference
4. Add human Ruth translation to augmented training data
"""

import json
import re
import sys
from pathlib import Path

import sacrebleu

BASE = Path(__file__).resolve().parent.parent


def parse_usfm_verses(filepath: str) -> dict[str, str]:
    """Parse USFM file into {ref: text} dict."""
    verses = {}
    current_book = None
    current_chapter = None
    current_verse = None
    verse_buffer = []

    def flush():
        nonlocal verse_buffer
        if current_book and current_chapter and current_verse:
            text = " ".join(verse_buffer).strip()
            # Clean USFM markers
            text = re.sub(r"\\f\s+.*?\\f\*", "", text)
            text = re.sub(r"\\x\s+.*?\\x\*", "", text)
            text = re.sub(r"\{[a-z]\{.*?$", "", text)  # footnote refs
            text = re.sub(r"\\[a-z0-9*]+\s*", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                ref = f"{current_book} {current_chapter}:{current_verse}"
                verses[ref] = text
        verse_buffer = []

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            m = re.match(r"\\id\s+(\w+)", line)
            if m:
                flush()
                current_book = m.group(1)
                current_chapter = None
                current_verse = None
                continue

            m = re.match(r"\\c\s+(\d+)", line)
            if m:
                flush()
                current_chapter = int(m.group(1))
                current_verse = None
                continue

            # Handle multiple \v in one line
            parts = re.split(r"\\v\s+(\d+)\s*", line)
            if len(parts) > 1:
                # First part is text before first \v (append to current verse)
                if parts[0].strip() and current_verse is not None:
                    verse_buffer.append(parts[0].strip())

                # Process each \v
                for i in range(1, len(parts), 2):
                    flush()
                    current_verse = int(parts[i])
                    if i + 1 < len(parts) and parts[i + 1].strip():
                        text = parts[i + 1].strip()
                        # Remove inline markers
                        text = re.sub(r"\\[a-z0-9]+\s*", " ", text)
                        verse_buffer.append(text)
                continue

            # Regular text line
            if current_verse is not None:
                if not line.startswith("\\") or re.match(r"\\[qmspfl]", line):
                    cleaned = re.sub(r"\\[a-z0-9+*]+\s*", " ", line)
                    if cleaned.strip():
                        verse_buffer.append(cleaned.strip())

    flush()
    return verses


def main():
    print("=" * 60)
    print("  Ruth Translation Comparison")
    print("=" * 60)

    # Parse files
    human_path = BASE / "data" / "nmx" / "08-RUTnmx.usfm"
    model_path = BASE / "output" / "drafts" / "RUT_draft_nmx.usfm"
    eng_path = BASE / "data" / "eng" / "09-RUTengwebp.usfm"

    human_verses = parse_usfm_verses(str(human_path))
    model_verses = parse_usfm_verses(str(model_path))
    eng_verses = parse_usfm_verses(str(eng_path))

    print(f"\n  Human translation: {len(human_verses)} verses")
    print(f"  Model translation: {len(model_verses)} verses")
    print(f"  English source:    {len(eng_verses)} verses")

    # Find common verses for comparison
    common = sorted(set(human_verses.keys()) & set(model_verses.keys()))
    print(f"  Common verses:     {len(common)}")

    # Compute metrics
    predictions = [model_verses[ref] for ref in common]
    references = [human_verses[ref] for ref in common]

    chrf = sacrebleu.corpus_chrf(predictions, [references])
    bleu = sacrebleu.corpus_bleu(predictions, [references])

    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  Model vs Human Translation          │")
    print(f"  │  BLEU:  {bleu.score:6.2f}                      │")
    print(f"  │  chrF:  {chrf.score:6.2f}                      │")
    print(f"  └─────────────────────────────────────┘")

    # Show sample comparisons
    print(f"\n  ─── Sample Comparisons (first 5 verses) ───")
    for ref in common[:5]:
        print(f"\n  [{ref}]")
        if ref in eng_verses:
            print(f"  ENG:   {eng_verses[ref][:100]}")
        print(f"  HUMAN: {human_verses[ref][:100]}")
        print(f"  MODEL: {model_verses[ref][:100]}")

    # Build parallel pairs from human Ruth translation
    print(f"\n{'=' * 60}")
    print(f"  Building Ruth parallel data")
    print(f"{'=' * 60}")

    ruth_pairs = []
    common_eng = sorted(set(human_verses.keys()) & set(eng_verses.keys()))
    for ref in common_eng:
        book = ref.split()[0]
        ch_v = ref.split()[1]
        ch, v = ch_v.split(":")
        ruth_pairs.append({
            "ref": ref,
            "book": book,
            "chapter": int(ch),
            "verse": int(v),
            "nama": human_verses[ref],
            "english": eng_verses[ref],
            "aligned": True,
            "source": "human_ruth",
        })

    print(f"  Ruth parallel pairs: {len(ruth_pairs)}")

    # Add to augmented dataset
    aug_path = BASE / "data" / "corpus" / "nama_eng_augmented.json"
    with open(aug_path, encoding="utf-8") as f:
        augmented = json.load(f)

    # Remove any existing RUT entries
    augmented = [e for e in augmented if e.get("book") != "RUT"]
    augmented.extend(ruth_pairs)

    with open(aug_path, "w", encoding="utf-8") as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)

    print(f"  Added to augmented dataset: {len(augmented)} total pairs")

    # Save comparison report
    report = {
        "metrics": {
            "bleu": round(bleu.score, 2),
            "chrf": round(chrf.score, 2),
        },
        "verse_counts": {
            "human": len(human_verses),
            "model": len(model_verses),
            "english": len(eng_verses),
            "common": len(common),
        },
        "ruth_pairs_added": len(ruth_pairs),
        "total_augmented": len(augmented),
    }

    report_path = BASE / "reports" / "ruth_comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  Report saved to {report_path}")


if __name__ == "__main__":
    main()
