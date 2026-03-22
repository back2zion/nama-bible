#!/usr/bin/env python3
"""Build multilingual parallel corpus from PNG Bible USFM files.

Aligns each PNG language with English (WEB) at the verse level,
producing a combined JSON dataset for multilingual transfer learning
to boost Nama (nmx) translation quality.
"""

import json
import re
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
BASE = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
from parse_usfm import parse_usfm

# ── Configuration ────────────────────────────────────────────────────────────

DATA = BASE / "data"
ENG_DIR = DATA / "eng"
MULTI_DIR = DATA / "multilingual"

# Languages to process: (dir_name, lang_code, nllb_code_or_none, base_dir_override_or_none)
LANGUAGES = [
    ("tpi", "tpi", "tpi_Latn", None),   # Tok Pisin
    ("hmo", "hmo", None, None),          # Hiri Motu
    ("beo", "beo", None, None),          # Bedamuni
    ("bon", "bon", None, None),          # Bine
    ("gdr", "gdr", None, None),          # Wipi
    ("xla", "xla", None, None),          # Kamula
    ("tof", "tof", None, None),          # Gizrra
    ("grk", "grk", None, "grk"),         # Greek NT (UGNT)
    ("heb", "heb", None, "heb"),         # Hebrew OT (UHB)
]

# USFM book code → sort order (NT only for now, plus selected OT)
BOOK_ORDER = {
    # OT (selected)
    "GEN": 1, "EXO": 2, "LEV": 3, "NUM": 4, "DEU": 5,
    "JOS": 6, "JDG": 7, "RUT": 8, "1SA": 9, "2SA": 10,
    "1KI": 11, "2KI": 12, "1CH": 13, "2CH": 14,
    "EZR": 15, "NEH": 16, "EST": 17, "JOB": 18,
    "PSA": 19, "PRO": 20, "ECC": 21, "SNG": 22,
    "ISA": 23, "JER": 24, "LAM": 25, "EZK": 26,
    "DAN": 27, "HOS": 28, "JOL": 29, "AMO": 30,
    "OBA": 31, "JON": 32, "MIC": 33, "NAM": 34,
    "HAB": 35, "ZEP": 36, "HAG": 37, "ZEC": 38, "MAL": 39,
    # NT
    "MAT": 40, "MRK": 41, "LUK": 42, "JHN": 43, "ACT": 44,
    "ROM": 45, "1CO": 46, "2CO": 47, "GAL": 48, "EPH": 49,
    "PHP": 50, "COL": 51, "1TH": 52, "2TH": 53,
    "1TI": 54, "2TI": 55, "TIT": 56, "PHM": 57,
    "HEB": 58, "JAS": 59, "1PE": 60, "2PE": 61,
    "1JN": 62, "2JN": 63, "3JN": 64, "JUD": 65, "REV": 66,
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def parse_all_usfm(directory: Path) -> dict:
    """Parse all USFM files in a directory into {BOOK: {ch: {v: text}}}."""
    combined = {}
    files = sorted(directory.glob("*.usfm")) + sorted(directory.glob("*.SFM"))
    for f in files:
        parsed = parse_usfm(str(f))
        combined.update(parsed)
    return combined


def align_verses(lang_data: dict, eng_data: dict) -> list[dict]:
    """Align a language with English at verse level."""
    pairs = []
    for book in lang_data:
        if book not in eng_data:
            continue
        for ch in lang_data[book]:
            if ch not in eng_data[book]:
                continue
            for v in lang_data[book][ch]:
                if v not in eng_data[book][ch]:
                    continue
                lang_text = lang_data[book][ch][v].strip()
                eng_text = eng_data[book][ch][v].strip()
                if lang_text and eng_text:
                    pairs.append({
                        "ref": f"{book} {ch}:{v}",
                        "book": book,
                        "english": eng_text,
                        "target": lang_text,
                    })
    # Sort by book order, chapter, verse
    pairs.sort(key=lambda p: (
        BOOK_ORDER.get(p["book"], 99),
        int(p["ref"].split()[1].split(":")[0]),
        int(p["ref"].split(":")[1]),
    ))
    return pairs


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  Multilingual Parallel Corpus Builder")
    print("=" * 60)

    # Parse English
    print("\nParsing English (WEB)...")
    eng_data = parse_all_usfm(ENG_DIR)
    eng_books = sorted(eng_data.keys(), key=lambda b: BOOK_ORDER.get(b, 99))
    eng_verses = sum(
        len(v) for chs in eng_data.values() for v in chs.values()
    )
    print(f"  {len(eng_books)} books, {eng_verses:,} verses")

    # Process each language
    all_pairs = []
    stats = {}

    for dir_name, lang_code, nllb_code, base_override in LANGUAGES:
        if base_override:
            lang_dir = DATA / base_override
        else:
            lang_dir = MULTI_DIR / dir_name
        if not lang_dir.exists():
            print(f"\n  SKIP {lang_code}: directory not found")
            continue

        print(f"\nProcessing {lang_code}...")
        lang_data = parse_all_usfm(lang_dir)
        lang_books = sorted(lang_data.keys(), key=lambda b: BOOK_ORDER.get(b, 99))
        lang_verses = sum(
            len(v) for chs in lang_data.values() for v in chs.values()
        )
        print(f"  {len(lang_books)} books, {lang_verses:,} verses")

        pairs = align_verses(lang_data, eng_data)
        for p in pairs:
            p["lang"] = lang_code
            p["nllb_code"] = nllb_code

        all_pairs.extend(pairs)
        stats[lang_code] = {
            "books": len(lang_books),
            "total_verses": lang_verses,
            "aligned_pairs": len(pairs),
            "book_list": lang_books,
        }
        print(f"  Aligned: {len(pairs):,} eng-{lang_code} pairs")

    # Also include existing Nama data
    print(f"\nIncluding Nama (nmx)...")
    nama_json = BASE / "data" / "corpus" / "nama_eng_parallel.json"
    with open(nama_json, encoding="utf-8") as f:
        nama_data = json.load(f)
    nama_pairs = [
        {
            "ref": d["ref"],
            "book": d["book"],
            "english": d["english"],
            "target": d["nama"],
            "lang": "nmx",
            "nllb_code": None,
        }
        for d in nama_data
        if d.get("aligned")
    ]
    all_pairs.extend(nama_pairs)
    stats["nmx"] = {"aligned_pairs": len(nama_pairs)}
    print(f"  {len(nama_pairs):,} eng-nmx pairs")

    # Save
    out_path = BASE / "data" / "corpus" / "multilingual_parallel.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Language':<12} {'Books':>6} {'Aligned':>10}")
    print("  " + "-" * 30)
    total = 0
    for lang, s in sorted(stats.items()):
        books = s.get("books", "-")
        aligned = s["aligned_pairs"]
        total += aligned
        print(f"  {lang:<12} {str(books):>6} {aligned:>10,}")
    print("  " + "-" * 30)
    print(f"  {'TOTAL':<12} {'':>6} {total:>10,}")
    print(f"\n  Output: {out_path}")
    print(f"  Size:   {out_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
