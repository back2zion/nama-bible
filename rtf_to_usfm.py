#!/usr/bin/env python3
"""Convert Paratext RTF export to USFM files.

Paratext exports RTF files with USFM markers encoded as RTF paragraph styles.
This script parses the RTF structure and reconstructs proper USFM files.

Usage:
    python rtf_to_usfm.py raw/*.rtf
    python rtf_to_usfm.py raw/마태복음부터\ 사도행전까지.rtf -o raw/nmx_usfm_new
"""

import re
import sys
import argparse
from pathlib import Path

BOOK_NUMBERS = {
    "GEN": "01", "EXO": "02", "LEV": "03", "NUM": "04", "DEU": "05",
    "JOS": "06", "JDG": "07", "RUT": "08", "1SA": "09", "2SA": "10",
    "1KI": "11", "2KI": "12", "1CH": "13", "2CH": "14", "EZR": "15",
    "NEH": "16", "EST": "17", "JOB": "18", "PSA": "19", "PRO": "20",
    "ECC": "21", "SNG": "22", "ISA": "23", "JER": "24", "LAM": "25",
    "EZK": "26", "DAN": "27", "HOS": "28", "JOL": "29", "AMO": "30",
    "OBA": "31", "JON": "32", "MIC": "33", "NAM": "34", "HAB": "35",
    "ZEP": "36", "HAG": "37", "ZEC": "38", "MAL": "39",
    "MAT": "41", "MRK": "42", "LUK": "43", "JHN": "44", "ACT": "45",
    "ROM": "46", "1CO": "47", "2CO": "48", "GAL": "49", "EPH": "50",
    "PHP": "51", "COL": "52", "1TH": "53", "2TH": "54", "1TI": "55",
    "2TI": "56", "TIT": "57", "PHM": "58", "HEB": "59", "JAS": "60",
    "1PE": "61", "2PE": "62", "1JN": "63", "2JN": "64", "3JN": "65",
    "JUD": "66", "REV": "67",
}

# Markers that are paragraph-level (no text follows on same line)
EMPTY_MARKERS = {"p", "b", "nb", "m"}

# All recognized USFM markers to output
KNOWN_MARKERS = {
    "id", "usfm", "ide", "h",
    "toc1", "toc2", "toc3", "toca1", "toca2", "toca3",
    "rem", "sts",
    "imt", "imt1", "imt2", "is", "is1", "iot", "io", "io1", "io2",
    "ip", "im", "ipi", "imi", "iex", "ie",
    "mt", "mt1", "mt2", "mt3",
    "mte", "mte1", "ms", "ms1", "ms2", "mr",
    "s", "s1", "s2", "s3", "s4", "sr", "r", "sp", "sd",
    "d", "c", "cl", "cp", "cd",
    "p", "m", "po", "pr", "cls", "pmo", "pm", "pmc", "pmr",
    "pi", "pi1", "pi2", "pi3", "pc", "mi", "nb",
    "q", "q1", "q2", "q3", "q4", "qc", "qr", "qm", "qm1", "qm2", "qm3", "qd",
    "b", "qa",
    "tr",
    "li", "li1", "li2", "li3", "li4", "lh", "lf",
    "lim", "lim1", "lim2",
    "lit", "pb", "periph",
}


def parse_stylesheet(content: str) -> dict[int, str]:
    """Extract style number -> USFM marker mapping from RTF stylesheet."""
    style_map = {}
    start = content.find("{\\stylesheet")
    if start == -1:
        raise ValueError("No stylesheet found in RTF")

    stylesheet_area = content[start:start + 100000]
    # Extract individual style entries between { }
    for entry in re.findall(r"\{([^{}]+)\}", stylesheet_area):
        m = re.search(r"\\s(\d+).*?\\fs\d+\s+(\S+)\s+-\s+", entry)
        if m:
            snum = int(m.group(1))
            marker = m.group(2)
            if marker != "DEPRECATED":
                style_map[snum] = marker

    return style_map


def decode_rtf_text(text: str) -> str:
    """Convert RTF unicode/hex escapes to actual characters."""
    def replace_unicode(m):
        code = int(m.group(1))
        if code < 0:
            code += 65536
        return chr(code)

    text = re.sub(r"\\u(-?\d+)\?", replace_unicode, text)
    text = re.sub(r"\\'([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), text)
    return text


def clean_paragraph_text(raw_text: str) -> str:
    """Remove RTF formatting from paragraph text, converting verse numbers to \\v."""
    text = raw_text

    # 1. Decode unicode FIRST (before RTF commands are stripped)
    text = decode_rtf_text(text)

    # 2. Extract verse numbers from {\cs57...\fs22 <num>~} groups
    text = re.sub(
        r"\{\\cs57[^}]*\\fs\d+\s+(\d+)\\~\}\s*\n?",
        lambda m: f"\x00v {m.group(1).strip()} ",  # use \x00 as placeholder
        text,
    )

    # 3. Remove other inline character style groups but keep their text content
    text = re.sub(r"\{\\cs\d+[^}]*\\fs\d+\s*", "", text)
    text = re.sub(r"\}", "", text)

    # 4. Remove remaining RTF control words (but not our \x00v placeholders)
    text = re.sub(r"\\[a-z]+\d*\s?", "", text)

    # 5. Restore \v markers
    text = text.replace("\x00v ", "\\v ")

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    return text


def parse_rtf_to_books(rtf_path: str) -> dict[str, list[str]]:
    """Parse a Paratext RTF file and return {book_code: [usfm_lines]}."""
    with open(rtf_path, "rb") as f:
        content = f.read().decode("latin-1")

    style_map = parse_stylesheet(content)

    # Find body start
    body_start = content.find("\\pard\\plain", 10000)
    body = content[body_start:]

    # Split into paragraphs
    paragraphs = re.split(r"\\par\\pard\\plain\s+|\\pard\\plain\s+", body)

    books: dict[str, list[str]] = {}
    current_book: str | None = None
    current_lines: list[str] = []

    for para in paragraphs:
        if not para.strip():
            continue

        style_match = re.match(r"\\s(\d+)\\", para)
        if not style_match:
            continue

        snum = int(style_match.group(1))
        marker = style_map.get(snum)
        if not marker:
            continue

        # Extract text after last \fsNN
        text_match = re.search(r"\\fs\d+\s*(.*)", para, re.DOTALL)
        if not text_match:
            continue

        raw_text = text_match.group(1).strip()
        text = clean_paragraph_text(raw_text)

        # Book boundary
        if marker == "id":
            if current_book and current_lines:
                books[current_book] = current_lines
            book_match = re.match(r"(\w{3})", text)
            if book_match:
                current_book = book_match.group(1)
                current_lines = [f"\\id {text}"]
            continue

        if current_book is None:
            continue

        # Build USFM line
        if marker == "c":
            current_lines.append(f"\\c {text}")
        elif marker in EMPTY_MARKERS:
            current_lines.append(f"\\{marker}")
            if text:
                current_lines.append(text)
        elif marker in KNOWN_MARKERS:
            if text:
                current_lines.append(f"\\{marker} {text}")
            else:
                current_lines.append(f"\\{marker}")
        else:
            if text:
                current_lines.append(f"\\{marker} {text}")

    if current_book and current_lines:
        books[current_book] = current_lines

    return books


def main():
    parser = argparse.ArgumentParser(description="Convert Paratext RTF to USFM")
    parser.add_argument("rtf_files", nargs="+", help="RTF file(s) to convert")
    parser.add_argument("-o", "--output", default="raw/nmx_usfm_rtf", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_books: dict[str, list[str]] = {}

    for rtf_path in args.rtf_files:
        print(f"\n{'='*60}")
        print(f"Processing: {rtf_path}")
        print(f"{'='*60}")
        books = parse_rtf_to_books(rtf_path)
        for code, lines in books.items():
            chapters = sum(1 for l in lines if l.startswith("\\c "))
            verses = sum(l.count("\\v ") for l in lines)
            print(f"  {code}: {chapters} chapters, {verses} verses")
            all_books[code] = lines

    print(f"\n{'='*60}")
    print(f"Writing {len(all_books)} books to {output_dir}/")
    print(f"{'='*60}")

    for book_code, lines in sorted(all_books.items(), key=lambda x: BOOK_NUMBERS.get(x[0], "99")):
        num = BOOK_NUMBERS.get(book_code, "00")
        filename = f"{num}-{book_code}nmx.usfm"
        filepath = output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  {filename}")

    print(f"\nDone! {len(all_books)} USFM files created.")


if __name__ == "__main__":
    main()
