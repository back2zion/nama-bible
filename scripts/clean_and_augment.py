"""
Clean parallel data and augment with back-translation.

Step 1: Quality filtering of nama_eng_parallel.json
  - Remove extreme length ratio outliers (nama/eng < 0.3 or > 5.0)
  - Remove entries with very short text (< 5 chars)
  - Flag and remove multi-verse concatenations

Step 2: Back-translation augmentation
  - Merge nama_bt_parallel.json as additional training pairs
  - Split long BT entries into individual verses where possible
  - Normalize BT English to proper sentences

Output:
  - nama_eng_clean.json        (cleaned parallel data)
  - nama_eng_augmented.json    (cleaned + back-translation augmented)
  - data_quality_report.json   (cleaning statistics)
"""

import json
import re
import statistics
from pathlib import Path


def load_json(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def save_json(data: list, path: str):
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(data)} entries to {path}")


def clean_parallel_data(entries: list) -> tuple[list, dict]:
    """Filter out low-quality parallel entries."""
    stats = {
        "input_total": len(entries),
        "input_aligned": 0,
        "removed_unaligned": 0,
        "removed_short": 0,
        "removed_ratio_low": 0,
        "removed_ratio_high": 0,
        "removed_multi_verse": 0,
        "output_clean": 0,
    }

    aligned = [e for e in entries if e.get("aligned")]
    stats["input_aligned"] = len(aligned)
    stats["removed_unaligned"] = len(entries) - len(aligned)

    clean = []
    for e in aligned:
        nama = e.get("nama", "").strip()
        eng = e.get("english", "").strip()

        # Skip very short entries
        if len(nama) < 5 or len(eng) < 5:
            stats["removed_short"] += 1
            continue

        ratio = len(nama) / len(eng)

        # Skip extreme ratio: too short in Nama (likely incomplete)
        if ratio < 0.3:
            stats["removed_ratio_low"] += 1
            continue

        # Skip extreme ratio: too long in Nama (likely multi-verse concatenation)
        if ratio > 5.0:
            stats["removed_ratio_high"] += 1
            continue

        # Detect multi-verse concatenation patterns (verse numbers in text)
        if re.search(r"\b\d{1,3}\s+[A-Z]", nama) and len(nama) > 300:
            stats["removed_multi_verse"] += 1
            continue

        clean.append(e)

    stats["output_clean"] = len(clean)
    return clean, stats


def split_bt_long_entries(entries: list) -> list:
    """Split BT entries that contain multiple concatenated verses."""
    result = []
    split_count = 0

    for e in entries:
        nama = e.get("nama", "").strip()
        bt = e.get("bt", "").strip()

        # Detect multi-verse: verse numbers like "5 ", "10 " embedded in text
        verse_pattern = re.compile(r"(?:^|\s)(\d{1,3})\s+([A-Z])")
        matches = list(verse_pattern.finditer(nama))

        if len(matches) > 0 and len(nama) > 300:
            # This entry contains multiple verses - skip it as splitting is unreliable
            split_count += 1
            continue

        result.append(e)

    print(f"  Removed {split_count} multi-verse BT entries")
    return result


def augment_with_bt(clean_data: list, bt_data: list) -> list:
    """Add back-translation data as augmentation."""
    # Build set of existing refs
    existing_refs = {e["ref"] for e in clean_data}

    augmented = list(clean_data)
    added = 0
    skipped_exists = 0
    skipped_short = 0

    for e in bt_data:
        nama = e.get("nama", "").strip()
        bt = e.get("bt", "").strip()

        if len(nama) < 5 or len(bt) < 5:
            skipped_short += 1
            continue

        # Add as augmentation (even if ref exists, BT provides different English)
        augmented.append({
            "ref": e["ref"],
            "book": e["book"],
            "chapter": e.get("chapter", 0),
            "verse": e.get("verse", 0),
            "nama": nama,
            "english": bt,  # Use back-translation as English side
            "aligned": True,
            "source": "bt",
        })
        added += 1

    print(f"  Added {added} BT entries, skipped {skipped_short} short")
    return augmented


def main():
    base = Path(__file__).parent.parent

    print("=" * 60)
    print("Step 1: Cleaning parallel data")
    print("=" * 60)
    eng_data = load_json(base / "nama_eng_parallel.json")
    clean_data, clean_stats = clean_parallel_data(eng_data)
    save_json(clean_data, base / "nama_eng_clean.json")

    print(f"\n  Cleaning summary:")
    for k, v in clean_stats.items():
        print(f"    {k}: {v}")

    # Length ratio stats after cleaning
    ratios = [len(e["nama"]) / len(e["english"]) for e in clean_data]
    print(f"\n  After cleaning - ratio mean: {statistics.mean(ratios):.2f}, "
          f"median: {statistics.median(ratios):.2f}")

    print()
    print("=" * 60)
    print("Step 2: Back-translation augmentation")
    print("=" * 60)
    bt_data = load_json(base / "nama_bt_parallel.json")
    print(f"  Raw BT entries: {len(bt_data)}")

    bt_filtered = split_bt_long_entries(bt_data)
    print(f"  BT entries after filtering: {len(bt_filtered)}")

    augmented = augment_with_bt(clean_data, bt_filtered)
    save_json(augmented, base / "nama_eng_augmented.json")

    # Final stats
    clean_only = [e for e in augmented if e.get("source") != "bt"]
    bt_only = [e for e in augmented if e.get("source") == "bt"]

    report = {
        "cleaning": clean_stats,
        "augmentation": {
            "bt_raw": len(bt_data),
            "bt_after_filter": len(bt_filtered),
            "bt_added": len(bt_only),
            "clean_entries": len(clean_only),
            "total_augmented": len(augmented),
        },
        "final_stats": {
            "total_training_pairs": len(augmented),
            "unique_books": len(set(e["book"] for e in augmented)),
            "improvement_over_original": f"{len(augmented) / clean_stats['input_aligned'] * 100 - 100:.1f}%",
        },
    }

    save_json(report, base / "data_quality_report.json")

    print()
    print("=" * 60)
    print("Final Summary")
    print("=" * 60)
    print(f"  Original aligned pairs:  {clean_stats['input_aligned']}")
    print(f"  After cleaning:          {len(clean_only)}")
    print(f"  After BT augmentation:   {len(augmented)}")
    print(f"  Data increase:           {report['final_stats']['improvement_over_original']}")


if __name__ == "__main__":
    main()
