# Nama Bible

English-to-Nama (nmx) Bible translation using [TranslateGemma](https://huggingface.co/google/translategemma-27b-it) — fine-tuning Google's translation LLM for an ultra-low-resource Papuan language of Papua New Guinea.

Nama is an ultra-low-resource Papuan language spoken in the Morehead district of Western Province, Papua New Guinea, belonging to the Yam (Morehead-Upper Maro) language family.

## Goal

Build a machine translation model that can generate draft translations of Old Testament books in Nama, to assist Bible translators working on this language. The model is trained on existing New Testament parallel data and will be expanded as more translated books become available.

## Progress

### v1 — NLLB-200 Baseline

- **Model:** facebook/nllb-200-distilled-600M
- **Data:** Luke + Acts only (2,151 aligned pairs from eBible.org)
- **GPU:** 1x RTX 3090

| Metric | Zero-shot | Fine-tuned | Improvement |
|--------|-----------|------------|-------------|
| BLEU   | 0.08      | 3.92       | +3.84       |
| chrF   | 13.39     | 33.27      | +19.88      |

### v2 — TranslateGemma 4B

- **Model:** google/translategemma-4b-it + QLoRA
- **Data:** Full NT 27 books — RTF sources from Paratext converted to USFM, parsed into 3,167 aligned verse pairs
- **GPU:** 1x RTX 3090

### v3 — TranslateGemma 27B (current)

- **Model:** google/translategemma-27b-it + QLoRA (4-bit quantization)
- **Data:** Full NT 27 books (3,167 aligned pairs) + multilingual transfer (7 PNG languages)
- **GPU:** 2x RTX 3090 (48 GB VRAM total)
- **First OT draft generated:** Book of Ruth (룻기) — 4 chapters, 85 verses

### v3.1 — Data Quality + Retrain

- **Cleaned** parallel data: removed 1,430 noisy entries (multi-verse concatenations, extreme length ratio outliers)
- **Augmented** with back-translation data: 1,403 BT pairs (Nama → English) added as training signal
- **Retrained** on 3,140 cleaned + augmented pairs

| Metric | v3 (before) | v3.1 (after cleaning) | Improvement |
|--------|-------------|----------------------|-------------|
| BLEU   | 3.92        | **34.75**            | +30.83      |
| chrF   | 33.27       | **59.77**            | +26.50      |

### v3.2 — Human Feedback Loop (current)

1. Generated first OT draft: **Book of Ruth** (룻기) — 85 verses
2. Shared draft with Nama Bible translation team for review
3. Received **human-translated Ruth** (룻기.rtf) from Nama language speakers
4. Compared model output vs human translation: **chrF 37.06**
5. Added human Ruth translation (85 pairs) to training data → **3,225 total pairs**
6. Retrained model with human feedback data
7. Re-translated Ruth with v3.2 model: **chrF 37.63, BLEU 9.23** (improved from 6.11)

| Metric | v3.1 (first draft) | v3.2 (after human feedback) |
|--------|-------------------|----------------------------|
| BLEU (Ruth vs human) | 6.11 | **9.23** |
| chrF (Ruth vs human) | 37.06 | **37.63** |
| BLEU (NT test set) | 34.75 | 28.01 |
| chrF (NT test set) | 59.77 | 58.31 |

**Translation Improvement Cycle:**
```
Model draft → Human review → Receive corrections → Add to training data → Retrain → Better draft
```

| Dataset | Entries | Description |
|---------|---------|-------------|
| `nama_eng_parallel.json` | 3,167 aligned | Raw parsed NT parallel data |
| `nama_eng_clean.json` | 1,737 | After quality filtering |
| `nama_eng_augmented.json` | 3,225 | Clean + BT + human Ruth |
| `nama_bt_parallel.json` | 2,622 | Raw back-translation corpus |

**chrF score guide** (max 100):

| Range | Quality |
|-------|---------|
| 50+   | Usable as draft for post-editing by translators |
| 60+   | Practically useful draft translation |
| 70+   | High-quality translation |

## Pipeline

1. **Convert** Paratext RTF sources to USFM format (`rtf_to_usfm.py`)
2. **Parse** USFM files into verse-aligned English-Nama parallel corpus (`parse_usfm.py`)
3. **Clean & Augment** parallel data with quality filtering and back-translation (`clean_and_augment.py`)
4. **Fine-tune** TranslateGemma 27B with QLoRA on parallel data (`train_v3.py`)
5. **Evaluate** with BLEU and chrF metrics on held-out test set (`evaluate_v3.py`)
6. **Generate** draft translations for untranslated OT books (`translate_draft.py`)

## Work History

### Phase 1: Data Acquisition & Parsing

1. Obtained Nama NT text from SIL PNG — 5 Paratext RTF files covering 27 NT books
2. Built RTF-to-USFM converter (`rtf_to_usfm.py`) to extract structured text from Paratext format
3. Obtained English WEB Bible (full OT+NT) from eBible.org as translation source
4. Built USFM parser (`parse_usfm.py`) to create verse-aligned English-Nama parallel corpus
5. Initial corpus: Luke + Acts (2,151 pairs) from eBible.org Nama source
6. Expanded to full NT (3,167 pairs) after parsing all 27 books from RTF sources

### Phase 2: Multilingual Data

1. Collected USFM data for 7 related PNG languages (Hiri Motu, Tok Pisin, Bedamuni, Bine, Gizrra, Gidra, Magi)
2. Built multilingual alignment pipeline (`build_multilingual.py`)
3. Created back-translation corpus (`nama_bt_parallel.json`) — 2,622 Nama-BT pairs from 26 books

### Phase 3: Model Training

1. **v1:** NLLB-200 baseline — chrF 33.27 on Luke+Acts data
2. **v2:** TranslateGemma 4B with QLoRA — trained on full NT
3. **v3:** TranslateGemma 27B with QLoRA (4-bit) on 2x RTX 3090 — current production model

### Phase 4: Data Quality Improvement

1. Analyzed parallel data quality: identified 1,430 noisy entries (multi-verse concatenations, length ratio outliers)
2. Built cleaning + augmentation pipeline (`clean_and_augment.py`)
3. Retrained v3 on cleaned data → **chrF jumped from 33.27 to 59.77** (near usable-draft threshold)

### Phase 5: Human-in-the-Loop Translation (2026-03-22)

1. Generated first OT draft: **Book of Ruth** (룻기) — 85 verses translated to Nama
2. Shared draft with translation team (박정석, QT IVF) for Nama speaker review
3. Received human-translated Ruth (룻기.rtf) from Nama language speakers
4. Converted RTF → USFM, parsed into 85 verse-aligned pairs
5. Compared model vs human translation: chrF 37.06 — model captures structure but differs in vocabulary choices
6. Added human Ruth data to training corpus (3,140 → 3,225 pairs)
7. Retrained model (v3.2) with human feedback → Ruth chrF improved 37.06 → 37.63
8. This establishes the **feedback loop**: draft → review → correct → retrain → better draft
9. More OT books from human translators will accelerate this cycle

## Project Structure

```
nama_bible/
├── scripts/                    # All Python scripts
│   ├── parse_usfm.py           # USFM parser & parallel corpus builder
│   ├── build_multilingual.py   # Multilingual data preparation
│   ├── rtf_to_usfm.py          # Paratext RTF → USFM converter
│   ├── clean_and_augment.py    # Data quality cleaning & BT augmentation
│   ├── compare_ruth.py         # Model vs human translation comparison
│   ├── train.py                # v1 (NLLB) training pipeline
│   ├── train_v3.py             # v3 (TranslateGemma 27B) training
│   ├── evaluate.py             # v1 evaluation
│   ├── evaluate_v3.py          # v3 evaluation
│   └── translate_draft.py      # OT draft translation generator
├── data/                       # All source data
│   ├── source_rtf/             # Paratext RTF originals
│   ├── nmx/                    # Nama USFM (27 NT + Ruth)
│   ├── eng/                    # English WEB USFM (full Bible)
│   ├── heb/                    # Hebrew OT originals
│   ├── grk/                    # Greek NT originals
│   ├── bt/                     # Back-translation USFM (26 books)
│   ├── multilingual/           # PNG language USFM corpora (7 languages)
│   ├── nmx_ebible/             # eBible.org originals (LUK, ACT)
│   └── corpus/                 # Processed parallel corpora (JSON/CSV)
├── docs/                       # Work history & technical documentation
├── reports/                    # Evaluation & analysis reports
├── output/                     # Model checkpoints & drafts (gitignored)
│   ├── translategemma-nama-v3/ # v3 LoRA adapter weights
│   └── drafts/                 # Generated OT draft translations (versioned)
├── pyproject.toml
└── README.md
```

## Hardware Requirements

- **GPU:** 2x NVIDIA RTX 3090 (48 GB VRAM total)
- **VRAM usage:** ~36 GB (4-bit quantized TranslateGemma 27B)
- **Model loading:** ~17 minutes
- **Translation speed:** ~85 verses in ~15 minutes

## Setup

Requires Python 3.10+ and CUDA-capable GPU(s).

```bash
# Clone
git clone https://github.com/back2zion/nama-bible.git
cd nama-bible

# Install dependencies
uv sync

# Build parallel corpus from USFM
uv run python scripts/parse_usfm.py

# Clean data and augment with back-translation
uv run python scripts/clean_and_augment.py

# Train model (v3)
uv run python scripts/train_v3.py

# Generate OT draft translation
uv run python scripts/translate_draft.py RUT
```

## Data Sources

- **Nama Bible (NT):** [png.bible](https://png.bible/details.php?id=nmx) - SIL Papua New Guinea
- **English World English Bible:** [eBible.org](https://ebible.org/Scriptures/engwebp_usfm.zip)
- **Multilingual PNG Bibles:** [png.bible](https://png.bible) - Hiri Motu, Tok Pisin, Bedamuni, Bine, Gizrra, Gidra, Magi

## License

- **Code:** MIT License
- **Nama Bible text:** copyright 2025 SIL Papua New Guinea, [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)
- **English WEB Bible:** Public Domain

## Acknowledgments

This project is a small helping hand for the expansion of God's Kingdom, supporting the Bible translation work of SIL Papua New Guinea among the Nama people of Western Province, PNG.
