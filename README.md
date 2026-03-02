# Nama Bible

Fine-tuning [NLLB-200](https://ai.meta.com/research/no-language-left-behind/) for English-to-Nama (nmx) Bible translation.

Nama is an ultra-low-resource Papuan language spoken in the Morehead district of Western Province, Papua New Guinea, belonging to the Yam (Morehead-Upper Maro) language family.

## Goal

Build a machine translation model that can generate draft translations of Old Testament books in Nama, to assist Bible translators working on this language. The model is trained on existing New Testament parallel data (Luke & Acts) and will be expanded as more translated books become available.

## Results

| Metric | Zero-shot | Fine-tuned | Improvement |
|--------|-----------|------------|-------------|
| BLEU   | 0.19      | 3.57       | +3.37       |
| chrF   | 12.73     | 23.53      | +10.80      |

*Baseline results using 2,151 aligned verse pairs (Luke + Acts). Further improvement is expected with additional New Testament data.*

## Pipeline

1. **Parse** USFM files into verse-aligned English-Nama parallel corpus
2. **Extend** NLLB-200 tokenizer with `nmx_Latn` language code
3. **Fine-tune** NLLB-200-distilled-600M on parallel data
4. **Evaluate** with BLEU and chrF metrics on held-out test set
5. **Generate** draft translations for untranslated books

## Project Structure

```
nama-bible/
├── main.py                  # Training & evaluation pipeline
├── parse_usfm.py            # USFM parser & parallel corpus builder
├── build_multilingual.py    # Multilingual data preparation (experimental)
├── evaluate.py              # Standalone evaluation script
├── nama_eng_parallel.json   # Aligned verse pairs (Luke + Acts)
├── nama_eng_parallel.csv    # Same data in CSV format
├── linguistics_report.json  # Nama linguistic analysis
├── pyproject.toml           # Python project config
└── raw/                     # USFM source files (not in repo)
```

## Setup

Requires Python 3.10+ and a CUDA-capable GPU (tested on RTX 3090).

```bash
# Clone
git clone https://github.com/woodcross/nama-bible.git
cd nama-bible

# Install dependencies
uv sync

# Download USFM data
mkdir -p raw/nmx_usfm raw/eng_usfm
# Get Nama USFM from https://png.bible/details.php?id=nmx
# Get English WEB USFM from https://ebible.org/Scriptures/engwebp_usfm.zip

# Build parallel corpus
uv run python parse_usfm.py

# Train model
uv run python main.py
```

## Training Configuration

- **Base model:** facebook/nllb-200-distilled-600M
- **Optimizer:** Adafactor (memory-efficient)
- **Batch size:** 8 (gradient accumulation 4, effective batch 32)
- **Epochs:** 20
- **FP16:** Enabled
- **Gradient checkpointing:** Enabled
- **GPU memory usage:** ~14.5 GB

## Data Sources

- **Nama Bible (Luke & Acts):** [png.bible](https://png.bible/details.php?id=nmx) - SIL Papua New Guinea
- **English World English Bible:** [eBible.org](https://ebible.org/Scriptures/engwebp_usfm.zip)

## License

- **Code:** MIT License
- **Nama Bible text:** copyright 2025 SIL Papua New Guinea, [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)
- **English WEB Bible:** Public Domain

## Acknowledgments

This project supports the Bible translation work of SIL Papua New Guinea among the Nama people of Western Province, PNG.
