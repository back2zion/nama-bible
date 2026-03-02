# Nama Bible

English-to-Nama (nmx) Bible translation using [TranslateGemma](https://huggingface.co/google/translategemma-4b-it) — fine-tuning Google's translation LLM for an ultra-low-resource Papuan language of Papua New Guinea.

Nama is an ultra-low-resource Papuan language spoken in the Morehead district of Western Province, Papua New Guinea, belonging to the Yam (Morehead-Upper Maro) language family.

## Goal

Build a machine translation model that can generate draft translations of Old Testament books in Nama, to assist Bible translators working on this language. The model is trained on existing New Testament parallel data (Luke & Acts) and will be expanded as more translated books become available.

## Results

| Metric | Zero-shot | Fine-tuned | Improvement |
|--------|-----------|------------|-------------|
| BLEU   | 0.08      | 3.92       | +3.84       |
| chrF   | 13.39     | 33.27      | +19.88      |

*Results using 2,151 aligned verse pairs (Luke + Acts, 20 epochs). Further improvement is expected with additional New Testament data.*

**chrF score guide** (max 100):

| Range | Quality |
|-------|---------|
| 50+   | Usable as draft for post-editing by translators |
| 60+   | Practically useful draft translation |
| 70+   | High-quality translation |
| 33    | Current — meaningful progress from 2,151 sentences, not yet usable as draft |

## Roadmap

| Phase | Model | GPU | Data | Target |
|-------|-------|-----|------|--------|
| **v1 (current)** | NLLB-200-distilled-600M | 1x RTX 3090 | Luke + Acts (2,151 pairs) | Baseline, chrF 33 |
| **v2 (planned)** | TranslateGemma 4B + QLoRA | 1x RTX 3090 | Full NT (~10k+ pairs) | chrF 50+ |
| **v3 (planned)** | TranslateGemma 27B + QLoRA | 2x RTX 3090 | Full NT (~10k+ pairs) | chrF 60+ (usable draft) |

v2/v3 are planned for when additional New Testament data becomes available.

## Pipeline

1. **Parse** USFM files into verse-aligned English-Nama parallel corpus
2. **Fine-tune** TranslateGemma 4B with QLoRA on parallel data
3. **Evaluate** with BLEU and chrF metrics on held-out test set
4. **Generate** draft translations for untranslated books

## Project Structure

```
nama_bible/
├── scripts/                 # All Python scripts
│   ├── parse_usfm.py        # USFM parser & parallel corpus builder
│   ├── build_multilingual.py # Multilingual data preparation
│   ├── rtf_to_usfm.py       # Paratext RTF → USFM converter
│   ├── train.py              # Training & evaluation pipeline
│   ├── evaluate.py           # Standalone evaluation script
│   └── translate_ruth.py     # Draft translation generator
├── data/                    # All source data
│   ├── source_rtf/           # Paratext RTF originals
│   ├── nmx/                  # Nama USFM (27 NT books)
│   ├── eng/                  # English WEB USFM (full Bible)
│   ├── multilingual/         # PNG language USFM corpora
│   └── nmx_ebible/           # eBible.org originals (LUK, ACT)
├── output/                  # Model checkpoints & drafts (gitignored)
├── pyproject.toml
└── README.md
```

## Hardware Requirements

### v1 (current) — NLLB-200 fine-tuning
- **GPU:** 1x NVIDIA RTX 3090 (24 GB VRAM)
- **VRAM usage:** ~14.5 GB
- **Training time:** ~30 minutes

### v2 (planned) — TranslateGemma 4B QLoRA
- **GPU:** 1x NVIDIA RTX 3090 (24 GB VRAM)
- **Method:** QLoRA (4-bit)
- **Libraries:** transformers, peft, bitsandbytes, accelerate

### v3 (planned) — TranslateGemma 27B QLoRA
- **GPU:** 2x NVIDIA RTX 3090 (48 GB VRAM total)
- **Method:** QLoRA (4-bit) + FSDP multi-GPU
- **Libraries:** transformers, peft, bitsandbytes, accelerate

## Setup

Requires Python 3.10+ and CUDA-capable GPU(s).

```bash
# Clone
git clone https://github.com/back2zion/nama-bible.git
cd nama-bible

# Install dependencies
uv sync

# Build parallel corpus
uv run python scripts/parse_usfm.py

# Train model
uv run python scripts/train.py
```

## Training Configuration (v1)

- **Base model:** facebook/nllb-200-distilled-600M
- **Optimizer:** Adafactor (memory-efficient)
- **Batch size:** 8 (gradient accumulation 4, effective batch 32)
- **Epochs:** 20
- **FP16:** Enabled
- **Gradient checkpointing:** Enabled

## Data Sources

- **Nama Bible (Luke & Acts):** [png.bible](https://png.bible/details.php?id=nmx) - SIL Papua New Guinea
- **English World English Bible:** [eBible.org](https://ebible.org/Scriptures/engwebp_usfm.zip)

## License

- **Code:** MIT License
- **Nama Bible text:** copyright 2025 SIL Papua New Guinea, [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)
- **English WEB Bible:** Public Domain

## Acknowledgments

This project is a small helping hand for the expansion of God's Kingdom, supporting the Bible translation work of SIL Papua New Guinea among the Nama people of Western Province, PNG.
