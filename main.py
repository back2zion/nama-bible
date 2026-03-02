"""NLLB-200 English→Nama fine-tuning pilot experiment.

Fine-tunes facebook/nllb-200-distilled-600M on ~2,151 aligned English-Nama
(nmx, Papua New Guinea) Bible verse pairs (Luke + Acts) to evaluate
translation feasibility for this ultra-low-resource language.
"""

import json
import random
import warnings

warnings.filterwarnings("ignore", message=".*tie_word_embeddings.*")
warnings.filterwarnings("ignore", message=".*not sharded.*")

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("accelerate.utils.modeling").setLevel(logging.ERROR)

import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import sacrebleu
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME = "facebook/nllb-200-distilled-600M"
NEW_LANG_CODE = "nmx_Latn"
PROXY_LANG_CODE = "tpi_Latn"  # Tok Pisin – PNG lingua franca
DATA_PATH = "nama_eng_parallel.json"
OUTPUT_DIR = "./output/nllb-nama-pilot"

SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# TEST_RATIO = 0.10 (remainder)

MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 5e-5
WARMUP_RATIO = 0.1
GRAD_ACCUM_STEPS = 4  # effective batch = 8 * 4 = 32
REPETITION_PENALTY = 1.3


# ── Data helpers ─────────────────────────────────────────────────────────────


def load_parallel_data() -> list[dict]:
    """Load aligned verse pairs from the JSON file."""
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    aligned = [d for d in data if d.get("aligned")]
    print(f"Loaded {len(aligned)} aligned verse pairs (from {len(data)} total)")
    return aligned


def split_data(
    data: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Randomly split data into train / val / test."""
    random.seed(SEED)
    shuffled = data.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]

    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# ── Zero-shot baseline ───────────────────────────────────────────────────────


def evaluate_zero_shot(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    test_data: list[dict],
) -> dict:
    """Translate test set using proxy language code (no fine-tuning)."""
    print("\n" + "=" * 60)
    print("ZERO-SHOT BASELINE (proxy language: {})".format(PROXY_LANG_CODE))
    print("=" * 60)

    tokenizer.src_lang = "eng_Latn"
    predictions = []

    for item in test_data:
        inputs = tokenizer(
            item["english"], return_tensors="pt", max_length=MAX_LENGTH, truncation=True
        ).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(PROXY_LANG_CODE),
                max_new_tokens=MAX_LENGTH,
                repetition_penalty=REPETITION_PENALTY,
            )
        pred = tokenizer.decode(generated[0], skip_special_tokens=True)
        predictions.append(pred)

    references = [item["nama"] for item in test_data]
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references])

    print(f"  BLEU:  {bleu.score:.2f}")
    print(f"  chrF:  {chrf.score:.2f}")
    return {"bleu": bleu.score, "chrf": chrf.score, "predictions": predictions}


# ── Tokenizer & model preparation ────────────────────────────────────────────


def extend_tokenizer(tokenizer: AutoTokenizer) -> AutoTokenizer:
    """Add nmx_Latn language code to the NLLB tokenizer."""
    existing = tokenizer.all_special_tokens
    if NEW_LANG_CODE not in existing:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": existing + [NEW_LANG_CODE]}
        )
    new_id = tokenizer.convert_tokens_to_ids(NEW_LANG_CODE)
    if hasattr(tokenizer, "lang_code_to_id"):
        tokenizer.lang_code_to_id[NEW_LANG_CODE] = new_id
    print(f"Added {NEW_LANG_CODE} to tokenizer (id={new_id})")
    return tokenizer


def prepare_model(
    model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer
) -> AutoModelForSeq2SeqLM:
    """Resize embeddings and initialise the new language token."""
    old_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    new_id = tokenizer.convert_tokens_to_ids(NEW_LANG_CODE)

    # Initialise new embedding as mean of all existing language embeddings
    with torch.no_grad():
        mean_emb = model.get_input_embeddings().weight[:old_size].mean(dim=0)
        model.get_input_embeddings().weight[new_id] = mean_emb
        # Also initialise the decoder-side (lm_head shares weight, but be explicit for the final layer)
        if model.get_output_embeddings() is not None:
            out_emb = model.get_output_embeddings()
            if out_emb.weight.shape[0] > old_size:
                out_emb.weight[new_id] = mean_emb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model on {device}, embeddings resized {old_size} → {len(tokenizer)}")
    return model


# ── Dataset construction ─────────────────────────────────────────────────────


def _tokenize(batch: dict, tokenizer: AutoTokenizer) -> dict:
    tokenizer.src_lang = "eng_Latn"
    model_inputs = tokenizer(
        batch["english"], max_length=MAX_LENGTH, truncation=True, padding=False
    )

    tokenizer.src_lang = NEW_LANG_CODE
    labels = tokenizer(
        batch["nama"], max_length=MAX_LENGTH, truncation=True, padding=False
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def build_datasets(
    tokenizer: AutoTokenizer,
    train: list[dict],
    val: list[dict],
    test: list[dict],
) -> tuple[Dataset, Dataset, Dataset]:
    """Convert raw verse pairs to tokenized HuggingFace Datasets."""
    def to_hf(split: list[dict]) -> Dataset:
        return Dataset.from_dict(
            {"english": [d["english"] for d in split], "nama": [d["nama"] for d in split]}
        )

    train_ds = to_hf(train).map(lambda b: _tokenize(b, tokenizer), batched=True, remove_columns=["english", "nama"])
    val_ds = to_hf(val).map(lambda b: _tokenize(b, tokenizer), batched=True, remove_columns=["english", "nama"])
    test_ds = to_hf(test).map(lambda b: _tokenize(b, tokenizer), batched=True, remove_columns=["english", "nama"])

    print(f"Datasets ready: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_ds, val_ds, test_ds


# ── Training ─────────────────────────────────────────────────────────────────


def train_model(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    train_ds: Dataset,
    val_ds: Dataset,
) -> Seq2SeqTrainer:
    """Fine-tune with Seq2SeqTrainer."""
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=int((len(train_ds) / BATCH_SIZE / GRAD_ACCUM_STEPS) * EPOCHS * WARMUP_RATIO),
        fp16=True,
        optim="adafactor",
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
        logging_steps=50,
        save_total_limit=2,
        seed=SEED,
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print("\n" + "=" * 60)
    print("FINE-TUNING")
    print("=" * 60)
    trainer.train()
    return trainer


# ── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_model(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    test_ds: Dataset,
    test_data: list[dict],
) -> dict:
    """Evaluate fine-tuned model on test set."""
    print("\n" + "=" * 60)
    print("FINE-TUNED EVALUATION")
    print("=" * 60)

    tokenizer.src_lang = "eng_Latn"
    forced_bos = tokenizer.convert_tokens_to_ids(NEW_LANG_CODE)
    predictions = []

    model.eval()
    for item in test_data:
        inputs = tokenizer(
            item["english"], return_tensors="pt", max_length=MAX_LENGTH, truncation=True
        ).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_new_tokens=MAX_LENGTH,
                repetition_penalty=REPETITION_PENALTY,
            )
        pred = tokenizer.decode(generated[0], skip_special_tokens=True)
        predictions.append(pred)

    references = [item["nama"] for item in test_data]
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references])

    print(f"  BLEU:  {bleu.score:.2f}")
    print(f"  chrF:  {chrf.score:.2f}")
    return {"bleu": bleu.score, "chrf": chrf.score, "predictions": predictions}


def print_sample_translations(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    test_data: list[dict],
    n: int = 10,
) -> None:
    """Print side-by-side sample translations from the test set."""
    print("\n" + "=" * 60)
    print(f"SAMPLE TRANSLATIONS ({n} verses)")
    print("=" * 60)

    tokenizer.src_lang = "eng_Latn"
    forced_bos = tokenizer.convert_tokens_to_ids(NEW_LANG_CODE)
    samples = test_data[:n]

    for i, item in enumerate(samples, 1):
        inputs = tokenizer(
            item["english"], return_tensors="pt", max_length=MAX_LENGTH, truncation=True
        ).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_new_tokens=MAX_LENGTH,
                repetition_penalty=REPETITION_PENALTY,
            )
        pred = tokenizer.decode(generated[0], skip_special_tokens=True)

        print(f"\n--- [{item['ref']}] ---")
        print(f"  ENG:   {item['english']}")
        print(f"  NAMA:  {item['nama']}")
        print(f"  PRED:  {pred}")


# ── Save results ─────────────────────────────────────────────────────────────


def save_results(
    zs_results: dict,
    ft_results: dict,
    test_data: list[dict],
) -> None:
    """Save evaluation results to JSON file."""
    results = {
        "zero_shot": {"bleu": zs_results["bleu"], "chrf": zs_results["chrf"]},
        "fine_tuned": {"bleu": ft_results["bleu"], "chrf": ft_results["chrf"]},
        "predictions": [
            {"ref": t["ref"], "english": t["english"], "nama": t["nama"], "predicted": p}
            for t, p in zip(test_data, ft_results["predictions"])
        ],
    }
    out_path = "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    # Step 1: Load & split data
    data = load_parallel_data()
    train, val, test = split_data(data)

    # Step 2: Load model/tokenizer & zero-shot baseline
    print(f"\nLoading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, tie_word_embeddings=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    zs_results = evaluate_zero_shot(model, tokenizer, test)

    # Move model back to CPU before modifying embeddings
    model = model.cpu()

    # Step 3: Extend tokenizer with nmx_Latn
    tokenizer = extend_tokenizer(tokenizer)

    # Step 4: Prepare model (resize embeddings, init new token, move to CUDA)
    model = prepare_model(model, tokenizer)

    # Step 5: Build tokenized datasets
    train_ds, val_ds, test_ds = build_datasets(tokenizer, train, val, test)

    # Step 6: Fine-tune
    trainer = train_model(model, tokenizer, train_ds, val_ds)

    # Step 7: Evaluate
    ft_results = evaluate_model(trainer.model, tokenizer, test_ds, test)

    # Comparison table
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<10} {'Zero-shot':>12} {'Fine-tuned':>12} {'Δ':>10}")
    print("-" * 46)
    for m in ("bleu", "chrf"):
        zs = zs_results[m]
        ft = ft_results[m]
        print(f"{m.upper():<10} {zs:>12.2f} {ft:>12.2f} {ft - zs:>+10.2f}")

    # Step 8: Sample translations
    print_sample_translations(trainer.model, tokenizer, test, n=10)

    # Step 9: Save results
    save_results(zs_results, ft_results, test)


if __name__ == "__main__":
    main()
