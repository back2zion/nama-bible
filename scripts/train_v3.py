#!/usr/bin/env python3
"""TranslateGemma 27B QLoRA fine-tuning for English→Nama translation.

Fine-tunes google/translategemma-27b-it with 4-bit QLoRA on ~15,000 aligned
English-Nama Bible verse pairs (27 NT books WEB + Back Translation).
Designed for 2x RTX 3090 (24GB each) with pipeline parallelism.
"""

import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# ── Configuration ────────────────────────────────────────────────────────────

BASE = Path(__file__).resolve().parent.parent
MODEL_NAME = "google/translategemma-27b-it"
OUTPUT_DIR = str(BASE / "output" / "translategemma-nama-v3")

SEED = 42
TRAIN_RATIO = 0.85
VAL_RATIO = 0.10
# TEST_RATIO = 0.05 (remainder)

MAX_SEQ_LEN = 256
PER_GPU_BATCH = 2
GRAD_ACCUM_STEPS = 8  # effective batch = 2 * 2 GPUs * 8 = 32
LEARNING_RATE = 1.5e-4
EPOCHS = 5
WARMUP_RATIO = 0.05

# LoRA config
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ── Data loading ─────────────────────────────────────────────────────────────


def load_data() -> list[dict]:
    """Load cleaned + augmented parallel corpus."""
    pairs = []

    # Use cleaned & augmented data if available, otherwise fall back to raw
    augmented_path = BASE / "nama_eng_augmented.json"
    raw_path = BASE / "nama_eng_parallel.json"

    if augmented_path.exists():
        print("Loading cleaned + augmented data...")
        with open(augmented_path, encoding="utf-8") as f:
            data = json.load(f)
        pairs = [
            {"ref": d["ref"], "source": d["english"], "target": d["nama"],
             "origin": d.get("source", "web")}
            for d in data
        ]
        print(f"Augmented pairs: {len(pairs)}")
    elif raw_path.exists():
        print("WARNING: Using raw data (run clean_and_augment.py first)")
        with open(raw_path, encoding="utf-8") as f:
            web_data = json.load(f)
        pairs = [
            {"ref": d["ref"], "source": d["english"], "target": d["nama"], "origin": "web"}
            for d in web_data if d.get("aligned")
        ]
        print(f"Raw WEB pairs: {len(pairs)}")

    print(f"Total pairs: {len(pairs)}")
    return pairs


def split_data(data: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Split into train/val/test by book to avoid data leakage."""
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


# ── Prompt formatting ────────────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "<start_of_turn>user\n"
    "Translate the following text from English to Nama.\n\n"
    "{source}<end_of_turn>\n"
    "<start_of_turn>model\n"
    "{target}<end_of_turn>"
)


def format_example(example: dict) -> str:
    """Format a single example into the TranslateGemma prompt format."""
    return PROMPT_TEMPLATE.format(source=example["source"], target=example["target"])


def build_dataset(split: list[dict]) -> Dataset:
    """Convert list of pairs to HuggingFace Dataset with formatted text."""
    texts = [format_example(ex) for ex in split]
    return Dataset.from_dict({"text": texts})


# ── Model setup ──────────────────────────────────────────────────────────────


def load_model_and_tokenizer():
    """Load TranslateGemma 27B with 4-bit quantization and LoRA."""
    print(f"\nLoading {MODEL_NAME} with 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_gpus = torch.cuda.device_count()
    print(f"  GPUs: {n_gpus}, device_map: auto")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ── Training ─────────────────────────────────────────────────────────────────


def train(model, tokenizer, train_ds: Dataset, val_ds: Dataset):
    """Fine-tune with SFTTrainer."""
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_length=MAX_SEQ_LEN,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_GPU_BATCH,
        per_device_eval_batch_size=PER_GPU_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        optim="paged_adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        save_total_limit=2,
        seed=SEED,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    print("\n" + "=" * 60)
    print("  TranslateGemma 27B QLoRA Fine-tuning")
    print("=" * 60)
    trainer.train()

    # Save adapter
    adapter_path = Path(OUTPUT_DIR) / "final_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nAdapter saved to {adapter_path}")

    return trainer


# ── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_samples(model, tokenizer, test_data: list[dict], n: int = 10):
    """Generate translations for test samples and print them."""
    import sacrebleu

    print("\n" + "=" * 60)
    print(f"  Sample Translations ({n} verses)")
    print("=" * 60)

    model.eval()
    predictions = []
    references = []
    samples = test_data[:n]

    for item in samples:
        prompt = (
            "<start_of_turn>user\n"
            "Translate the following text from English to Nama.\n\n"
            f"{item['source']}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_SEQ_LEN,
                do_sample=False,
                temperature=1.0,
            )
        # Decode only the generated part
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        pred = tokenizer.decode(generated, skip_special_tokens=True)
        # Strip end_of_turn marker if present
        pred = pred.replace("<end_of_turn>", "").strip()

        predictions.append(pred)
        references.append(item["target"])

        print(f"\n--- [{item['ref']}] ---")
        print(f"  SRC:  {item['source'][:100]}")
        print(f"  REF:  {item['target'][:100]}")
        print(f"  PRED: {pred[:100]}")

    # Compute metrics on the sample
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    print(f"\n  Sample BLEU: {bleu.score:.2f}")
    print(f"  Sample chrF: {chrf.score:.2f}")

    return {"bleu": bleu.score, "chrf": chrf.score}


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  Nama Bible TranslateGemma 27B QLoRA Training (v3)")
    print("=" * 60)

    # Load and split data
    data = load_data()
    if not data:
        print("ERROR: No training data found. Run parse_usfm.py first.")
        return
    train_split, val_split, test_split = split_data(data)

    # Build datasets
    train_ds = build_dataset(train_split)
    val_ds = build_dataset(val_split)
    print(f"\nDatasets: train={len(train_ds)}, val={len(val_ds)}")

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Train
    trainer = train(model, tokenizer, train_ds, val_ds)

    # Evaluate
    results = evaluate_samples(trainer.model, tokenizer, test_split, n=10)

    # Save test data for evaluate_v3.py
    test_json = Path(OUTPUT_DIR) / "test_split.json"
    test_json.parent.mkdir(parents=True, exist_ok=True)
    with open(test_json, "w", encoding="utf-8") as f:
        json.dump(test_split, f, ensure_ascii=False, indent=2)
    print(f"\nTest split saved to {test_json}")

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Adapter: {OUTPUT_DIR}/final_adapter")
    print(f"  BLEU: {results['bleu']:.2f}, chrF: {results['chrf']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
