#!/usr/bin/env python3
"""TranslateGemma 27B QLoRA fine-tuning for English→Nama translation (v3.5).

v3.5: 2-stage curriculum learning.
  Stage 1: FLEx dictionary pairs (2,573) — build Nama vocabulary/grammar intuition
  Stage 2: Augmented Bible pairs (22,658) — learn Bible translation style

Designed for 2x RTX 3090 (24GB each) with pipeline parallelism.
"""

import argparse
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
OUTPUT_DIR = str(BASE / "output" / "translategemma-nama-v3.5")

SEED = 42
TRAIN_RATIO = 0.85
VAL_RATIO = 0.10

MAX_SEQ_LEN = 256
PER_GPU_BATCH = 2
GRAD_ACCUM_STEPS = 8  # effective batch = 2 * 2 GPUs * 8 = 32

# Stage 1: FLEx — higher LR, fewer epochs (vocabulary acquisition)
STAGE1_LEARNING_RATE = 1.5e-4
STAGE1_EPOCHS = 2
STAGE1_WARMUP_RATIO = 0.10

# Stage 2: Bible — lower LR, more careful fine-tuning (style alignment)
STAGE2_LEARNING_RATE = 5e-5
STAGE2_EPOCHS = 2
STAGE2_WARMUP_RATIO = 0.05

# LoRA config
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ── Data loading ─────────────────────────────────────────────────────────────


def load_flex_data() -> list[dict]:
    """Load FLEx dictionary pairs for Stage 1."""
    flex_path = BASE / "data" / "corpus" / "flex_parallel.json"
    if not flex_path.exists():
        print("ERROR: FLEx data not found at", flex_path)
        return []

    with open(flex_path, encoding="utf-8") as f:
        flex_data = json.load(f)

    pairs = [
        {"ref": f"flex:{d.get('text_title', 'unknown')}",
         "source": d["english"], "target": d["nama"],
         "origin": "flex"}
        for d in flex_data
        if d.get("english") and d.get("nama")
    ]
    print(f"  FLEx dictionary pairs: {len(pairs)}")
    return pairs


def load_bible_data() -> list[dict]:
    """Load augmented Bible pairs for Stage 2."""
    augmented_path = BASE / "data" / "corpus" / "nama_eng_augmented.json"
    raw_path = BASE / "data" / "corpus" / "nama_eng_parallel.json"

    if augmented_path.exists():
        with open(augmented_path, encoding="utf-8") as f:
            data = json.load(f)
        pairs = [
            {"ref": d["ref"], "source": d["english"], "target": d["nama"],
             "origin": d.get("source", "web")}
            for d in data
        ]
        print(f"  Augmented Bible pairs: {len(pairs)}")
    elif raw_path.exists():
        print("  WARNING: Using raw data (run clean_and_augment.py first)")
        with open(raw_path, encoding="utf-8") as f:
            web_data = json.load(f)
        pairs = [
            {"ref": d["ref"], "source": d["english"], "target": d["nama"], "origin": "web"}
            for d in web_data if d.get("aligned")
        ]
        print(f"  Raw WEB pairs: {len(pairs)}")
    else:
        print("ERROR: No Bible data found")
        return []

    return pairs


def split_data(data: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Split into train/val/test."""
    random.seed(SEED)
    shuffled = data.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


# ── Prompt formatting ────────────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "<start_of_turn>user\n"
    "Translate the following text from English to Nama.\n\n"
    "{source}<end_of_turn>\n"
    "<start_of_turn>model\n"
    "{target}<end_of_turn>"
)


def format_example(example: dict) -> str:
    return PROMPT_TEMPLATE.format(source=example["source"], target=example["target"])


def build_dataset(split: list[dict]) -> Dataset:
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


def run_stage(stage_name, model, tokenizer, train_ds, val_ds,
              output_dir, learning_rate, epochs, warmup_ratio):
    """Run a single training stage."""
    print(f"\n{'=' * 60}")
    print(f"  {stage_name}")
    print(f"  LR: {learning_rate}, Epochs: {epochs}, Data: {len(train_ds)}")
    print(f"{'=' * 60}")

    training_args = SFTConfig(
        output_dir=output_dir,
        max_length=MAX_SEQ_LEN,
        num_train_epochs=epochs,
        per_device_train_batch_size=PER_GPU_BATCH,
        per_device_eval_batch_size=PER_GPU_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
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

    torch.cuda.empty_cache()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    trainer.train()

    # Log best eval loss
    best = trainer.state.best_metric
    print(f"\n  {stage_name} best eval_loss: {best:.4f}")

    return trainer


# ── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_samples(model, tokenizer, test_data: list[dict], n: int = 10):
    """Generate translations for test samples and print them."""
    import sacrebleu

    print("\n" + "=" * 60)
    print(f"  Sample Translations ({n} verses)")
    print("=" * 60)

    torch.cuda.empty_cache()
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
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        pred = tokenizer.decode(generated, skip_special_tokens=True)
        del inputs, outputs, generated
        torch.cuda.empty_cache()
        pred = pred.replace("<end_of_turn>", "").strip()

        predictions.append(pred)
        references.append(item["target"])

        print(f"\n--- [{item['ref']}] ---")
        print(f"  SRC:  {item['source'][:100]}")
        print(f"  REF:  {item['target'][:100]}")
        print(f"  PRED: {pred[:100]}")

    chrf = sacrebleu.corpus_chrf(predictions, [references])
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    print(f"\n  Sample BLEU: {bleu.score:.2f}")
    print(f"  Sample chrF: {chrf.score:.2f}")

    return {"bleu": bleu.score, "chrf": chrf.score}


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-stage1", action="store_true",
                        help="Skip Stage 1 (FLEx) and load existing adapter")
    parser.add_argument("--stage1-adapter", type=str, default=None,
                        help="Path to Stage 1 adapter (for skipping Stage 1)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Nama Bible TranslateGemma 27B QLoRA Training (v3.5)")
    print("  2-Stage: FLEx vocabulary → Bible translation")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    flex_data = load_flex_data()
    bible_data = load_bible_data()

    if not bible_data:
        print("ERROR: No Bible data found. Aborting.")
        return

    # Split both datasets
    flex_train, flex_val, _ = split_data(flex_data)
    bible_train, bible_val, bible_test = split_data(bible_data)

    print(f"\n  Stage 1 (FLEx):  train={len(flex_train)}, val={len(flex_val)}")
    print(f"  Stage 2 (Bible): train={len(bible_train)}, val={len(bible_val)}, test={len(bible_test)}")

    # Build datasets
    flex_train_ds = build_dataset(flex_train)
    flex_val_ds = build_dataset(flex_val)
    bible_train_ds = build_dataset(bible_train)
    bible_val_ds = build_dataset(bible_val)

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # ── Stage 1: FLEx vocabulary ──
    stage1_dir = str(Path(OUTPUT_DIR) / "stage1_flex")

    if args.skip_stage1:
        adapter_path = args.stage1_adapter or str(Path(stage1_dir) / "stage1_adapter")
        print(f"\nSkipping Stage 1, loading adapter from {adapter_path}")
    else:
        if not flex_data:
            print("WARNING: No FLEx data, skipping Stage 1")
        else:
            run_stage(
                "Stage 1: FLEx Vocabulary Acquisition",
                model, tokenizer,
                flex_train_ds, flex_val_ds,
                output_dir=stage1_dir,
                learning_rate=STAGE1_LEARNING_RATE,
                epochs=STAGE1_EPOCHS,
                warmup_ratio=STAGE1_WARMUP_RATIO,
            )

            # Save Stage 1 adapter
            s1_adapter = Path(stage1_dir) / "stage1_adapter"
            model.save_pretrained(str(s1_adapter))
            tokenizer.save_pretrained(str(s1_adapter))
            print(f"  Stage 1 adapter saved to {s1_adapter}")

    # ── Stage 2: Bible translation ──
    stage2_dir = str(Path(OUTPUT_DIR) / "stage2_bible")

    trainer = run_stage(
        "Stage 2: Bible Translation Style",
        model, tokenizer,
        bible_train_ds, bible_val_ds,
        output_dir=stage2_dir,
        learning_rate=STAGE2_LEARNING_RATE,
        epochs=STAGE2_EPOCHS,
        warmup_ratio=STAGE2_WARMUP_RATIO,
    )

    # Save final adapter
    adapter_path = Path(OUTPUT_DIR) / "final_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nFinal adapter saved to {adapter_path}")

    # ── Evaluate ──
    results = evaluate_samples(trainer.model, tokenizer, bible_test, n=10)

    # Save test split
    test_json = Path(OUTPUT_DIR) / "test_split.json"
    test_json.parent.mkdir(parents=True, exist_ok=True)
    with open(test_json, "w", encoding="utf-8") as f:
        json.dump(bible_test, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("  Training complete! (v3.5 — 2-stage)")
    print(f"  Stage 1: FLEx ({len(flex_data)} pairs, {STAGE1_EPOCHS} epochs, LR={STAGE1_LEARNING_RATE})")
    print(f"  Stage 2: Bible ({len(bible_data)} pairs, {STAGE2_EPOCHS} epochs, LR={STAGE2_LEARNING_RATE})")
    print(f"  Final adapter: {adapter_path}")
    print(f"  BLEU: {results['bleu']:.2f}, chrF: {results['chrf']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
