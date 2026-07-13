"""
LoRA instruction-tuning for security tool-call classification.

Trains a small LLM (≤7B) on the instruct dataset using LoRA via TRL's
SFTTrainer. After training the LoRA adapter is merged into the base model
and saved so it can be loaded directly by vLLM for evaluation.

Usage:
  python src/train_instruct.py --model Qwen/Qwen2.5-7B-Instruct
  python src/train_instruct.py --model meta-llama/Llama-3.2-3B-Instruct --epochs 5
  python src/train_instruct.py --model microsoft/Phi-3.5-mini-instruct --lr 2e-4

Supported models:
  Qwen/Qwen2.5-7B-Instruct          ← recommended
  Qwen/Qwen2.5-3B-Instruct
  meta-llama/Llama-3.2-3B-Instruct
  meta-llama/Llama-3.2-1B-Instruct
  microsoft/Phi-3.5-mini-instruct
  mistralai/Mistral-7B-Instruct-v0.3
  google/gemma-2-2b-it
"""

import argparse
import json
import os
import time

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "data", "processed", "instruct")
OUTPUTS_DIR = os.path.join(ROOT, "outputs", "instruct")


# ── data ───────────────────────────────────────────────────────────────────────

def load_split(split: str) -> list[dict]:
    path = os.path.join(DATA_DIR, f"{split}.jsonl")
    with open(path) as f:
        return [json.loads(l) for l in f]


def apply_chat_template(examples: dict, tokenizer) -> dict:
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        for msgs in examples["messages"]
    ]
    return {"text": texts}


# ── LoRA config ────────────────────────────────────────────────────────────────

def make_lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules="all-linear",  # targets all linear layers automatically
        bias="none",
    )


# ── training ───────────────────────────────────────────────────────────────────

def train(args):
    model_slug = args.model.replace("/", "__")
    run_name   = f"{model_slug}_lr{args.lr}_r{args.lora_r}_ep{args.epochs}"
    output_dir = os.path.join(OUTPUTS_DIR, run_name)
    adapter_dir = os.path.join(output_dir, "adapter")
    merged_dir  = os.path.join(output_dir, "merged")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Model  : {args.model}")
    print(f"  LR     : {args.lr}")
    print(f"  LoRA r : {args.lora_r}  alpha : {args.lora_alpha}")
    print(f"  Batch  : {args.batch_size}  grad_accum : {args.grad_accum}")
    print(f"  Epochs : {args.epochs}")
    print(f"  Output : {output_dir}")
    print(f"{'='*70}\n")

    # ── tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── datasets ──────────────────────────────────────────────────────────────
    train_raw = Dataset.from_list(load_split("train"))
    val_raw   = Dataset.from_list(load_split("validation"))

    train_ds = train_raw.map(
        lambda ex: apply_chat_template(ex, tokenizer),
        batched=True, remove_columns=["messages"],
    )
    val_ds = val_raw.map(
        lambda ex: apply_chat_template(ex, tokenizer),
        batched=True, remove_columns=["messages"],
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,   # fp16 for V100 compatibility
        device_map="auto",
        trust_remote_code=True,
    )
    model.enable_input_require_grads()

    lora_cfg = make_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── training args ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=adapter_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        report_to="none",
        save_total_limit=2,
        seed=42,
        dataloader_num_workers=0,
    )

    # ── train ─────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=args.max_length,
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print(f"\nTraining finished in {train_time:.1f}s")

    # ── merge adapter into base model and save ────────────────────────────────
    print("\nMerging LoRA adapter into base model...")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged model saved to {merged_dir}")

    # ── save run metadata ─────────────────────────────────────────────────────
    meta = {
        "model":        args.model,
        "lr":           args.lr,
        "lora_r":       args.lora_r,
        "lora_alpha":   args.lora_alpha,
        "batch_size":   args.batch_size,
        "grad_accum":   args.grad_accum,
        "epochs":       args.epochs,
        "train_time_s": round(train_time, 1),
        "merged_dir":   merged_dir,
    }
    with open(os.path.join(output_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nRun metadata saved to {output_dir}/train_meta.json")
    print(f"Run evaluation with:")
    print(f"  python src/evaluate_instruct.py --merged_dir {merged_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA instruction-tune a small LLM")
    parser.add_argument("--model",       type=str,   default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--lora_r",      type=int,   default=16)
    parser.add_argument("--lora_alpha",  type=int,   default=32)
    parser.add_argument("--lora_dropout",type=float, default=0.05)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--grad_accum",  type=int,   default=4,  help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--max_length",  type=int,   default=512)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
