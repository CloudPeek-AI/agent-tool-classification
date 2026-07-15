"""
Encoder-only classifier training for security tool-call classification.

Supported models (pass via --model):
  bert-base-uncased                   BERT base — baseline
  roberta-base                        RoBERTa — stronger BERT variant
  microsoft/deberta-v3-base           DeBERTa-v3 — best general classifier (recommended)
  answerdotai/ModernBERT-base         ModernBERT — modern efficient architecture
  google/electra-base-discriminator   ELECTRA — very parameter-efficient

Usage:
  python src/train_classifier.py --model microsoft/deberta-v3-base
  python src/train_classifier.py --model answerdotai/ModernBERT-base --epochs 10
  python src/train_classifier.py --model bert-base-uncased --lr 3e-5 --batch_size 16
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
RESULTS_DIR = os.path.join(ROOT, "results")

# ── data loading ───────────────────────────────────────────────────────────────

def load_split(split: str, data_dir: str) -> Dataset:
    path = os.path.join(data_dir, f"{split}.jsonl")
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            records.append({"text": r["text"], "label": int(r["label"])})
    return Dataset.from_list(records)


def load_datasets(data_dir: str) -> DatasetDict:
    return DatasetDict({
        "train":      load_split("train",      data_dir),
        "validation": load_split("validation", data_dir),
        "test":       load_split("test",       data_dir),
    })


# ── tokenisation ───────────────────────────────────────────────────────────────

def tokenize(batch, tokenizer, max_length):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_length,
    )


# ── metrics ────────────────────────────────────────────────────────────────────

def make_compute_metrics(id2label):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        macro_f1  = f1_score(labels, preds, average="macro",    zero_division=0)
        weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        accuracy  = float(np.mean(preds == labels))
        return {
            "accuracy":    round(accuracy,    4),
            "macro_f1":    round(macro_f1,    4),
            "weighted_f1": round(weighted_f1, 4),
        }
    return compute_metrics


# ── training ───────────────────────────────────────────────────────────────────

def train(args):
    # ── resolve data dir and label map ────────────────────────────────────────
    data_dir  = os.path.abspath(args.data_dir)
    label_map = os.path.join(os.path.dirname(data_dir), "label_map.json")
    with open(label_map) as f:
        _lmap = json.load(f)
    label2id   = _lmap["label2id"]
    id2label   = {int(k): v for k, v in _lmap["id2label"].items()}
    num_labels = len(label2id)

    model_slug = args.model.replace("/", "__")
    run_name   = f"{model_slug}_lr{args.lr}_bs{args.batch_size}_ep{args.epochs}"
    output_dir = os.path.join(OUTPUTS_DIR, run_name)
    result_dir = os.path.join(RESULTS_DIR, model_slug)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Model    : {args.model}")
    print(f"  Data dir : {data_dir}")
    print(f"  LR       : {args.lr}")
    print(f"  Batch    : {args.batch_size}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Output   : {output_dir}")
    print(f"  Results  : {result_dir}")
    print(f"{'='*70}\n")

    # ── tokenizer & model ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    max_length = min(args.max_length, tokenizer.model_max_length)

    raw = load_datasets(data_dir)
    tokenized = raw.map(
        lambda b: tokenize(b, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
    )
    tokenized.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── training args ──────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=50,
        lr_scheduler_type="cosine",

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        logging_steps=20,
        report_to="none",
        save_total_limit=2,
        seed=42,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        dataloader_num_workers=0,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(id2label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    # ── train ──────────────────────────────────────────────────────────────────
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print(f"\nTraining finished in {train_time:.1f}s")

    # ── evaluate on test set ───────────────────────────────────────────────────
    print("\n--- Test set evaluation ---")
    test_preds_out = trainer.predict(tokenized["test"])
    preds  = np.argmax(test_preds_out.predictions, axis=-1)
    labels = test_preds_out.label_ids

    label_names = [id2label[i] for i in range(num_labels)]
    report = classification_report(labels, preds, target_names=label_names, digits=4)
    cm     = confusion_matrix(labels, preds)

    print(report)
    print("Confusion matrix (rows=true, cols=pred):")
    print("Labels:", label_names)
    print(cm)

    # ── save results ───────────────────────────────────────────────────────────
    results = {
        "model":          args.model,
        "lr":             args.lr,
        "batch_size":     args.batch_size,
        "epochs":         args.epochs,
        "train_time_s":   round(train_time, 1),
        "test_accuracy":  round(float(np.mean(preds == labels)), 4),
        "test_macro_f1":  round(f1_score(labels, preds, average="macro",    zero_division=0), 4),
        "test_weighted_f1": round(f1_score(labels, preds, average="weighted", zero_division=0), 4),
        "per_class_f1":   {
            name: round(f1_score(labels == i, preds == i, average="binary", zero_division=0), 4)
            for i, name in id2label.items()
        },
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    results_path = os.path.join(result_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ── save best model & tokenizer ────────────────────────────────────────────
    best_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"Best model saved to {best_dir}")

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train an encoder-only classifier")

    parser.add_argument(
        "--data_dir", type=str,
        default=os.path.join(ROOT, "data", "processed_enhanced", "classifier"),
        help="Path to the classifier dataset directory (must contain train/validation/test.jsonl and ../label_map.json)",
    )
    parser.add_argument(
        "--model", type=str, default="microsoft/deberta-v3-base",
        help=(
            "HuggingFace model ID. Recommended options:\n"
            "  microsoft/deberta-v3-base          (best accuracy, recommended)\n"
            "  answerdotai/ModernBERT-base         (modern, efficient)\n"
            "  roberta-base                        (solid baseline)\n"
            "  bert-base-uncased                   (classic baseline)\n"
            "  google/electra-base-discriminator   (parameter-efficient)"
        ),
    )
    parser.add_argument("--lr",           type=float, default=2e-5,  help="Peak learning rate")
    parser.add_argument("--batch_size",   type=int,   default=16,    help="Per-device train batch size")
    parser.add_argument("--epochs",       type=int,   default=15,    help="Max training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01,  help="AdamW weight decay")
    parser.add_argument("--max_length",   type=int,   default=256,   help="Max token length (texts avg ~80 tokens)")
    parser.add_argument("--patience",     type=int,   default=3,     help="Early stopping patience (epochs)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
