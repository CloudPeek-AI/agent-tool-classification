"""
Run all recommended encoder-only models sequentially and produce a
side-by-side comparison table saved to outputs/benchmark_results.json.

Usage:
  python src/benchmark_models.py                  # run all models
  python src/benchmark_models.py --models deberta modernbert   # run subset
  python src/benchmark_models.py --epochs 10 --batch_size 32

Model keys (pass to --models):
  bert        bert-base-uncased
  roberta     roberta-base
  electra     google/electra-base-discriminator
  modernbert  answerdotai/ModernBERT-base
  deberta     microsoft/deberta-v3-base           ← recommended
"""

import argparse
import json
import os
import subprocess
import sys

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
TRAIN_SCRIPT = os.path.join(ROOT, "src", "train_classifier.py")

MODEL_REGISTRY = {
    "bert":       "bert-base-uncased",
    "roberta":    "roberta-base",
    "electra":    "google/electra-base-discriminator",
    "modernbert": "answerdotai/ModernBERT-base",
    "deberta":    "microsoft/deberta-v3-base",
}

# Good defaults per model based on literature
MODEL_DEFAULTS = {
    "bert-base-uncased":                  {"lr": "2e-5", "batch_size": "16"},
    "roberta-base":                       {"lr": "2e-5", "batch_size": "16"},
    "google/electra-base-discriminator":  {"lr": "3e-5", "batch_size": "32"},
    "answerdotai/ModernBERT-base":        {"lr": "5e-5", "batch_size": "32"},
    "microsoft/deberta-v3-base":          {"lr": "1e-5", "batch_size": "16"},
}


def run_model(model_id: str, epochs: int, batch_size: str | None, lr: str | None) -> dict:
    defaults = MODEL_DEFAULTS.get(model_id, {"lr": "2e-5", "batch_size": "16"})
    lr_val    = lr         or defaults["lr"]
    bs_val    = batch_size or defaults["batch_size"]

    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--model",      model_id,
        "--epochs",     str(epochs),
        "--lr",         lr_val,
        "--batch_size", bs_val,
    ]
    print(f"\n{'#'*70}")
    print(f"  Running: {model_id}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'#'*70}\n")

    result = subprocess.run(cmd, check=True)

    # Load results written by train_classifier.py
    model_slug = model_id.replace("/", "__")
    results_path = os.path.join(
        OUTPUTS_DIR,
        f"{model_slug}_lr{lr_val}_bs{bs_val}_ep{epochs}",
        "test_results.json",
    )
    with open(results_path) as f:
        return json.load(f)


def print_table(all_results: list[dict]) -> None:
    print(f"\n{'='*90}")
    print(f"{'MODEL':<42} {'ACC':>6} {'MACRO F1':>9} {'WEIGHTED F1':>12} {'TIME(s)':>8}")
    print(f"{'-'*90}")
    for r in sorted(all_results, key=lambda x: -x["test_macro_f1"]):
        name = r["model"].split("/")[-1]
        print(
            f"{name:<42} "
            f"{r['test_accuracy']:>6.4f} "
            f"{r['test_macro_f1']:>9.4f} "
            f"{r['test_weighted_f1']:>12.4f} "
            f"{r['train_time_s']:>8.1f}s"
        )
    print(f"{'='*90}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark all encoder-only classifiers")
    parser.add_argument(
        "--models", nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        default=list(MODEL_REGISTRY.keys()),
        help="Which models to run (default: all)",
    )
    parser.add_argument("--epochs",     type=int,   default=15,  help="Max epochs per model")
    parser.add_argument("--batch_size", type=str,   default=None, help="Override batch size for all models")
    parser.add_argument("--lr",         type=str,   default=None, help="Override LR for all models")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    all_results = []
    failed = []

    for key in args.models:
        model_id = MODEL_REGISTRY[key]
        try:
            result = run_model(model_id, args.epochs, args.batch_size, args.lr)
            all_results.append(result)
        except Exception as e:
            print(f"\n[ERROR] {model_id} failed: {e}")
            failed.append(model_id)

    if all_results:
        print_table(all_results)

        summary_path = os.path.join(OUTPUTS_DIR, "benchmark_results.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Full results saved to {summary_path}")

    if failed:
        print(f"\nFailed models: {failed}")


if __name__ == "__main__":
    main()
