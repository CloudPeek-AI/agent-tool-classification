"""
Evaluate a fine-tuned (merged) LLM on the instruct test set using vLLM.

Loads the merged model, runs inference on the test split, parses the
predicted label from the output, and reports accuracy + F1 per class.

Usage:
  python src/evaluate_instruct.py --merged_dir outputs/instruct/<run>/merged

  # Or point to any merged HF model directory:
  python src/evaluate_instruct.py --merged_dir /path/to/merged_model
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from vllm import LLM, SamplingParams

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "data", "processed_enhanced", "instruct")
LABEL_MAP_PATH = os.path.join(ROOT, "data", "processed_enhanced", "label_map.json")
RESULTS_DIR = os.path.join(ROOT, "results")

with open(LABEL_MAP_PATH) as f:
    _lmap = json.load(f)
LABEL2ID   = _lmap["label2id"]
ID2LABEL   = {int(k): v for k, v in _lmap["id2label"].items()}
VALID_LABELS = set(LABEL2ID.keys())


# ── data ───────────────────────────────────────────────────────────────────────

def load_test() -> list[dict]:
    path = os.path.join(DATA_DIR, "test.jsonl")
    with open(path) as f:
        return [json.loads(l) for l in f]


def build_prompt(messages: list[dict], tokenizer) -> str:
    """Build the prompt without the assistant turn (add_generation_prompt=True)."""
    prompt_msgs = [m for m in messages if m["role"] != "assistant"]
    return tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True,
    )


# ── label parsing ──────────────────────────────────────────────────────────────

def parse_label(output: str) -> str:
    """
    Extract the predicted label from the model's raw output.
    Tries exact match first, then checks if any valid label appears in the text.
    Falls back to UNKNOWN if nothing matches.
    """
    text = output.strip().upper()
    # Exact match on first word/line
    first = text.split()[0] if text.split() else ""
    if first in VALID_LABELS:
        return first
    # Scan for any valid label in the output
    for label in VALID_LABELS:
        if label in text:
            return label
    return "UNKNOWN"


# ── evaluation ─────────────────────────────────────────────────────────────────

def evaluate(args):
    print(f"\n{'='*70}")
    print(f"  Model  : {args.merged_dir}")
    print(f"  Batch  : {args.batch_size}")
    print(f"{'='*70}\n")

    # ── load tokenizer (for chat template) ────────────────────────────────────
    from transformers import AutoTokenizer
    merged_dir = os.path.abspath(args.merged_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        merged_dir, trust_remote_code=True, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── load test data ─────────────────────────────────────────────────────────
    test_data  = load_test()
    prompts    = [build_prompt(r["messages"], tokenizer) for r in test_data]
    gold_labels = [r["messages"][2]["content"].strip() for r in test_data]  # assistant turn

    # ── vLLM inference ─────────────────────────────────────────────────────────
    print(f"Loading model with vLLM ({len(prompts)} test examples)...")
    llm = LLM(
        model=merged_dir,
        dtype="float16",          # fp16 for V100 compatibility
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,    # greedy — deterministic classification
        max_tokens=10,      # label names are short (longest is 24 chars)
        stop=["\n", "<|eot_id|>", "<|im_end|>", "</s>"],
    )

    print("Running inference...")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    raw_outputs = [o.outputs[0].text for o in outputs]

    # ── parse predictions ──────────────────────────────────────────────────────
    pred_labels = [parse_label(o) for o in raw_outputs]

    # ── metrics ────────────────────────────────────────────────────────────────
    label_names = [ID2LABEL[i] for i in range(len(ID2LABEL))]
    unknown_count = sum(1 for p in pred_labels if p == "UNKNOWN")

    # Replace UNKNOWN with a dummy class for metric computation
    gold_ids = [LABEL2ID.get(g, -1) for g in gold_labels]
    pred_ids = [LABEL2ID.get(p, -1) for p in pred_labels]

    valid_mask = [g != -1 and p != -1 for g, p in zip(gold_ids, pred_ids)]
    g_filtered = [g for g, v in zip(gold_ids, valid_mask) if v]
    p_filtered = [p for p, v in zip(pred_ids, valid_mask) if v]

    accuracy    = float(np.mean([g == p for g, p in zip(gold_ids, pred_ids)]))
    macro_f1    = f1_score(g_filtered, p_filtered, average="macro",    zero_division=0)
    weighted_f1 = f1_score(g_filtered, p_filtered, average="weighted", zero_division=0)

    report = classification_report(
        g_filtered, p_filtered,
        target_names=label_names,
        labels=list(range(len(label_names))),
        digits=4,
    )
    cm = confusion_matrix(g_filtered, p_filtered, labels=list(range(len(label_names))))

    print(f"\nUnknown / unparseable outputs: {unknown_count}/{len(pred_labels)}")
    print(report)
    print("Confusion matrix (rows=true, cols=pred):")
    print("Labels:", label_names)
    print(cm)

    # ── save results ───────────────────────────────────────────────────────────
    results = {
        "merged_dir":     args.merged_dir,
        "test_accuracy":  round(accuracy, 4),
        "test_macro_f1":  round(macro_f1, 4),
        "test_weighted_f1": round(weighted_f1, 4),
        "unknown_outputs": unknown_count,
        "per_class_f1": {
            name: round(f1_score(
                [g == i for g in g_filtered],
                [p == i for p in p_filtered],
                average="binary", zero_division=0,
            ), 4)
            for i, name in ID2LABEL.items()
        },
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "raw_samples": [
            {"gold": g, "pred": p, "raw_output": r}
            for g, p, r in zip(gold_labels[:20], pred_labels[:20], raw_outputs[:20])
        ],
    }

    model_slug = os.path.basename(os.path.dirname(os.path.normpath(args.merged_dir)))
    result_dir = os.path.join(RESULTS_DIR, model_slug)
    os.makedirs(result_dir, exist_ok=True)
    out_path = os.path.join(result_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate merged LLM with vLLM")
    parser.add_argument("--merged_dir",    type=str, required=True,
                        help="Path to the merged model directory")
    parser.add_argument("--batch_size",    type=int, default=32,
                        help="vLLM inference batch size")
    parser.add_argument("--max_model_len", type=int, default=512,
                        help="Max sequence length for vLLM")
    parser.add_argument("--gpu_mem_util",  type=float, default=0.85,
                        help="vLLM GPU memory utilisation fraction")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
