#!/usr/bin/env python3
import os
import json
import argparse
from typing import Dict, List

from datasets import Dataset, DatasetDict, Features, Value
from huggingface_hub import create_repo


def str_to_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}

parser = argparse.ArgumentParser(description="Format Medbullets data and push to HF with aligned splits")
parser.add_argument("--hf_username", default=os.environ.get("HF_USERNAME", "mkieffer"))
parser.add_argument("--hf_repo_name", default=os.environ.get("HF_REPO_NAME", "Medbullets"))
parser.add_argument("--private", default=os.environ.get("PRIVATE", "false"),
                    help="Whether the HF dataset repo should be private (true/false)")
parser.add_argument("--eval_frac", type=float, default=float(os.environ.get("EVAL_FRAC", 0.20)),
                    help="Fraction of each split to put in eval (default 0.20)")
args = parser.parse_args()

HF_USERNAME = args.hf_username
HF_REPO_NAME = args.hf_repo_name
PRIVATE = str_to_bool(args.private)
EVAL_FRAC = float(args.eval_frac)
HF_REPO_ID = f"{HF_USERNAME}/{HF_REPO_NAME}"

DATA_DIR = "data"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# schema shared by all splits (includes idx)
FEATURES = Features({
    "idx": Value("string"),
    "question": Value("string"),
    "options": {
        "A": Value("string"),
        "B": Value("string"),
        "C": Value("string"),
        "D": Value("string"),
        "E": Value("string"),  # empty "" for 4-option questions
    },
    "answer": Value("string"),
    "explanation": Value("string"),
    "link": Value("string"),
})

def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

def _to_str(x):
    return "" if x is None else str(x)

def build_idx_map(op4_raw: dict, op5_raw: dict) -> Dict[str, str]:
    """
    Deterministic mapping from link -> idx shared by OP4 and OP5.
    idx is a 3-digit zero-padded string so lexicographic sort == numeric order.
    """
    def to_set(d):
        if isinstance(d.get("link"), dict):
            return set(d["link"].values())
        return set(d.get("link", []))

    all_links = sorted(to_set(op4_raw).union(to_set(op5_raw)))
    return {lnk: f"{i+1:03d}" for i, lnk in enumerate(all_links)}  # <-- 3 digits

def format_dataset(filepath: str, idx_map: Dict[str, str]):
    data = load_json(filepath)
    if isinstance(data["link"], dict):
        index_keys = sorted(data["link"].keys(), key=lambda s: int(s))
        get = lambda col, i: data[col].get(i, "")
    else:
        index_keys = [str(i) for i in range(len(data["link"]))]
        get = lambda col, i: data[col][int(i)]

    formatted = []
    print(f"Formatting {len(index_keys)} entries from {filepath}")
    has_ope_array = "ope" in data

    for k in index_keys:
        link_val = _to_str(get("link", k))
        e_val = _to_str(get("ope", k)) if has_ope_array else ""
        formatted.append({
            "idx": idx_map.get(link_val, f"UNALIGNED_{k}"),
            "question": _to_str(get("question", k)),
            "options": {
                "A": _to_str(get("opa", k)),
                "B": _to_str(get("opb", k)),
                "C": _to_str(get("opc", k)),
                "D": _to_str(get("opd", k)),
                "E": e_val,
            },
            "answer": _to_str(get("answer_idx", k)),  # keep the letter as the answer
            "explanation": _to_str(get("explanation", k)),
            "link": link_val,
        })

    # Sort entire dataset ascending by idx (then link as tiebreaker)
    formatted.sort(key=lambda e: (e["idx"], e["link"]))
    return formatted

def split_aligned(op4_examples: List[dict], op5_examples: List[dict], eval_frac: float):
    """
    Create aligned train/eval splits by idx (shared across OP4/OP5).
    """
    idx4 = {e["idx"] for e in op4_examples}
    idx5 = {e["idx"] for e in op5_examples}
    common = sorted(idx4.intersection(idx5))  # ascending by idx

    n = len(common)
    n_eval = max(0, min(n, round(eval_frac * n)))
    eval_ids = set(common[-n_eval:]) if n_eval > 0 else set()
    train_ids = set(common[:-n_eval]) if n_eval > 0 else set(common)

    op4_train = [e for e in op4_examples if e["idx"] in train_ids]
    op4_eval  = [e for e in op4_examples if e["idx"] in eval_ids]
    op5_train = [e for e in op5_examples if e["idx"] in train_ids]
    op5_eval  = [e for e in op5_examples if e["idx"] in eval_ids]

    key = lambda e: (e["idx"], e["link"])
    op4_train.sort(key=key); op4_eval.sort(key=key)
    op5_train.sort(key=key); op5_eval.sort(key=key)

    return op4_train, op4_eval, op5_train, op5_eval

if __name__ == "__main__":
    # Build shared idx from union of links
    op4_raw = load_json(os.path.join(DATA_DIR, "medbullets_op4.json"))
    op5_raw = load_json(os.path.join(DATA_DIR, "medbullets_op5.json"))
    idx_map = build_idx_map(op4_raw, op5_raw)

    # Format both datasets with shared idx
    medbullets_op4 = format_dataset(os.path.join(DATA_DIR, "medbullets_op4.json"), idx_map)
    medbullets_op5 = format_dataset(os.path.join(DATA_DIR, "medbullets_op5.json"), idx_map)

    # Aligned splits (by idx), sorted ascending by idx
    op4_train, op4_eval, op5_train, op5_eval = split_aligned(medbullets_op4, medbullets_op5, EVAL_FRAC)
    print(f"Aligned splits:")
    print(f"  op4 -> train: {len(op4_train)} | eval: {len(op4_eval)}")
    print(f"  op5 -> train: {len(op5_train)} | eval: {len(op5_eval)}")

    # Optional local dump
    os.makedirs(OUT_DIR, exist_ok=True)
    save_json(medbullets_op4, os.path.join(OUT_DIR, "Medbullets-4.json"))
    save_json(medbullets_op5, os.path.join(OUT_DIR, "Medbullets-5.json"))

    print(f"\nPushing dataset to Hugging Face Hub as {HF_REPO_ID} (private={PRIVATE})...")
    create_repo(HF_REPO_ID, repo_type="dataset", private=PRIVATE, exist_ok=True)

    dsd = DatasetDict({
        "op4_train": Dataset.from_list(op4_train, features=FEATURES),
        "op4_eval":  Dataset.from_list(op4_eval,  features=FEATURES),
        "op5_train": Dataset.from_list(op5_train, features=FEATURES),
        "op5_eval":  Dataset.from_list(op5_eval,  features=FEATURES),
    })

    dsd.push_to_hub(HF_REPO_ID, private=PRIVATE)
    print(f"Dataset pushed to https://huggingface.co/datasets/{HF_REPO_ID}")
