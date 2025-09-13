#!/usr/bin/env python3
import os
import json
import argparse

from datasets import Dataset, DatasetDict, Features, Value
from huggingface_hub import create_repo


def str_to_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}

parser = argparse.ArgumentParser(description="Format Medbullets data and push to Hugging Face Hub")
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

# schema shared by all splits
FEATURES = Features({
    "question": Value("string"),
    "options": {
        "A": Value("string"),
        "B": Value("string"),
        "C": Value("string"),
        "D": Value("string"),
        "E": Value("string"),  # always a string; empty "" for 4-option questions
    },
    "answer": Value("string"),
    "explanation": Value("string"),
    "link": Value("string"),
})

def load_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def save_json(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

def _to_str(x):
    return "" if x is None else str(x)

def format_dataset(filepath: str):
    data = load_json(filepath)
    formatted_data = []
    num_entries = len(data["link"])
    print(f"Formatting {num_entries} entries from {filepath}")
    for i in range(num_entries):
        idx = str(i)
        has_ope = "ope" in data and data.get("ope") is not None and idx in data.get("ope", {})
        e_val = data["ope"][idx] if has_ope else ""
        entry = {
            "question": _to_str(data["question"][idx]),
            "options": {
                "A": _to_str(data["opa"][idx]),
                "B": _to_str(data["opb"][idx]),
                "C": _to_str(data["opc"][idx]),
                "D": _to_str(data["opd"][idx]),
                "E": _to_str(e_val),
            },
            "answer": _to_str(data["answer_idx"][idx]),  # keep the letter as the answer
            "explanation": _to_str(data["explanation"][idx]),
            "link": _to_str(data["link"][idx]),
        }
        formatted_data.append(entry)
    return formatted_data

def split_dataset(examples, eval_frac: float):
    """Use the last eval_frac of the dataset for eval; the rest is train."""
    n = len(examples)
    n_eval = max(0, min(n, round(eval_frac * n)))
    if n_eval == 0:
        return examples, []
    return examples[:-n_eval], examples[-n_eval:]

if __name__ == "__main__":
    filenames = ["medbullets_op4.json", "medbullets_op5.json"]
    data = [format_dataset(os.path.join(DATA_DIR, filename)) for filename in filenames]

    medbullets_op4 = data[0]
    medbullets_op5 = data[1]

    # drop E if it's empty, for the saved "op4" file only
    # remove_opt_e = lambda d: [
    #     {**e, "options": {k: v for k, v in e["options"].items()
    #                       if not (k == "E" and (v is None or v == ""))}}
    #     for e in d
    # ]

    # save_json(remove_opt_e(medbullets_op4), os.path.join(OUT_DIR, "Medbullets-4-options.json"))
    # print(f"Saved {len(medbullets_op4)} entries to {os.path.join(OUT_DIR, 'Medbullets-4-options.json')}")
    # save_json(medbullets_op5, os.path.join(OUT_DIR, "Medbullets-5-options.json"))
    # print(f"Saved {len(medbullets_op5)} entries to {os.path.join(OUT_DIR, 'Medbullets-5-options.json')}")

    # combined_data = [item for sublist in data for item in sublist]
    # save_json(combined_data, os.path.join(OUT_DIR, "Medbullets-all.json"))
    # print(f"Saved {len(combined_data)} entries to {os.path.join(OUT_DIR, 'Medbullets-all.json')}")

    # make per-file splits using the simple tail rule
    op4_train, op4_eval = split_dataset(medbullets_op4, EVAL_FRAC)
    op5_train, op5_eval = split_dataset(medbullets_op5, EVAL_FRAC)
    print(f"op4 -> train: {len(op4_train)} | eval: {len(op4_eval)}")
    print(f"op5 -> train: {len(op5_train)} | eval: {len(op5_eval)}")

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
