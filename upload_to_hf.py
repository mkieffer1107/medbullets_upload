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
args = parser.parse_args()

HF_USERNAME = args.hf_username
HF_REPO_NAME = args.hf_repo_name
PRIVATE = str_to_bool(args.private)
HF_REPO_ID = f"{HF_USERNAME}/{HF_REPO_NAME}"

DATA_DIR = "data"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# schema shared by all splits
FEATURES = Features({
    "link": Value("string"),
    "question": Value("string"),
    "options": {
        "A": Value("string"),
        "B": Value("string"),
        "C": Value("string"),
        "D": Value("string"),
        "E": Value("string"),  # always a string; empty "" for 4-option questions
    },
    "answer_idx": Value("string"),
    "answer": Value("string"),
    "explanation": Value("string"),
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

        # make E always a string ("" if missing), so features are consistent
        has_ope = "ope" in data and data.get("ope") is not None and idx in data.get("ope", {})
        e_val = data["ope"][idx] if has_ope else ""

        entry = {
            "link": _to_str(data["link"][idx]),
            "question": _to_str(data["question"][idx]),
            "options": {
                "A": _to_str(data["opa"][idx]),
                "B": _to_str(data["opb"][idx]),
                "C": _to_str(data["opc"][idx]),
                "D": _to_str(data["opd"][idx]),
                "E": _to_str(e_val),   # "" when there is no E option
            },
            "answer_idx": _to_str(data["answer_idx"][idx]),
            "answer": _to_str(data["answer"][idx]),
            "explanation": _to_str(data["explanation"][idx]),
        }
        formatted_data.append(entry)
    
    return formatted_data

def push_as_hf_dataset(repo_id: str, private: bool = False):
    """HF auth via `huggingface-cli login`"""
    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    # Build datasets with explicit FEATURES so all splits match
    ds_op4 = Dataset.from_list(medbullets_op4, features=FEATURES)
    ds_op5 = Dataset.from_list(medbullets_op5, features=FEATURES)

    dsd = DatasetDict({
        "op4": ds_op4,
        "op5": ds_op5,
    })

    dsd.push_to_hub(repo_id, private=private)
    print(f"Dataset pushed to https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    filenames = ["medbullets_op4.json", "medbullets_op5.json"]
    data = [format_dataset(os.path.join(DATA_DIR, filename)) for filename in filenames]

    medbullets_op4 = data[0]
    medbullets_op5 = data[1]

    # keep your original save logic; tweak remove_opt_e so it drops E if it's empty
    remove_opt_e = lambda d: [
        {
            **e,
            "options": {k: v for k, v in e["options"].items() if not (k == "E" and (v is None or v == ""))}
        }
        for e in d
    ]
    
    save_filename_4 = "Medbullets-4-options.json"
    save_filename_5 = "Medbullets-5-options.json"

    save_json(remove_opt_e(medbullets_op4), os.path.join(OUT_DIR, save_filename_4))
    print(f"Saved {len(medbullets_op4)} entries to {os.path.join(OUT_DIR, save_filename_4)}")
    save_json(medbullets_op5, os.path.join(OUT_DIR, save_filename_5))
    print(f"Saved {len(medbullets_op5)} entries to {os.path.join(OUT_DIR, save_filename_5)}")

    combined_data = [item for sublist in data for item in sublist]
    save_filename = "Medbullets-all.json"
    
    save_json(combined_data, os.path.join(OUT_DIR, save_filename))
    print(f"Saved {len(combined_data)} entries to {os.path.join(OUT_DIR, save_filename)}")

    print(f"\nPushing dataset to Hugging Face Hub as {HF_REPO_ID} (private={PRIVATE})...")
    push_as_hf_dataset(HF_REPO_ID, private=PRIVATE)
