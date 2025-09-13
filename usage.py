import json
from datasets import load_dataset, Dataset, DatasetDict

def _strip_E(split):
    for ex in split:
        ex = dict(ex)
        ex["options"] = {k: v for k, v in ex["options"].items() if k != "E"}
        yield ex

if __name__ == "__main__":
    dataset = load_dataset("mkieffer/Medbullets")

    # rebuild just the op4 splits to remove the "E" option
    op4_train_noE = Dataset.from_generator(lambda: _strip_E(dataset["op4_train"]))
    op4_eval_noE  = Dataset.from_generator(lambda: _strip_E(dataset["op4_eval"]))

    dataset = DatasetDict({
        "op4_train": op4_train_noE,
        "op4_eval":  op4_eval_noE,
        "op5_train": dataset["op5_train"],
        "op5_eval":  dataset["op5_eval"],
    })

    print(json.dumps(dataset["op4_train"][0], indent=2))
    print(json.dumps(dataset["op4_eval"][0], indent=2))
    print(json.dumps(dataset["op5_train"][0], indent=2))
    print(json.dumps(dataset["op5_eval"][0], indent=2))