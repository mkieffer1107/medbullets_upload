import json
from datasets import load_dataset, Dataset, DatasetDict

def _strip_E(split):
    for ex in split:
        ex = dict(ex)
        ex["options"] = {k: v for k, v in ex["options"].items() if k != "E"}
        yield ex

if __name__ == "__main__":
    # load all data
    # dataset = load_dataset("mkieffer/Medbullets")

    # load only op4 splits
    op4_train, op4_eval = load_dataset("mkieffer/Medbullets", split=["op4_train", "op4_eval"])

    # remove the "E" option from op4 splits
    op4_train = Dataset.from_generator(lambda: _strip_E(op4_train))
    op4_eval  = Dataset.from_generator(lambda: _strip_E(op4_eval))

    # load only op5 splits
    op5_train, op5_eval = load_dataset("mkieffer/Medbullets", split=["op5_train", "op5_eval"])

    # optionally, combine them into a single dataset
    dataset = DatasetDict({
        "op4_train": op4_train,
        "op4_eval":  op4_eval,
        "op5_train": op5_train,
        "op5_eval":  op5_eval,
    })

    print("\nop4_train:\n", json.dumps(dataset["op4_train"][0], indent=2))
    print("\nop4_eval:\n", json.dumps(dataset["op4_eval"][0], indent=2))
    print("\nop5_train:\n", json.dumps(dataset["op5_train"][0], indent=2))
    print("\nop5_eval:\n", json.dumps(dataset["op5_eval"][0], indent=2))