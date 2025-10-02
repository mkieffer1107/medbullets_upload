import json
from datasets import load_dataset, Dataset

def _strip_E(split):
    for ex in split:
        ex = dict(ex)
        ex["options"] = {k: v for k, v in ex["options"].items() if k != "E"}
        yield ex

if __name__ == "__main__":
    op4_test, op5_test = load_dataset("mkieffer/Medbullets", split=["op4_test", "op5_test"])

    # remove the "E" option from op4 split
    op4_test = Dataset.from_generator(lambda: _strip_E(op4_test))

    print("\nop4_test:\n", json.dumps(op4_test[0], indent=2))
    print("\nop5_test:\n", json.dumps(op5_test[0], indent=2))