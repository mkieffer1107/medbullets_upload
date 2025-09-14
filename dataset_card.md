---
dataset_info:
  features:
  - name: idx
    dtype: string
  - name: question
    dtype: string
  - name: options
    struct:
    - name: A
      dtype: string
    - name: B
      dtype: string
    - name: C
      dtype: string
    - name: D
      dtype: string
    - name: E
      dtype: string
  - name: answer
    dtype: string
  - name: explanation
    dtype: string
  - name: link
    dtype: string
  splits:
  - name: op4_train
    num_bytes: 947036
    num_examples: 246
  - name: op4_eval
    num_bytes: 263957
    num_examples: 62
  - name: op5_train
    num_bytes: 952919
    num_examples: 246
  - name: op5_eval
    num_bytes: 265144
    num_examples: 62
  download_size: 1262330
  dataset_size: 2429056
configs:
- config_name: default
  data_files:
  - split: op4_train
    path: data/op4_train-*
  - split: op4_eval
    path: data/op4_eval-*
  - split: op5_train
    path: data/op5_train-*
  - split: op5_eval
    path: data/op5_eval-*
task_categories:
- question-answering
tags:
- medical,
- clinical,
- multiple-choice
- usmle
---

# Medbullets

HuggingFace upload of a multiple-choice QA dataset of USMLE Step 2 and Step 3 style questions sourced from [Medbullets](https://step2.medbullets.com/). If used, please cite the original authors using the citation below.

## Dataset Details

### Dataset Description

The dataset contains four splits:
  - **op4_train**: four-option multiple-choice QA (choices A-D)
  - **op4_eval**: four-option multiple-choice QA (choices A-D)
  - **op5_train**: five-option multiple-choice QA (choices A-E)
  - **op5_eval**: five-option multiple-choice QA (choices A-E)

`op5` splits contain the same content as `op4` splits, but with one additional answer choice to increase difficulty. Note that while the content is the same, the letter choice corresponding to the correct answer is sometimes different between these splits.

The train/eval splits for a given number of options are 80/20, where the last 20% of the original, complete dataset is used for evals.

### Dataset Sources

- **Repository:** https://github.com/HanjieChen/ChallengeClinicalQA
- **Paper:** https://arxiv.org/pdf/2402.18060v3

### Direct Use

```python
import json
from datasets import load_dataset, Dataset

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

    print("\nop4_train:\n", json.dumps(op4_train[0], indent=2))
    print("\nop4_eval:\n", json.dumps(op4_eval[0], indent=2))
    print("\nop5_train:\n", json.dumps(op5_train[0], indent=2))
    print("\nop5_eval:\n", json.dumps(op5_eval[0], indent=2))
```



## Citation 

```
@inproceedings{chen-etal-2025-benchmarking,
    title = "Benchmarking Large Language Models on Answering and Explaining Challenging Medical Questions",
    author = "Chen, Hanjie  and
      Fang, Zhouxiang  and
      Singla, Yash  and
      Dredze, Mark",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.182/",
    doi = "10.18653/v1/2025.naacl-long.182",
    pages = "3563--3599",
    ISBN = "979-8-89176-189-6",
    abstract = "LLMs have demonstrated impressive performance in answering medical questions, such as achieving passing scores on medical licensing examinations. However, medical board exams or general clinical questions do not capture the complexity of realistic clinical cases. Moreover, the lack of reference explanations means we cannot easily evaluate the reasoning of model decisions, a crucial component of supporting doctors in making complex medical decisions. To address these challenges, we construct two new datasets: JAMA Clinical Challenge and Medbullets. JAMA Clinical Challenge consists of questions based on challenging clinical cases, while Medbullets comprises simulated clinical questions. Both datasets are structured as multiple-choice question-answering tasks, accompanied by expert-written explanations. We evaluate seven LLMs on the two datasets using various prompts. Experiments demonstrate that our datasets are harder than previous benchmarks. In-depth automatic and human evaluations of model-generated explanations provide insights into the promise and deficiency of LLMs for explainable medical QA."
}
```