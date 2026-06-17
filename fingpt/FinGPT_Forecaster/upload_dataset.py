"""
Upload a local HuggingFace DatasetDict to the Hub.

Usage:
  # Upload (private by default)
  python upload_dataset.py \
      --dataset_path chatgpt_cot \
      --repo_id YOUR_HF_USERNAME/fingpt-forecaster-dow30-cot

  # Make an already-uploaded private dataset repo public
  python upload_dataset.py --repo_id YOUR_HF_USERNAME/fingpt-forecaster-dow30-cot --make_public
"""

import argparse
import os
from pathlib import Path

from datasets import load_from_disk
from huggingface_hub import HfApi


DATASET_CARD_TEMPLATE = """\
---
license: apache-2.0
language:
- en
tags:
- finance
- stock-forecasting
- chain-of-thought
- dow30
pretty_name: FinGPT Forecaster DOW30 CoT
task_categories:
- text-generation
dataset_info:
  features:
  - name: symbol
    dtype: string
  - name: period
    dtype: string
  - name: prompt
    dtype: string
  - name: answer
    dtype: string
  - name: label
    dtype: string
  splits:
  - name: train
  - name: test
---

# FinGPT Forecaster DOW30 — Chain-of-Thought Dataset

Weekly stock price movement predictions for DOW-30 constituents, augmented with
**Chain-of-Thought (CoT) reasoning** generated via GPT-4.

Built on top of the base dataset:
[FinGPT/fingpt-forecaster-dow30-202305-202405](https://huggingface.co/datasets/FinGPT/fingpt-forecaster-dow30-202305-202405)

## Schema

| Field | Description |
|-------|-------------|
| `symbol` | DOW-30 ticker (e.g. `AXP`, `MSFT`) |
| `period` | Forecast week, e.g. `2023-05-14 to 2023-05-21` |
| `prompt` | Full instruction prompt fed to the model |
| `answer` | CoT reasoning + structured final answer (`assistantfinal` marker separates them) |
| `label` | Ground-truth direction: `up` / `down` / `stable` |

## Answer format

```
<CoT reasoning preamble>
assistantfinal
[Positive Developments]:
1. ...
[Potential Concerns]:
1. ...
[Prediction & Analysis]:
...
```

## Citation

```bibtex
@misc{fingpt2023,
  title={FinGPT: Open-Source Financial Large Language Models},
  author={Yang, Hongyang and Liu, Xiao-Yang and Wang, Christina Dan},
  year={2023},
  url={https://github.com/AI4Finance-Foundation/FinGPT}
}
```
"""

from typing import Optional

def upload_dataset(dataset_path: str, repo_id: str, private: bool, token: Optional[str] = None):
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    print(f"Loading dataset from {path} ...")
    ds = load_from_disk(str(path))
    print(ds)

    print(f"Pushing to hub: {repo_id}  (private={private}) ...")
    ds.push_to_hub(repo_id, private=private, token=token)

    # Upload the dataset card
    api = HfApi(token=token)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        card_path = Path(tmp) / "README.md"
        card_path.write_text(DATASET_CARD_TEMPLATE)
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )

    print(f"Done: https://huggingface.co/datasets/{repo_id}")


def make_public(repo_id: str, token: Optional[str] = None):
    api = HfApi(token=token)
    api.update_repo_visibility(repo_id=repo_id, private=False, repo_type="dataset")
    print(f"Dataset repo is now public: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="Local path to the dataset folder (saved with save_to_disk)")
    parser.add_argument("--repo_id", type=str, required=True, help="HF repo id, e.g. username/dataset-name")
    parser.add_argument("--public", action="store_true", help="Create / keep the repo public (default: private)")
    parser.add_argument("--make_public", action="store_true", help="Flip an existing private repo to public and exit")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"), help="HF write token (default: $HF_TOKEN)")
    args = parser.parse_args()

    if args.make_public:
        make_public(args.repo_id, args.token)
        return

    if not args.dataset_path:
        parser.error("--dataset_path is required unless --make_public is set")

    upload_dataset(
        dataset_path=args.dataset_path,
        repo_id=args.repo_id,
        private=not args.public,
        token=args.token,
    )


if __name__ == "__main__":
    main()
