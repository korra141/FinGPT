"""
Upload a LoRA adapter checkpoint to the Hugging Face Hub.

Usage:
  # Upload (private by default, skips intermediate checkpoint-* dirs)
  python upload_model.py \
      --model_path finetuned_models/dow30-llama3-cot-4bit-78osjvnb_202606092117 \
      --repo_id YOUR_HF_USERNAME/fingpt-forecaster-llama3-cot-lora

  # Make an already-uploaded private repo public
  python upload_model.py --repo_id YOUR_HF_USERNAME/fingpt-forecaster-llama3-cot-lora --make_public
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo


MODEL_CARD_TEMPLATE = """\
---
base_model: {base_model}
library_name: peft
datasets:
- FinGPT/fingpt-forecaster-dow30-202305-202405
tags:
- finance
- stock-forecasting
- lora
- peft
- llm
language:
- en
---

# {repo_name}

LoRA adapter fine-tuned on the [FinGPT DOW30 forecaster dataset](https://huggingface.co/datasets/FinGPT/fingpt-forecaster-dow30-202305-202405)
for weekly stock price movement prediction on DOW-30 constituents.

The model produces structured outputs with:
- **Positive Developments** (2–4 bullet points)
- **Potential Concerns** (2–4 bullet points)
- **Prediction & Analysis** with a directional label (`up` / `down` / `stable`)

## Base model

`{base_model}`

## Training data

[FinGPT/fingpt-forecaster-dow30-202305-202405](https://huggingface.co/datasets/FinGPT/fingpt-forecaster-dow30-202305-202405)
— DOW-30 weekly forecasting dataset covering May 2023 – April 2025.

## Usage

```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("{base_model}", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base, "{repo_id}")
```

## Citation

```bibtex
@misc{{fingpt2023,
  title={{FinGPT: Open-Source Financial Large Language Models}},
  author={{Yang, Hongyang and Liu, Xiao-Yang and Wang, Christina Dan}},
  year={{2023}},
  url={{https://github.com/AI4Finance-Foundation/FinGPT}}
}}
```
"""

from typing import Optional
def get_base_model(model_path: Path) -> str:
    cfg = model_path / "adapter_config.json"
    if cfg.exists():
        with open(cfg) as f:
            return json.load(f).get("base_model_name_or_path", "unknown")
    return "unknown"


def upload_model(model_path: str, repo_id: str, private: bool, token: Optional[str] = None):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path not found: {path}")

    api = HfApi(token=token)

    create_repo(repo_id, repo_type="model", private=private, exist_ok=True, token=token)
    print(f"Repo ready: https://huggingface.co/{repo_id}  (private={private})")

    base_model = get_base_model(path)
    repo_name = repo_id.split("/")[-1]

    # Write a filled-in model card into a temp file then upload it alongside
    # the adapter weights. We do NOT overwrite the path on disk.
    card_text = MODEL_CARD_TEMPLATE.format(
        base_model=base_model,
        repo_name=repo_name,
        repo_id=repo_id,
    )

    with tempfile.TemporaryDirectory() as tmp:
        card_path = Path(tmp) / "README.md"
        card_path.write_text(card_text)

        # Upload the generated card first
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )

    # Upload everything in model_path except intermediate checkpoint dirs
    print(f"Uploading adapter weights from {path} ...")
    api.upload_folder(
        folder_path=str(path),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=[
            "checkpoint-*",   # intermediate training saves
            "global_step*",
            "rng_state_*.pth",
            "latest",
            "*.sif",
            "*.img",
            "__pycache__",
        ],
    )
    print(f"Done: https://huggingface.co/{repo_id}")


def make_public(repo_id: str, token: Optional[str] = None):
    api = HfApi(token=token)
    api.update_repo_visibility(repo_id=repo_id, private=False, repo_type="model")
    print(f"Repo is now public: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Local path to the finetuned model folder")
    parser.add_argument("--repo_id", type=str, required=True, help="HF repo id, e.g. username/model-name")
    parser.add_argument("--public", action="store_true", help="Create / keep the repo public (default: private)")
    parser.add_argument("--make_public", action="store_true", help="Flip an existing private repo to public and exit")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"), help="HF write token (default: $HF_TOKEN)")
    args = parser.parse_args()

    if args.make_public:
        make_public(args.repo_id, args.token)
        return

    if not args.model_path:
        parser.error("--model_path is required unless --make_public is set")

    upload_model(
        model_path=args.model_path,
        repo_id=args.repo_id,
        private=not args.public,
        token=args.token,
    )


if __name__ == "__main__":
    main()
