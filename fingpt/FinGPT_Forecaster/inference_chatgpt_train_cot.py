"""
Generate CoT-augmented training data using a HuggingFace-hosted model.

For each train-split sample the saved answer becomes:
    {gpt_cot_trace}
    assistantfinal
    {original_gt_answer}

The dataset is saved in HuggingFace arrow format (save_to_disk) so it can be
loaded with datasets.load_from_disk() or pushed to the Hub later.
"""

from transformers import pipeline
from transformers.utils.import_utils import is_triton_available
from huggingface_hub import login
import torch
import os
import re
import json
import argparse
import datasets as hf_datasets
from tqdm import tqdm
import wandb

from utils import load_dataset

LLAMA2_SYS_RE = re.compile(
    r'\[INST\]<<SYS>>\n(.*?)\n<</SYS>>\n\n(.*?)\[/INST\]',
    re.DOTALL,
)

COT_SYSTEM_SUFFIX = (
    "\n\nBefore giving your final answer, reason step by step about the "
    "financial data. Write your reasoning freely, then output the exact "
    "string \"assistantfinal\" on its own line, followed immediately by "
    "your structured final answer."
)


def extract_messages(raw_prompt: str):
    m = LLAMA2_SYS_RE.search(raw_prompt)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, raw_prompt.strip()


def extract_cot_only(raw_output: str) -> str:
    """Return everything up to and including 'assistantfinal', or '' if absent."""
    m = re.search(r'assistantfinal', raw_output, re.IGNORECASE)
    if m:
        return raw_output[:m.end()]
    return ''


def run(args):
    print(f"GPU Detected: {torch.cuda.is_available()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"Triton Available: {is_triton_available()}")

    wandb.init(project='fingpt-forecaster', name=args.run_name, config=vars(args))

    hf_token = args.hf_token or os.getenv('HF_TOKEN') or os.getenv('HF_USER_ACCESS_TOKEN')
    if hf_token:
        login(token=hf_token)

    print(f"Loading model: {args.model_id}")
    pipe = pipeline(
        "text-generation",
        model=args.model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    dataset_list = load_dataset(args.dataset, from_remote=True)
    dataset_train = hf_datasets.concatenate_datasets([d['train'] for d in dataset_list])
    dataset_test  = hf_datasets.concatenate_datasets([d['test']  for d in dataset_list])

    if args.num_samples > 0:
        dataset_train = dataset_train.shuffle(seed=42).select(
            range(min(args.num_samples, len(dataset_train)))
        )

    print(f"Generating CoT traces for {len(dataset_train)} train samples...")

    # Build all inputs up front so the pipeline can batch them on GPU
    all_features = list(dataset_train)
    all_messages = []
    for feature in all_features:
        system_msg, user_msg = extract_messages(feature['prompt'])
        combined_system = (system_msg or "") + COT_SYSTEM_SUFFIX
        messages = []
        if combined_system:
            messages.append({"role": "system", "content": combined_system})
        messages.append({"role": "user", "content": user_msg})
        all_messages.append(messages)

    all_outputs = pipe(
        all_messages,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    augmented_rows = []
    no_cot_count = 0

    for feature, outputs in tqdm(zip(all_features, all_outputs), total=len(all_features)):
        gt_answer = feature['answer']

        last = outputs[0]["generated_text"][-1]
        raw_output = last["content"] if isinstance(last, dict) else str(last)

        cot_trace = extract_cot_only(raw_output)

        if cot_trace:
            combined_answer = cot_trace + "\n" + gt_answer
        else:
            # No CoT marker — keep plain GT so the sample is still usable
            no_cot_count += 1
            combined_answer = gt_answer

        augmented_rows.append({
            "prompt":  feature['prompt'],       # original prompt unchanged
            "answer":  combined_answer,          # cot_trace + gt  (or plain gt)
            "label":   feature.get('label', ''),
            "symbol":  feature.get('symbol', ''),
            "period":  feature.get('period', ''),
        })

    if no_cot_count:
        print(f"\nWARNING: {no_cot_count}/{len(augmented_rows)} samples had no CoT marker — plain GT used.")

    # Build HuggingFace DatasetDict (arrow format → load_from_disk / push_to_hub)
    augmented_train = hf_datasets.Dataset.from_list(augmented_rows)
    augmented = hf_datasets.DatasetDict({
        'train': augmented_train,
        'test':  dataset_test,   # test split is unchanged — GT answers for eval
    })

    os.makedirs(args.output_dir, exist_ok=True)
    augmented.save_to_disk(args.output_dir)
    print(f"\nSaved CoT-augmented dataset to {args.output_dir}")
    print("To push to Hub later:  dataset.push_to_hub('<your-repo>')")

    # Small JSON preview for quick inspection
    preview_path = os.path.join(args.output_dir, "preview.json")
    preview = [
        {
            "symbol": augmented_train[i].get("symbol", ""),
            "period": augmented_train[i].get("period", ""),
            "answer_preview": augmented_train[i]["answer"][:600],
        }
        for i in range(min(5, len(augmented_train)))
    ]
    with open(preview_path, 'w') as f:
        json.dump(preview, f, indent=2)
    print(f"Saved 5-sample preview to {preview_path}")

    wandb.log({
        "total_samples":    len(augmented_rows),
        "no_cot_count":     no_cot_count,
        "cot_success_rate": 1 - no_cot_count / max(len(augmented_rows), 1),
    })

    dataset_artifact = wandb.Artifact(f"{args.run_name}-dataset", type="dataset")
    dataset_artifact.add_dir(args.output_dir)
    wandb.log_artifact(dataset_artifact)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name",       default="chatgpt-train-cot",   type=str)
    parser.add_argument("--model_id",       default="openai/gpt-oss-20b",  type=str,
                        help="HuggingFace model ID of the CoT teacher")
    parser.add_argument("--dataset",        required=True,                 type=str,
                        help="Dataset name passed to load_dataset (same as train_lora.py)")
    parser.add_argument("--output_dir",     required=True,                 type=str,
                        help="Local path to save the HuggingFace dataset")
    parser.add_argument("--num_samples",    default=-1,                    type=int,
                        help="Limit train samples for testing (-1 for all)")
    parser.add_argument("--max_new_tokens", default=1024,                  type=int)
    parser.add_argument("--batch_size",     default=8,                     type=int,
                        help="Pipeline batch size for GPU inference")
    parser.add_argument("--hf_token",       default=None,                  type=str,
                        help="HuggingFace API token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    run(args)
