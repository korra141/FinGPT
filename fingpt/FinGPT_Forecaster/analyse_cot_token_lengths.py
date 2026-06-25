"""
Analyse prompt, answer, and total token lengths in a FinGPT forecaster dataset.

Usage (inside your Apptainer container, from the FinGPT_Forecaster directory):

  # Compare LLaMA-2 and LLaMA-3 tokenizers side-by-side (default dataset: dow30):
  python3 analyse_cot_token_lengths.py --compare

  # Single tokenizer:
  python3 analyse_cot_token_lengths.py --tokenizer meta-llama/Llama-2-7b-hf
  python3 analyse_cot_token_lengths.py --tokenizer meta-llama/Meta-Llama-3-8B

  # Different dataset:
  python3 analyse_cot_token_lengths.py --compare --dataset dow30-202305-202405

  # Limit to one split:
  python3 analyse_cot_token_lengths.py --compare --split train
"""

import argparse
import os
import statistics

import datasets as hf_datasets

LLAMA2_MODEL = 'meta-llama/Llama-2-7b-hf'
LLAMA3_MODEL = 'meta-llama/Meta-Llama-3-8B'


def make_tokenizer(name: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    return lambda text: len(tok.encode(text, add_special_tokens=False))


def percentile(data, p):
    data = sorted(data)
    k = (len(data) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(data) - 1)
    return data[f] + (data[c] - data[f]) * (k - f)


def summarise(values, label):
    if not values:
        print(f"  {label}: no data")
        return
    print(f"  {label}:")
    print(f"    n       = {len(values)}")
    print(f"    mean    = {statistics.mean(values):.1f}")
    print(f"    median  = {statistics.median(values):.1f}")
    print(f"    stdev   = {statistics.stdev(values):.1f}")
    print(f"    p75     = {percentile(values, 75):.1f}")
    print(f"    p90     = {percentile(values, 90):.1f}")
    print(f"    p95     = {percentile(values, 95):.1f}")
    print(f"    max     = {max(values)}")


def analyse_split(split_ds, split_name, count_tokens, tok_label):
    n = len(split_ds)
    print(f"\n{'=' * 60}")
    print(f"  Split: {split_name}  ({n} samples)  [{tok_label}]")
    print('=' * 60)

    prompt_lens, answer_lens, total_lens = [], [], []

    for row in split_ds:
        p = count_tokens(row['prompt'])
        a = count_tokens(row['answer'])
        prompt_lens.append(p)
        answer_lens.append(a)
        total_lens.append(p + a)

    print()
    summarise(prompt_lens, "Prompt tokens")
    print()
    summarise(answer_lens, "Answer tokens")
    print()
    summarise(total_lens, "Total tokens (prompt + answer)")

    avg_prompt_pct = statistics.mean(p / t * 100 for p, t in zip(prompt_lens, total_lens))
    print(f"\n  Average split:  prompt {avg_prompt_pct:.1f}%  /  answer {100 - avg_prompt_pct:.1f}%")

    for budget in (4096, 8192):
        over = sum(1 for t in total_lens if t > budget)
        print(f"  Exceeds {budget} tokens: {over}/{n} ({100*over/n:.1f}%)")


def load_hf_dataset(dataset_name: str):
    hf_name = f'FinGPT/fingpt-forecaster-{dataset_name}'
    print(f"Loading dataset: {hf_name}")
    ds = hf_datasets.load_dataset(hf_name)
    if 'test' not in ds:
        ds = ds['train'].train_test_split(0.2, shuffle=True, seed=42)
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dow30-202305-202405',
                        help='Dataset suffix after "FinGPT/fingpt-forecaster-"')
    parser.add_argument('--tokenizer', default=LLAMA2_MODEL,
                        help='HuggingFace model name for the tokenizer')
    parser.add_argument('--compare', action='store_true',
                        help='Run both LLaMA-2 and LLaMA-3 tokenizers side-by-side')
    parser.add_argument('--split', default='both', choices=['train', 'test', 'both'])
    parser.add_argument('--hf_token', default=None,
                        help='HuggingFace token for gated models (or set HF_TOKEN env var)')
    args = parser.parse_args()

    token = args.hf_token or os.environ.get('HF_TOKEN') or os.environ.get('HF_USER_ACCESS_TOKEN')
    if token:
        os.environ['HF_TOKEN'] = token

    ds = load_hf_dataset(args.dataset)
    splits = [s for s in (['train', 'test'] if args.split == 'both' else [args.split]) if s in ds]

    tokenizers = (
        [(LLAMA2_MODEL, 'LLaMA-2'), (LLAMA3_MODEL, 'LLaMA-3')]
        if args.compare
        else [(args.tokenizer, args.tokenizer.split('/')[-1])]
    )

    for model_name, label in tokenizers:
        print(f"\nBuilding tokenizer: {model_name}")
        count_tokens = make_tokenizer(model_name)
        for split in splits:
            analyse_split(ds[split], split, count_tokens, label)

    print(f"\n{'=' * 60}\n  Done.\n{'=' * 60}\n")


if __name__ == '__main__':
    main()
