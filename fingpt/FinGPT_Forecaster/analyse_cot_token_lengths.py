"""
Analyse CoT vs final-answer token lengths in the chatgpt_cot dataset.

Usage (inside your Apptainer container, from the FinGPT_Forecaster directory):

  # With LLaMA-2 tokenizer (matches training for llama2 runs):
  python3 analyse_cot_token_lengths.py --tokenizer meta-llama/Llama-2-7b-hf

  # With LLaMA-3 tokenizer (matches training for llama3 runs):
  python3 analyse_cot_token_lengths.py --tokenizer meta-llama/Meta-Llama-3-8B

  # If you don't want to download a tokenizer, fall back to tiktoken cl100k (GPT-4):
  python3 analyse_cot_token_lengths.py --tokenizer tiktoken

  # Limit to one split for a quick check:
  python3 analyse_cot_token_lengths.py --tokenizer tiktoken --split train
"""

import argparse
import re
import statistics
import sys

import datasets

COT_END_RE = re.compile(r'assistantfinal', re.IGNORECASE)


def split_cot_final(answer: str):
    """Return (cot_text, final_text). final_text is '' if marker absent."""
    m = COT_END_RE.search(answer)
    if m:
        return answer[:m.start()], answer[m.end():]
    return answer, ''


def make_tokenizer(name: str):
    if name == 'tiktoken':
        import tiktoken
        enc = tiktoken.get_encoding('cl100k_base')
        return lambda text: len(enc.encode(text))
    else:
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
    print(f"    p25     = {percentile(values, 25):.1f}")
    print(f"    p75     = {percentile(values, 75):.1f}")
    print(f"    p90     = {percentile(values, 90):.1f}")
    print(f"    min     = {min(values)}")
    print(f"    max     = {max(values)}")


def analyse_split(split_ds, split_name, count_tokens):
    n = len(split_ds)
    print(f"\n{'=' * 60}")
    print(f"  Split: {split_name}  ({n} samples)")
    print('=' * 60)

    cot_tokens, final_tokens, total_tokens = [], [], []
    no_marker = 0

    for row in split_ds:
        answer = row['answer']
        cot, final = split_cot_final(answer)

        total = count_tokens(answer)
        total_tokens.append(total)

        if final:
            cot_tokens.append(count_tokens(cot))
            final_tokens.append(count_tokens(final))
        else:
            no_marker += 1
            cot_tokens.append(0)
            final_tokens.append(total)

    print(f"\n  Rows missing 'assistantfinal' marker: {no_marker}/{n} ({100*no_marker/n:.1f}%)")

    print()
    summarise(total_tokens, "Total answer tokens")
    print()
    summarise([v for v in cot_tokens if v > 0], "CoT reasoning tokens  (before 'assistantfinal')")
    print()
    summarise(final_tokens, "Final answer tokens   (after 'assistantfinal')")

    # Ratio: what fraction of the answer is CoT vs final
    ratios = []
    for c, f in zip(cot_tokens, final_tokens):
        total = c + f
        if total > 0:
            ratios.append(c / total)
    if ratios:
        avg_cot_pct = statistics.mean(ratios) * 100
        avg_final_pct = 100 - avg_cot_pct
        print(f"\n  Average token split:")
        print(f"    CoT reasoning : {avg_cot_pct:.1f}%")
        print(f"    Final answer  : {avg_final_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='chatgpt_cot/',
                        help='Path to DatasetDict saved with save_to_disk')
    parser.add_argument('--tokenizer', default='tiktoken',
                        help='"tiktoken" or a HuggingFace model name like meta-llama/Meta-Llama-3-8B')
    parser.add_argument('--split', default='both', choices=['train', 'test', 'both'])
    args = parser.parse_args()

    print(f"Loading dataset from: {args.dataset}")
    ds = datasets.load_from_disk(args.dataset)
    print(f"Splits: {list(ds.keys())}")

    print(f"\nBuilding tokenizer: {args.tokenizer}")
    count_tokens = make_tokenizer(args.tokenizer)
    print("Tokenizer ready.\n")

    splits = list(ds.keys()) if args.split == 'both' else [args.split]
    for split in splits:
        if split not in ds:
            print(f"WARNING: split '{split}' not found, skipping.")
            continue
        analyse_split(ds[split], split, count_tokens)

    print(f"\n{'=' * 60}\n  Done.\n{'=' * 60}\n")


if __name__ == '__main__':
    main()
