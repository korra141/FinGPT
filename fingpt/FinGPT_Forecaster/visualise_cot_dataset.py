"""
Visualise the chatgpt_cot DatasetDict saved by inference_chatgpt_train_cot.py.

Usage:
    python3 visualise_cot_dataset.py --dataset chatgpt_cot/
    python3 visualise_cot_dataset.py --dataset chatgpt_cot/ --split train --n_samples 5
"""

import argparse
import re
import sys
from collections import Counter, defaultdict

import datasets


# ── parsing helpers (mirrors inference_chatgpt.py) ────────────────────────────

COT_END = 'assistantfinal'

_HEADER_RE = re.compile(
    r'^\s*\[Positive Developments\]:\s*(.*?)\s*\[Potential Concerns\]:\s*(.*?)\s*'
    r'\[Prediction\s*(?:&|and)\s*Analysis\]:\s*(.*?)\s*$',
    re.DOTALL | re.IGNORECASE,
)
_PRED_RE = re.compile(r'^Prediction:\s*(.*?)\s*Analysis:\s*(.*)\s*$', re.DOTALL | re.IGNORECASE)
_PCT_RANGE_RE = re.compile(r'(\d+)\s*-\s*(\d+)\s*%')
_PCT_RE = re.compile(r'(\d+)\s*%')


def split_cot_final(answer: str):
    """Return (cot_text, final_text). final_text is '' if marker absent."""
    m = re.search(COT_END, answer, re.IGNORECASE)
    if m:
        return answer[:m.start()], answer[m.end():]
    return answer, ''


def normalise(text: str) -> str:
    """Fix common model formatting quirks before regex parsing."""
    # Add missing colon after section headers
    text = re.sub(r'(\[(?:Positive Developments|Potential Concerns|Prediction\s*(?:&|and)\s*Analysis)\])\s*\n',
                  r'\1:\n', text, flags=re.IGNORECASE)
    # Strip bold markdown: **[Header]** and **SubHeader:**
    text = re.sub(r'\*+\[([^\]]+)\]\*+', r'[\1]', text)
    text = re.sub(r'\*+(Prediction|Analysis):\**', r'\1:', text, flags=re.IGNORECASE)
    # Non-breaking spaces / narrow no-break spaces → normal space
    text = text.replace(' ', ' ').replace(' ', ' ')
    return text


def parse_final(final_text: str):
    """Parse the structured final answer. Returns dict or None."""
    text = normalise(final_text)
    m = _HEADER_RE.match(text)
    if not m:
        return None
    pros, cons, pna = m.group(1), m.group(2), m.group(3)
    mp = _PRED_RE.match(pna)
    if not mp:
        return None
    pred_str, analysis = mp.group(1), mp.group(2)
    pred_lower = pred_str.lower()
    if re.search(r'\bup\b|increase|rise|higher|bullish', pred_lower):
        sign = 1
    elif re.search(r'\bdown\b|decrease|decline|fall|lower|bearish', pred_lower):
        sign = -1
    else:
        sign = 0
    mr = _PCT_RANGE_RE.search(pred_str)
    if mr:
        magnitude = (int(mr.group(1)) + int(mr.group(2))) / 2
    else:
        mp2 = _PCT_RE.search(pred_str)
        magnitude = float(mp2.group(1)) if mp2 else 0.0
    return {
        'positive_developments': pros,
        'potential_concerns': cons,
        'prediction_binary': sign,
        'prediction': sign * magnitude,
        'analysis': analysis,
    }


# ── display helpers ────────────────────────────────────────────────────────────

def bar(label, value, total, width=40):
    filled = int(width * value / max(total, 1))
    pct = 100 * value / max(total, 1)
    return f"  {label:<20} {'█' * filled:<{width}} {value:>5} ({pct:5.1f}%)"


def section(title):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print('═' * 60)


# ── main ───────────────────────────────────────────────────────────────────────

def analyse(ds_split, split_name: str, n_samples: int):
    section(f"Split: {split_name}  ({len(ds_split)} samples)")

    cot_lengths, final_lengths, answer_lengths = [], [], []
    has_cot, parse_ok = 0, 0
    direction_counts = Counter()
    symbol_counts = Counter()
    period_counts = Counter()
    parse_failures = []

    for row in ds_split:
        answer = row['answer']
        symbol_counts[row.get('symbol', '?')] += 1
        period_counts[row.get('period', '?')] += 1
        answer_lengths.append(len(answer))

        cot, final = split_cot_final(answer)
        if final:
            has_cot += 1
            cot_lengths.append(len(cot))
            final_lengths.append(len(final))
        else:
            final = cot  # no marker — treat whole answer as final
            cot_lengths.append(0)
            final_lengths.append(len(final))

        parsed = parse_final(final)
        if parsed:
            parse_ok += 1
            direction_counts[parsed['prediction_binary']] += 1
        else:
            parse_failures.append({'symbol': row.get('symbol'), 'period': row.get('period'),
                                   'final_preview': final[:200]})

    n = len(ds_split)

    # CoT coverage
    print(f"\n  Samples with CoT marker ('assistantfinal'):  {has_cot}/{n}  ({100*has_cot/n:.1f}%)")
    print(f"  Samples parsed successfully:                 {parse_ok}/{n}  ({100*parse_ok/n:.1f}%)")

    # Length stats
    def stats(lst):
        if not lst:
            return 'n/a'
        return f"min={min(lst)}  avg={sum(lst)//len(lst)}  max={max(lst)}"

    print(f"\n  Answer length (chars):  {stats(answer_lengths)}")
    print(f"  CoT length    (chars):  {stats([l for l in cot_lengths if l])}")
    print(f"  Final length  (chars):  {stats([l for l in final_lengths if l])}")

    # Prediction direction
    section("Prediction Direction")
    labels = {1: 'Up', -1: 'Down', 0: 'Neutral/unclear'}
    for sign in [1, -1, 0]:
        print(bar(labels[sign], direction_counts[sign], parse_ok))

    # Symbols
    section("Top Symbols")
    for sym, cnt in symbol_counts.most_common(15):
        print(bar(sym, cnt, n))

    # Parse failures
    if parse_failures:
        section(f"Parse Failures ({len(parse_failures)} samples)")
        for f in parse_failures[:5]:
            print(f"  [{f['symbol']} | {f['period']}]")
            print(f"  {repr(f['final_preview'])}\n")

    # Sample rows
    if n_samples > 0:
        section(f"Sample Rows (n={n_samples})")
        for i in range(min(n_samples, n)):
            row = ds_split[i]
            cot, final = split_cot_final(row['answer'])
            parsed = parse_final(final or cot)
            print(f"\n  ── Sample {i+1} ──────────────────────────────")
            print(f"  Symbol : {row.get('symbol')}  |  Period: {row.get('period')}")
            print(f"  CoT    : {cot[:120].strip()!r}{'...' if len(cot) > 120 else ''}")
            print(f"  Final  : {(final or cot)[:200].strip()!r}{'...' if len(final or cot) > 200 else ''}")
            if parsed:
                sign_str = {1: 'UP', -1: 'DOWN', 0: 'NEUTRAL'}[parsed['prediction_binary']]
                print(f"  Parsed : direction={sign_str}  magnitude={parsed['prediction']:.1f}%")
            else:
                print(f"  Parsed : FAILED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='chatgpt_cot/', help='Path to DatasetDict (save_to_disk format)')
    parser.add_argument('--split', default='both', choices=['train', 'test', 'both'])
    parser.add_argument('--n_samples', default=3, type=int, help='Sample rows to print per split (0 to skip)')
    args = parser.parse_args()

    print(f"Loading dataset from: {args.dataset}")
    ds = datasets.load_from_disk(args.dataset)
    print(f"Splits found: {list(ds.keys())}")

    splits = list(ds.keys()) if args.split == 'both' else [args.split]
    for split in splits:
        if split not in ds:
            print(f"WARNING: split '{split}' not found, skipping.")
            continue
        analyse(ds[split], split, args.n_samples)

    print(f"\n{'═' * 60}\n  Done.\n{'═' * 60}\n")


if __name__ == '__main__':
    main()
