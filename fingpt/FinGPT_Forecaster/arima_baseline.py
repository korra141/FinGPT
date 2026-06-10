"""
ARIMA(2,1,1) baseline for FinGPT-Forecaster.

For each prompt, extracts the two price points embedded in the text, fits
ARIMA(2,1,1) on them, and forecasts the next price.  With only 2 price
points the model has 1 differenced observation — insufficient to estimate
the 4 parameters of ARIMA(2,1,1).  Those samples fall back to linear
extrapolation: forecast = p2 + (p2 - p1), which is the maximum-likelihood
estimate under a random-walk-with-drift given 2 observations.

Ground truth is NOT used to fit ARIMA. The historical prices inside the
prompt are the fitting data; the parsed GT percentage change is used only
for evaluation (bin_acc and MSE).

Metrics are computed consistently with baseline_eval.py and calc_metrics:
  - bin_acc  : directional accuracy (-1/0/+1) vs GT binary direction
  - mse      : mean squared error of predicted % change vs GT % change
"""

import re
import argparse
import warnings
import numpy as np
import datasets
import wandb
from collections import Counter
from sklearn.metrics import accuracy_score, mean_squared_error

from utils import parse_answer, parse_answer_base, load_dataset
import datasets as hf_datasets

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not installed — all samples will use linear extrapolation fallback.")


# Matches "stock price increased/decreased/... from 150.55 to 145.90"
PRICE_RE = re.compile(
    r"stock price \w+ from ([\d,]+\.?\d*) to ([\d,]+\.?\d*)",
    re.IGNORECASE,
)


def extract_prices(prompt):
    """Return (start_price, end_price) from the prompt, or (None, None)."""
    m = PRICE_RE.search(prompt)
    if not m:
        return None, None
    p1 = float(m.group(1).replace(',', ''))
    p2 = float(m.group(2).replace(',', ''))
    return p1, p2


def arima_forecast(prices):
    """
    Fit ARIMA(2,1,1) and return a 1-step-ahead forecast price.

    With fewer than 4 price points the model cannot be identified — falls
    back to linear extrapolation (trend continuation):
        forecast = p2 + (p2 - p1)
    which is the same as projecting the last observed first-difference forward.

    Returns (forecast_price, used_fallback).
    """
    last = prices[-1]
    drift = prices[-1] - prices[-2]

    if not HAS_STATSMODELS or len(prices) < 4:
        return last + drift, True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = ARIMA(prices, order=(2, 1, 1)).fit()
            return float(fit.forecast(steps=1)[0]), False
        except Exception:
            return last + drift, True


def parse_gt(answer):
    """Return (gt_bin, gt_margin) or (None, None) if unparseable."""
    parsed = parse_answer(answer) or parse_answer_base(answer)
    if parsed and parsed['prediction'] is not None:
        return parsed['prediction_binary'], parsed['prediction']
    return None, None


def main(args):
    # ds = datasets.load_from_disk(args.dataset)
    # split = ds[args.split]
    dataset_list = load_dataset(args.dataset, True)
    split = hf_datasets.concatenate_datasets([d['test'] for d in dataset_list])

    print(f"Loaded '{args.split}' split: {len(split)} samples")

    pred_bins, pred_margins = [], []
    gt_bins, gt_margins     = [], []
    n_fallback = 0
    n_skip     = 0

    for row in split:
        # Ground truth — only used for evaluation, not for fitting
        gt_bin, gt_margin = parse_gt(row['answer'])
        if gt_bin is None:
            n_skip += 1
            continue

        # Extract price series from prompt
        p1, p2 = extract_prices(row['prompt'])
        if p1 is None:
            n_skip += 1
            continue

        # Fit and forecast
        forecast_price, fallback = arima_forecast([p1, p2])
        if fallback:
            n_fallback += 1

        # Convert to percentage change relative to last observed price
        pct_change = (forecast_price - p2) / p2 * 100

        # Binary direction from last observed value
        if pct_change > 0:
            pred_bin = 1
        elif pct_change < 0:
            pred_bin = -1
        else:
            pred_bin = 0

        pred_bins.append(pred_bin)
        pred_margins.append(pct_change)
        gt_bins.append(gt_bin)
        gt_margins.append(gt_margin)

    n = len(pred_bins)
    print(f"Evaluated : {n}")
    print(f"Skipped   : {n_skip}  (GT unparseable or no prices in prompt)")
    print(f"Fallback  : {n_fallback} / {n}  (linear extrapolation, insufficient points for ARIMA)")

    dist = Counter(pred_bins)
    total = sum(dist.values())
    print(f"\nPredicted class distribution:")
    for cls in sorted(dist):
        label = {-1: "down (-1)", 0: "neutral (0)", 1: "up (+1)"}.get(cls, str(cls))
        print(f"  {label}: {dist[cls]}  ({100*dist[cls]/total:.1f}%)")

    bin_acc = accuracy_score(gt_bins, pred_bins)
    mse     = mean_squared_error(gt_margins, pred_margins)

    print(f"\n{'='*50}")
    print(f"  ARIMA(2,1,1) [linear extrap fallback for <4 pts]")
    print(f"{'='*50}")
    print(f"  Binary accuracy : {bin_acc:.4f}")
    print(f"  MSE (margin)    : {mse:.4f}")
    print()

    if not args.no_wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={
                "dataset":      args.dataset,
                "split":        args.split,
                "arima_order":  "(2,1,1)",
                "n_evaluated":  n,
                "n_skipped":    n_skip,
                "n_fallback":   n_fallback,
            },
        )
        wandb.log({
            "arima/bin_acc":    bin_acc,
            "arima/mse":        mse,
            "arima/n_evaluated": n,
            "arima/n_fallback":  n_fallback,
            "class_dist/up":      dist.get(1,  0) / total,
            "class_dist/neutral": dist.get(0,  0) / total,
            "class_dist/down":    dist.get(-1, 0) / total,
        })
        run.finish()
        print(f"Logged to wandb: {run.url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",       required=True,  type=str,
                        help="Path to dataset saved with save_to_disk")
    parser.add_argument("--split",         default="test", choices=["train", "test"])
    parser.add_argument("--wandb_project", default="fingpt-forecaster")
    parser.add_argument("--run_name",      default="arima-2-1-1-baseline")
    parser.add_argument("--no_wandb",      action="store_true",
                        help="Skip wandb logging, print only")
    args = parser.parse_args()
    main(args)
