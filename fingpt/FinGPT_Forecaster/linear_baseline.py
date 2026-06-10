"""
linear_baseline.py

Learned linear baseline for FinGPT-Forecaster.

Replaces the fixed extrapolation rule in arima_baseline.py
    forecast = p2 + (p2 - p1)   [no training data used]
with parameters fitted on the training split:
  - LinearRegression   → predicts price-change magnitude (%)
  - LogisticRegression → predicts direction (-1 / 0 / +1)

Features (extracted from the prompt, identical to what arima_baseline sees):
  pct_change_last_week  (p2-p1)/p1 * 100   — last observed weekly return
  price_level           log(p2)            — valuation regime

Both models are fitted on the training split and applied to the test split.
No test-set information leaks into training.

Metrics match arima_baseline.py and baseline_eval.py: bin_acc, MSE.

Usage
-----
  python linear_baseline.py --dataset dow30-2023 [--no_wandb]
"""

import re
import argparse
import numpy as np
from collections import Counter
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import wandb
import datasets as hf_datasets

from utils import parse_answer, parse_answer_base, load_dataset


# ---------------------------------------------------------------------------
# Shared regex — same pattern as arima_baseline.py
# ---------------------------------------------------------------------------
_PRICE_RE = re.compile(
    r"stock price \w+ from ([\d,]+\.?\d*) to ([\d,]+\.?\d*)",
    re.IGNORECASE,
)


def extract_features(prompt):
    """
    Return (pct_change_last_week, price_level) from the prompt,
    or None if the price line is absent.
    """
    m = _PRICE_RE.search(prompt)
    if not m:
        return None
    p1 = float(m.group(1).replace(',', ''))
    p2 = float(m.group(2).replace(',', ''))
    if p1 <= 0 or p2 <= 0:
        return None
    pct_change  = (p2 - p1) / p1 * 100.0
    price_level = np.log(p2)
    return [pct_change, price_level]


def parse_gt(answer):
    """Return (gt_bin, gt_margin) or (None, None)."""
    parsed = parse_answer(answer) or parse_answer_base(answer)
    if parsed and parsed['prediction'] is not None:
        return parsed['prediction_binary'], float(parsed['prediction'])
    return None, None


def collect_split(dataset_split):
    """
    Extract features and labels from a dataset split.
    Returns (X, bins, margins, n_skipped).
    """
    X, bins, margins = [], [], []
    n_skip = 0
    for row in dataset_split:
        feats = extract_features(row['prompt'])
        gt_bin, gt_margin = parse_gt(row['answer'])
        if feats is None or gt_bin is None:
            n_skip += 1
            continue
        X.append(feats)
        bins.append(gt_bin)
        margins.append(gt_margin)
    return np.array(X, dtype=np.float32), bins, margins, n_skip


def report(name, bin_acc, mse, n):
    print(f"\n{'='*52}")
    print(f"  {name}")
    print(f"{'='*52}")
    print(f"  N test          : {n}")
    print(f"  Binary accuracy : {bin_acc:.4f}")
    print(f"  MSE (margin %)  : {mse:.4f}")


def main(args):
    dataset_list = load_dataset(args.dataset, True)
    train_split  = hf_datasets.concatenate_datasets([d['train'] for d in dataset_list])
    test_split   = hf_datasets.concatenate_datasets([d['test']  for d in dataset_list])

    print(f"Raw  — train: {len(train_split)}  test: {len(test_split)}")

    X_train, train_bins, train_margins, n_skip_tr = collect_split(train_split)
    X_test,  test_bins,  test_margins,  n_skip_te = collect_split(test_split)

    print(f"Parsed — train: {len(train_bins)} (skipped {n_skip_tr})  "
          f"test: {len(test_bins)} (skipped {n_skip_te})")

    dist  = Counter(train_bins)
    total = sum(dist.values())
    print(f"\nTrain label distribution:")
    for cls in sorted(dist):
        label = {-1: "down (-1)", 0: "neutral (0)", 1: "up (+1)"}.get(cls, str(cls))
        print(f"  {label}: {dist[cls]}  ({100*dist[cls]/total:.1f}%)")

    y_train_margins = np.array(train_margins, dtype=np.float32)
    y_test_margins  = np.array(test_margins,  dtype=np.float32)

    # ---- Magnitude: LinearRegression -----------------------------------------
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train_margins)

    pred_margins = lin_reg.predict(X_test)
    mse = mean_squared_error(y_test_margins, pred_margins)

    # Direction from regression sign (sign of predicted margin → binary class)
    pred_bins_from_reg = [1 if p > 0 else (-1 if p < 0 else 0) for p in pred_margins]
    bin_acc_from_reg = accuracy_score(test_bins, pred_bins_from_reg)

    report("LinearRegression  (direction = sign of predicted margin)",
           bin_acc_from_reg, mse, len(test_bins))

    coef = lin_reg.coef_
    print(f"\n  Learnt coefficients:")
    print(f"    pct_change_last_week : {coef[0]:+.4f}")
    print(f"    price_level (log p2) : {coef[1]:+.4f}")
    print(f"    intercept            : {lin_reg.intercept_:+.4f}")

    # ---- Direction: LogisticRegression (3-class) ------------------------------
    log_reg = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
    )
    log_reg.fit(X_train, train_bins)

    pred_bins_logit = log_reg.predict(X_test)
    bin_acc_logit   = accuracy_score(test_bins, pred_bins_logit)

    # For MSE use the linear regression magnitude even with logistic direction
    report("LogisticRegression  (direction only, magnitude from LinearRegression)",
           bin_acc_logit, mse, len(test_bins))

    pred_dist = Counter(pred_bins_logit)
    print(f"\nPredicted class distribution (LogisticRegression, test):")
    for cls in sorted(pred_dist):
        label = {-1: "down (-1)", 0: "neutral (0)", 1: "up (+1)"}.get(cls, str(cls))
        print(f"  {label}: {pred_dist[cls]}  ({100*pred_dist[cls]/len(test_bins):.1f}%)")

    # ---- wandb ----------------------------------------------------------------
    if not args.no_wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={
                "dataset":      args.dataset,
                "features":     ["pct_change_last_week", "price_level"],
                "n_train":      len(train_bins),
                "n_test":       len(test_bins),
                "n_skip_train": n_skip_tr,
                "n_skip_test":  n_skip_te,
                "lin_reg_coef_pct_change": float(coef[0]),
                "lin_reg_coef_price_level": float(coef[1]),
                "lin_reg_intercept": float(lin_reg.intercept_),
            },
        )
        wandb.log({
            "linear_reg/bin_acc": bin_acc_from_reg,
            "linear_reg/mse":     mse,
            "logistic_reg/bin_acc": bin_acc_logit,
            "logistic_reg/mse":     mse,
            "class_dist/train_up":      dist.get(1,  0) / total,
            "class_dist/train_neutral": dist.get(0,  0) / total,
            "class_dist/train_down":    dist.get(-1, 0) / total,
        })
        run.finish()
        print(f"\nLogged to wandb run: {run.url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Learned linear baseline for FinGPT-Forecaster"
    )
    parser.add_argument(
        "--dataset", required=True, type=str,
        help="Dataset name/path — same format as arima_baseline.py (e.g. dow30-2023)",
    )
    parser.add_argument("--wandb_project", default="fingpt-forecaster")
    parser.add_argument("--run_name",      default="linear-regression-baseline")
    parser.add_argument("--no_wandb",      action="store_true",
                        help="Skip wandb logging, print only")
    args = parser.parse_args()
    main(args)
