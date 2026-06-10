"""
test_arima.py

Evaluates per-stock ARIMA models against the FinGPT-Forecaster test prompts.

For each test prompt the script:
  1.  Parses the ticker, the 2 embedded price points (P1, P2), the week dates,
      and the forecast-week end date.
  2.  Looks up the matching 42-price window from the preprocessed dataset
      (arima_preprocess.py output) by (ticker, window_end ≈ current week_end).
  3.  Computes steps_ahead = round((forecast_end − window_end).days / 7).
        steps_ahead = 1  → normal case: prompt's forecast is one week out
        steps_ahead > 1  → gap (holiday, missing week, or test set starts
                           several weeks after training cut-off); ARIMA must
                           iterate to reach the target date.
  4.  Applies iterative ARIMA forecasting:
        for each step:
          forecast = ARIMA(current_42_window)        # 1-step ahead
          window   = window[1:] + [forecast]         # slide: drop oldest
      Error compounds with each additional step; steps_ahead is reported
      so you can filter or weight results accordingly.
  5.  Converts the final forecast to a percentage change from P2
      (= prices[-1] in the original window, the last KNOWN price):
        predicted_pct = (forecast_price − P2) / P2 × 100
  6.  Parses the ground-truth percentage from the FinGPT answer and computes
      MSE and directional accuracy, both globally and stratified by steps_ahead.

Usage:
  python test_arima.py \\
      --fingpt_dataset  data/fingpt-forecaster-dow30-2023 \\
      --arima_dataset   data/arima_dow30_42w \\
      --arima_params    results/arima_per_stock.json \\
      --output          results/arima_test_eval.json
"""

import re
import json
import argparse
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import wandb
import datasets
from sklearn.metrics import accuracy_score, mean_squared_error

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not installed — using linear-extrapolation fallback.")

from utils import parse_answer, parse_answer_base
from utils import load_dataset as load_fingpt_dataset
from indices import DOW_30

WINDOW_SIZE = 42
D_FIXED     = 1


def direction_bin(pct: float) -> int:
    return 1 if pct > 0 else -1


# ---------------------------------------------------------------------------
# Prompt parsing
# ---------------------------------------------------------------------------

# "From 2023-05-07 to 2023-05-14, AXP's stock price decreased from 150.55 to 145.90"
_WEEK_PRICE_RE = re.compile(
    r"From\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2}),\s+"
    r"(\w+)'s stock price \w+ from ([\d,]+\.?\d*) to ([\d,]+\.?\d*)",
    re.IGNORECASE,
)

# "next week (2023-05-14 to 2023-05-21)"
_FORECAST_WEEK_RE = re.compile(
    r"next week \((\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})\)",
    re.IGNORECASE,
)


def parse_test_prompt(prompt: str):
    """
    Extract fields needed for ARIMA evaluation from a FinGPT test prompt.

    Returns dict with keys:
      ticker          str   e.g. "AXP"
      p1              float start-of-week price  (not used for prediction)
      p2              float end-of-week price = last known price = prices[-1]
      window_end      str   ISO date of p2 (end of the current week)
      forecast_end    str   ISO date of the END of the forecast week
    Returns None if any required field is missing.
    """
    m_price = _WEEK_PRICE_RE.search(prompt)
    if not m_price:
        return None

    m_forecast = _FORECAST_WEEK_RE.search(prompt)
    if not m_forecast:
        return None

    return {
        "ticker":       m_price.group(3).upper(),
        "p1":           float(m_price.group(4).replace(",", "")),
        "p2":           float(m_price.group(5).replace(",", "")),
        "window_end":   m_price.group(2),          # end of current week
        "forecast_end": m_forecast.group(2),       # end of forecast week
    }


def parse_gt_pct(answer: str):
    """Return the ground-truth percentage from a FinGPT answer, or None."""
    parsed = parse_answer(answer) or parse_answer_base(answer)
    if parsed and parsed.get("prediction") is not None:
        return float(parsed["prediction"])
    return None


# ---------------------------------------------------------------------------
# Steps-ahead computation
# ---------------------------------------------------------------------------

def compute_steps_ahead(window_end: str, forecast_end: str) -> int:
    """
    How many weekly ARIMA steps are needed to reach forecast_end from window_end?

    Normal case: forecast_end = window_end + 7 days → steps_ahead = 1
    Gap case:    holiday or missing weeks → steps_ahead ≥ 2
    The result is rounded to the nearest integer to absorb ±1 day calendar drift.
    """
    delta_days = (pd.Timestamp(forecast_end) - pd.Timestamp(window_end)).days
    return max(1, round(delta_days / 7))


# ---------------------------------------------------------------------------
# ARIMA forecasting
# ---------------------------------------------------------------------------

def _linear_extrapolation(prices: list) -> float:
    return prices[-1] + (prices[-1] - prices[-2])


def arima_one_step(prices: list, p: int, q: int) -> tuple:
    """Fit ARIMA(p, D_FIXED, q) and return (forecast_price, used_fallback)."""
    if not HAS_STATSMODELS or len(prices) < p + D_FIXED + q + 2:
        return _linear_extrapolation(prices), True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit      = ARIMA(prices, order=(p, D_FIXED, q)).fit()
            forecast = float(fit.forecast(steps=1).iloc[0])
            return forecast, False
        except Exception:
            return _linear_extrapolation(prices), True


def multi_step_forecast(prices: list, p: int, q: int, steps: int) -> tuple:
    """
    Iteratively apply ARIMA for `steps` one-step-ahead forecasts.

    Each step:
      1. Fit ARIMA(p, 1, q) on the current 42-price window.
      2. Produce a 1-step forecast.
      3. Slide the window: drop prices[0], append the forecast.

    After `steps` iterations the final predicted price is at
    window_end + steps × 1 week.

    Returns (final_forecast_price, n_fallbacks_used).
    """
    window    = list(prices)       # shallow copy — do not mutate caller's list
    fallbacks = 0

    for _ in range(steps):
        next_price, fell_back = arima_one_step(window, p, q)
        if fell_back:
            fallbacks += 1
        window = window[1:] + [next_price]   # slide: len stays WINDOW_SIZE

    # final forecast = the value just appended in the last iteration
    return next_price, fallbacks


# ---------------------------------------------------------------------------
# 42-price window lookup
# ---------------------------------------------------------------------------

def build_window_index(arima_dataset_path: str) -> dict:
    """
    Load the preprocessed arima dataset and index test rows by
    (ticker, window_end) for O(1) lookup.

    Returns dict: (ticker, window_end_str) → list[float] of 42 prices
    """
    ds    = datasets.load_from_disk(arima_dataset_path)
    index = {}

    for split in ("train", "test"):
        if split not in ds:
            continue
        for row in ds[split]:
            key = (row["ticker"], row["window_end"])
            index[key] = row["prices"]

    return index


def find_window(ticker: str, window_end: str, index: dict,
                tolerance_days: int = 4) -> list | None:
    """
    Look up the 42-price window for (ticker, window_end).

    Tries an exact date match first, then searches within ±tolerance_days
    to handle minor calendar mismatches between the prompt date and the
    Friday-close date stored in the preprocessed dataset.
    """
    key = (ticker, window_end)
    if key in index:
        return index[key]

    # Fuzzy search within ±tolerance_days
    target_ts = pd.Timestamp(window_end)
    for delta in range(1, tolerance_days + 1):
        for sign in (+1, -1):
            candidate = (target_ts + pd.DateOffset(days=sign * delta)).strftime("%Y-%m-%d")
            k = (ticker, candidate)
            if k in index:
                return index[k]

    return None


# ---------------------------------------------------------------------------
# Load per-stock ARIMA (p, q) from train_arima.py output
# ---------------------------------------------------------------------------

def load_arima_params(params_path: str) -> dict:
    """
    Read per-stock ARIMA orders from the JSON written by train_arima.py.
    Returns dict: ticker → (p, q)
    Falls back to (2, 1) for any ticker not found.
    """
    try:
        with open(params_path) as f:
            data = json.load(f)
        result = {}
        for entry in data.get("per_ticker", []):
            order_str = entry["arima_order"]          # e.g. "(2,1,1)"
            digits    = re.findall(r"\d+", order_str)
            if len(digits) >= 3:
                p, _, q = int(digits[0]), int(digits[1]), int(digits[2])
                result[entry["ticker"]] = (p, q)
        return result
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(args):
    print("Loading FinGPT test prompts ...")
    # load_fingpt_dataset mirrors arima_baseline.py: accepts a HuggingFace short
    # name ("dow30-2023") or a local path, returns a list of DatasetDicts.
    dataset_list = load_fingpt_dataset(args.fingpt_dataset, from_remote=True)
    test_split   = datasets.concatenate_datasets(
        [ds["test"] for ds in dataset_list if "test" in ds]
    )
    print(f"  {len(test_split)} test rows\n")

    print("Indexing preprocessed 42-price windows ...")
    window_index = build_window_index(args.arima_dataset)
    print(f"  {len(window_index)} (ticker, window_end) entries indexed\n")

    arima_params = load_arima_params(args.arima_params) if args.arima_params else {}
    if arima_params:
        print(f"Loaded ARIMA orders for {len(arima_params)} tickers from {args.arima_params}\n")
    else:
        print("No --arima_params supplied — using ARIMA(2,1,1) for all tickers\n")

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    pred_pcts:  list[float] = []
    gt_pcts:    list[float] = []
    pred_bins:  list[int]   = []
    gt_bins:    list[int]   = []
    steps_list: list[int]   = []

    n_skip_parse   = 0
    n_skip_window  = 0
    n_skip_gt      = 0
    details        = []   # per-row detail for debugging

    for row in test_split:
        # --- 1. Parse prompt ---
        parsed = parse_test_prompt(row["prompt"])
        if parsed is None:
            n_skip_parse += 1
            continue

        ticker      = parsed["ticker"]
        p2          = parsed["p2"]
        window_end  = parsed["window_end"]
        forecast_end = parsed["forecast_end"]

        # --- 2. Ground truth ---
        gt_pct = parse_gt_pct(row.get("answer", ""))
        if gt_pct is None:
            n_skip_gt += 1
            continue

        # --- 3. 42-price window lookup ---
        prices = find_window(ticker, window_end, window_index)
        if prices is None:
            n_skip_window += 1
            continue

        # --- 4. Steps ahead ---
        steps = compute_steps_ahead(window_end, forecast_end)

        # --- 5. ARIMA order ---
        p, q = arima_params.get(ticker, (2, 1))

        # --- 6. Iterative forecast ---
        forecast_price, n_fallbacks = multi_step_forecast(prices, p, q, steps)

        # --- 7. Percentage change relative to P2 (last known price) ---
        if p2 == 0:
            n_skip_parse += 1
            continue
        predicted_pct = (forecast_price - p2) / p2 * 100.0

        pred_pcts.append(predicted_pct)
        gt_pcts.append(gt_pct)
        pred_bins.append(direction_bin(predicted_pct))
        gt_bins.append(direction_bin(gt_pct))
        steps_list.append(steps)

        details.append({
            "ticker":        ticker,
            "window_end":    window_end,
            "forecast_end":  forecast_end,
            "steps_ahead":   steps,
            "p2":            p2,
            "arima_order":   f"({p},{D_FIXED},{q})",
            "predicted_pct": round(predicted_pct, 4),
            "gt_pct":        round(gt_pct, 4),
            "correct_dir":   int(direction_bin(predicted_pct) == direction_bin(gt_pct)),
            "n_fallbacks":   n_fallbacks,
        })

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    n = len(pred_pcts)
    if n == 0:
        print("No valid predictions — check dataset and window index alignment.")
        return

    mse     = mean_squared_error(gt_pcts, pred_pcts)
    bin_acc = accuracy_score(gt_bins, pred_bins)

    print(f"\n{'='*60}")
    print(f"  ARIMA per-stock — test evaluation")
    print(f"{'='*60}")
    print(f"  Evaluated    : {n}")
    print(f"  Skipped      : parse={n_skip_parse}  no_window={n_skip_window}  no_gt={n_skip_gt}")
    print(f"  MSE (pct)    : {mse:.4f}")
    print(f"  Bin accuracy : {bin_acc:.4f}")

    # Stratify by steps_ahead
    steps_counts = defaultdict(int)
    for s in steps_list:
        steps_counts[s] += 1
    print(f"\n  Steps-ahead distribution:")
    for s in sorted(steps_counts):
        label = "  (adjacent)" if s == 1 else f"  (gap = {s} weeks)"
        print(f"    steps={s}: {steps_counts[s]} samples{label}")

    # Stratified accuracy
    if max(steps_counts) > 1:
        print(f"\n  Accuracy by steps_ahead:")
        for s in sorted(steps_counts):
            mask  = [i for i, st in enumerate(steps_list) if st == s]
            s_acc = accuracy_score(
                [gt_bins[i]   for i in mask],
                [pred_bins[i] for i in mask],
            )
            s_mse = mean_squared_error(
                [gt_pcts[i]   for i in mask],
                [pred_pcts[i] for i in mask],
            )
            print(f"    steps={s}: acc={s_acc:.4f}  mse={s_mse:.4f}  n={steps_counts[s]}")

    # Per-ticker summary
    ticker_groups = defaultdict(list)
    for d in details:
        ticker_groups[d["ticker"]].append(d)

    print(f"\n  {'Ticker':<8} {'Order':<12} {'Acc':>6} {'MSE':>8} {'N':>5} {'Avg_steps':>10}")
    print("  " + "-" * 50)
    for ticker in sorted(ticker_groups):
        rows_t   = ticker_groups[ticker]
        t_gt     = [r["gt_pct"]        for r in rows_t]
        t_pred   = [r["predicted_pct"] for r in rows_t]
        t_gbins  = [direction_bin(r["gt_pct"])          for r in rows_t]
        t_pbins  = [direction_bin(r["predicted_pct"])   for r in rows_t]
        t_acc    = accuracy_score(t_gbins, t_pbins)
        t_mse    = mean_squared_error(t_gt, t_pred)
        t_steps  = sum(r["steps_ahead"] for r in rows_t) / len(rows_t)
        order    = rows_t[0]["arima_order"]
        print(
            f"  {ticker:<8} {order:<12} {t_acc:>6.3f} {t_mse:>8.3f} "
            f"{len(rows_t):>5} {t_steps:>10.1f}"
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output = {
        "global": {
            "n_evaluated": n,
            "n_skip_parse": n_skip_parse,
            "n_skip_window": n_skip_window,
            "n_skip_gt": n_skip_gt,
            "mse": round(mse, 4),
            "bin_acc": round(bin_acc, 4),
        },
        "steps_distribution": {str(k): v for k, v in sorted(steps_counts.items())},
        "per_row": details,
    }

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # ------------------------------------------------------------------
    # wandb logging
    # ------------------------------------------------------------------
    if not args.no_wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={
                "fingpt_dataset": args.fingpt_dataset,
                "arima_dataset":  args.arima_dataset,
                "arima_params":   args.arima_params or "default(2,1,1)",
                "window_size":    WINDOW_SIZE,
                "d_fixed":        D_FIXED,
            },
        )

        # Global metrics
        wandb.log({
            "test/bin_acc":       bin_acc,
            "test/mse":           mse,
            "test/n_evaluated":   n,
            "test/n_skip_parse":  n_skip_parse,
            "test/n_skip_window": n_skip_window,
            "test/n_skip_gt":     n_skip_gt,
        })

        # Stratified metrics by steps_ahead
        for s in sorted(steps_counts):
            mask  = [i for i, st in enumerate(steps_list) if st == s]
            s_acc = accuracy_score([gt_bins[i]  for i in mask], [pred_bins[i] for i in mask])
            s_mse = mean_squared_error([gt_pcts[i] for i in mask], [pred_pcts[i] for i in mask])
            wandb.log({
                f"test/steps_{s}/bin_acc": s_acc,
                f"test/steps_{s}/mse":     s_mse,
                f"test/steps_{s}/n":       steps_counts[s],
            })

        # Per-ticker table
        tbl = wandb.Table(columns=["ticker", "arima_order", "bin_acc", "mse", "n", "avg_steps"])
        for ticker in sorted(ticker_groups):
            rows_t  = ticker_groups[ticker]
            t_gt    = [r["gt_pct"]        for r in rows_t]
            t_pred  = [r["predicted_pct"] for r in rows_t]
            t_gbins = [direction_bin(r["gt_pct"])        for r in rows_t]
            t_pbins = [direction_bin(r["predicted_pct"]) for r in rows_t]
            tbl.add_data(
                ticker,
                rows_t[0]["arima_order"],
                round(accuracy_score(t_gbins, t_pbins), 4),
                round(mean_squared_error(t_gt, t_pred), 4),
                len(rows_t),
                round(sum(r["steps_ahead"] for r in rows_t) / len(rows_t), 2),
            )
        wandb.log({"test/per_ticker": tbl})

        run.finish()
        print(f"Logged to wandb run: {run.url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ARIMA models against FinGPT prompts")
    parser.add_argument(
        "--fingpt_dataset",
        required=True,
        help="Path to the FinGPT DatasetDict (train + test splits)",
    )
    parser.add_argument(
        "--arima_dataset",
        required=True,
        help="Path to the preprocessed 42-price dataset (arima_preprocess.py output)",
    )
    parser.add_argument(
        "--arima_params",
        default=None,
        help="Path to per-stock ARIMA orders JSON (train_arima.py output). "
             "If omitted, ARIMA(2,1,1) is used for every stock.",
    )
    parser.add_argument(
        "--output",
        default="results/arima_test_eval.json",
        help="Where to write the evaluation JSON",
    )
    parser.add_argument("--wandb_project", default="fingpt-forecaster")
    parser.add_argument("--run_name",      default="arima-test")
    parser.add_argument("--no_wandb",      action="store_true")
    args = parser.parse_args()
    main(args)
