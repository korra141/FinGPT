"""
train_arima.py

Fits one ARIMA model per DOW-30 stock and evaluates directional + magnitude
accuracy against the FinGPT HF DOW-30 dataset answers.

Dataset schema (from arima_preprocess.py)
──────────────────────────────────────────
  symbol        str    stock ticker
  sequence      list   42 consecutive weekly prices, oldest first
  window_start  str    ISO date of sequence[0]
  window_end    str    ISO date of sequence[-1]

  Loaded as a flat Dataset (no built-in split).  A chronological 80/20 split
  is applied per ticker inside this script: early windows → train (p,q
  selection by AIC); later windows → test (evaluation against HF GT).

ARIMA fitting
─────────────
  d = 1 fixed (prices are I(1)).
  p, q selected per ticker by AIC grid-search on the first training window.
  Forecast is N-step autoregressive: ARIMA.forecast(N) feeds each predicted
  price back as input for the next step.  N = weeks(window_end → period_end).

Ground truth (test only, no fallback)
──────────────────────────────────────
  Loaded from --hf_dataset (test split).  Each row's 'answer' field is parsed
  with parse_answer / parse_answer_base to extract:
    gt_bin      +1 (up) / -1 (down)    ← prediction_binary
    gt_abs_pct  unsigned midpoint pct   ← abs(prediction), e.g. "Up 2-3%" → 2.5

  Matched by (symbol, period_start_date) where period_start ≈ window_end.
  Test windows with no matching HF GT row are skipped (logged as n_skipped).

MSE objective
─────────────
  pred_abs_pct = |forecast_price − last_price| / last_price × 100
  gt_abs_pct   = abs(parsed pct from HF answer)
  MSE          = mean((pred_abs_pct − gt_abs_pct)²)   [unsigned magnitude]

  Direction accuracy = mean(sign(forecast_price − last_price) == gt_bin)

Usage:
  python train_arima.py --dataset data/arima_dow30_42w \\
                        --hf_dataset dow30-202305-202405
"""

import re
import json
import argparse
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import wandb
import datasets
from sklearn.metrics import accuracy_score, mean_squared_error

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not installed — falling back to linear extrapolation.")

from indices import DOW_30
from utils import parse_answer, parse_answer_base, load_dataset as load_fingpt_dataset

WINDOW_SIZE  = 42
D_FIXED      = 1
P_MAX        = 3
Q_MAX        = 3
MAX_ORDER    = 4   # cap p+q to avoid overfitting on 41 differenced obs
TRAIN_RATIO  = 0.8


# ---------------------------------------------------------------------------
# ARIMA utilities
# ---------------------------------------------------------------------------

def select_pq(prices: list) -> tuple:
    """
    Grid-search ARIMA(p, 1, q) for p ∈ [0, P_MAX], q ∈ [0, Q_MAX],
    p+q ≤ MAX_ORDER.  Returns (best_p, best_q) by AIC.
    Falls back to (2, 1) if every fit fails.
    """
    if not HAS_STATSMODELS:
        return 2, 1

    best_aic = np.inf
    best_pq  = (2, 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in range(0, P_MAX + 1):
            for q in range(0, Q_MAX + 1):
                if p + q > MAX_ORDER or (p == 0 and q == 0):
                    continue
                try:
                    fit = ARIMA(prices, order=(p, D_FIXED, q)).fit()
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_pq  = (p, q)
                except Exception:
                    continue

    return best_pq


def arima_forecast_n(prices: list, p: int, q: int, steps: int = 1):
    """
    Fit ARIMA(p, 1, q) on `prices` and return an N-step autoregressive forecast.
    Each predicted value feeds the next step automatically via statsmodels.

    Returns (forecast_list, used_fallback).
    Fallback = linear extrapolation with drift when statsmodels is unavailable
    or the fit fails.
    """
    last  = prices[-1]
    drift = prices[-1] - prices[-2]

    if not HAS_STATSMODELS or len(prices) < p + D_FIXED + q + 2:
        return [last + drift * k for k in range(1, steps + 1)], True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit      = ARIMA(prices, order=(p, D_FIXED, q)).fit()
            forecast = fit.forecast(steps=steps).tolist()
            return forecast, False
        except Exception:
            return [last + drift * k for k in range(1, steps + 1)], True


def weeks_between(date_a: str, date_b: str) -> int:
    """Whole weeks from date_a to date_b (minimum 1)."""
    try:
        d_a = datetime.fromisoformat(date_a)
        d_b = datetime.fromisoformat(date_b)
        return max(1, round((d_b - d_a).days / 7))
    except Exception:
        return 1


# ---------------------------------------------------------------------------
# HF ground-truth loader
# ---------------------------------------------------------------------------

def _extract_period_dates(period: str):
    """
    Return (start_ISO, end_ISO) from a period field like "2024-02-02 to 2024-02-09".
    Handles ISO (YYYY-MM-DD) and compact (YYYYMMDD) forms.
    Returns (None, None) if neither pattern matches.
    """
    m = re.search(r'(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})', period)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r'(\d{8})\s+to\s+(\d{8})', period)
    if m:
        def fmt(s): return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return fmt(m.group(1)), fmt(m.group(2))
    return None, None


def build_hf_gt_lookup(hf_dataset_name: str, from_remote: bool = False) -> dict:
    """
    Load the HF DOW-30 forecaster dataset (test split only) and return:

        (symbol_upper, period_start_ISO) -> {
            "period_end":  str,    ISO date of the week being predicted
            "gt_bin":      int,    +1 up / -1 down
            "gt_abs_pct":  float,  unsigned midpoint pct, e.g. 2.5
        }

    period_start ≈ window_end in the ARIMA dataset, so this key lets us
    match each test window directly by (symbol, window_end).

    Only the test split is used — we do not evaluate on training windows.
    """
    try:
        dataset_list = load_fingpt_dataset(hf_dataset_name, from_remote)
    except Exception as exc:
        raise RuntimeError(f"Could not load HF dataset '{hf_dataset_name}': {exc}")

    # Collect only the test split from every dataset in the list
    test_splits = []
    for ds in dataset_list:
        if "test" in ds:
            test_splits.append(ds["test"])
        elif "train" in ds:
            # dataset had no test split — skip; train windows are not evaluated
            pass
    if not test_splits:
        raise RuntimeError("HF dataset has no test split — cannot build GT lookup.")

    combined = datasets.concatenate_datasets(test_splits)

    lookup   = {}
    n_parsed = 0

    n_skipped_neutral = 0

    for row in combined:
        answer = row.get("answer", "")
        parsed = parse_answer(answer) or parse_answer_base(answer)
        if not parsed or parsed.get("prediction") is None:
            continue

        # prediction_binary is set by parse_answer/parse_answer_base from words
        # like "increase/up" (+1) or "decrease/down" (-1) in the Prediction text.
        # Skip neutral (0) rows — direction cannot be determined.
        gt_bin = int(parsed.get("prediction_binary", 0))
        if gt_bin == 0:
            n_skipped_neutral += 1
            continue

        gt_abs_pct = abs(float(parsed["prediction"]))
        symbol     = row.get("symbol", "").upper().strip()
        period     = row.get("period", "")
        p_start, p_end = _extract_period_dates(period)

        if symbol and p_start and p_end:
            lookup[(symbol, p_start)] = {
                "period_end": p_end,
                "gt_bin":     gt_bin,
                "gt_abs_pct": gt_abs_pct,
            }
            n_parsed += 1

    n_symbols = len({k[0] for k in lookup})
    print(f"  HF GT lookup: {n_parsed} test rows for {n_symbols} symbols "
          f"({n_skipped_neutral} skipped — neutral/unparseable direction)")
    return lookup


# ---------------------------------------------------------------------------
# Per-ticker evaluation
# ---------------------------------------------------------------------------

def evaluate_ticker(ticker: str, train_rows: list, test_rows: list,
                    gt_lookup: dict) -> dict:
    """
    Training phase
    ──────────────
    Select ARIMA(p, 1, q) by AIC on the first training window's 42-price
    sequence.  ARIMA fits via MLE (L-BFGS-B internally) — no explicit n+1
    target is needed; the price at position n+1 is what the trained model
    will forecast.

    Inference phase (test windows only)
    ────────────────────────────────────
    For each test window:
      1. Match to HF GT by (symbol, window_end) → period_end, gt_bin, gt_abs_pct.
         Skip if no match.
      2. n_steps = weeks(window_end → period_end).
      3. Fit ARIMA(p, 1, q) on sequence; autoregressively forecast n_steps ahead.
      4. pred_abs_pct = |forecast[-1] − last| / last × 100.
      5. pred_bin     = +1 if forecast[-1] > last else -1.

    Metrics
    ───────
      mse      = mean((pred_abs_pct − gt_abs_pct)²)   [unsigned magnitude]
      bin_acc  = mean(pred_bin == gt_bin)              [direction]
    """
    if not train_rows:
        return {}

    p, q = select_pq(train_rows[0]["sequence"])

    pred_abs_pcts = []
    gt_abs_pcts   = []
    pred_bins     = []
    gt_bins       = []
    predictions   = []
    n_fallback    = 0
    n_skipped     = 0

    for row in test_rows:
        prices     = row["sequence"]
        last       = prices[-1]
        window_end = row["window_end"]
        symbol     = row["symbol"].upper()

        gt_entry = gt_lookup.get((symbol, window_end))
        if gt_entry is None:
            n_skipped += 1
            continue

        period_end = gt_entry["period_end"]
        gt_bin     = gt_entry["gt_bin"]
        gt_abs_pct = gt_entry["gt_abs_pct"]

        n_steps = weeks_between(window_end, period_end)
        forecasts, fallback = arima_forecast_n(prices, p, q, steps=n_steps)
        if fallback:
            n_fallback += 1

        forecast_price = forecasts[-1]
        pred_bin       = 1 if forecast_price > last else -1
        pred_abs_pct   = abs(forecast_price - last) / last * 100.0 if last != 0 else 0.0

        pred_abs_pcts.append(pred_abs_pct)
        gt_abs_pcts.append(gt_abs_pct)
        pred_bins.append(pred_bin)
        gt_bins.append(gt_bin)
        predictions.append({
            "window_end":     window_end,
            "period_end":     period_end,
            "last_price":     round(last, 4),
            "forecast_price": round(forecast_price, 4),
            "pred_abs_pct":   round(pred_abs_pct, 4),
            "gt_abs_pct":     round(gt_abs_pct, 4),
            "pred_bin":       pred_bin,
            "gt_bin":         gt_bin,
            "correct_dir":    int(pred_bin == gt_bin),
            "n_steps":        n_steps,
            "fallback":       fallback,
        })

    n = len(pred_abs_pcts)
    if n == 0:
        return {}

    mse     = mean_squared_error(gt_abs_pcts, pred_abs_pcts)
    bin_acc = accuracy_score(gt_bins, pred_bins)

    return {
        "ticker":      ticker,
        "arima_order": f"({p},{D_FIXED},{q})",
        "n_test":      n,
        "n_skipped":   n_skipped,
        "n_fallback":  n_fallback,
        "mse":         round(mse, 4),
        "bin_acc":     round(bin_acc, 4),
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(args):
    # ------------------------------------------------------------------
    # Load ARIMA windows (flat Dataset from arima_preprocess.py)
    # ------------------------------------------------------------------
    raw = datasets.load_from_disk(args.dataset)

    # Support both a flat Dataset and a DatasetDict (legacy)
    if isinstance(raw, datasets.DatasetDict):
        all_rows = list(raw["train"]) + list(raw["test"])
    else:
        all_rows = list(raw)

    # Group by ticker and sort chronologically
    by_ticker: dict = defaultdict(list)
    for row in all_rows:
        by_ticker[row["symbol"]].append(row)
    for rows in by_ticker.values():
        rows.sort(key=lambda r: r["window_start"])

    # Chronological 80/20 split per ticker
    train_by_ticker: dict = {}
    test_by_ticker: dict  = {}
    for ticker, rows in by_ticker.items():
        cut = max(1, int(len(rows) * TRAIN_RATIO))
        train_by_ticker[ticker] = rows[:cut]
        test_by_ticker[ticker]  = rows[cut:]

    # ------------------------------------------------------------------
    # Ground-truth lookup from HF test split (required)
    # ------------------------------------------------------------------
    print(f"Loading HF ground truth from: {args.hf_dataset}")
    gt_lookup = build_hf_gt_lookup(args.hf_dataset, from_remote=args.from_remote)

    # ------------------------------------------------------------------
    # Per-ticker ARIMA fit + evaluation
    # ------------------------------------------------------------------
    results = []

    for ticker in DOW_30:
        print(f"  {ticker:<6} ...", end=" ", flush=True)
        train_rows = train_by_ticker.get(ticker, [])
        test_rows  = test_by_ticker.get(ticker, [])

        if not test_rows:
            print("no test windows — skip")
            continue

        res = evaluate_ticker(ticker, train_rows, test_rows, gt_lookup)
        if not res:
            print("no matched GT rows — skip")
            continue

        results.append(res)
        print(
            f"order={res['arima_order']}  "
            f"bin_acc={res['bin_acc']:.3f}  "
            f"mse={res['mse']:.4f}  "
            f"n={res['n_test']}  skipped={res['n_skipped']}  fallback={res['n_fallback']}"
        )

    if not results:
        raise RuntimeError("No results produced — check dataset and HF GT paths.")

    total_n   = sum(r["n_test"] for r in results)
    macro_acc = sum(r["bin_acc"] * r["n_test"] for r in results) / total_n
    macro_mse = sum(r["mse"]     * r["n_test"] for r in results) / total_n

    print(f"\n{'='*60}")
    print(f"  ARIMA per-stock  (d={D_FIXED} fixed, p/q by AIC per stock)")
    print(f"{'='*60}")
    print(f"  Stocks evaluated : {len(results)} / {len(DOW_30)}")
    print(f"  Total test rows  : {total_n}")
    print(f"  Macro bin_acc    : {macro_acc:.4f}")
    print(f"  Macro MSE (|Δpct|): {macro_mse:.4f}")
    print()
    print(f"{'Ticker':<8} {'Order':<12} {'Acc':>6} {'MSE':>9} {'N':>5} {'Skip':>5}")
    print("-" * 50)
    for r in sorted(results, key=lambda x: -x["bin_acc"]):
        print(
            f"{r['ticker']:<8} {r['arima_order']:<12} "
            f"{r['bin_acc']:>6.3f} {r['mse']:>9.4f} "
            f"{r['n_test']:>5} {r['n_skipped']:>5}"
        )

    output = {
        "global": {
            "macro_bin_acc":  round(macro_acc, 4),
            "macro_mse":      round(macro_mse, 4),
            "total_n_test":   total_n,
            "gt_source":      "hf_dataset_answer_unsigned",
        },
        "per_ticker": results,
    }
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
                "dataset":     args.dataset,
                "hf_dataset":  args.hf_dataset,
                "d_fixed":     D_FIXED,
                "p_max":       P_MAX,
                "q_max":       Q_MAX,
                "max_order":   MAX_ORDER,
                "window_size": WINDOW_SIZE,
                "train_ratio": TRAIN_RATIO,
            },
        )
        wandb.log({
            "eval/macro_bin_acc": macro_acc,
            "eval/macro_mse":     macro_mse,
            "eval/n_stocks":      len(results),
            "eval/total_n_test":  total_n,
        })

        summary_tbl = wandb.Table(
            columns=["ticker", "arima_order", "bin_acc", "mse",
                     "n_test", "n_skipped", "n_fallback"]
        )
        for r in sorted(results, key=lambda x: -x["bin_acc"]):
            summary_tbl.add_data(
                r["ticker"], r["arima_order"], r["bin_acc"],
                r["mse"], r["n_test"], r["n_skipped"], r["n_fallback"],
            )
        wandb.log({"eval/per_ticker": summary_tbl})

        detail_tbl = wandb.Table(columns=[
            "ticker", "arima_order",
            "window_end", "period_end",
            "last_price", "forecast_price",
            "pred_abs_pct", "gt_abs_pct",
            "pred_bin", "gt_bin", "correct_dir",
            "n_steps", "fallback",
        ])
        for r in results:
            for p in r.get("predictions", []):
                detail_tbl.add_data(
                    r["ticker"], r["arima_order"],
                    p["window_end"], p["period_end"],
                    p["last_price"], p["forecast_price"],
                    p["pred_abs_pct"], p["gt_abs_pct"],
                    p["pred_bin"], p["gt_bin"], p["correct_dir"],
                    p["n_steps"], p["fallback"],
                )
        wandb.log({"eval/predictions_detail": detail_tbl})

        run.finish()
        print(f"Logged to wandb run: {run.url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit one ARIMA per DOW-30 stock and evaluate")
    parser.add_argument("--dataset", default="data/arima_dow30_42w",
                        help="Path to dataset saved by arima_preprocess.py")
    parser.add_argument("--hf_dataset", required=True,
                        help="FinGPT DOW-30 dataset for GT "
                             "(short name e.g. 'dow30-202305-202405', local path, or HF ID)")
    parser.add_argument("--from_remote", action="store_true",
                        help="Load --hf_dataset from the HuggingFace Hub")
    parser.add_argument("--output", default="results/arima_per_stock.json",
                        help="JSON file to write per-ticker and global results")
    parser.add_argument("--wandb_project", default="fingpt-forecaster")
    parser.add_argument("--run_name",      default="arima-train")
    parser.add_argument("--no_wandb",      action="store_true")
    args = parser.parse_args()
    main(args)
