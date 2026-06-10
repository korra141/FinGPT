"""
arima_preprocess.py

Builds a sliding-window dataset of 42 weekly Friday-close prices for every
DOW-30 stock, starting from 2023-05-07 (the FinGPT-Forecaster corpus start).

Each row:
  ticker        str    stock symbol
  prices        list   42 consecutive Friday-close prices, oldest first
  target_price  float  the following week's Friday close  (ARIMA forecast target)
  target_pct    float  (target_price - prices[-1]) / prices[-1] * 100
                       ← this is the ground truth the FinGPT answer expresses
  target_bin    int    +1 (pct > 0), -1 (pct ≤ 0)
  window_end    str    ISO date of prices[-1]
  target_date   str    ISO date of target_price

Price sources
─────────────
  --fingpt_dataset NAME  (recommended)
      Short HuggingFace dataset name, e.g. "dow30-2023".
      The load_dataset utility in utils.py prepends "FinGPT/fingpt-forecaster-"
      and calls datasets.load_dataset() — same pattern used by arima_baseline.py.
      Pass a local path instead if you have the dataset on disk (must be saved
      with save_to_disk, i.e. a DatasetDict directory).
      Prices are extracted from the embedded prompt text; the last training
      answer's realized bin gives price[43] = price[42] × (1 + mid_pct/100),
      yielding one extra sliding window (approximate ±0.5%).

  default (no flag)
      Downloads weekly adjusted closes from yfinance.

Preprocessed output (--output_dir)
───────────────────────────────────
  Saved as a HuggingFace DatasetDict via save_to_disk.  Contains two splits:
    train  — windows whose target_date < 80th-percentile date (chronological)
    test   — windows whose target_date ≥ 80th-percentile date
  Each row: ticker, prices (list[float] len=42), target_price, target_pct,
            target_bin (+1/-1), window_end (str), target_date (str).
  Load with: datasets.load_from_disk(output_dir)

Usage:
  python arima_preprocess.py --fingpt_dataset dow30-2023
  python arima_preprocess.py                               # yfinance fallback
"""

import re
import argparse
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import wandb
import yfinance as yf
import datasets
from datasets import Dataset

from indices import DOW_30
from utils import parse_answer, parse_answer_base
from utils import load_dataset as load_fingpt_dataset

WINDOW_SIZE   = 42
CORPUS_START  = "2023-05-07"   # first week in the FinGPT-Forecaster corpus

# Fetch far enough back so every window whose END is >= CORPUS_START is full.
_FETCH_START = (
    pd.Timestamp(CORPUS_START) - pd.DateOffset(weeks=WINDOW_SIZE + 8)
).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Regex patterns for parsing FinGPT prompts
# ---------------------------------------------------------------------------
# "From 2023-05-07 to 2023-05-14, AXP's stock price decreased from 150.55 to 145.90"
_WEEK_PRICE_RE = re.compile(
    r"From\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2}),\s+"
    r"(\w+)'s stock price \w+ from ([\d,]+\.?\d*) to ([\d,]+\.?\d*)",
    re.IGNORECASE,
)


def _parse_prompt_price(prompt: str):
    """
    Return (ticker, week_start, week_end, p1, p2) from the first price line
    in a FinGPT prompt, or None if the line is absent.
    week_end is the end-of-week price we use in the 42-price series.
    """
    m = _WEEK_PRICE_RE.search(prompt)
    if not m:
        return None
    return (
        m.group(3).upper(),                          # ticker
        m.group(1),                                  # week_start  (ISO str)
        m.group(2),                                  # week_end    (ISO str)
        float(m.group(4).replace(",", "")),          # p1
        float(m.group(5).replace(",", "")),          # p2  ← the price we store
    )


def _parse_answer_pct(answer: str):
    """Return the mid-point realized percentage from a FinGPT answer, or None."""
    parsed = parse_answer(answer) or parse_answer_base(answer)
    if parsed and parsed.get("prediction") is not None:
        return float(parsed["prediction"])
    return None


# ---------------------------------------------------------------------------
# Build weekly price series from a FinGPT dataset split
# ---------------------------------------------------------------------------

def extract_prices_from_fingpt(dataset_name: str) -> dict:
    """
    Load a FinGPT dataset (local path or HuggingFace short name) and extract
    per-ticker weekly price series from the embedded prompt text.

    Uses the same load_fingpt_dataset() utility as arima_baseline.py:
      - local path that exists on disk → load_from_disk
      - short name like "dow30-2023"   → FinGPT/fingpt-forecaster-dow30-2023
                                         loaded from the HuggingFace Hub

    Returns
    -------
    dict  ticker → list of (date_str, price) sorted chronologically

    The "+1 price" trick
    --------------------
    The last training prompt per ticker has a realized-return answer
    ("Up by 2-3%" → mid = +2.5%).  We back-calculate
        price[43] = price[42] × (1 + mid_pct / 100)
    and append it as an extra data point (approximate ±0.5%).
    """
    # load_fingpt_dataset returns a list of DatasetDicts (one per comma-separated
    # name).  We pass from_remote=True so it hits the HuggingFace Hub when the
    # name is not a local path — identical behaviour to arima_baseline.py.
    dataset_list = load_fingpt_dataset(dataset_name, from_remote=True)

    # Collect rows across all datasets and both splits.
    all_rows = []   # (ticker, week_end_str, p2, answer_or_None, split_name)

    for ds in dataset_list:
        for split_name in ("train", "test"):
            if split_name not in ds:
                continue
            for row in ds[split_name]:
                parsed = _parse_prompt_price(row["prompt"])
                if parsed is None:
                    continue
                ticker, _, week_end, _, p2 = parsed
                answer = row.get("answer", "") if split_name == "train" else None
                all_rows.append((ticker, week_end, p2, answer, split_name))

    by_ticker = defaultdict(list)
    for ticker, week_end, p2, answer, split in all_rows:
        by_ticker[ticker].append((week_end, p2, answer, split))

    result = {}
    for ticker, entries in by_ticker.items():
        entries.sort(key=lambda x: x[0])

        series = [(date, price) for date, price, _, _ in entries]

        # "+1 price": use the last TRAIN entry's answer to extend by one week
        train_entries = [(d, p, a) for d, p, a, s in entries if s == "train"]
        if train_entries:
            last_date, last_price, last_answer = train_entries[-1]
            pct = _parse_answer_pct(last_answer) if last_answer else None
            if pct is not None:
                p43 = last_price * (1.0 + pct / 100.0)
                p43_date = (pd.Timestamp(last_date) + pd.DateOffset(weeks=1)).strftime("%Y-%m-%d")
                # Only append if this date is not already present
                existing_dates = {d for d, _ in series}
                if p43_date not in existing_dates:
                    series.append((p43_date, p43))
                    series.sort(key=lambda x: x[0])

        result[ticker] = series

    return result


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def direction_bin(pct: float) -> int:
    return 1 if pct > 0 else -1


def fetch_weekly_prices(ticker: str, start: str, end: str) -> pd.Series:
    """
    Return adjusted Friday-close prices resampled to weekly.
    W-FRI: week ends on Friday; last() keeps the last trading day in each week
    so holiday weeks use Thursday or Wednesday close transparently.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            ticker, start=start, end=end,
            progress=False, auto_adjust=True, actions=False,
        )
    if raw.empty:
        return pd.Series(dtype=float, name=ticker)

    closes = raw["Close"]
    if isinstance(closes, pd.DataFrame):
        closes = closes.squeeze()

    weekly = closes.resample("W-FRI").last().dropna()
    return weekly


def build_windows(ticker: str, weekly: pd.Series) -> list:
    """
    Slide a 42-price window over the series.
    Only emit windows whose prices[-1] (window_end) >= CORPUS_START,
    so every prediction aligns with a week in the FinGPT corpus.
    """
    corpus_ts = pd.Timestamp(CORPUS_START)
    prices    = weekly.values.astype(float).tolist()
    dates     = weekly.index.tolist()
    rows      = []

    for i in range(WINDOW_SIZE, len(prices)):
        window_end_date = dates[i - 1]
        if window_end_date < corpus_ts:
            continue

        window = prices[i - WINDOW_SIZE : i]   # indices [i-42 .. i-1]  (42 items)
        target = prices[i]                      # index  i  (week after window)
        last   = window[-1]

        if last <= 0:
            continue

        pct = (target - last) / last * 100.0

        rows.append({
            "ticker":       ticker,
            "prices":       window,               # list[float], len=42
            "target_price": round(float(target), 4),
            "target_pct":   round(float(pct), 6),
            "target_bin":   direction_bin(pct),
            "window_end":   str(window_end_date.date()),
            "target_date":  str(dates[i].date()),
        })

    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(args):
    all_rows = []

    # ------------------------------------------------------------------
    # Price source: FinGPT prompts (preferred) or yfinance (fallback)
    # ------------------------------------------------------------------
    fingpt_prices = {}
    if args.fingpt_dataset:
        print(f"Extracting prices from FinGPT dataset: {args.fingpt_dataset}")
        fingpt_prices = extract_prices_from_fingpt(args.fingpt_dataset)
        print(f"  Found price series for {len(fingpt_prices)} tickers\n")

    for ticker in DOW_30:
        print(f"  {ticker:<6} ...", end=" ", flush=True)

        if ticker in fingpt_prices:
            # Build a pd.Series from the (date, price) list extracted from prompts
            entries = fingpt_prices[ticker]
            weekly = pd.Series(
                [p for _, p in entries],
                index=pd.to_datetime([d for d, _ in entries]),
                name=ticker,
            )
        else:
            weekly = fetch_weekly_prices(ticker, _FETCH_START, args.end_date)

        if len(weekly) < WINDOW_SIZE + 1:
            print(f"SKIP  (only {len(weekly)} weekly bars available)")
            continue

        rows = build_windows(ticker, weekly)
        all_rows.extend(rows)
        src = "fingpt" if ticker in fingpt_prices else "yfinance"
        print(f"{len(rows)} windows  [{src}]")

    if not all_rows:
        raise RuntimeError("No rows produced — check tickers and date range.")

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["target_date", "ticker"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Chronological 80/20 split — never shuffle; time order must be kept
    # so train always precedes test (no lookahead).
    # ------------------------------------------------------------------
    unique_dates = sorted(df["target_date"].unique())
    cutoff_idx   = int(0.8 * len(unique_dates))
    split_date   = unique_dates[cutoff_idx]

    train_df = df[df["target_date"] <  split_date].reset_index(drop=True)
    test_df  = df[df["target_date"] >= split_date].reset_index(drop=True)

    print(f"\nTotal windows : {len(df)}")
    print(f"Train (< {split_date})  : {len(train_df)}")
    print(f"Test  (≥ {split_date}) : {len(test_df)}")

    label_counts = df["target_bin"].value_counts().sort_index()
    print(f"\nLabel distribution:")
    for bin_val, count in label_counts.items():
        label = {1: "up (+1)", 0: "neutral (0)", -1: "down (-1)"}.get(bin_val, str(bin_val))
        print(f"  {label}: {count}  ({100*count/len(df):.1f}%)")

    ds = datasets.DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "test":  Dataset.from_pandas(test_df,  preserve_index=False),
    })

    ds.save_to_disk(args.output_dir)
    print(f"\nDataset saved to: {args.output_dir}")

    # ------------------------------------------------------------------
    # wandb logging
    # ------------------------------------------------------------------
    if not args.no_wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={
                "window_size":    WINDOW_SIZE,
                "corpus_start":   CORPUS_START,
                "price_source":   "fingpt" if args.fingpt_dataset else "yfinance",
                "fingpt_dataset": args.fingpt_dataset or "",
                "split_date":     split_date,
                "output_dir":     args.output_dir,
            },
        )

        wandb.log({
            "dataset/total_windows": len(df),
            "dataset/train_size":    len(train_df),
            "dataset/test_size":     len(test_df),
            "dataset/n_tickers":     df["ticker"].nunique(),
            "dataset/label_up_pct":  100 * (df["target_bin"] == 1).mean(),
            "dataset/label_dn_pct":  100 * (df["target_bin"] == -1).mean(),
        })

        # Per-ticker window counts as a table
        ticker_counts = (
            df.groupby("ticker")
              .agg(n_windows=("target_date", "count"),
                   n_train=("target_date", lambda x: (x < split_date).sum()),
                   n_test =("target_date", lambda x: (x >= split_date).sum()))
              .reset_index()
        )
        tbl = wandb.Table(
            columns=["ticker", "n_windows", "n_train", "n_test"],
            data=ticker_counts.values.tolist(),
        )
        wandb.log({"dataset/per_ticker": tbl})

        run.finish()
        print(f"Logged to wandb run: {run.url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build 42-week ARIMA windows for DOW-30")
    parser.add_argument(
        "--fingpt_dataset",
        default=None,
        type=str,
        help="Path to a FinGPT DatasetDict saved with save_to_disk. "
             "When provided, prices are extracted from the prompt text "
             "(with +1 price from the last training answer) instead of yfinance.",
    )
    parser.add_argument(
        "--end_date",
        default=datetime.today().strftime("%Y-%m-%d"),
        help="Last date for yfinance fetch — ignored when --fingpt_dataset is set",
    )
    parser.add_argument(
        "--output_dir",
        default="data/arima_dow30_42w",
        help="Where to save the HuggingFace DatasetDict",
    )
    parser.add_argument("--wandb_project", default="fingpt-forecaster")
    parser.add_argument("--run_name",      default="arima-preprocess")
    parser.add_argument("--no_wandb",      action="store_true")
    args = parser.parse_args()
    main(args)
