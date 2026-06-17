"""
arima_preprocess.py

Builds a sliding-window dataset of 42 consecutive weekly prices for every
DOW-30 stock from the FinGPT-Forecaster corpus (2023-05-07 onward).

Each row:
  symbol        str    stock ticker
  sequence      list   42 consecutive weekly prices, oldest first
  window_start  str    ISO date of sequence[0]
  window_end    str    ISO date of sequence[-1]

Price extraction from FinGPT prompts
──────────────────────────────────────
Every prompt contains one or more price lines of the form:
  "From 2023-05-07 to 2023-05-14, AXP's stock price decreased from 150.55 to 145.90"

All such lines in every prompt are parsed (not just the first).  For each line
BOTH the start price (150.55 @ 2023-05-07) and the end price (145.90 @ 2023-05-14)
are recorded, so the corpus-start price (2023-05-07) is captured from the very
first prompt.

Consecutive prompts overlap by one week, so the same date appears multiple
times across prompts — duplicates are silently dropped; conflicting values for
the same date are logged and the first value seen is kept.

Any 42-price window that contains a gap (> 9 days between successive dates) is
skipped and reported at the end.

Usage:
  python arima_preprocess.py --fingpt_dataset dow30-2023
  python arima_preprocess.py                               # yfinance fallback
"""

import re
import logging
import argparse
import warnings
from collections import defaultdict
from datetime import datetime

import pandas as pd
import datasets
from datasets import Dataset

from indices import DOW_30
from utils import load_dataset as load_fingpt_dataset

WINDOW_SIZE  = 42
CORPUS_START = "2023-05-07"

_FETCH_START = (
    pd.Timestamp(CORPUS_START) - pd.DateOffset(weeks=WINDOW_SIZE + 8)
).strftime("%Y-%m-%d")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# "From 2023-05-07 to 2023-05-14, AXP's stock price decreased from 150.55 to 145.90"
_WEEK_PRICE_RE = re.compile(
    r"From\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2}),\s+"
    r"(\w+)'s stock price \w+ from ([\d,]+\.?\d*) to ([\d,]+\.?\d*)",
    re.IGNORECASE,
)


def extract_prices_from_fingpt(dataset_name: str) -> dict:
    """
    Load a FinGPT dataset and extract per-ticker weekly price series.

    Scans ALL price lines in every prompt via finditer (not just the first).
    Records BOTH p1 @ week_start and p2 @ week_end so the very first corpus
    date (2023-05-07) is captured from the first prompt's start price.
    Deduplicates by date (first-seen wins); logs value conflicts.

    Returns
    -------
    dict  ticker -> sorted list of (date_str, price)
    """
    dataset_list = load_fingpt_dataset(dataset_name, from_remote=True)

    by_ticker: dict = defaultdict(dict)   # ticker -> {date_str: price}
    conflicts: dict = defaultdict(list)   # ticker -> [(date, kept, seen)]

    for ds in dataset_list:
        for split_name in ("train", "test"):
            if split_name not in ds:
                continue
            for row in ds[split_name]:
                for m in _WEEK_PRICE_RE.finditer(row["prompt"]):
                    ticker   = m.group(3).upper()
                    wk_start = m.group(1)
                    wk_end   = m.group(2)
                    p1       = float(m.group(4).replace(",", ""))
                    p2       = float(m.group(5).replace(",", ""))

                    for date, price in ((wk_start, p1), (wk_end, p2)):
                        existing = by_ticker[ticker].get(date)
                        if existing is None:
                            by_ticker[ticker][date] = price
                        elif abs(existing - price) > 0.01:
                            conflicts[ticker].append((date, existing, price))

    for ticker, confs in conflicts.items():
        for date, kept, seen in confs:
            log.warning(
                "  [CONFLICT] %s %s: keeping %.4f, ignoring %.4f",
                ticker, date, kept, seen,
            )
    if conflicts:
        total = sum(len(v) for v in conflicts.values())
        log.warning("  %d date-price conflicts total (first-seen kept)", total)

    result = {}
    for ticker, date_price in by_ticker.items():
        result[ticker] = sorted(date_price.items(), key=lambda x: x[0])
    return result


def fetch_weekly_prices(ticker: str, start: str, end: str) -> list:
    """Return sorted (date_str, price) pairs from yfinance weekly adjusted closes."""
    import yfinance as yf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            ticker, start=start, end=end,
            progress=False, auto_adjust=True, actions=False,
        )
    if raw.empty:
        return []

    closes = raw["Close"]
    if isinstance(closes, pd.DataFrame):
        closes = closes.squeeze()

    weekly = closes.resample("W-FRI").last().dropna()
    return [(str(d.date()), float(p)) for d, p in zip(weekly.index, weekly.values)]


def build_windows(ticker: str, series: list, skipped: list) -> list:
    """
    Slide a 42-price window over `series` (sorted (date_str, price) pairs).

    Any window that contains a gap (> 9 days between consecutive dates) is
    skipped; a record is appended to `skipped` for end-of-run logging.

    Returns list of dicts with keys: symbol, sequence, window_start, window_end.
    """
    dates  = [d for d, _ in series]
    prices = [p for _, p in series]
    rows   = []

    for i in range(len(prices) - WINDOW_SIZE + 1):
        wdates  = dates[i : i + WINDOW_SIZE]
        wprices = prices[i : i + WINDOW_SIZE]

        gap = None
        for j in range(1, WINDOW_SIZE):
            delta = (pd.Timestamp(wdates[j]) - pd.Timestamp(wdates[j - 1])).days
            if delta > 9:
                gap = (wdates[j - 1], wdates[j], delta)
                break

        if gap:
            skipped.append({
                "ticker":       ticker,
                "window_start": wdates[0],
                "window_end":   wdates[-1],
                "gap_from":     gap[0],
                "gap_to":       gap[1],
                "gap_days":     gap[2],
            })
            continue

        rows.append({
            "symbol":       ticker,
            "sequence":     wprices,
            "window_start": wdates[0],
            "window_end":   wdates[-1],
        })

    return rows


def main(args):
    all_rows = []
    skipped  = []

    fingpt_prices = {}
    if args.fingpt_dataset:
        print(f"Extracting prices from FinGPT dataset: {args.fingpt_dataset}")
        fingpt_prices = extract_prices_from_fingpt(args.fingpt_dataset)
        print(f"  Found price series for {len(fingpt_prices)} tickers\n")

    for ticker in DOW_30:
        print(f"  {ticker:<6} ...", end=" ", flush=True)

        if ticker in fingpt_prices:
            series = fingpt_prices[ticker]
        else:
            series = fetch_weekly_prices(ticker, _FETCH_START, args.end_date)

        if len(series) < WINDOW_SIZE:
            print(f"SKIP  (only {len(series)} prices, need {WINDOW_SIZE})")
            continue

        src  = "fingpt" if ticker in fingpt_prices else "yfinance"
        rows = build_windows(ticker, series, skipped)
        all_rows.extend(rows)
        print(f"{len(rows)} windows  [{src}]  ({len(series)} prices)")

    if not all_rows:
        raise RuntimeError("No rows produced — check tickers and date range.")

    if skipped:
        print(f"\nSkipped {len(skipped)} windows due to gaps:")
        for s in skipped:
            print(
                f"  {s['ticker']:<6}  {s['window_start']} → {s['window_end']}  "
                f"gap: {s['gap_from']} → {s['gap_to']} ({s['gap_days']} days)"
            )

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["window_start", "symbol"]).reset_index(drop=True)

    print(f"\nTotal windows : {len(df)}")
    print(f"Tickers       : {df['symbol'].nunique()}")
    print(f"Date range    : {df['window_start'].min()}  →  {df['window_end'].max()}")

    ds = Dataset.from_pandas(df, preserve_index=False)
    ds.save_to_disk(args.output_dir)
    print(f"Dataset saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build 42-week price windows for DOW-30")
    parser.add_argument("--fingpt_dataset", default=None, type=str,
                        help="FinGPT dataset name or local path; omit to use yfinance")
    parser.add_argument("--end_date", default=datetime.today().strftime("%Y-%m-%d"),
                        help="Last date for yfinance fetch (ignored when --fingpt_dataset set)")
    parser.add_argument("--output_dir", default="data/arima_dow30_42w",
                        help="Where to save the HuggingFace Dataset")
    args = parser.parse_args()
    main(args)
