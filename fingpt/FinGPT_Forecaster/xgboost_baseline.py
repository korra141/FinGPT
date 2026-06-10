"""
xgboost_baseline.py

XGBoost baseline for FinGPT-Forecaster.

Operates on the same FinGPT dataset format used by baseline_eval.py and
arima_baseline.py.  Extracts features from the prompt text and trains two
models:
  - XGBClassifier  → binary direction  (-1 / 0 / +1)
  - XGBRegressor   → price-change magnitude (%)

Features
--------
  Numerical (8):
    pct_change_last_week  (p2-p1)/p1*100  — prior-week return in the prompt
    price_level           log(p2)         — valuation regime proxy
    pos_word_count        bullish financial word count in full prompt
    neg_word_count        bearish financial word count in full prompt
    net_sentiment         pos - neg
    n_headlines           count of numbered news items in prompt
    month                 calendar month of week-start date
    week_of_year          ISO week number

  TF-IDF (--tfidf_features, default 200):
    Unigrams + bigrams from the headline text preceding the price line.
    Fitted on the train split only — no lookahead into test.

Metrics match baseline_eval.py: bin_acc, MSE (margin).

Usage
-----
  python xgboost_baseline.py --dataset dow30-2023 [--no_wandb]
"""

import re
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
import wandb
import datasets as hf_datasets

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    raise ImportError("xgboost is required — install with: pip install xgboost")

from utils import parse_answer, parse_answer_base, load_dataset


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# "From 2023-05-07 to 2023-05-14, AXP's stock price decreased from 150.55 to 145.90"
_PRICE_DATE_RE = re.compile(
    r"From\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2}),\s+"
    r"(\w+)'s stock price \w+ from ([\d,]+\.?\d*) to ([\d,]+\.?\d*)",
    re.IGNORECASE,
)

_POS_RE = re.compile(
    r'\b(?:up|higher|increas\w+|rise|risen|rising|rose|gain\w*|climb\w*|surg\w*|'
    r'rally\w*|soar\w*|jump\w*|rebound\w*|advanc\w*|expand\w*|'
    r'outperform\w*|bullish|upward|upside|beat\w*|exceed\w*|strengthen\w*|recover\w*)\b',
    re.IGNORECASE,
)

_NEG_RE = re.compile(
    r'\b(?:down|lower|decreas\w*|declin\w*|fall\w*|fell|fallen|loss\w*|los[et]\w*|'
    r'drop\w*|plung\w*|slid\w*|slip\w*|tumbl\w*|sink\w*|sank|sunk|'
    r'retreat\w*|contract\w*|underperform\w*|bearish|downward|downside|'
    r'miss\w*|weaken\w*)\b',
    re.IGNORECASE,
)

# Fixed order so train/test feature columns always align
_NUM_FEATURE_NAMES = [
    'month',
    'n_headlines',
    'neg_word_count',
    'net_sentiment',
    'pct_change_last_week',
    'pos_word_count',
    'price_level',
    'week_of_year',
]

# XGBoost multiclass needs 0-indexed integer labels
_LABEL_MAP  = {-1: 0, 0: 1, 1: 2}
_LABEL_RMAP = {0: -1, 1: 0, 2: 1}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_rows(dataset_split):
    """
    Parse each row into a numerical feature dict, headline text, bin label,
    and margin label.  Rows with unparseable GT or missing price line are skipped.

    Returns (num_rows, texts, bins, margins, n_skipped).
    """
    num_rows, texts, bins, margins = [], [], [], []
    n_skip = 0

    for row in dataset_split:
        parsed = parse_answer(row['answer']) or parse_answer_base(row['answer'])
        if not parsed or parsed['prediction'] is None:
            n_skip += 1
            continue

        m = _PRICE_DATE_RE.search(row['prompt'])
        if m is None:
            n_skip += 1
            continue

        p1 = float(m.group(4).replace(',', ''))
        p2 = float(m.group(5).replace(',', ''))
        dt = pd.Timestamp(m.group(1))

        pct_change  = (p2 - p1) / p1 * 100.0 if p1 > 0 else 0.0
        price_level = float(np.log(p2)) if p2 > 0 else 0.0

        full_text   = row['prompt']
        pos_count   = len(_POS_RE.findall(full_text))
        neg_count   = len(_NEG_RE.findall(full_text))
        n_headlines = len(re.findall(r'^\s*\d+\.', full_text, re.MULTILINE))

        num_rows.append({
            'pct_change_last_week': pct_change,
            'price_level':          price_level,
            'pos_word_count':       float(pos_count),
            'neg_word_count':       float(neg_count),
            'net_sentiment':        float(pos_count - neg_count),
            'n_headlines':          float(n_headlines),
            'month':                float(dt.month),
            'week_of_year':         float(int(dt.isocalendar().week)),
        })

        # Headlines = everything before the price line
        texts.append(full_text[:m.start()].strip())
        bins.append(parsed['prediction_binary'])
        margins.append(float(parsed['prediction']))

    return num_rows, texts, bins, margins, n_skip


def build_X(num_rows, texts, tfidf, fit=False):
    """Combine numerical and TF-IDF features into a single float32 array."""
    X_num = np.array(
        [[r[k] for k in _NUM_FEATURE_NAMES] for r in num_rows],
        dtype=np.float32,
    )
    X_tfidf = (tfidf.fit_transform(texts) if fit else tfidf.transform(texts)).toarray()
    return np.hstack([X_num, X_tfidf.astype(np.float32)])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(name, bin_acc, mse, n):
    print(f"\n{'='*52}")
    print(f"  {name}")
    print(f"{'='*52}")
    print(f"  N test          : {n}")
    print(f"  Binary accuracy : {bin_acc:.4f}")
    print(f"  MSE (margin %)  : {mse:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    dataset_list = load_dataset(args.dataset, True)
    train_split  = hf_datasets.concatenate_datasets([d['train'] for d in dataset_list])
    test_split   = hf_datasets.concatenate_datasets([d['test']  for d in dataset_list])

    print(f"Raw  — train: {len(train_split)}  test: {len(test_split)}")

    train_num, train_texts, train_bins, train_margins, n_skip_tr = extract_rows(train_split)
    test_num,  test_texts,  test_bins,  test_margins,  n_skip_te = extract_rows(test_split)

    print(f"Parsed — train: {len(train_bins)} (skipped {n_skip_tr})  "
          f"test: {len(test_bins)} (skipped {n_skip_te})")

    dist  = Counter(train_bins)
    total = sum(dist.values())
    print(f"\nTrain label distribution:")
    for cls in sorted(dist):
        label = {-1: "down (-1)", 0: "neutral (0)", 1: "up (+1)"}.get(cls, str(cls))
        print(f"  {label}: {dist[cls]}  ({100*dist[cls]/total:.1f}%)")

    # ---- Feature matrix -------------------------------------------------------
    tfidf = TfidfVectorizer(
        max_features=args.tfidf_features,
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
    )
    X_train = build_X(train_num, train_texts, tfidf, fit=True)
    X_test  = build_X(test_num,  test_texts,  tfidf, fit=False)

    n_num   = len(_NUM_FEATURE_NAMES)
    n_tfidf = X_train.shape[1] - n_num
    print(f"\nFeature matrix: {X_train.shape[1]} features "
          f"({n_num} numerical + {n_tfidf} TF-IDF)")

    y_train_cls = np.array([_LABEL_MAP[b] for b in train_bins], dtype=np.int32)
    y_test_cls  = np.array([_LABEL_MAP[b] for b in test_bins],  dtype=np.int32)
    y_train_reg = np.array(train_margins, dtype=np.float32)
    y_test_reg  = np.array(test_margins,  dtype=np.float32)

    # Per-class sample weights to handle class imbalance
    counts  = np.bincount(y_train_cls, minlength=3).astype(float)
    weights = len(y_train_cls) / (3.0 * np.where(counts > 0, counts, 1.0))
    sample_weights = np.array([weights[y] for y in y_train_cls], dtype=np.float32)

    # ---- Direction classifier -------------------------------------------------
    print("\nTraining XGBClassifier (direction) ...")
    clf = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    clf.fit(
        X_train, y_train_cls,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test_cls)],
        verbose=False,
    )

    pred_bins_raw   = clf.predict(X_test)
    pred_bins_named = np.array([_LABEL_RMAP[int(p)] for p in pred_bins_raw])
    bin_acc = accuracy_score(test_bins, pred_bins_named)

    # ---- Magnitude regressor --------------------------------------------------
    print("Training XGBRegressor (magnitude) ...")
    reg = XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        eval_metric='rmse',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    reg.fit(
        X_train, y_train_reg,
        eval_set=[(X_test, y_test_reg)],
        verbose=False,
    )

    pred_margins = reg.predict(X_test)
    mse = mean_squared_error(y_test_reg, pred_margins)

    report("XGBoost  (TF-IDF + price/sentiment features)", bin_acc, mse, len(test_bins))

    pred_dist = Counter(pred_bins_named)
    print(f"\nPredicted class distribution (test):")
    for cls in sorted(pred_dist):
        label = {-1: "down (-1)", 0: "neutral (0)", 1: "up (+1)"}.get(cls, str(cls))
        print(f"  {label}: {pred_dist[cls]}  ({100*pred_dist[cls]/len(test_bins):.1f}%)")

    # ---- Feature importance (top 10) ------------------------------------------
    tfidf_names  = [f'tfidf:{t}' for t in tfidf.get_feature_names_out()]
    all_feat_names = _NUM_FEATURE_NAMES + tfidf_names
    importances  = clf.feature_importances_
    top10        = np.argsort(importances)[::-1][:10]
    print(f"\nTop-10 classifier features by importance:")
    for idx in top10:
        if idx < len(all_feat_names):
            print(f"  {all_feat_names[idx]:<40}  {importances[idx]:.4f}")

    # ---- wandb ----------------------------------------------------------------
    if not args.no_wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={
                "dataset":        args.dataset,
                "n_estimators":   args.n_estimators,
                "max_depth":      args.max_depth,
                "lr":             args.lr,
                "tfidf_features": args.tfidf_features,
                "n_train":        len(train_bins),
                "n_test":         len(test_bins),
                "n_skip_train":   n_skip_tr,
                "n_skip_test":    n_skip_te,
            },
        )
        wandb.log({
            "xgboost/bin_acc":          bin_acc,
            "xgboost/mse":              mse,
            "xgboost/n_test":           len(test_bins),
            "class_dist/train_up":      dist.get(1,  0) / total,
            "class_dist/train_neutral": dist.get(0,  0) / total,
            "class_dist/train_down":    dist.get(-1, 0) / total,
        })

        feat_tbl = wandb.Table(columns=["feature", "importance"])
        for idx in np.argsort(importances)[::-1][:20]:
            if idx < len(all_feat_names):
                feat_tbl.add_data(all_feat_names[idx], float(importances[idx]))
        wandb.log({"xgboost/feature_importance": feat_tbl})

        run.finish()
        print(f"\nLogged to wandb run: {run.url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XGBoost baseline for FinGPT-Forecaster"
    )
    parser.add_argument(
        "--dataset", required=True, type=str,
        help="Dataset name/path — same format as baseline_eval.py (e.g. dow30-2023)",
    )
    parser.add_argument("--n_estimators",   default=300,  type=int,
                        help="Number of boosting rounds")
    parser.add_argument("--max_depth",      default=4,    type=int,
                        help="Max tree depth")
    parser.add_argument("--lr",             default=0.05, type=float,
                        help="Learning rate")
    parser.add_argument("--tfidf_features", default=200,  type=int,
                        help="Vocabulary size for TF-IDF")
    parser.add_argument("--wandb_project",  default="fingpt-forecaster")
    parser.add_argument("--run_name",       default="xgboost-baseline")
    parser.add_argument("--no_wandb",       action="store_true",
                        help="Skip wandb logging, print only")
    args = parser.parse_args()
    main(args)
