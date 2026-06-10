"""
Naive forecast baselines for FinGPT-Forecaster evaluation.

Two baselines:
  driftless  — driftless random walk: always predict 0 direction, 0% change
  always_up  — always predict up (+1 direction), magnitude = mean of positive gt margins
"""

import argparse
import datasets
import wandb
from collections import Counter
from sklearn.metrics import accuracy_score, mean_squared_error

from utils import parse_answer, parse_answer_base, load_dataset
import datasets as hf_datasets


def parse_gt_labels(dataset_split):
    gt_bins, gt_margins = [], []
    unparsed = 0
    for row in dataset_split:
        parsed = parse_answer(row['answer']) or parse_answer_base(row['answer'])
        if parsed and parsed['prediction'] is not None:
            gt_bins.append(parsed['prediction_binary'])
            gt_margins.append(parsed['prediction'])
        else:
            unparsed += 1
    return gt_bins, gt_margins, unparsed


def compute_metrics(gt_bins, gt_margins, pred_bins, pred_margins):
    return {
        "bin_acc": accuracy_score(gt_bins, pred_bins),
        "mse":     mean_squared_error(gt_margins, pred_margins),
    }


def report(name, metrics):
    print(f"\n{'='*50}")
    print(f"  Baseline: {name}")
    print(f"{'='*50}")
    print(f"  Binary accuracy : {metrics['bin_acc']:.4f}")
    print(f"  MSE (margin)    : {metrics['mse']:.4f}")


def main(args):
    # ds = datasets.load_from_disk(args.dataset)

    dataset_list = load_dataset(args.dataset, True)
    split = hf_datasets.concatenate_datasets([d['test'] for d in dataset_list])

    gt_bins, gt_margins, unparsed = parse_gt_labels(split)
    n = len(gt_bins)
    print(f"Parsed: {n}  |  Unparseable: {unparsed}")

    dist = Counter(gt_bins)
    total = sum(dist.values())
    print(f"\nGround-truth class distribution:")
    for cls, count in sorted(dist.items()):
        label = {-1: "down (-1)", 0: "neutral (0)", 1: "up (+1)"}.get(cls, str(cls))
        print(f"  {label}: {count}  ({100*count/total:.1f}%)")

    # --- Driftless random walk: predict no movement ---
    driftless_metrics = compute_metrics(
        gt_bins, gt_margins,
        [0] * n, [0.0] * n,
    )
    report("Driftless random walk (always 0)", driftless_metrics)

    # --- Always-up: predict +1, magnitude = mean positive gt margin ---
    pos_margins = [m for b, m in zip(gt_bins, gt_margins) if b == 1]
    mean_pos_margin = sum(pos_margins) / len(pos_margins) if pos_margins else 1.0
    print(f"\n  [always-up] mean positive gt margin used as predicted magnitude: {mean_pos_margin:.4f}%")
    always_up_metrics = compute_metrics(
        gt_bins, gt_margins,
        [1] * n, [mean_pos_margin] * n,
    )
    report("Always-up (+1, mean positive margin)", always_up_metrics)

    print()

    if not args.no_wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={
                "dataset": args.dataset,
                "split":   'test',
                "n_parsed": n,
                "n_unparsed": unparsed,
                "class_dist": dict(dist),
                "always_up_mean_pos_margin": mean_pos_margin,
            },
        )
        wandb.log({f"driftless/{k}": v for k, v in driftless_metrics.items()})
        wandb.log({f"always_up/{k}": v for k, v in always_up_metrics.items()})
        wandb.log({
            "class_dist/up":      dist.get(1,  0) / total,
            "class_dist/neutral": dist.get(0,  0) / total,
            "class_dist/down":    dist.get(-1, 0) / total,
        })
        run.finish()
        print(f"Logged to wandb run: {run.url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str,
                        help="Path to dataset saved with save_to_disk")
    parser.add_argument("--wandb_project", default="fingpt-forecaster", type=str)
    parser.add_argument("--run_name", default="baselines", type=str)
    parser.add_argument("--no_wandb", action="store_true",
                        help="Skip wandb logging, print only")
    args = parser.parse_args()
    main(args)
