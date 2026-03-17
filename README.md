# FinGPT — Reproducibility Notes (March 2026)

This document tracks the reproducibility status of the FinGPT codebase.

---

## Modules

### FinGPT Forecaster (`fingpt/FinGPT_Forecaster/`)
- Training with LoRA fine-tuning is functional (`train_lora.py`, `train.sh`)
- Inference script available (`run_inference.py`)

### FinGPT Trading (`fingpt/FinGPT_Others/FinGPT_Trading/`)
- ChatGPT-based trading bot exists but RL trading agent is **not implemented** (see below)

---

## Known Gaps vs. Paper

### Reinforcement Learning for Stock Price Prediction
**Status: Not reproducible — code not present**

The paper references reinforcement learning applied to stock price prediction/trading. No such implementation exists in this codebase. The trading module (`chatgpt-trading-v2/README.md`) has an explicit TODO to train a FinRL agent on GPT-generated sentiment scores, but it was never built.

The only RL-related code found is:
- RLHF reward modeling (`fingpt/FinGPT_RAG/`) — applies RL to LLM alignment, not stock trading
- PyTorch bundled RL examples inside Singularity container build artifacts — not project code

To reproduce the RL results from the paper, an integration with the [FinRL library](https://github.com/AI4Finance-Foundation/FinRL) would need to be built from scratch, using FinGPT sentiment scores as input features to a trading agent.

---

## Known Code Issues

- `--from_remote` argument uses `type=bool` (broken — `--from_remote False` still evaluates to `True`). Affects `train_lora.py` and `run_inference.py`.
- `wandb.init()` is called twice in `train_lora.py` (manually on line 84 and again via `report_to='wandb'` in `TrainingArguments`), causing metrics to not be logged correctly.
- HuggingFace token is hardcoded in `run_inference.py` — the fallback to env vars is dead code.
