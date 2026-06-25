
# FinGPT Forecaster — Reproducibility & Extensions (June 2026)

> **TL;DR** FinGPT fine-tuned on Llama-3 achieves 0.61 directional accuracy on DOW30 forecasting — a 34% relative improvement over GPT-4 and competitive with ARIMA on MSE — while operating purely from natural language inputs. This repo reproduces the original results, adds chain-of-thought (CoT) training, and ablates the contribution of financial language vs. numerical signals.

---

## Motivation

Two questions drive this work:

1. **Does FinGPT Forecaster capture genuine predictive signal, or is it a proxy for sentiment analysis?**
2. **What does the model actually rely on — financial language, or numerical price/volume data?**

These questions matter because LLMs process numbers as subword tokens, not as numeric values, which fundamentally limits their ability to reason over magnitudes. Understanding where the signal comes from informs both model design and the interpretation of evaluation metrics. For a deeper treatment of tokenization and numerical representation, see [this blog post](https://korra141.github.io/#projects).

---

## Background: Tokenization and Numerical Representation

Numbers are not first-class citizens in subword tokenizers. The same integer can be split differently depending on its value, surrounding context, and tokenizer vocabulary:

| Number | FinBERT (WordPiece) | LLaMA 2 (BPE/SentencePiece) | GPT-4 (tiktoken cl100k) |
|--------|--------------------|-----------------------------|--------------------------|
| `100` | `['100']` | `['100']` | `['100']` |
| `1024` | `['102', '##4']` | `['1024']` | `['1024']` |
| `0.08` | `['0', '.', '08']` | `['0', '.', '08']` | `['0.08']` |
| `142.50` | `['142', '.', '50']` | `['142', '.', '50']` | `['142.50']` |

This fragmentation means magnitude arithmetic (e.g. computing percentage change) is not directly grounded in the token representations — the model must learn it implicitly from co-occurrence statistics.

---

## Dataset

Experiments use the [FinGPT DOW30 Forecaster dataset](https://huggingface.co/datasets/FinGPT/fingpt-forecaster-dow30-202305-202405) (May 2023 – April 2024, ~42 weeks of DOW30 constituents). Each sample contains:

- **Prompt**: stock fundamentals (market cap, sector), prior-week price trend, and recent news headlines
- **Target**: structured prediction with three sections — `[Positive Developments]`, `[Potential Concerns]`, and `[Prediction & Analysis]`, the last including a directional call and an estimated percentage move

Evaluation combines three metrics:
- **Directional Accuracy** — binary classification accuracy on predicted price direction (up/down)
- **MSE** — mean squared error on the predicted percentage price change magnitude
- **ROUGE** — text similarity (ROUGE-1/2/L average) between generated and reference `[Positive Developments]`, `[Potential Concerns]`, and `[Analysis]` sections

---

## Implementations

Beyond reproducing the [original paper](https://arxiv.org/pdf/2306.06031), this repo adds:

- **BERTScore evaluation** alongside ROUGE for richer text quality assessment
- **GPT-4 inference** baseline with structured output prompting
- **ARIMA dataset creation** — 42-week rolling forecasts on DOW30 price series
- **XGBoost training** on engineered price/volume features
- **Linear and Logistic Regression** baselines
- **Chain-of-Thought (CoT) dataset generation** via GPT-4 distillation
- **CoT LoRA fine-tuning** with a two-phase inference pipeline (CoT reasoning → structured answer)
- **Unlikelihood penalty** on CoT token repetitions during training to reduce degenerate loops
- **Ablations**: masking financial terms, masking numerical values, token length sensitivity
- **Data analysis scripts** for distribution, coverage, and format validation

### Two-Phase CoT Inference

The CoT inference pipeline separates reasoning from answering:

1. **Phase 1** — the model generates free-form chain-of-thought reasoning (up to `--cot_budget` tokens), prompted with a reasoning instruction injected before `[/INST]`
2. **Phase 2** — if the model does not self-terminate with the `assistantfinal` marker, it is forced closed and the model generates the structured `[Positive Developments] / [Potential Concerns] / [Prediction & Analysis]` answer from the completed reasoning context

This decouples the reasoning budget from the answer format, keeping the structured output clean for parsing regardless of CoT length.

### Reproducibility Notes

- LoRA fine-tuning is functional (`train_lora.py`, `train_lora_cot.py`, `train.sh`)
- Inference scripts available for standard and CoT pipelines (`run_inference.py`, `run_inference_cot.py`)
- Reinforcement learning from stock price signal (described in the original paper) is out of scope for this study

---

## Results

Evaluated on the [DOW30 held-out test split](https://huggingface.co/datasets/FinGPT/fingpt-forecaster-dow30-202305-202405).

### Summary Comparison

| Model | Dir. Acc. | MSE | ROUGE |
|---|---|---|---|
| FinGPT (Llama-3) | 0.6122 | 7.2653 | 0.2467 |
| FinGPT (Llama-2) | 0.5102 | 9.7142 | 0.2425 |
| Llama-3 (base) | 0.4568 | 19.9748 | 0.2387 |
| Llama-2 (base) | 0.4201 | 28.4471 | 0.2023 |
| GPT-4 | 0.3506 | 24.5682 | 0.1674 |
| FinBERT | 0.4107 | 17.9348 | --- |
| ARIMA | 0.5111 | 8.2926 | --- |
| XGBoost | 0.4782 | 8.8607 | --- |
| Linear Regression | 0.4600 | 7.4170 | --- |
| Driftless Random Walk | --- | 7.1150 | --- |
| Class Distribution | 0.5040 | --- | --- |

**Key observations:**

- FinGPT (Llama-3) achieves the highest directional accuracy (0.61), a **34% relative improvement** over GPT-4 (0.35) and **20% over the base Llama-3 model** (0.46), confirming that LoRA fine-tuning on domain data provides genuine signal beyond instruction following
- The MSE of 7.27 is within 0.15 points of the driftless random walk (7.12) — the theoretical lower bound for a zero-signal predictor — suggesting the model captures directional tendency without overfitting to magnitude
- ARIMA (0.51 Dir. Acc., 8.29 MSE) is competitive with FinGPT (Llama-2) on both metrics, highlighting that classical time-series baselines remain strong on short-horizon forecasting; the LLM advantage materialises with the stronger base model and language grounding
- GPT-4 underperforms all fine-tuned and most classical models on directional accuracy, consistent with the hypothesis that out-of-the-box instruction models treat this as a language generation task rather than a forecasting task

---

## Computation and Resources

Training was conducted on H100 nodes (4× H100 80 GB per node) with multi-node DDP via DeepSpeed ZeRO.

| Configuration | Training Time | Notes |
|---|---|---|
| LoRA rank 8, Llama-2-7B | 3:27 hours | 1 gpu, 4 cpus, 4096 token length , 350 steps |
| LoRA rank 8, Llama-3-8B | 2:01 hours | 4 gpu, 48 cpus, 8192 token length , 360 steps |
| LoRA rank 32 + CoT, Llama-2-7B |  4:00 hours | 12 gpus, 48 x 3 cpus, 5000 token length, 240 steps |

Base model footprint: Llama-2-7B in fp16 requires ~22 GB per GPU; with LoRA adapters (rank 32) and optimizer states, peak VRAM per GPU is approximately 15 GB.

---

## Hugging Face Artifacts

| Artifact | Link |
|---|---|
| FinGPT Llama-3 (fine-tuned) | https://huggingface.co/korra141/fingpt-forecaster-llama3-lora |
| FinGPT Llama-2 (fine-tuned) | https://huggingface.co/korra141/fingpt-forecaster-llama2-lora |
| FinGPT CoT Llama-2 | https://huggingface.co/korra141/fingpt-forecaster-llama2-COT-lora |
| DOW30 42-week ARIMA dataset | https://huggingface.co/datasets/korra141/fingpt-forecaster-dow30-sequential_2 |
| DOW30 GPT CoT dataset | https://huggingface.co/datasets/korra141/fingpt-dow30-cot-reasoning |
