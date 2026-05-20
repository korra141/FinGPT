# FinGPT — Reproducibility Notes (March 2026)

This document tracks the reproducibility status of the FinGPT codebase and studies finpgt financial forecasting.

The dataset looks like this 
\href{https://huggingface.co/datasets/FinGPT/fingpt-forecaster-dow30-202305-202405} 
The prompt carries information about the stock like market cap, the trend of last week and the news related to it and asks to predict the next two weeks outcome. The outcome is seen in prediction of the stock and the analysis of the stock, outline the positive developement and concerns.

The metrics are calculated with how close the prediction of text from LLM is to the reference text.

<explain the metrics>

---

## Implementations

### FinGPT Forecaster (`fingpt/FinGPT_Forecaster/`)
- Training with LoRA fine-tuning is functional (`train_lora.py`, `train.sh`)
- Inference script available (`run_inference.py`)
- RL Stock Price is not implemented as per the paper. 
- Chain of Thought has been implemented by this repo.
---

The results from their paper mainly talk about sentiment analysis, in this we explore the finanicial forecasting 

## Results

On dataset ([DOW30 at hugging face](https://huggingface.co/datasets/FinGPT/fingpt-forecaster-dow30-202305-202405))

  ### Summary Comparison

  | Model | Binary Accuracy | Pros ROUGE-L | Analysis ROUGE-L | Cons ROUGE-L |
  |---|---|---|---|---|
  | llama2 base | 42.04% | 21.68 | 20.53 | 18.49 |
  | FinGPT (llama2, finetuned) | 53.06% | 25.88 | 21.43 | 24.99 |
  | llama3 base | 42.04% | 21.68 | 20.53 | 18.49 |
  | FinGPT (llama3, finetuned) | 53.06% | 25.88 | 21.43 | 24.99 |
  | chatgpt | 42.04% | 21.68 | 20.53 | 18.49 |

 
   <explain the results>
  

  ---

  ### Baseline llama2 [wandb_link](https://wandb.ai/korra141/fingpt-forecaster/runs/x7fw8mh5)                                                                   
                                                                                          
  ### FinGPT (LORA finetuned llama2)                                                              [wandb_link](https://wandb.ai/korra141/fingpt-forecaster/runs/741zp8pi/overview?nw=nwuserriaarora333)                                                               
## Computation and Resources 

## Ablation with the dataset insights

Llama 2 and 3 with different token lengths

Decoder style why the token length matters

## Chain of Thought 

## RAG, the prompt is so detailed. 

## Research Investigation: Do LLMs Actually Understand Numbers?

### Motivation

<screenshots showing the problem with numbers> 

intuition converting everything to language representation token, numerical understanding.

The core pipeline of FinGPT (and models like FinBERT, GPT-4) treats financial forecasting as a text classification or text generation task. News headlines and summaries are fed in, and the model outputs a directional prediction (up/down/flat).

But financial text is saturated with numbers: price changes, EPS figures, revenue growth percentages, P/E ratios. A human analyst reads "earnings missed by $0.08" and immediately computes its significance relative to expectations. Does the model do the same — or does it recognize the *phrase pattern* "missed by" and ignore the magnitude entirely?

This matters practically: if the model is insensitive to the actual numbers, then predictions may be driven by linguistic framing rather than quantitative signal. A story that says "missed by $0.01" might get the same prediction as one that says "missed by $2.00".


---

### Background: Tokenization and Numerical Representation

Numbers are not first-class citizens in subword tokenizers. The same integer can be tokenized differently depending on its value, surrounding context, and the specific tokenizer vocabulary. For example:

| Number | FinBERT (WordPiece) | LLaMA 2 (BPE/SentencePiece) | GPT-4 (tiktoken cl100k) |
|--------|--------------------|-----------------------------|--------------------------|
| `100`  | `['100']`          | `['100']`                   | `['100']`                |
| `1024` | `['102', '##4']`   | `['1024']`                  | `['1024']`               |
| `0.08` | `['0', '.', '08']` | `['0', '.', '08']`          | `['0.08']`               |
| `142.50` | `['142', '.', '50']` | `['142', '.', '50']`    | `['142.50']`             |

Key observations:
- **No shared embedding space for numeric magnitude.** `1` and `1000` share no representational similarity despite being numerically close (relative to, say, `1` and `1000000`).
- **Inconsistent granularity.** Some tokenizers merge multi-digit numbers as single tokens; others split them. The split boundary is arbitrary from an arithmetic standpoint.
- **Decimal points are often separate tokens.** `0.08` as three tokens means the model must attend across token boundaries to understand it as a single quantity.
- **Positional encoding leaks in.** In transformer models, the position of each digit-token affects its representation, so `1` at position 4 is different from `1` at position 12.

This means LLMs have no inherent sense of numeric order, magnitude, or arithmetic — unless they have learned it implicitly from patterns in training data.

---
Another experiment is to remove all leading word from the number in the dataset to check the performance for base and finetune.

### Hypothesis

**H1 (Context Dominance):** LLM forecasting predictions are primarily driven by the *linguistic context surrounding numbers* (e.g., "missed", "beat", "declined", "surged") rather than the numeric values themselves. Replacing a number with a different one of the same sign but different magnitude will not significantly change model output.

**H2 (Threshold Blindness):** Models are insensitive to quantitative thresholds. "Missed by $0.01" and "missed by $5.00" will produce similar sentiment/direction predictions because the framing word "missed" dominates.

**H3 (Tokenizer Effect):** Models using tokenizers with more consistent number tokenization (e.g., GPT-4's tiktoken, which often encodes common numbers as single tokens) will show *marginally* better numerical sensitivity than models using aggressive subword splitting, but the gap will be small relative to the context-dominance effect.

**H4 (Fine-tuning Doesn't Fix It):** LoRA fine-tuning on financial text (as done in FinGPT Forecaster) will improve domain-specific language understanding but will *not* improve numerical sensitivity, because the fine-tuning signal is still at the sentence/label level and doesn't teach the model to reason about magnitude.

---

### Tokenizer Comparison Across Baselines

| Model    | Tokenizer Type | Vocab Size | Number Handling Strategy |
|----------|---------------|------------|--------------------------|
| **FinBERT** | WordPiece (BERT-based) | ~30K | Aggressive subword split; numbers rarely appear as whole tokens unless in vocab |
| **LLaMA 2** | BPE via SentencePiece | ~32K | Common numbers (1–999) often single tokens; larger numbers split unpredictably |
| **LLaMA 3** | tiktoken (BPE) | ~128K | Larger vocab; more numbers represented as single tokens; closer to GPT-4 behavior |
| **GPT-4 (cl100k)** | tiktoken (BPE) | ~100K | Most consistent: many 4-digit numbers are single tokens; decimals still split |

The vocabulary size difference is meaningful: a 128K vocab (LLaMA 3) can afford to dedicate tokens to more number patterns than a 30K vocab (FinBERT), reducing fragmentation.

---

### Experiment Plan

#### Experiment 1: Magnitude Perturbation Test
**Goal:** Test H1 and H2 — does changing only the number in a sentence change the model's prediction?

**Method:**
1. Take a sample of test inputs from the FinGPT Forecaster dataset.
2. For each input, identify numeric values (prices, percentages, EPS figures).
3. Perturb the numbers: scale by 10x, 0.1x, negate, replace with 0, replace with extreme values.
4. Keep all surrounding text identical.
5. Run inference on original and all perturbed versions.
6. Measure: what fraction of predictions change? By how much?

**Expected result (if H1 holds):** Fewer than 20% of predictions change across perturbations.

**Implementation note:** Can be built on top of `run_inference.py` with a preprocessing step that applies regex-based number substitution to the `prompt` field before tokenization.

---

#### Experiment 2: Context Stripping Test
**Goal:** Measure how much of the prediction signal comes from *only* the numbers vs. *only* the surrounding words.

**Method:**
1. Create three versions of each test input:
   - **Original:** full text
   - **Numbers only:** replace all non-numeric tokens with `[MASK]` or a neutral filler
   - **Words only:** replace all numeric values with a neutral placeholder (e.g., `X`)
2. Run inference on all three versions and compare prediction distributions.

**Expected result:** "Words only" performance will be close to "Original"; "Numbers only" performance will be near-random.

---

#### Experiment 3: Tokenizer Fragmentation Audit
**Goal:** Quantify how differently each tokenizer handles the numeric content in the FinGPT dataset.

**Method:**
1. Extract all numeric strings from the dataset (regex: `\d+\.?\d*%?`).
2. Tokenize each with FinBERT's WordPiece, LLaMA 2's SentencePiece, LLaMA 3's tiktoken, and GPT-4's tiktoken.
3. Measure:
   - **Fragmentation rate:** % of numbers tokenized as 2+ tokens
   - **Vocabulary coverage:** % of unique numbers that appear as single tokens
   - **Consistency:** do numerically adjacent values (e.g., 1.00, 1.01, 1.02) tokenize similarly?

**Tooling:** `transformers.AutoTokenizer` for FinBERT/LLaMA; `tiktoken` library for GPT-4.

---

#### Experiment 4: Attention Attribution on Numbers
**Goal:** Directly measure how much attention the model places on numeric tokens during inference.

**Method:**
1. Run inference with `output_attentions=True` on the final layer(s).
2. For each input, identify which tokens are numeric.
3. Compute average attention weight on numeric tokens vs. non-numeric tokens (normalized by token count).
4. Compare across: base model vs. LoRA fine-tuned model.

**Expected result (if H4 holds):** Fine-tuning will not increase relative attention on numeric tokens.

**Caveat:** Attention weights are not a reliable proxy for "importance" in transformers (attention ≠ gradient-based saliency). Should be paired with gradient-based attribution (e.g., integrated gradients via `captum`) for stronger claims.

---

#### Experiment 5: Breaking the Correlation — Number Normalization Preprocessing
**Goal:** If H1-H2 hold, can we improve sensitivity by preprocessing numbers into a normalized form before tokenization?

**Approaches to test:**
- **Verbalization:** Convert `0.08` → `eight cents` or `$0.08 (small miss)`. Forces the model to use language patterns for magnitude.
- **Relative scaling:** Replace raw values with percentile ranks within the dataset (e.g., `top 10% earnings miss`). Strips absolute value, encodes relative significance.
- **Structured tagging:** Wrap numbers with semantic tags: `<EPS_MISS magnitude=small>0.08</EPS_MISS>`. Tests whether explicit structure helps.
- **Number-to-word expansion:** `142.50` → `one hundred forty two dollars and fifty cents`. Forces single-token representation per digit but loses compactness.

**Evaluation:** Compare FinGPT Forecaster accuracy (existing `calc_metrics`) before and after each preprocessing on the same test set.

---

### Success Criteria

| Hypothesis | Confirmed if | Disconfirmed if |
|------------|-------------|-----------------|
| H1: Context dominates | <20% prediction change on magnitude perturbation | >50% change |
| H2: Threshold blind | "missed by $0.01" ≈ "missed by $5.00" in output distribution | Statistically different outputs |
| H3: Tokenizer effect | <5% accuracy gap between FinBERT and LLaMA 3 on number-heavy inputs | >10% gap attributable to tokenization |
| H4: Fine-tuning doesn't fix it | Fine-tuned attention on numerics ≈ base model attention on numerics | Fine-tuned model shows >2x attention weight on numeric tokens |
