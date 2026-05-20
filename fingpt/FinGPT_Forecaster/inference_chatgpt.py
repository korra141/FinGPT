from transformers import pipeline
from transformers.utils.import_utils import is_triton_available
from huggingface_hub import login
import torch
import os
import re
import json
import argparse
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, mean_squared_error
import datasets as hf_datasets
import wandb

from utils import load_dataset, calc_rouge_score, parse_answer

LLAMA2_SYS_RE = re.compile(
    r'\[INST\]<<SYS>>\n(.*?)\n<</SYS>>\n\n(.*?)\[/INST\]',
    re.DOTALL,
)
def extract_messages(raw_prompt: str):
    """Return (system_msg, user_msg) parsed from a Llama2-format dataset prompt."""
    m = LLAMA2_SYS_RE.search(raw_prompt)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, raw_prompt.strip()
def parse_answer(answer):
    
    match_res = re.match(r"^\s*\[Positive Developments\]:?\s*(.*)\s*\[Potential Concerns\]:?\s*(.*)\s*-*\s*\[Prediction (&|and) Analysis\]:?\s*(.*)\s*$", answer, flags=re.DOTALL)
    if not match_res:
        return None
    
    pros, cons, pna = match_res.group(1), match_res.group(2), match_res.group(4)
        
    match_res = re.match(r'^Prediction:\s*(.*)\s*Analysis:\s*(.*)\s*$', pna, flags=re.DOTALL)
    if not match_res:
        return None
        
    pred, anal = match_res.group(1), match_res.group(2)
        
    if re.search(r'up|increase', pred.lower()):
        pred_bin = 1
    elif re.search(r'down|decrease|decline', pred.lower()):
        pred_bin = -1
    else:
        pred_bin = 0
            
    cleaned = pred.replace('\u202f', '').replace('–', '-') 
    match_res = re.search(r'([\d.]+)%[–\-]([\d.]+)%', cleaned)
    if not match_res:
        match_res = re.search(r'(?:more than )?(\d)+?%', cleaned)    
        
    if match_res and match_res.lastindex == 2:
        pred_margin = pred_bin * (float(match_res.group(1)) + float(match_res.group(2))) / 2
    elif match_res:
        pred_margin = pred_bin * (float(match_res.group(1)) + 0.5)
    else:
        pred_margin = 0.
        
    return {
        "positive developments": pros,
        "potential concerns": cons,
        "prediction": pred_margin,
        "prediction_binary": pred_bin,
        "analysis": anal
    }


#**[Positive Developments]:**  \n1. **Growing Investor Interest** \u2013 Recent Zacks.com coverage and \u201cAmgen Inc. (AMGN) is Attracting Investor Attention\u201d suggest increasing analyst and retail focus on the stock.  \n2. **High Dividend Yield** \u2013 The headline \u201cCan This High\u2011Yield Dividend Stock Keep Beating the S&P 500?\u201d highlights Amgen\u2019s attractive dividend, appealing to income\u2011seeking investors.  \n3. **Solid Liquidity Position** \u2013 Cash ratio (0.60) and current ratio (1.65) indicate Amgen can comfortably meet short\u2011term obligations.  \n4. **Potential Undervaluation** \u2013 \u201cAmgen\u2019s shares have come under pressure this year, making it a compelling bargain buy\u201d implies that the market may be overlooking the company\u2019s fundamentals.\n\n**[Potential Concerns]:**  \n1. **Anticipated Earnings Decline** \u2013 Analysts predict a drop in earnings (\u201cAnalysts Estimate Amgen (AMGN) to Report a Decline in Earnings\u201d), raising doubts about near\u2011term profitability.  \n2. **High Leverage & Valuation** \u2013 Total debt\u2011to\u2011equity (10.37x) and a P/B of 24.7 suggest Amgen is heavily leveraged and possibly over\u2011valued.  \n3. **Low Free\u2011Cash\u2011Flow Margin** \u2013 FCF margin of only 3.53% limits the company\u2019s ability to fund dividends and reinvest without external financing.  \n4. **Sector\u2011Wide Earnings Pressure** \u2013 The article \u201cWill Healthcare ETFs Lose Momentum as Q1 Earnings Unfold?\u201d signals broader weakness in the healthcare space, which could weigh on Amgen\u2019s share price.\n\n---\n\n**[Prediction & Analysis]**\n\n**Prediction:**  \n- **Short\u2011term (2024\u201104\u201128 to 2024\u201105\u201105):** Slight **downward** movement of **\u2248\u202f0.5\u202f%\u20131.0\u202f%**.\n\n**Analysis:**  \n- **Upcoming Earnings**: Amgen\u2019s next earnings release is likely to be scrutinized heavily after the analyst forecast of a decline. Any miss will probably trigger a modest sell\u2011off.  \n- **Debt & Dividend Sustainability**: With an 8.6x net debt\u2011to\u2011equity and a 68\u202f% payout ratio, the company\u2019s leverage limits upside potential and could temper investor enthusiasm if cash flows don\u2019t improve.  \n- **Market Sentiment**: The positive headlines (dividend appeal, investor attention) provide a buying bias, but the sector\u2011wide caution and valuation concerns dominate the short\u2011term outlook.  \n- **Liquidity Cushion**: The healthy current and quick ratios should cushion the stock against a sharp drop, reinforcing the expectation of only a modest decline rather than a significant tumble.  \n\nOverall, the confluence of a likely earnings dip, high leverage, and sector pressure is expected to produce a small, temporary dip in Amgen\u2019s share price during the coming week, followed by a potential rebound if earnings beat expectations or if the broader healthcare sector stabilizes."

def parse_chatgpt_answer(answer: str):
    """
    Parse ChatGPT output.

    Handles two quirks:
    - Chain-of-thought preamble before the actual answer: use the LAST
      occurrence of the Positive Developments header so format-example
      mentions in the reasoning are skipped.
    - Bold markdown headers (**[Positive Developments]**) produced by some
      models: stripped to plain bracket form before delegating to parse_answer.
    """
    # Strip chain-of-thought: split at 'assistantfinal' if present
    cot_split = re.search(r'assistantfinal', answer, re.IGNORECASE)
    if cot_split:
        answer = answer[cot_split.end():]
    else:
        # Fallback: use last occurrence of the header to skip format-example in reasoning
        last_match = None
        for match in re.finditer(r'\[Positive Developments\]', answer, re.IGNORECASE):
            last_match = match
        if last_match:
            answer = answer[last_match.start():]

    # Normalise bold markdown headers → plain bracket form
    answer = re.sub(r'\*+\[([^\]]+)\]\*+', r'[\1]', answer)
    answer = re.sub(r'\*+', '', answer)


    return parse_answer(answer)


def log_parse_failures(answers, failure_log_path: str, max_log: int = 20):
    """
    Write unparseable outputs to a JSON file so the actual model template
    can be inspected and the regex updated for the next run.

    Each entry contains:
      - "raw": the full answer string
      - "first_200": quick preview
      - "headers_found": any bracket-style section headers detected
    """
    failures = []
    header_re = re.compile(r'\[[^\]]{3,40}\]\s*:', re.IGNORECASE)

    for i, answer in enumerate(answers):
        if parse_chatgpt_answer(answer) is None:
            failures.append({
                "index": i,
                "first_200": answer[:200],
                "raw": answer,
                "headers_found": header_re.findall(answer),
            })
            if len(failures) >= max_log:
                break

    if failures:
        with open(failure_log_path, 'w') as f:
            json.dump(failures, f, indent=2)
        print(f"\nLogged {len(failures)} parse failure(s) to {failure_log_path}")
        print("Inspect that file to identify the actual output template and update parse_chatgpt_answer.")
    else:
        print("\nAll outputs parsed successfully — no failure log written.")


def calc_metrics_chatgpt(answers, gts, failure_log_path: str = None):
    answers_dict = defaultdict(list)
    gts_dict = defaultdict(list)

    for answer, gt in zip(answers, gts):
        answer_dict = parse_chatgpt_answer(answer)
        gt_dict = parse_answer(gt)

        if answer_dict and gt_dict:
            for k in answer_dict:
                answers_dict[k].append(answer_dict[k])
                gts_dict[k].append(gt_dict[k])

    total = len(answers)
    parsed = len(answers_dict['prediction'])
    print(f"\nParsed {parsed}/{total} samples successfully ({100*parsed/total:.1f}%)")

    if failure_log_path and parsed < total:
        log_parse_failures(answers, failure_log_path)

    if not answers_dict['prediction']:
        print("WARNING: No samples parsed — check model output format.")
        return {}

    bin_acc = accuracy_score(gts_dict['prediction_binary'], answers_dict['prediction_binary'])
    mse = mean_squared_error(gts_dict['prediction'], answers_dict['prediction'])
    pros_rouge = calc_rouge_score(gts_dict['positive developments'], answers_dict['positive developments'])
    cons_rouge = calc_rouge_score(gts_dict['potential concerns'], answers_dict['potential concerns'])
    anal_rouge = calc_rouge_score(gts_dict['analysis'], answers_dict['analysis'])

    print(f"Binary Accuracy: {bin_acc:.2f}  |  Mean Square Error: {mse:.2f}")
    print(f"Rouge Score of Positive Developments: {pros_rouge}")
    print(f"Rouge Score of Potential Concerns: {cons_rouge}")
    print(f"Rouge Score of Summary Analysis: {anal_rouge}")

    return {
        "valid_count": parsed,
        "bin_acc": bin_acc,
        "mse": mse,
        "pros_rouge_scores": pros_rouge,
        "cons_rouge_scores": cons_rouge,
        "anal_rouge_scores": anal_rouge,
    }


def run_inference(args):
    print(f"GPU Detected: {torch.cuda.is_available()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"Triton Available: {is_triton_available()}")

    hf_token = args.hf_token or os.getenv('HF_TOKEN') or os.getenv('HF_USER_ACCESS_TOKEN')
    if hf_token:
        login(token=hf_token)

    print(f"Loading model: {args.model_id}")
    pipe = pipeline(
        "text-generation",
        model=args.model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    dataset_list = load_dataset(args.dataset, from_remote=True)
    dataset_test = hf_datasets.concatenate_datasets([d['test'] for d in dataset_list])

    if args.num_samples > 0:
        dataset_test = dataset_test.shuffle(seed=42).select(
            range(min(args.num_samples, len(dataset_test)))
        )

    print(f"Running inference on {len(dataset_test)} samples...")

    generated_texts, reference_texts, prompt_records = [], [], []

    for feature in tqdm(dataset_test):
        raw_prompt = feature['prompt']
        gt = feature['answer']

        system_msg, user_msg = extract_messages(raw_prompt)
        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": user_msg})

        outputs = pipe(messages, max_new_tokens=2048)
        # pipeline returns the full conversation; last entry is the assistant reply
        last = outputs[0]["generated_text"][-1]
        answer = last["content"] if isinstance(last, dict) else str(last)

        generated_texts.append(answer)
        reference_texts.append(gt)
        prompt_records.append({"system": system_msg, "user": user_msg})

    output_path = f"{args.run_name}_outputs.json"
    with open(output_path, 'w') as f:
        json.dump([
            {"prompt": p, "generated": g, "reference": r}
            for p, g, r in zip(prompt_records, generated_texts, reference_texts)
        ], f, indent=2)
    print(f"\nSaved raw outputs to {output_path}")

    print("\n=== Sample Outputs (first 3) ===")
    for i in range(min(3, len(generated_texts))):
        print(f"\n--- Sample {i+1} ---")
        print(f"[GENERATED]:\n{generated_texts[i][:500]}")
        print(f"[REFERENCE]:\n{reference_texts[i][:300]}")

    failure_log = f"{args.run_name}_parse_failures.json"
    print("\n=== Inference Metrics ===")
    metrics = calc_metrics_chatgpt(generated_texts, reference_texts, failure_log_path=failure_log)
    print(metrics)

    
    wandb.init(project='fingpt-forecaster', name=args.run_name, config=vars(args))
    wandb.log(metrics)
    if os.path.exists(output_path):
        artifact = wandb.Artifact(f"{args.run_name}-outputs", type="predictions")
        artifact.add_file(output_path)
        wandb.log_artifact(artifact)
    if os.path.exists(failure_log):
        artifact = wandb.Artifact(f"{args.run_name}-failures", type="logs")
        artifact.add_file(failure_log)
        wandb.log_artifact(artifact)
    wandb.finish()

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default='inference-chatgpt', type=str)
    parser.add_argument("--model_id", default='openai/gpt-oss-20b', type=str,
                        help="HuggingFace model ID")
    parser.add_argument("--dataset", default='dow30-202305-202405', type=str,
                        help="Dataset name passed to load_dataset (e.g. dow-30)")
    parser.add_argument("--num_samples", default=-1, type=int,
                        help="Number of test samples to evaluate (-1 for all)")
    parser.add_argument("--hf_token", default=None, type=str,
                        help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--wandb", action="store_true",
                        help="Log metrics and output artifacts to Weights & Biases")
    args = parser.parse_args()

    run_inference(args)
