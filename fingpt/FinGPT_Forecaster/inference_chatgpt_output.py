
import argparse
import json
import os
import re
from collections import defaultdict
from sklearn.metrics import accuracy_score, mean_squared_error
import wandb
import datasets as hf_datasets
from utils import load_dataset, calc_rouge_score, parse_answer
import pdb

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default='inference_chatgpt_outputs.json', type=str)
    parser.add_argument("--run_name", default='inference-chatgpt', type=str)
    parser.add_argument("--wandb", action='store_true')
    args = parser.parse_args()

    wandb.init(project='fingpt-forecaster', name=args.run_name, config=vars(args))
    run_inference_chatgpt(args)

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

def run_inference_chatgpt(args):
    with open(args.output_path, 'w') as f:
        json.dump([
            {"prompt": p, "generated": g, "reference": r}
            for p, g, r in zip(prompt_records, generated_texts, reference_texts)
        ], f, indent=2)
    print(f"\nSaved raw outputs to {args.output_path}")

    print("\n=== Sample Outputs (first 3) ===")
    for i in range(min(3, len(generated_texts))):
        print(f"\n--- Sample {i+1} ---")
        print(f"[GENERATED]:\n{generated_texts[i][:500]}")
        print(f"[REFERENCE]:\n{reference_texts[i][:300]}")

    pdb.set_trace()  # Pause here to allow inspection of outputs before metrics calculation

    failure_log = f"{args.run_name}_parse_failures.json"
    print("\n=== Inference Metrics ===")
    metrics = calc_metrics_chatgpt(generated_texts, reference_texts, failure_log_path=failure_log)
    print(metrics)

    if args.wandb:
        wandb.log(metrics)
        if os.path.exists(args.output_path):
            artifact = wandb.Artifact(f"{args.run_name}-outputs", type="predictions")
            artifact.add_file(args.output_path)
            wandb.log_artifact(artifact)
        if os.path.exists(failure_log):
            artifact = wandb.Artifact(f"{args.run_name}-failures", type="logs")
            artifact.add_file(failure_log)
            wandb.log_artifact(artifact)
        wandb.finish()