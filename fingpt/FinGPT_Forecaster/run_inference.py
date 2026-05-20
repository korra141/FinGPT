from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import os
import re
import json
import argparse
import wandb
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, mean_squared_error
from utils import parse_model_name, load_dataset, calc_rouge_score
import datasets as hf_datasets


def parse_answer_base(answer):
    """
    Parse the base model's unstructured output format:

        Positive Developments:
        1. ...

        Potential Concerns:
        1. ...

        Prediction & Analysis:
        Based on ... we predict ... [single paragraph]
    """
    # Split on section headers, case-insensitive, brackets optional
    section_re = re.compile(
        r'\[?Positive Developments\]?\s*:',
        re.IGNORECASE
    )
    concerns_re = re.compile(r'\[?Potential Concerns\]?\s*:', re.IGNORECASE)
    pna_re = re.compile(r'\[?Prediction\s*[&and]+\s*Analysis\]?\s*:', re.IGNORECASE)
    note_re = re.compile(r'\n\s*Note\s*:', re.IGNORECASE)

    pos_match = section_re.search(answer)
    con_match = concerns_re.search(answer)
    pna_match = pna_re.search(answer)

    if not (pos_match and con_match and pna_match):
        return None

    pros = answer[pos_match.end():con_match.start()].strip()
    cons = answer[con_match.end():pna_match.start()].strip()
    pna_text = answer[pna_match.end():].strip()

    # Strip trailing Note: section
    note_match = note_re.search(pna_text)
    if note_match:
        pna_text = pna_text[:note_match.start()].strip()

    if not pna_text:
        return None

    # Direction from the combined paragraph
    pna_lower = pna_text.lower()
    if re.search(r'\bup\b|increase|rise|higher|outperform|bullish|upside', pna_lower):
        pred_bin = 1
    elif re.search(r'\bdown\b|decrease|decline|fall|lower|bearish|downside', pna_lower):
        pred_bin = -1
    else:
        pred_bin = 0

    # Percentage magnitude
    match_pct = re.search(r'(\d+)\s*-\s*(\d+)\s*%', pna_text)
    if match_pct:
        pred_margin = pred_bin * (int(match_pct.group(1)) + 0.5)
    else:
        match_pct = re.search(r'(?:more than\s+)?(\d+)\s*%', pna_text)
        pred_margin = pred_bin * (int(match_pct.group(1)) + 0.5) if match_pct else 0.0

    return {
        "positive developments": pros,
        "potential concerns": cons,
        "prediction": pred_margin,
        "prediction_binary": pred_bin,
        "analysis": pna_text,
    }


def calc_metrics_base(answers, gts):
    """calc_metrics for base-model output format; gts still use the bracketed format."""
    answers_dict = defaultdict(list)
    gts_dict = defaultdict(list)

    from utils import parse_answer  # GT answers use the original format

    for answer, gt in zip(answers, gts):
        answer_dict = parse_answer_base(answer)
        gt_dict = parse_answer(gt)

        if answer_dict and gt_dict:
            for k in answer_dict:
                answers_dict[k].append(answer_dict[k])
                gts_dict[k].append(gt_dict[k])

    total = len(answers)
    parsed = len(answers_dict['prediction'])
    print(f"\nParsed {parsed}/{total} samples successfully ({100*parsed/total:.1f}%)")

    if not answers_dict['prediction']:
        print("WARNING: No samples parsed — check if model output format matches parse_answer_base expectations.")
        return {}

    bin_acc = accuracy_score(gts_dict['prediction_binary'], answers_dict['prediction_binary'])
    mse = mean_squared_error(gts_dict['prediction'], answers_dict['prediction'])

    pros_rouge = calc_rouge_score(gts_dict['positive developments'], answers_dict['positive developments'])
    cons_rouge = calc_rouge_score(gts_dict['potential concerns'], answers_dict['potential concerns'])
    anal_rouge = calc_rouge_score(gts_dict['analysis'], answers_dict['analysis'])

    print(f"\nValid samples parsed: {len(answers_dict['prediction'])}/{len(answers)}")
    print(f"Binary Accuracy: {bin_acc:.2f}  |  Mean Square Error: {mse:.2f}")
    print(f"Rouge Score of Positive Developments: {pros_rouge}")
    print(f"Rouge Score of Potential Concerns: {cons_rouge}")
    print(f"Rouge Score of Summary Analysis: {anal_rouge}")

    return {
        "valid_count": len(answers_dict['prediction']),
        "bin_acc": bin_acc,
        "mse": mse,
        "pros_rouge_scores": pros_rouge,
        "cons_rouge_scores": cons_rouge,
        "anal_rouge_scores": anal_rouge,
    }


def run_inference(args):

    token = args.hf_token or os.environ.get('HF_TOKEN') or os.environ.get('HF_USER_ACCESS_TOKEN')

    model_name = parse_model_name(args.base_model, args.from_remote)

    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs = dict(quantization_config=bnb_config)
    elif args.load_in_8bit:
        model_kwargs = dict(load_in_8bit=True)
    else:
        model_kwargs = dict(torch_dtype=torch.float16)

    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        trust_remote_code=True,
        token=token,
        **model_kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.peft_model:
        print(f"Loading LoRA adapter: {args.peft_model}")
        model = PeftModel.from_pretrained(model, args.peft_model, token=token)

    model = model.eval()

    # Load test data
    dataset_list = load_dataset(args.dataset, args.from_remote)
    
    dataset_test = hf_datasets.concatenate_datasets([d['test'] for d in dataset_list])

    if args.num_samples > 0:
        dataset_test = dataset_test.shuffle(seed=42).select(range(min(args.num_samples, len(dataset_test))))

    print(f"Running inference on {len(dataset_test)} samples...")

    generated_texts, reference_texts, prompt_texts = [], [], []

    for feature in tqdm(dataset_test):
        prompt = feature['prompt']
        gt = feature['answer']

        inputs = tokenizer(
            prompt, return_tensors='pt',
            padding=False, max_length=args.max_length, truncation=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            res = model.generate(**inputs, max_new_tokens=512, use_cache=True)
        output = tokenizer.decode(res[0], skip_special_tokens=True)

        # Llama2 uses [/INST] as a special token; Llama3 may encode it differently.
        # Try [/INST] first, then fall back to stripping the prompt text directly.
        if '[/INST]' in output:
            answer = re.sub(r'.*\[/INST\]\s*', '', output, flags=re.DOTALL)
        else:
            # Strip the prompt prefix verbatim
            answer = output[len(prompt):].strip() if output.startswith(prompt) else output

        generated_texts.append(answer)
        reference_texts.append(gt)
        prompt_texts.append(prompt)

    # Save raw outputs for offline inspection
    output_path = f"{args.run_name}_outputs.json"
    with open(output_path, 'w') as f:
        json.dump([
            {"prompt": p, "generated": g, "reference": r}
            for p, g, r in zip(prompt_texts, generated_texts, reference_texts)
        ], f, indent=2)
    print(f"\nSaved raw outputs to {output_path}")

    # Log a few generated samples for quick sanity check
    print("\n=== Sample Outputs (first 3) ===")
    for i in range(min(3, len(generated_texts))):
        print(f"\n--- Sample {i+1} ---")
        print(f"[PROMPT END]:\n...{prompt_texts[i][-200:]}")
        print(f"[GENERATED]:\n{generated_texts[i][:500]}")
        print(f"[REFERENCE]:\n{reference_texts[i][:300]}")

    print("\n=== Inference Metrics ===")
    metrics = calc_metrics_base(generated_texts, reference_texts)
    print(metrics)

    if args.wandb:
        wandb.init(project='fingpt-forecaster', name=args.run_name)
        wandb.log(metrics)
        if output_path and os.path.exists(output_path):
            artifact = wandb.Artifact(f"{args.run_name}-outputs", type="predictions")
            artifact.add_file(output_path)
            wandb.log_artifact(artifact)
        wandb.finish()

    return metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default='inference', type=str)
    parser.add_argument("--base_model", required=True, type=str, choices=['chatglm2', 'llama2', 'llama3'])
    parser.add_argument("--peft_model", default=None, type=str, help="Path or HF repo of LoRA adapter (optional)")
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--max_length", default=4096, type=int)
    parser.add_argument("--num_samples", default=-1, type=int, help="Number of test samples to evaluate, -1 for all")
    parser.add_argument("--from_remote", default=True, type=bool)
    parser.add_argument("--hf_token", default=None, type=str, help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization (~4GB VRAM)")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit quantization (~7GB VRAM)")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to wandb")
    args = parser.parse_args()

    run_inference(args)
