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
from utils import (
    parse_model_name, load_dataset, calc_rouge_score, calc_bert_score,
    parse_answer_base, parse_answer, mask_numbers_in_prompt, mask_fin_words_in_prompt,
    randomize_numbers_in_prompt,
)
import datasets as hf_datasets


def calc_metrics_base(answers, gts, parsed_answers=None):
    """calc_metrics for base-model output format; gts use the original bracketed format."""
    answers_dict = defaultdict(list)
    gts_dict = defaultdict(list)
    mse_preds, mse_gts_list = [], []

    if parsed_answers is None:
        parsed_answers = [parse_answer_base(a) for a in answers]

    for answer_dict, gt in zip(parsed_answers, gts):
        gt_dict = parse_answer(gt)
        if answer_dict and gt_dict:
            for k in answer_dict:
                if k == 'prediction' and (answer_dict['prediction'] is None or gt_dict['prediction'] is None):
                    continue
                answers_dict[k].append(answer_dict[k])
                gts_dict[k].append(gt_dict[k])
            # Only include in MSE when both sides have a real numeric prediction
            # (prediction is None when [NUM] masking prevented extraction)
            if answer_dict['prediction'] is not None and gt_dict['prediction'] is not None:
                mse_preds.append(answer_dict['prediction'])
                mse_gts_list.append(gt_dict['prediction'])

    total = len(answers)
    parsed = len(answers_dict['prediction_binary'])
    print(f"\nParsed {parsed}/{total} samples successfully ({100*parsed/total:.1f}%)")

    if not parsed:
        print("WARNING: No samples parsed — check if model output format matches parse_answer_base expectations.")
        return {}

    bin_acc = accuracy_score(gts_dict['prediction_binary'], answers_dict['prediction_binary'])

    if mse_preds:
        mse = mean_squared_error(mse_gts_list, mse_preds)
        print(f"Binary Accuracy: {bin_acc:.2f}  |  MSE: {mse:.2f} ({len(mse_preds)}/{parsed} samples had numeric predictions)")
    else:
        mse = None
        print(f"Binary Accuracy: {bin_acc:.2f}  |  MSE: N/A (no numeric predictions — [NUM] masked?)")

    pros_rouge = calc_rouge_score(gts_dict['positive developments'], answers_dict['positive developments'])
    cons_rouge = calc_rouge_score(gts_dict['potential concerns'], answers_dict['potential concerns'])
    anal_rouge = calc_rouge_score(gts_dict['analysis'], answers_dict['analysis'])

    pros_bert = calc_bert_score(gts_dict['positive developments'], answers_dict['positive developments'])
    cons_bert = calc_bert_score(gts_dict['potential concerns'], answers_dict['potential concerns'])
    anal_bert = calc_bert_score(gts_dict['analysis'], answers_dict['analysis'])

    print(f"Rouge Score of Positive Developments: {pros_rouge}")
    print(f"Rouge Score of Potential Concerns: {cons_rouge}")
    print(f"Rouge Score of Summary Analysis: {anal_rouge}")
    print(f"BERTScore of Positive Developments: {pros_bert}")
    print(f"BERTScore of Potential Concerns: {cons_bert}")
    print(f"BERTScore of Summary Analysis: {anal_bert}")

    return {
        "valid_count": parsed,
        "bin_acc": bin_acc,
        "mse": mse,
        "pros_rouge_scores": pros_rouge,
        "cons_rouge_scores": cons_rouge,
        "anal_rouge_scores": anal_rouge,
        "pros_bert_scores": pros_bert,
        "cons_bert_scores": cons_bert,
        "anal_bert_scores": anal_bert,
    }


def run_inference(args):

    token = args.hf_token or os.environ.get('HF_TOKEN') or os.environ.get('HF_USER_ACCESS_TOKEN')

    model_name = parse_model_name('llama2', args.from_remote)

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
        model_kwargs = dict(torch_dtype=torch.float32)

    # Omitting "cpu" from max_memory prevents device_map="auto" from
    # offloading layers to CPU RAM — all layers stay on GPU.
    n_gpus = torch.cuda.device_count()
    max_memory = {i: "75GiB" for i in range(n_gpus)} if n_gpus > 0 else None

    print(f"Loading base model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        max_memory=max_memory,
        token=token,
        **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.peft_model:
        print(f"Loading LoRA adapter: {args.peft_model}")
        model = PeftModel.from_pretrained(base_model, args.peft_model, token=token)
    else:
        print("Running base model without PEFT adapter")
        model = base_model
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
        if args.mask_numbers:
            prompt = mask_numbers_in_prompt(prompt)
        if args.mask_fin_words:
            prompt = mask_fin_words_in_prompt(prompt)
        gt = feature['answer']

        inputs = tokenizer(
            prompt, return_tensors='pt',
            padding=False, max_length=args.max_length, truncation=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            res = model.generate(**inputs, max_new_tokens=512, use_cache=True)
        output = tokenizer.decode(res[0], skip_special_tokens=True)

        if '[/INST]' in output:
            answer = re.sub(r'.*\[/INST\]\s*', '', output, flags=re.DOTALL)
        else:
            answer = output[len(prompt):].strip() if output.startswith(prompt) else output

        generated_texts.append(answer)
        reference_texts.append(gt)
        prompt_texts.append(prompt)

    # Fine-tuned llama2 produces bracketed output ([Positive Developments]: ...);
    # base llama2 produces freetext (Positive Developments: ...).
    if args.peft_model:
        parsed_answers = [parse_answer(g) for g in generated_texts]
    else:
        parsed_answers = [parse_answer_base(g) for g in generated_texts]

    # Save raw outputs to disk
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.run_name}_outputs.json")
    with open(output_path, 'w') as f:
        json.dump([
            {"prompt": p, "generated": g, "parsed": parsed, "reference": r}
            for p, g, parsed, r in zip(prompt_texts, generated_texts, parsed_answers, reference_texts)
        ], f, indent=2)
    print(f"\nSaved raw outputs to {output_path}")

    unparsed_path = os.path.join(args.output_dir, f"{args.run_name}_unparsed.json")
    with open(unparsed_path, 'w') as f:
        json.dump(
            [{"prompt": p, "generated": g, "reference": r}
             for p, g, parsed, r in zip(prompt_texts, generated_texts, parsed_answers, reference_texts)
             if parsed is None],
            f, indent=2,
        )
    print(f"Saved unparsed outputs to {unparsed_path}")

    # Log a few generated samples for quick sanity check
    print("\n=== Sample Outputs (first 3) ===")
    for i in range(min(3, len(generated_texts))):
        print(f"\n--- Sample {i+1} ---")
        print(f"[PROMPT END]:\n...{prompt_texts[i][-200:]}")
        print(f"[GENERATED]:\n{generated_texts[i][:500]}")
        print(f"[REFERENCE]:\n{reference_texts[i][:300]}")

    print("\n=== Inference Metrics ===")
    metrics = calc_metrics_base(generated_texts, reference_texts, parsed_answers=parsed_answers)
    print(metrics)

    if args.wandb:
        run_config = dict(
            base_model='llama2',
            peft_model=args.peft_model,
            dataset=args.dataset,
            num_samples=len(dataset_test),
            max_length=args.max_length,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            mask_numbers=args.mask_numbers,
        mask_fin_words=args.mask_fin_words,
        )
        wandb.init(project='fingpt-forecaster', name=args.run_name, config=run_config)
        wandb.log(metrics)
        artifact = wandb.Artifact(f"{args.run_name}-outputs", type="predictions")
        artifact.add_file(output_path)
        wandb.log_artifact(artifact)
        wandb.finish()

    return metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default='inference', type=str)
    parser.add_argument(
        "--peft_model", default=None, type=str,
        help="HF repo ID or local path of a LoRA adapter (optional); omit to run the base llama2 model"
    )
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--max_length", default=4096, type=int)
    parser.add_argument("--num_samples", default=-1, type=int, help="Number of test samples (-1 for all)")
    parser.add_argument("--from_remote", default=True, type=bool)
    parser.add_argument("--output_dir", default="outputs", type=str, help="Directory to save generated outputs")
    parser.add_argument("--hf_token", default=None, type=str, help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization (~4GB VRAM)")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit quantization (~7GB VRAM)")
    parser.add_argument("--mask_numbers", action="store_true",
                        help="Replace all numbers (and surrounding ±$%%) in prompts with [NUM]")
    parser.add_argument("--mask_fin_words", action="store_true",
                        help="Replace financial directional/sentiment words in prompts with [WORD]")
    parser.add_argument("--wandb", action="store_true", help="Log metrics and outputs artifact to wandb")
    args = parser.parse_args()

    run_inference(args)
