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
    parse_answer_base, parse_answer,
)
import datasets as hf_datasets

COT_END = 'assistantfinal'

COT_TRIGGER = "Assistant: Let me reason through this step by step."

REASONING_INSTRUCTION = (
    "You are a financial analyst. Before giving a forecast, "
    "reason through the available signals step by step."
)


def inject_reasoning_instruction(prompt: str) -> str:
    """Insert REASONING_INSTRUCTION before [/INST] in a LLaMA-2 chat prompt."""
    marker = '[/INST]'
    idx = prompt.rfind(marker)
    if idx == -1:
        return prompt.rstrip() + '\n' + REASONING_INSTRUCTION
    return prompt[:idx].rstrip() + '\n' + REASONING_INSTRUCTION + '\n' + marker


def build_cot_content(cot_text: str) -> str:
    """Assemble the full CoT content string ending with COT_END."""
    return f"{COT_TRIGGER}\n{cot_text}\n{COT_END}"


def extract_cot_answer(text: str) -> str:
    """Strip CoT preamble and return only the final structured answer."""
    split = re.search(r'assistantfinal', text, re.IGNORECASE)
    if split:
        text = text[split.end():]
    else:
        last = None
        for m in re.finditer(r'\[Positive Developments\]', text, re.IGNORECASE):
            last = m
        if last:
            text = text[last.start():]
    text = re.sub(r'\*+\[([^\]]+)\]\*+', r'[\1]', text)
    return text.strip()


def extract_generated_cot(text: str) -> str:
    """Return the CoT reasoning the model generated (up to 'assistantfinal')."""
    end = re.search(r'assistantfinal', text, re.IGNORECASE)
    if end:
        return text[:end.start()].strip()
    fallback = re.search(r'\[Positive Developments\]', text, re.IGNORECASE)
    if fallback:
        return text[:fallback.start()].strip()
    return ''


def calc_metrics_cot(answers, gts, parsed_answers=None):
    answers_dict = defaultdict(list)
    gts_dict = defaultdict(list)
    mse_preds, mse_gts_list = [], []

    if parsed_answers is None:
        parsed_answers = [parse_answer(a) or parse_answer_base(a) for a in answers]

    for answer_dict, gt in zip(parsed_answers, gts):
        gt_dict = parse_answer(gt) or parse_answer_base(gt)
        if answer_dict and gt_dict:
            for k in answer_dict:
                if k == 'prediction' and (answer_dict['prediction'] is None or gt_dict['prediction'] is None):
                    continue
                answers_dict[k].append(answer_dict[k])
                gts_dict[k].append(gt_dict[k])
            if answer_dict['prediction'] is not None and gt_dict['prediction'] is not None:
                mse_preds.append(answer_dict['prediction'])
                mse_gts_list.append(gt_dict['prediction'])

    total = len(answers)
    parsed = len(answers_dict['prediction_binary'])
    print(f"\nParsed {parsed}/{total} samples successfully ({100*parsed/total:.1f}%)")

    if not parsed:
        print("WARNING: No samples parsed — check model output format.")
        return {}

    bin_acc = accuracy_score(gts_dict['prediction_binary'], answers_dict['prediction_binary'])

    if mse_preds:
        mse = mean_squared_error(mse_gts_list, mse_preds)
        print(f"Binary Accuracy: {bin_acc:.2f}  |  MSE: {mse:.2f} ({len(mse_preds)}/{parsed} samples had numeric predictions)")
    else:
        mse = None
        print(f"Binary Accuracy: {bin_acc:.2f}  |  MSE: N/A")

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


def run_inference_cot(args):

    token = args.hf_token or os.environ.get('HF_TOKEN') or os.environ.get('HF_USER_ACCESS_TOKEN')

    # model_name = parse_model_name('llama2', args.from_remote)
    model_name = "meta-llama/Llama-2-7b-hf"

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

    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        if args.max_memory_per_gpu:
            max_memory = {i: args.max_memory_per_gpu for i in range(n_gpus)}
        else:
            # auto-detect: leave ~2 GiB headroom per GPU
            free_mem = []
            for i in range(n_gpus):
                props = torch.cuda.get_device_properties(i)
                free_mem.append(f"{int(props.total_memory / 1024**3) - 2}GiB")
            max_memory = {i: free_mem[i] for i in range(n_gpus)}
        print(f"Using {n_gpus} GPU(s) with max_memory={max_memory}")
    else:
        max_memory = None

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
    tokenizer.padding_side = "left"  # left-pad so all generated tokens align at the same offset

    if args.peft_model:
        print(f"Loading LoRA adapter: {args.peft_model}")
        model = PeftModel.from_pretrained(base_model, args.peft_model, token=token)
    else:
        print("Running base model without PEFT adapter")
        model = base_model
    model = model.eval()
    input_device = next(model.parameters()).device

    dataset_list = load_dataset(args.dataset, args.from_remote)
    dataset_test = hf_datasets.concatenate_datasets([d['test'] for d in dataset_list])

    if args.num_samples > 0:
        dataset_test = dataset_test.shuffle(seed=42).select(range(min(args.num_samples, len(dataset_test))))

    print(f"Running inference on {len(dataset_test)} samples (batch_size={args.batch_size})...")

    generated_texts, cot_texts, reference_texts, prompt_texts = [], [], [], []

    samples = list(dataset_test)
    for batch_start in tqdm(range(0, len(samples), args.batch_size)):
        batch = samples[batch_start:batch_start + args.batch_size]
        raw_prompts = [s['prompt'] for s in batch]
        gts = [s['answer'] for s in batch]

        # Phase 1: generate CoT for all samples in the batch
        phase1_prompts = [
            inject_reasoning_instruction(p) + ' ' + COT_TRIGGER
            for p in raw_prompts
        ]
        enc1 = tokenizer(
            phase1_prompts, return_tensors='pt',
            padding=True, truncation=True, max_length=args.max_length,
        )
        in_len1 = enc1['input_ids'].shape[1]
        enc1 = {k: v.to(input_device) for k, v in enc1.items()}

        with torch.no_grad():
            out1 = model.generate(
                **enc1, use_cache=True,
                max_new_tokens=args.cot_budget,
                do_sample=False, temperature=None, top_p=None,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
            )

        # With left-padding, generated tokens start at in_len1 for every sample
        cot_texts_batch = []
        for i in range(len(batch)):
            cot_raw = tokenizer.decode(out1[i][in_len1:], skip_special_tokens=True).strip()
            for _stop in ('[INST]', '<<SYS>>', '<s>'):
                _idx = cot_raw.find(_stop)
                if _idx != -1:
                    cot_raw = cot_raw[:_idx].strip()
            cot_texts_batch.append(cot_raw)

        # Phase 2: only for samples where CoT didn't close itself
        needs_phase2 = [
            (i, phase1_prompts[i], cot_texts_batch[i])
            for i in range(len(batch))
            if COT_END.lower() not in cot_texts_batch[i].lower()
        ]

        phase2_answers = {}
        if needs_phase2:
            phase2_prompts = [
                p1 + ' ' + ct + '\n' + COT_END + '\n'
                for _, p1, ct in needs_phase2
            ]
            enc2 = tokenizer(
                phase2_prompts, return_tensors='pt',
                padding=True, truncation=True, max_length=args.max_length,
            )
            in_len2 = enc2['input_ids'].shape[1]
            enc2 = {k: v.to(input_device) for k, v in enc2.items()}

            with torch.no_grad():
                out2 = model.generate(
                    **enc2, use_cache=True,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False, temperature=None, top_p=None,
                    repetition_penalty=args.repetition_penalty,
                    pad_token_id=tokenizer.eos_token_id,
                )

            for j, (orig_i, _, _ct) in enumerate(needs_phase2):
                answer_raw = tokenizer.decode(out2[j][in_len2:], skip_special_tokens=True).strip()
                for _stop in ('[INST]', '<<SYS>>', '<s>'):
                    _idx = answer_raw.find(_stop)
                    if _idx != -1:
                        answer_raw = answer_raw[:_idx].strip()
                phase2_answers[orig_i] = answer_raw

        # Assemble results for this batch
        for i in range(len(batch)):
            ct = cot_texts_batch[i]
            if i in phase2_answers:
                raw_generation = ct + '\n' + COT_END + '\n' + phase2_answers[i]
            else:
                raw_generation = ct

            cot_part = extract_generated_cot(raw_generation)
            answer = extract_cot_answer(raw_generation)

            generated_texts.append(answer)
            cot_texts.append(cot_part)
            reference_texts.append(gts[i])
            prompt_texts.append(raw_prompts[i])

    # Try strict bracketed parser first, fall back to lenient
    parsed_answers = [parse_answer(g) or parse_answer_base(g) for g in generated_texts]

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.run_name}_outputs.json")
    with open(output_path, 'w') as f:
        json.dump([
            {"prompt": p, "cot_reasoning": cot, "generated": g, "parsed": parsed, "reference": r}
            for p, cot, g, parsed, r in zip(prompt_texts, cot_texts, generated_texts, parsed_answers, reference_texts)
        ], f, indent=2)
    print(f"\nSaved outputs to {output_path}")

    unparsed_path = os.path.join(args.output_dir, f"{args.run_name}_unparsed.json")
    with open(unparsed_path, 'w') as f:
        json.dump(
            [{"prompt": p, "cot_reasoning": cot, "generated": g, "reference": r}
             for p, cot, g, parsed, r in zip(prompt_texts, cot_texts, generated_texts, parsed_answers, reference_texts)
             if parsed is None],
            f, indent=2,
        )
    print(f"Saved unparsed outputs to {unparsed_path}")

    print("\n=== Sample Outputs (first 3) ===")
    for i in range(min(3, len(generated_texts))):
        print(f"\n--- Sample {i+1} ---")
        print(f"[PROMPT END]:\n...{prompt_texts[i][-200:]}")
        if cot_texts[i]:
            print(f"[COT REASONING]:\n{cot_texts[i][:300]}...")
        print(f"[GENERATED]:\n{generated_texts[i][:500]}")
        print(f"[REFERENCE]:\n{reference_texts[i][:300]}")

    print("\n=== Inference Metrics ===")
    metrics = calc_metrics_cot(generated_texts, reference_texts, parsed_answers=parsed_answers)
    print(metrics)

    if args.wandb:
        run_config = dict(
            base_model='llama2',
            peft_model=args.peft_model,
            dataset=args.dataset,
            num_samples=len(dataset_test),
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            batch_size=args.batch_size,
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
    parser.add_argument("--run_name", default='inference-cot', type=str)
    parser.add_argument(
        "--peft_model", default=None, type=str,
        help="HF repo ID or local path of a CoT LoRA adapter (optional)"
    )
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--max_length", default=4096, type=int,
                        help="Max input sequence length (prompt truncation)")
    parser.add_argument("--max_new_tokens", default=2048, type=int,
                        help="Max tokens to generate; CoT outputs are long so set higher than standard inference")
    parser.add_argument("--num_samples", default=-1, type=int, help="Number of test samples (-1 for all)")
    parser.add_argument("--from_remote", default=True, type=bool)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--hf_token", default=None, type=str, help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--max_memory_per_gpu", default="80GiB", type=str,
                        help="Max memory per GPU, e.g. '40GiB'. If unset, auto-detected from GPU total memory.")
    parser.add_argument("--wandb", action="store_true", help="Log metrics and outputs to wandb")
    parser.add_argument("--cot_budget", default=1000, type=int,
                        help="Max new tokens for CoT reasoning in phase-1 generation")
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Number of samples to process in parallel per generate call")
    parser.add_argument("--repetition_penalty", default=1.2, type=float,
                        help="Repetition penalty for generation (>1.0 discourages loops; 1.3 is aggressive)")
    args = parser.parse_args()

    run_inference_cot(args)
