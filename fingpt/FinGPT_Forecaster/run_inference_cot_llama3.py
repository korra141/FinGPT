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

LLAMA2_SYS_RE = re.compile(
    r'\[INST\]<<SYS>>\n(.*?)\n<</SYS>>\n\n(.*?)\[/INST\]',
    re.DOTALL,
)


def inject_reasoning_instruction(prompt: str) -> str:
    """Insert REASONING_INSTRUCTION before [/INST] in a LLaMA-2 chat prompt."""
    marker = '[/INST]'
    idx = prompt.rfind(marker)
    if idx == -1:
        return prompt.rstrip() + '\n' + REASONING_INSTRUCTION
    return prompt[:idx].rstrip() + '\n' + REASONING_INSTRUCTION + '\n' + marker


def build_cot_assistant_content(cot_text: str) -> str:
    """Assemble the full assistant CoT content string ending with COT_END."""
    return f"{COT_TRIGGER}\n{cot_text}\n{COT_END}"


def build_llama3_prompt(tokenizer, raw_prompt: str, assistant_content: str = None) -> str:
    """Convert a Llama2-format prompt to a Llama3 chat-template prompt.

    If assistant_content is provided it is included as an assistant-role
    message (e.g. the full CoT block ending with COT_END), and
    add_generation_prompt=True appends the header for the next assistant turn.
    """
    m = LLAMA2_SYS_RE.search(raw_prompt)
    if m:
        messages = [
            {"role": "system", "content": m.group(1).strip()},
            {"role": "user",   "content": m.group(2).strip()},
        ]
        if assistant_content is not None:
            messages.append({"role": "assistant", "content": assistant_content})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    return raw_prompt


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

    model_name = parse_model_name('llama3', args.from_remote)  # <-- llama3

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

    n_gpus = torch.cuda.device_count()
    max_memory = {i: "78GiB" for i in range(n_gpus)} if n_gpus > 0 else None

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
    # With device_map="auto", layers are spread across GPUs; inputs must go to the first device.
    input_device = next(model.parameters()).device

    dataset_list = load_dataset(args.dataset, args.from_remote)
    dataset_test = hf_datasets.concatenate_datasets([d['test'] for d in dataset_list])

    if args.num_samples > 0:
        dataset_test = dataset_test.shuffle(seed=42).select(range(min(args.num_samples, len(dataset_test))))

    print(f"Running inference on {len(dataset_test)} samples...")

    generated_texts, cot_texts, reference_texts, prompt_texts = [], [], [], []

    for feature in tqdm(dataset_test):
        raw_prompt = feature['prompt']
        gt = feature['answer']

        # Phase 1: prime with COT_TRIGGER, generate CoT reasoning
        phase1_prompt = (
            build_llama3_prompt(tokenizer, inject_reasoning_instruction(raw_prompt))
            + ' ' + COT_TRIGGER
        )
        enc1 = tokenizer(
            phase1_prompt, return_tensors='pt',
            padding=False, max_length=args.max_length, truncation=True,
        )
        enc1 = {k: v.to(input_device) for k, v in enc1.items()}
        in_len1 = enc1['input_ids'].shape[1]

        with torch.no_grad():
            out1 = model.generate(
                **enc1, use_cache=True,
                max_new_tokens=args.cot_budget,
                do_sample=False, temperature=None, top_p=None,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        cot_text = tokenizer.decode(out1[0][in_len1:], skip_special_tokens=True).strip()

        for _stop in ('<|eot_id|>', '<|start_header_id|>', '<|end_header_id|>'):
            _idx = cot_text.find(_stop)
            if _idx != -1:
                cot_text = cot_text[:_idx].strip()

        # Phase 2: if CoT didn't close itself, force COT_END then generate the answer
        if COT_END.lower() not in cot_text.lower():
            phase2_prompt = build_llama3_prompt(
                tokenizer,
                inject_reasoning_instruction(raw_prompt),
                assistant_content=build_cot_assistant_content(cot_text),
            )
            enc2 = tokenizer(
                phase2_prompt, return_tensors='pt',
                padding=False, max_length=args.max_length, truncation=True,
            )
            enc2 = {k: v.to(input_device) for k, v in enc2.items()}
            in_len2 = enc2['input_ids'].shape[1]
            with torch.no_grad():
                out2 = model.generate(
                    **enc2, use_cache=True,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False, temperature=None, top_p=None,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                )
            answer_text = tokenizer.decode(out2[0][in_len2:], skip_special_tokens=True).strip()
            for _stop in ('<|eot_id|>', '<|start_header_id|>', '<|end_header_id|>'):
                _idx = answer_text.find(_stop)
                if _idx != -1:
                    answer_text = answer_text[:_idx].strip()
            raw_generation = cot_text + '\n' + COT_END + '\n' + answer_text
        else:
            raw_generation = cot_text

        cot_part = extract_generated_cot(raw_generation)
        answer = extract_cot_answer(raw_generation)

        generated_texts.append(answer)
        cot_texts.append(cot_part)
        reference_texts.append(gt)
        prompt_texts.append(raw_prompt)

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
            base_model='llama3',
            peft_model=args.peft_model,
            dataset=args.dataset,
            num_samples=len(dataset_test),
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
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
    parser.add_argument("--run_name", default='inference-cot-llama3', type=str)
    parser.add_argument(
        "--peft_model", default=None, type=str,
        help="HF repo ID or local path of a CoT LoRA adapter (optional)"
    )
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--max_length", default=8192, type=int,
                        help="Max input sequence length (Llama-3 native context is 8192)")
    parser.add_argument("--max_new_tokens", default=2048, type=int,
                        help="Max tokens to generate; CoT outputs are long so set higher than standard inference")
    parser.add_argument("--num_samples", default=-1, type=int, help="Number of test samples (-1 for all)")
    parser.add_argument("--from_remote", default=True, type=bool)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--hf_token", default=None, type=str, help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Log metrics and outputs to wandb")
    parser.add_argument("--cot_budget", default=700, type=int,
                        help="Max new tokens for CoT reasoning in phase-1 generation")
    args = parser.parse_args()

    run_inference_cot(args)
