import pdb

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import os
import re
import json
import argparse
import wandb
from tqdm import tqdm
from utils import parse_model_name, load_dataset, calc_metrics, calc_rouge_score, calc_bert_score, parse_answer, parse_answer_base
from collections import defaultdict
from sklearn.metrics import accuracy_score, mean_squared_error


LLAMA2_SYS_RE = re.compile(
    r'\[INST\]<<SYS>>\n(.*?)\n<</SYS>>\n\n(.*?)\[/INST\]',
    re.DOTALL,
)

ASSISTANT_HEADER = '<|start_header_id|>assistant<|end_header_id|>'
CHATML_ASSISTANT_HEADER = '<|im_start|>assistant'
CHATML_END = '<|im_end|>'


CHATML_JUNK_RE = re.compile(
    r'<\|im_end\|>.*|<\|im_start\|>.*|<\|end_of_text\|>.*|<\|eot_id\|>.*',
    re.DOTALL | re.IGNORECASE,
)

# Matches numbers with optional leading sign/currency and trailing percent,
# e.g. 42, 3.5%, $1,234.56, +0.8%, -12.3
NUMBER_RE = re.compile(r'[\+\-\$]?\d[\d,.]*%?')


def mask_numbers_in_prompt(prompt: str, mask_token: str = '[NUM]') -> str:
    return NUMBER_RE.sub(mask_token, prompt)


def strip_chatml_artifacts(text: str) -> str:
    """Remove <|im_end|> and any text that follows (fake continuation turns)."""
    return CHATML_JUNK_RE.sub('', text).strip()


def extract_llama3_answer(full_output: str) -> str:
    """Extract assistant reply, supporting both ChatML and Llama3-native formats."""
    # ChatML format: <|im_start|>assistant\n...<|im_end|>
    idx = full_output.rfind(CHATML_ASSISTANT_HEADER)
    if idx != -1:
        start = idx + len(CHATML_ASSISTANT_HEADER)
        end = full_output.find(CHATML_END, start)
        return full_output[start:end].strip() if end != -1 else full_output[start:].strip()
    # Llama3-native format
    idx = full_output.rfind(ASSISTANT_HEADER)
    if idx != -1:
        return full_output[idx + len(ASSISTANT_HEADER):].strip()
    return full_output.strip()


def build_llama3_prompt(tokenizer, raw_prompt: str) -> str:
    """
    Convert a Llama2-format prompt to a Llama3 chat-template prompt.
    Falls back to the raw prompt if the format isn't recognised.
    """
    m = LLAMA2_SYS_RE.search(raw_prompt)
    if m:
        system_msg = m.group(1).strip()
        user_msg = m.group(2).strip()
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    print("Could not preprocess the prompt, passing it as it is.")
    # Fallback: strip [/INST] and treat rest as the raw user turn
    return raw_prompt


def calc_metrics_llama3(answers, gts):
    answers_dict = defaultdict(list)
    gts_dict = defaultdict(list)

    for answer, gt in zip(answers, gts):
        answer_dict = parse_answer_base(answer)
        gt_dict = parse_answer(gt)  # GT uses the original bracketed format

        if answer_dict and gt_dict:
            for k in answer_dict:
                answers_dict[k].append(answer_dict[k])
                gts_dict[k].append(gt_dict[k])

    total = len(answers)
    parsed = len(answers_dict['prediction'])
    print(f"\nParsed {parsed}/{total} samples successfully ({100*parsed/total:.1f}%)")

    if not answers_dict['prediction']:
        print("WARNING: No samples parsed — check model output format.")
        return {}

    bin_acc = accuracy_score(gts_dict['prediction_binary'], answers_dict['prediction_binary'])
    mse = mean_squared_error(gts_dict['prediction'], answers_dict['prediction'])

    pros_rouge = calc_rouge_score(gts_dict['positive developments'], answers_dict['positive developments'])
    cons_rouge = calc_rouge_score(gts_dict['potential concerns'], answers_dict['potential concerns'])
    anal_rouge = calc_rouge_score(gts_dict['analysis'], answers_dict['analysis'])

    pros_bert = calc_bert_score(gts_dict['positive developments'], answers_dict['positive developments'])
    cons_bert = calc_bert_score(gts_dict['potential concerns'], answers_dict['potential concerns'])
    anal_bert = calc_bert_score(gts_dict['analysis'], answers_dict['analysis'])

    print(f"Binary Accuracy: {bin_acc:.2f}  |  Mean Square Error: {mse:.2f}")
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

    # model_name = args.base_model_path if args.base_model_path else parse_model_name(args.base_model, args.from_remote)

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

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

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        trust_remote_code=True,
        token=token,
        **model_kwargs,
    )

    if args.peft_model:
        print(f"Loading LoRA adapter: {args.peft_model}")
        model = PeftModel.from_pretrained(model, args.peft_model, token=token)

    model = model.eval()

    dataset_list = load_dataset(args.dataset, args.from_remote)
    import datasets as hf_datasets
    dataset_test = hf_datasets.concatenate_datasets([d['test'] for d in dataset_list])

    if args.num_samples > 0:
        dataset_test = dataset_test.shuffle(seed=42).select(range(min(args.num_samples, len(dataset_test))))

    print(f"Running inference on {len(dataset_test)} samples...")

    generated_texts, reference_texts, prompt_texts, raw_outputs = [], [], [], []

    for feature in tqdm(dataset_test):
        raw_prompt = feature['prompt']
        if args.mask_numbers:
            raw_prompt = mask_numbers_in_prompt(raw_prompt)
        gt = feature['answer']

        formatted_prompt = build_llama3_prompt(tokenizer, raw_prompt)

        inputs = tokenizer(
            formatted_prompt,
            return_tensors='pt',
            padding=False,
            max_length=args.max_length,
            truncation=True,
        )
        input_len = inputs['input_ids'].shape[1]
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        eot_id = tokenizer.convert_tokens_to_ids('<|eot_id|>')
        eos_ids = [tokenizer.eos_token_id]
        if eot_id and eot_id != tokenizer.eos_token_id:
            eos_ids.append(eot_id)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=eos_ids,
            )

        # Raw output before any preprocessing
        raw_new_tokens = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=False)
        raw_full_output = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        raw_outputs.append({"raw_new_tokens": raw_new_tokens, "raw_full_output": raw_full_output})

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][input_len:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # pdb.set_trace()
        # answer = strip_chatml_artifacts(answer)

        # If the tokenizer didn't cleanly separate (e.g. prompt re-included), fall back
        # if not answer:
            # answer = strip_chatml_artifacts(extract_llama3_answer(raw_full_output))

        generated_texts.append(answer)
        reference_texts.append(gt)
        prompt_texts.append(formatted_prompt)

    raw_output_path = f"{args.run_name}_raw_outputs.json"
    with open(raw_output_path, 'w') as f:
        json.dump(raw_outputs[:10], f, indent=2)
    print(f"Saved raw outputs (first 10) to {raw_output_path}")

    output_path = f"{args.run_name}_outputs.json"
    with open(output_path, 'w') as f:
        json.dump([
            {"prompt": p, "generated": g, "reference": r}
            for p, g, r in zip(prompt_texts, generated_texts, reference_texts)
        ], f, indent=2)
    print(f"\nSaved raw outputs to {output_path}")

    print("\n=== Sample Outputs (first 3) ===")
    for i in range(min(3, len(generated_texts))):
        print(f"\n--- Sample {i+1} ---")
        print(f"[GENERATED]:\n{generated_texts[i][:500]}")
        print(f"[REFERENCE]:\n{reference_texts[i][:300]}")

    print("\n=== Inference Metrics ===")
    metrics = calc_metrics_llama3(generated_texts, reference_texts)
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
    parser.add_argument("--run_name", default='inference-llama3', type=str)
    parser.add_argument("--base_model", default='llama3', type=str, choices=['chatglm2', 'llama2', 'llama3'],
                        help="Base model key resolved via parse_model_name (e.g. llama3)")
    parser.add_argument("--base_model_path", default=None, type=str,
                        help="Local path to base model weights; overrides --base_model HF lookup")
    parser.add_argument("--peft_model", default=None, type=str,
                        help="Path or HF repo of LoRA adapter (optional)")
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--max_length", default=4096, type=int)
    parser.add_argument("--num_samples", default=-1, type=int,
                        help="Number of test samples (-1 for all)")
    parser.add_argument("--from_remote", default=True, type=bool)
    parser.add_argument("--hf_token", default=None, type=str,
                        help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--mask_numbers", action="store_true",
                        help="Replace all numbers (and surrounding ±$%%) in prompts with [NUM]")
    args = parser.parse_args()

    run_inference(args)
