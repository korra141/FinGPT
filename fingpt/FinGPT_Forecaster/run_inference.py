from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import os
import re
import argparse
import wandb
from tqdm import tqdm
from functools import partial
from utils import parse_model_name, load_dataset, calc_metrics


def run_inference(args):

    token = "REDACTED_HF_TOKEN"
    if token is None:
        token = os.environ.get('HF_TOKEN') or os.environ.get('HF_USER_ACCESS_TOKEN')

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
        device_map="auto",
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
    import datasets as hf_datasets
    dataset_test = hf_datasets.concatenate_datasets([d['test'] for d in dataset_list])

    if args.num_samples > 0:
        dataset_test = dataset_test.shuffle(seed=42).select(range(min(args.num_samples, len(dataset_test))))

    print(f"Running inference on {len(dataset_test)} samples...")

    generated_texts, reference_texts = [], []

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
        answer = re.sub(r'.*\[/INST\]\s*', '', output, flags=re.DOTALL)

        generated_texts.append(answer)
        reference_texts.append(gt)

    print("\n=== Inference Metrics ===")
    metrics = calc_metrics(reference_texts, generated_texts)

    if args.wandb:
        wandb.init(project='fingpt-forecaster', name=args.run_name)
        wandb.log(metrics)
        wandb.finish()

    return metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default='inference', type=str)
    parser.add_argument("--base_model", required=True, type=str, choices=['chatglm2', 'llama2', 'llama3'])
    parser.add_argument("--peft_model", default=None, type=str, help="Path or HF repo of LoRA adapter (optional)")
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--max_length", default=4096, type=int)
    parser.add_argument("--num_samples", default=50, type=int, help="Number of test samples to evaluate, -1 for all")
    parser.add_argument("--from_remote", default=True, type=bool)
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization (~4GB VRAM)")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit quantization (~7GB VRAM)")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to wandb")
    args = parser.parse_args()

    run_inference(args)