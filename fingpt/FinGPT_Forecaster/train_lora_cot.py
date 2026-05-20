from transformers.integrations import TensorBoardCallback
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer import TRAINING_ARGS_NAME
from torch.utils.tensorboard import SummaryWriter
import datasets
import torch
import os
import re
import sys
import wandb
import argparse
from datetime import datetime
from functools import partial
from tqdm import tqdm
from utils import *

from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

os.environ['WANDB_PROJECT'] = 'fingpt-forecaster'

# Markers used by the CoT-trained model (matches inference_chatgpt.py)
COT_START = 'analysis'
COT_END = 'assistantfinal'


def extract_cot_answer(text: str) -> str:
    """Strip CoT preamble and return only the final structured answer."""
    split = re.search(r'assistantfinal', text, re.IGNORECASE)
    if split:
        text = text[split.end():]
    else:
        # Fallback: last occurrence of the first section header
        last = None
        for m in re.finditer(r'\[Positive Developments\]', text, re.IGNORECASE):
            last = m
        if last:
            text = text[last.start():]
    # Normalise bold markdown headers → plain bracket form
    text = re.sub(r'\*+\[([^\]]+)\]\*+', r'[\1]', text)
    return text.strip()


def extract_cot_reasoning(dataset):
    """Extract only the CoT reasoning portion of a CoT-format answer."""
    def _add_cot(feature):
        raw_answer = feature['answer'].strip()
        split = re.search(r'assistantfinal', raw_answer, re.IGNORECASE)
        if split:
            return {'cot_reasoning': raw_answer[:split.start()].strip()}
        print(f"Warning: No CoT marker found in answer: {raw_answer}")
        return {'cot_reasoning': ''}
    return dataset.map(_add_cot)


def tokenize_cot(args, tokenizer, feature):
    """
    Tokenize a CoT-format sample.

    The answer field is expected to contain:
        analysis{reasoning}assistantfinal{structured answer}

    When args.mask_cot_loss is True (recommended), loss is computed only on
    the final answer tokens — the CoT reasoning tokens are masked out.
    When False, loss is computed over the full answer (CoT + final answer).
    """
    prompt_ids = tokenizer.encode(
        feature['prompt'].strip(), padding=False,
        max_length=args.max_length, truncation=True
    )

    cot_text = feature['cot_reasoning'] if 'cot_reasoning' in feature else None
    final_text = feature['answer'].strip()
    
    cot_ids = tokenizer.encode(
        cot_text, padding=False,
        max_length=args.max_length, truncation=True, add_special_tokens=False
    ) if cot_text else []

    final_ids = tokenizer.encode(
        final_text, padding=False,
        max_length=args.max_length, truncation=True, add_special_tokens=False
    )

    target_ids = cot_ids + final_ids

    input_ids = prompt_ids + target_ids

    if args.mask_cot_loss:
        exceed_max_length = len(input_ids) >= args.max_length

        if input_ids[-1] != tokenizer.eos_token_id and not exceed_max_length:
            input_ids.append(tokenizer.eos_token_id)
            final_ids.append(tokenizer.eos_token_id)

        # Mask prompt and CoT tokens; only supervise on final answer
        label_ids = (
            [tokenizer.pad_token_id] * (len(prompt_ids) + len(cot_ids))
            + input_ids[len(prompt_ids) + len(cot_ids):]
        )
    else:
        # Train on full answer including CoT (same structure as original tokenize)

        exceed_max_length = len(input_ids) >= args.max_length

        if input_ids[-1] != tokenizer.eos_token_id and not exceed_max_length:
            input_ids.append(tokenizer.eos_token_id)

        label_ids = [tokenizer.pad_token_id] * len(prompt_ids) + input_ids[len(prompt_ids):]

    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "exceed_max_length": exceed_max_length,
    }


class GenerationEvalCallback(TrainerCallback):

    def __init__(self, eval_dataset, ignore_until_epoch=0):
        self.eval_dataset = eval_dataset
        self.ignore_until_epoch = ignore_until_epoch

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.epoch is None or state.epoch + 1 < self.ignore_until_epoch:
            return

        if state.is_local_process_zero:
            model = kwargs['model']
            tokenizer = kwargs['tokenizer']
            generated_texts, reference_texts = [], []

            count = 0
            for feature in tqdm(self.eval_dataset):
                prompt = feature['prompt']
                gt = feature['answer']

                inputs = tokenizer(
                    prompt, return_tensors='pt',
                    padding=False, max_length=4096
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                res = model.generate(
                    **inputs,
                    use_cache=True,
                    max_new_tokens=1024,  # longer budget for CoT + answer
                )
                full_output = tokenizer.decode(res[0], skip_special_tokens=True)

                # Strip prompt portion then extract final answer past CoT
                stripped = re.sub(r'.*\[/INST\]\s*', '', full_output, flags=re.DOTALL)
                answer = extract_cot_answer(stripped)

                # Reference may also be CoT-format; extract final answer for fair comparison
                gt_answer = extract_cot_answer(gt) if COT_END in gt else gt

                if count < 3:  # sanity check a few examples each eval
                    print(f"\nGenerated Answer:\n{answer}\n")
                    print(f"Ground Truth Answer:\n{gt_answer}\n")

                count+=1

                generated_texts.append(answer)
                reference_texts.append(gt_answer)
                torch.cuda.empty_cache()

            metrics = calc_metrics(reference_texts, generated_texts)
            print(f"Step {state.global_step} evaluation metrics: {metrics}")

            if wandb.run is None:
                wandb.init()
            wandb.log(metrics, step=state.global_step)
            torch.cuda.empty_cache()



def load_local_dataset(path):
    """Load a DatasetDict saved with save_to_disk (e.g. chatgpt_cot/).
    Returns a one-element list to match the format expected by the caller."""
    ds = datasets.load_from_disk(path)
    if 'train' not in ds or 'test' not in ds:
        raise ValueError(f"{path} must contain 'train' and 'test' splits, found: {list(ds.keys())}")
    return ds


def main(args):
    run = wandb.init(project=os.environ['WANDB_PROJECT'], name=args.run_name)
    args.run_name = f"{args.run_name}-{run.id}"

    model_name = parse_model_name(args.base_model, False)
    token = args.hf_token or os.environ.get('HF_TOKEN') or os.environ.get('HF_USER_ACCESS_TOKEN')

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": local_rank},
        trust_remote_code=True,
        token=token,
    )
    if args.local_rank == 0:
        print(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if os.path.isdir(args.cot_dataset):
        cot_dataset = load_local_dataset(args.cot_dataset)
    else:
        print(f"There is not cot dataset at {args.cot_dataset}. Please provide a valid path to a local dataset saved with save_to_disk, or specify a HuggingFace Hub dataset identifier.")
        sys.exit(1)
    
    cot_dataset_train = extract_cot_reasoning(cot_dataset['train'])
    cot_dataset_test =  extract_cot_reasoning(cot_dataset['test'])


    if args.dataset:
        dataset_ = load_dataset(args.dataset, True)[0]

    cot_subset_train = cot_dataset_train.select_columns(['cot_reasoning'])
    cot_subset_test = cot_dataset_test.select_columns(['cot_reasoning'])

    
    dataset_train_untoken = datasets.concatenate_datasets([dataset_['train'], cot_subset_train],axis=1 ).shuffle(seed=42)

    dataset_test_untoken = datasets.concatenate_datasets([dataset_['test'], cot_subset_test],axis=1 )

    tokenize_fn = partial(tokenize_cot, args, tokenizer)

    def build_dataset(source):
        tokenized = [tokenize_fn(feature) for feature in source]
        return datasets.Dataset.from_dict({
            'input_ids': [t['input_ids'] for t in tokenized],
            'labels':    [t['labels']    for t in tokenized],
        })

    dataset_train = build_dataset(dataset_train_untoken)
    dataset_test  = build_dataset(dataset_test_untoken)

    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M')

    training_args = TrainingArguments(
        output_dir=f'finetuned_models/{args.run_name}_{formatted_time}',
        logging_steps=args.log_interval,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        fp16=True,
        evaluation_strategy=args.evaluation_strategy,
        remove_unused_columns=False,
        report_to='wandb',
        run_name=args.run_name,
    )

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.model.config.use_cache = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=lora_module_dict[args.base_model],
        bias='none',
    )
    model = get_peft_model(model, peft_config)
    model.is_parallelizable = True
    model.model_parallel = True

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, return_tensors="pt"
        ),
        callbacks=[
            GenerationEvalCallback(
                eval_dataset=dataset_test,
                ignore_until_epoch=round(0.3 * args.num_epochs),
            )
        ],
    )

    torch.cuda.empty_cache()
    trainer.train()
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--run_name", default='local-test-cot', type=str)
    parser.add_argument("--cot_dataset", required=True, type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--base_model", required=True, type=str, choices=['chatglm2', 'llama2', 'llama3'])
    parser.add_argument("--max_length", default=2048, type=int,
                        help="Increased from 512 to accommodate CoT traces")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--num_epochs", default=5, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--log_interval", default=20, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--warmup_ratio", default=0.03, type=float)
    parser.add_argument("--ds_config", default='./config_new.json', type=str)
    parser.add_argument("--scheduler", default='linear', type=str)
    parser.add_argument("--instruct_template", default='default')
    parser.add_argument("--evaluation_strategy", default='steps', type=str)
    parser.add_argument("--eval_steps", default=0.1, type=float)
    parser.add_argument("--hf_token", default=None, type=str,
                        help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--mask_cot_loss", action="store_true",
                        help="Mask loss on CoT reasoning tokens; supervise only on final answer")
    args = parser.parse_args()

    wandb.login()
    main(args)
