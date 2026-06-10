from transformers.integrations import TensorBoardCallback
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from transformers import TrainerCallback, TrainerState, TrainerControl
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
from collections import defaultdict
from sklearn.metrics import accuracy_score, mean_squared_error
from utils import load_dataset, calc_rouge_score, calc_bert_score, parse_answer, parse_answer_base

from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
)

os.environ['WANDB_PROJECT'] = 'fingpt-forecaster'

LLAMA2_SYS_RE = re.compile(
    r'\[INST\]<<SYS>>\n(.*?)\n<</SYS>>\n\n(.*?)\[/INST\]',
    re.DOTALL,
)

ASSISTANT_HEADER = '<|start_header_id|>assistant<|end_header_id|>'
CHATML_ASSISTANT_HEADER = '<|im_start|>assistant'
CHATML_END = '<|im_end|>'


def build_llama3_prompt(tokenizer, raw_prompt: str) -> str:
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
    return raw_prompt


def extract_llama3_answer(full_output: str) -> str:
    idx = full_output.rfind(CHATML_ASSISTANT_HEADER)
    if idx != -1:
        start = idx + len(CHATML_ASSISTANT_HEADER)
        end = full_output.find(CHATML_END, start)
        return full_output[start:end].strip() if end != -1 else full_output[start:].strip()
    idx = full_output.rfind(ASSISTANT_HEADER)
    if idx != -1:
        return full_output[idx + len(ASSISTANT_HEADER):].strip()
    return full_output.strip()


def calc_metrics_llama3(answers, gts, parsed_answers=None):
    answers_dict = defaultdict(list)
    gts_dict = defaultdict(list)
    mse_preds, mse_gts_list = [], []

    if parsed_answers is None:
        parsed_answers = [parse_answer_base(a) for a in answers]

    for answer_dict, gt in zip(parsed_answers, gts):
        gt_dict = parse_answer(gt)

        if answer_dict and gt_dict:
            for k in answer_dict:
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
        print(f"Binary Accuracy: {bin_acc:.2f}  |  MSE: N/A (no numeric predictions)")

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


LLAMA3_INSTRUCT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

LORA_TARGET_MODULES = [
    'q_proj', 'k_proj', 'v_proj',
    'o_proj', 'gate_proj', 'up_proj', 'down_proj',
]


def tokenize_llama3(args, tokenizer, feature):
    formatted_prompt = build_llama3_prompt(tokenizer, feature['prompt'])

    prompt_ids = tokenizer.encode(
        formatted_prompt, padding=False,
        max_length=args.max_length, truncation=True
    )

    target_ids = tokenizer.encode(
        feature['answer'].strip(), padding=False,
        max_length=args.max_length, truncation=True, add_special_tokens=False
    )

    input_ids = prompt_ids + target_ids
    exceed_max_length = len(input_ids) >= args.max_length

    if input_ids[-1] != tokenizer.eos_token_id and not exceed_max_length:
        input_ids.append(tokenizer.eos_token_id)

    # Mask prompt tokens in labels so loss is only computed on the answer
    label_ids = [tokenizer.pad_token_id] * len(prompt_ids) + input_ids[len(prompt_ids):]

    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "exceed_max_length": exceed_max_length,
    }


class GenerationEvalCallback(TrainerCallback):

    def __init__(self, eval_dataset, tokenizer, ignore_until_epoch=0):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.ignore_until_epoch = ignore_until_epoch

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.epoch is None or state.epoch + 1 < self.ignore_until_epoch:
            return

        if not state.is_local_process_zero:
            return

        model = kwargs['model']
        tokenizer = self.tokenizer

        eot_id = tokenizer.convert_tokens_to_ids('<|eot_id|>')
        eos_ids = [tokenizer.eos_token_id]
        if eot_id and eot_id != tokenizer.eos_token_id:
            eos_ids.append(eot_id)

        generated_texts, reference_texts = [], []

        for feature in tqdm(self.eval_dataset):
            raw_prompt = feature['prompt']
            gt = feature['answer']

            formatted_prompt = build_llama3_prompt(tokenizer, raw_prompt)
            inputs = tokenizer(
                formatted_prompt, return_tensors='pt',
                padding=False, max_length=4096, truncation=True
            )
            input_len = inputs['input_ids'].shape[1]
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=eos_ids,
                )

            new_tokens = output_ids[0][input_len:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            generated_texts.append(answer)
            reference_texts.append(gt)
            torch.cuda.empty_cache()

        metrics = calc_metrics_llama3(generated_texts, reference_texts)

        if wandb.run is None:
            wandb.init()
        wandb.log(metrics, step=state.global_step)
        torch.cuda.empty_cache()


def main(args):
    run = wandb.init(project=os.environ['WANDB_PROJECT'], name=args.run_name)
    args.run_name = f"{args.run_name}-{run.id}"

    token = args.hf_token or os.environ.get('HF_TOKEN') or os.environ.get('HF_USER_ACCESS_TOKEN')

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    tokenizer = AutoTokenizer.from_pretrained(
        LLAMA3_INSTRUCT_MODEL, trust_remote_code=True, token=token
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA3_INSTRUCT_MODEL,
        torch_dtype=torch.float16,
        device_map={"": local_rank},
        trust_remote_code=True,
        token=token,
    )

    if args.local_rank == 0:
        print(model)

    # Load DOW (or any) dataset
    dataset_list = load_dataset(args.dataset, args.from_remote)
    dataset_train = datasets.concatenate_datasets([d['train'] for d in dataset_list]).shuffle(seed=42)

    if args.test_dataset:
        test_dataset_list = load_dataset(args.test_dataset, args.from_remote)
    else:
        test_dataset_list = dataset_list
    dataset_test = datasets.concatenate_datasets([d['test'] for d in test_dataset_list])

    original_dataset = datasets.DatasetDict({'train': dataset_train, 'test': dataset_test})
    eval_dataset = original_dataset['test'].shuffle(seed=42).select(range(min(50, len(dataset_test))))

    dataset = original_dataset.map(partial(tokenize_llama3, args, tokenizer))
    print('original dataset length: ', len(dataset['train']))
    dataset = dataset.filter(lambda x: not x['exceed_max_length'])
    print('filtered dataset length: ', len(dataset['train']))
    dataset = dataset.remove_columns(
        ['prompt', 'answer', 'label', 'symbol', 'period', 'exceed_max_length']
    )

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

    # model.gradient_checkpointing_enable()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.model.config.use_cache = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=LORA_TARGET_MODULES,
        bias='none',
    )
    model = get_peft_model(model, peft_config)
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, return_tensors="pt"
        ),
        callbacks=[
            GenerationEvalCallback(
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
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
    parser.add_argument("--run_name", default='llama3-dow', type=str)
    parser.add_argument("--dataset", required=True, type=str,
                        help="Dataset name(s), e.g. 'dow-jones' or local path")
    parser.add_argument("--test_dataset", type=str)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--num_epochs", default=8, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--log_interval", default=20, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
    parser.add_argument("--warmup_ratio", default=0.05, type=float)
    parser.add_argument("--scheduler", default='linear', type=str)
    parser.add_argument("--evaluation_strategy", default='steps', type=str)
    parser.add_argument("--eval_steps", default=0.1, type=float)
    parser.add_argument("--from_remote", default=True, type=bool)
    parser.add_argument("--hf_token", default=None, type=str,
                        help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--ds_config", default='./config_new.json', type=str)
    args = parser.parse_args()

    wandb.login()
    main(args)
