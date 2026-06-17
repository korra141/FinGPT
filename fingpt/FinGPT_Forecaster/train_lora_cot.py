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
from utils import lora_module_dict, parse_model_name, load_dataset, calc_metrics

from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
from transformers import BitsAndBytesConfig

os.environ['WANDB_PROJECT'] = 'fingpt-forecaster'

COT_END = 'assistantfinal'

# Trigger phrase inserted between the prompt and CoT reasoning.
# Teaches the model to begin reasoning before answering, so that at inference
# time (no CoT provided) it still starts with this phrase and reasons through.
COT_TRIGGER = "Assistant: Let me reason through this step by step."

REASONING_INSTRUCTION = (
    "You are a financial analyst. Before giving a forecast, "
    "reason through the available signals step by step."
)


def build_cot_content(cot_text: str) -> str:
    """Assemble the full CoT content string ending with COT_END."""
    return f"{COT_TRIGGER}\n{cot_text}\n{COT_END}"


def inject_reasoning_instruction(prompt: str) -> str:
    """Insert REASONING_INSTRUCTION before [/INST] in a LLaMA-2 chat prompt."""
    marker = '[/INST]'
    idx = prompt.rfind(marker)
    if idx == -1:
        return prompt.rstrip() + '\n' + REASONING_INSTRUCTION
    return prompt[:idx].rstrip() + '\n' + REASONING_INSTRUCTION + '\n' + marker


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


def extract_generated_cot(text: str) -> str:
    """Return the CoT reasoning the model generated (up to 'assistantfinal')."""
    end = re.search(r'assistantfinal', text, re.IGNORECASE)
    if end:
        return text[:end.start()].strip()
    fallback = re.search(r'\[Positive Developments\]', text, re.IGNORECASE)
    if fallback:
        return text[:fallback.start()].strip()
    return ''


def extract_cot_reasoning(dataset):
    """Extract only the CoT reasoning portion of a CoT-format answer.

    Strips the trailing COT_END ('assistantfinal') marker so cot_reasoning
    contains pure reasoning text. COT_END is appended by build_cot_content
    during tokenization.
    """
    def _add_cot(feature):
        raw_answer = feature['answer'].strip()
        split = re.search(r'assistantfinal', raw_answer, re.IGNORECASE)
        if split:
            return {'cot_reasoning': raw_answer[:split.start()].strip()}
        print(f"Warning: No CoT marker found in answer: {raw_answer}")
        return {'cot_reasoning': ''}
    return dataset.map(_add_cot)


IGNORE_INDEX = -100


def tokenize_cot(args, tokenizer, feature):
    """
    Tokenize a CoT-format sample.

    When cot_reasoning is available, build_cot_content assembles the full CoT
    block (COT_TRIGGER + cot body + COT_END) as a single string appended after
    the prompt. The final structured answer follows.

    Default: loss on CoT reasoning AND final answer (mask only the prompt).
    --mask_cot_loss: loss on final answer only (mask prompt + CoT content).

    Falls back to plain prompt + answer tokenization when no cot_reasoning is
    available (e.g. test split) so eval loss stays meaningful.
    """
    prompt_ids = tokenizer.encode(
        inject_reasoning_instruction(feature['prompt'].strip()), padding=False,
        max_length=args.max_length, truncation=True
    )

    cot_text = feature.get('cot_reasoning') or None
    final_text = feature['answer'].strip()

    final_ids = tokenizer.encode(
        final_text, padding=False,
        max_length=args.max_length, truncation=True, add_special_tokens=False
    )

    if not cot_text:
        # No CoT reasoning available (e.g. test split) — fall back to plain
        # prompt + answer tokenization so eval loss stays meaningful.
        budget = args.max_length - len(final_ids) - 1
        if budget < 0:
            final_ids = final_ids[:args.max_length - 1]
            prompt_ids = []
        else:
            prompt_ids = prompt_ids[:budget]
        input_ids = prompt_ids + final_ids
        exceed_max_length = len(input_ids) >= args.max_length
        if not exceed_max_length:
            input_ids.append(tokenizer.eos_token_id)
        ignore_all = [IGNORE_INDEX] * len(input_ids)
        label_ids = [IGNORE_INDEX] * len(prompt_ids) + input_ids[len(prompt_ids):]
        return {
            "input_ids":     input_ids,
            "labels":        label_ids,
            "cot_labels":    ignore_all,
            "answer_labels": ignore_all,
            "exceed_max_length": exceed_max_length,
        }

    cot_content_ids = tokenizer.encode(
        build_cot_content(cot_text), padding=False,
        max_length=args.max_length, truncation=True, add_special_tokens=False
    )

    # Guarantee total sequence fits within max_length (reserve 1 slot for EOS).
    # Trim cot_content first (context), then prompt as last resort.
    budget = args.max_length - len(final_ids) - 1
    if budget < 0:
        final_ids = final_ids[:args.max_length - 1]
        prompt_ids = []
        cot_content_ids = []
    elif len(prompt_ids) + len(cot_content_ids) > budget:
        cot_content_ids = cot_content_ids[:max(0, budget - len(prompt_ids))]
        prompt_ids = prompt_ids[:budget - len(cot_content_ids)]

    input_ids = prompt_ids + cot_content_ids + final_ids
    exceed_max_length = len(input_ids) >= args.max_length
    if not exceed_max_length:
        input_ids.append(tokenizer.eos_token_id)

    n_prompt      = len(prompt_ids)        # base prompt (masked in all cases)
    n_cot_content = len(cot_content_ids)   # trigger + cot body + COT_END

    # Default: loss on CoT reasoning AND final answer (mask only the prompt).
    # --mask_cot_loss: loss on final answer only (mask prompt + CoT content).
    if args.mask_cot_loss:
        label_ids = [IGNORE_INDEX] * (n_prompt + n_cot_content) + input_ids[n_prompt + n_cot_content:]
    else:
        label_ids = [IGNORE_INDEX] * n_prompt + input_ids[n_prompt:]

    # Segment label masks for CotTrainer.compute_loss monitoring (logging only).
    cot_labels = (
        [IGNORE_INDEX] * n_prompt
        + input_ids[n_prompt : n_prompt + n_cot_content]
        + [IGNORE_INDEX] * (len(input_ids) - n_prompt - n_cot_content)
    )
    answer_labels = [IGNORE_INDEX] * (n_prompt + n_cot_content) + input_ids[n_prompt + n_cot_content:]

    return {
        "input_ids":     input_ids,
        "labels":        label_ids,
        "cot_labels":    cot_labels,
        "answer_labels": answer_labels,
        "exceed_max_length": exceed_max_length,
    }


class CotDataCollator:
    """Wraps DataCollatorForSeq2Seq and pads the extra cot_labels/answer_labels
    tensors with IGNORE_INDEX so they match the padded input_ids length."""

    def __init__(self, base_collator, ignore_index=-100):
        self.base         = base_collator
        self.ignore_index = ignore_index

    def __call__(self, features):
        cot_labels_list    = [f.pop('cot_labels',    None) for f in features]
        answer_labels_list = [f.pop('answer_labels', None) for f in features]
        batch   = self.base(features)
        max_len = batch['input_ids'].shape[1]
        for name, labels_list in [
            ('cot_labels',    cot_labels_list),
            ('answer_labels', answer_labels_list),
        ]:
            padded = torch.full((len(features), max_len), self.ignore_index, dtype=torch.long)
            for i, labels in enumerate(labels_list):
                if labels is not None:
                    t = torch.tensor(labels[:max_len], dtype=torch.long)
                    padded[i, :len(t)] = t
            batch[name] = padded
        return batch


class CotTrainer(Trainer):
    """Trainer that logs CoT-body and answer-segment CE losses alongside the
    normal training loss.

    Segment losses are computed from the same logits the training forward pass
    already produced — no extra forward passes, no extra allgathers, safe with
    DeepSpeed ZeRO-3.  cot_labels / answer_labels must be columns in the
    training dataset (added by build_dataset via tokenize_cot).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cot_sum    = 0.0
        self._answer_sum = 0.0
        self._seg_n      = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        cot_labels    = inputs.pop('cot_labels',    None)
        answer_labels = inputs.pop('answer_labels', None)

        outputs = model(**inputs)
        loss    = outputs.loss

        # Accumulate segment losses from the same logits — no extra forward pass.
        if model.training and cot_labels is not None and outputs.logits is not None:
            logits = outputs.logits
            # Causal-LM shift: predict token[i+1] from position i
            shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

            cot_shift    = cot_labels[:, 1:].contiguous().view(-1)
            answer_shift = answer_labels[:, 1:].contiguous().view(-1)

            if (cot_shift != IGNORE_INDEX).any():
                self._cot_sum    += loss_fct(shift_logits, cot_shift).item()
                self._answer_sum += loss_fct(shift_logits, answer_shift).item()
                self._seg_n      += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, **kwargs):
        if self._seg_n > 0 and 'loss' in logs:
            logs['train_cot_loss']    = round(self._cot_sum    / self._seg_n, 4)
            logs['train_answer_loss'] = round(self._answer_sum / self._seg_n, 4)
            self._cot_sum = self._answer_sum = 0.0
            self._seg_n   = 0
        super().log(logs, **kwargs)


class GenerationEvalCallback(TrainerCallback):

    def __init__(self, eval_dataset, max_length=4096, eval_batch_size=4, cot_budget=400):
        self.eval_dataset = eval_dataset
        self.max_length = max_length
        self.eval_batch_size = eval_batch_size
        self.cot_budget = cot_budget

    def _run_generation_metrics(self, state: TrainerState, model, tokenizer):
        global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f'cuda:{local_rank}'

        dataset_list = list(self.eval_dataset)[:3]
        gts = [f['answer'] for f in dataset_list]

        generated_texts, reference_texts, example_rows = [], [], []

        orig_padding_side = tokenizer.padding_side
        tokenizer.padding_side = 'left'

        torch.cuda.empty_cache()

        for i, (f, gt) in enumerate(zip(dataset_list, gts)):
            # --- Phase 1: prime the model with COT_TRIGGER, give it cot_budget tokens ---
            phase1_prompt = (
                inject_reasoning_instruction(f['prompt'].strip())
                + ' ' + COT_TRIGGER
            )
            enc1 = tokenizer(
                [phase1_prompt], return_tensors='pt',
                padding=True, truncation=True, max_length=self.max_length,
            )
            enc1 = {k: v.to(device) for k, v in enc1.items()}
            in_len1 = enc1['input_ids'].shape[1]

            with torch.no_grad():
                out1 = model.generate(
                    **enc1, use_cache=True,
                    max_new_tokens=self.cot_budget,
                    do_sample=False, temperature=None, top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            cot_tokens = out1[0][in_len1:]
            cot_text = tokenizer.decode(cot_tokens, skip_special_tokens=True).strip()

            # If the model echoes its own context (common in early training), cut off
            # at the first prompt-structure marker before it contaminates extraction.
            for _stop in ('[INST]', '<<SYS>>', '<s>'):
                _idx = cot_text.find(_stop)
                if _idx != -1:
                    cot_text = cot_text[:_idx].strip()

            # --- Phase 2: if model didn't close CoT, force COT_END then answer ---
            # Synchronize this decision across all ranks to avoid asymmetric
            # NCCL op counts under ZeRO-3 (fp16 can yield different outputs per rank).
            run_phase2 = torch.tensor(
                int(COT_END.lower() not in cot_text.lower()), device=device
            )
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(run_phase2, op=torch.distributed.ReduceOp.MAX)
            if run_phase2.item():
                phase2_prompt = phase1_prompt + ' ' + cot_text + '\n' + COT_END + '\n'
                enc2 = tokenizer(
                    [phase2_prompt], return_tensors='pt',
                    padding=True, truncation=True, max_length=self.max_length,
                )
                enc2 = {k: v.to(device) for k, v in enc2.items()}
                in_len2 = enc2['input_ids'].shape[1]
                with torch.no_grad():
                    out2 = model.generate(
                        **enc2, use_cache=True,
                        max_new_tokens=512,
                        do_sample=False, temperature=None, top_p=None,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                answer_text = tokenizer.decode(out2[0][in_len2:], skip_special_tokens=True).strip()
                for _stop in ('[INST]', '<<SYS>>', '<s>'):
                    _idx = answer_text.find(_stop)
                    if _idx != -1:
                        answer_text = answer_text[:_idx].strip()
                raw_output = cot_text + '\n' + COT_END + '\n' + answer_text
            else:
                raw_output = cot_text

            cot_reasoning = extract_generated_cot(raw_output)
            answer = extract_cot_answer(raw_output)
            gt_answer = extract_cot_answer(gt) if COT_END in gt else gt

            print(f"\n[Example {i}] Raw Output:\n{raw_output}\n")
            print(f"\n[Example {i}] CoT Reasoning (len={len(cot_reasoning)}):\n{cot_reasoning}\n")
            print(f"\n[Example {i}] Generated Answer:\n{answer}\n")
            print(f"[Example {i}] Ground Truth:\n{gt_answer}\n")

            generated_texts.append(answer)
            reference_texts.append(gt_answer)
            example_rows.append([i, len(cot_reasoning), raw_output, cot_reasoning, answer, gt_answer])

        tokenizer.padding_side = orig_padding_side

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if global_rank == 0:
            metrics = calc_metrics(reference_texts, generated_texts)
            print(f"Step {state.global_step} generation metrics: {metrics}")
            if wandb.run is None:
                wandb.init()
            wandb.log(
                {
                    **metrics,
                    "generation_examples": wandb.Table(
                        columns=["idx", "cot_length", "raw_output", "cot_reasoning",
                                 "generated", "ground_truth"],
                        data=example_rows,
                    ),
                },
                step=state.global_step,
            )
        torch.cuda.empty_cache()

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self._run_generation_metrics(state, kwargs['model'], kwargs['tokenizer'])



def load_local_dataset(path):
    """Load a DatasetDict saved with save_to_disk (e.g. chatgpt_cot/).
    Returns a one-element list to match the format expected by the caller."""
    ds = datasets.load_from_disk(path)
    if 'train' not in ds or 'test' not in ds:
        raise ValueError(f"{path} must contain 'train' and 'test' splits, found: {list(ds.keys())}")
    return ds


def log_sample_inputs(tokenizer, dataset_tok, n=3):
    """Print and log to wandb the supervised portion of n training samples."""
    rows = []
    for i in range(min(n, len(dataset_tok))):
        tok = dataset_tok[i]
        input_ids = tok['input_ids']
        labels    = tok['labels']

        supervised_ids = [t for t, l in zip(input_ids, labels) if l != IGNORE_INDEX]
        supervised_text = tokenizer.decode(supervised_ids, skip_special_tokens=False)

        seq_len      = len(input_ids)
        n_supervised = len(supervised_ids)
        n_masked     = seq_len - n_supervised

        print(f"\n{'='*60}")
        print(f"[Sample {i}] seq_len={seq_len}  supervised={n_supervised}  masked={n_masked}")
        print(f"--- supervised tokens ---\n{supervised_text}")

        rows.append([i, seq_len, n_supervised, n_masked, supervised_text])

    if wandb.run is not None:
        wandb.log({
            "train_input_samples": wandb.Table(
                columns=["idx", "seq_len", "n_supervised", "n_masked", "supervised_text"],
                data=rows,
            )
        })


def main(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    if local_rank == 0:
        run = wandb.init(project=os.environ['WANDB_PROJECT'], name=args.run_name)
        args.run_name = f"{args.run_name}-{run.id}"

    # model_name = parse_model_name(args.base_model, False)
    model_name = "meta-llama/Llama-2-7b-hf"
    token = args.hf_token or os.environ.get('HF_TOKEN') or os.environ.get('HF_USER_ACCESS_TOKEN')

    # Stagger loading: rank 0 loads first, then releases barrier for other ranks.
    # Prevents all processes hammering the CUDA allocator simultaneously.
    if local_rank != 0:
        torch.distributed.barrier()

    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"": local_rank},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            token=token,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": local_rank},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            token=token,
        )

    if local_rank == 0:
        torch.distributed.barrier()
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
            'input_ids':     [t['input_ids']     for t in tokenized],
            'labels':        [t['labels']        for t in tokenized],
            'cot_labels':    [t['cot_labels']    for t in tokenized],
            'answer_labels': [t['answer_labels'] for t in tokenized],
        })

    dataset_train = build_dataset(dataset_train_untoken)
    dataset_test  = build_dataset(dataset_test_untoken)

    if local_rank == 0:
        log_sample_inputs(tokenizer, dataset_train)

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
        ddp_find_unused_parameters=False,
        deepspeed=args.ds_config if args.ds_config else None,
    )

    if not args.load_in_4bit:
        # prepare_model_for_kbit_training already enables gradient checkpointing for 4-bit
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    model.model.config.use_cache = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=lora_module_dict[args.base_model],
        bias='none',
    )
    model = get_peft_model(model, peft_config)
    if not args.load_in_4bit:
        # PEFT inherits the base model's fp16 dtype; GradScaler requires fp32 grads.
        # 4-bit: bitsandbytes upcasts automatically during compute, no manual cast needed.
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)

    trainer = CotTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        tokenizer=tokenizer,
        data_collator=CotDataCollator(
            DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt")
        ),
        callbacks=[
            GenerationEvalCallback(
                eval_dataset=dataset_test_untoken,
                max_length=args.max_length,
                eval_batch_size=args.gen_eval_batch_size,
                cot_budget=args.cot_budget,
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
    parser.add_argument("--max_length", default=4096, type=int,
                        help="Total sequence length (prompt + CoT + final answer). Hard limit for LLaMA-2; use 8192 for LLaMA-3.")
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
    parser.add_argument("--gen_eval_batch_size", default=4, type=int,
                        help="Batch size for GenerationEvalCallback at train end")
    parser.add_argument("--mask_cot_loss", action="store_true",
                        help="Supervise only the final answer tokens; by default loss is on both CoT reasoning and final answer")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="QLoRA: load base model in 4-bit NF4 to cut VRAM ~50%% and allow larger batches")
    parser.add_argument("--cot_budget", default=700, type=int,
                        help="Max new tokens for CoT reasoning in phase-1 generation. "
                             "If model doesn't emit assistantfinal within this budget, "
                             "it is forcibly appended and answer generation continues.")
    args = parser.parse_args()

    wandb.login()
    main(args)
