from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from transformers import TrainerCallback, TrainerState, TrainerControl
import gc
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
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import BitsAndBytesConfig

os.environ['WANDB_PROJECT'] = 'fingpt-forecaster'

IGNORE_INDEX = -100

COT_END = 'assistantfinal'

LLAMA2_SYS_RE = re.compile(
    r'\[INST\]<<SYS>>\n(.*?)\n<</SYS>>\n\n(.*?)\[/INST\]',
    re.DOTALL,
)


def build_llama3_prompt(tokenizer, raw_prompt: str) -> str:
    """Convert a Llama2-format prompt to a Llama3 chat-template prompt."""
    m = LLAMA2_SYS_RE.search(raw_prompt)
    if m:
        messages = [
            {"role": "system", "content": m.group(1).strip()},
            {"role": "user",   "content": m.group(2).strip()},
        ]
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
    Tokenize a CoT-format sample for Llama3.

    Converts the raw (Llama2-format) prompt to Llama3 chat template before
    encoding. When args.mask_cot_loss is True, loss is computed only on the
    final answer tokens; otherwise the full CoT + answer is supervised.
    """
    llama3_prompt = build_llama3_prompt(tokenizer, feature['prompt'])
    prompt_ids = tokenizer.encode(
        llama3_prompt, padding=False,
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

    # Guarantee total sequence fits within max_length (reserve 1 slot for EOS).
    # Priority: keep final_ids intact (supervision signal), trim cot first
    # (context only), then prompt as last resort.
    budget = args.max_length - len(final_ids) - 1
    if budget < 0:
        final_ids = final_ids[:args.max_length - 1]
        prompt_ids = []
        cot_ids = []
    elif len(prompt_ids) + len(cot_ids) > budget:
        cot_ids = cot_ids[:max(0, budget - len(prompt_ids))]
        prompt_ids = prompt_ids[:budget - len(cot_ids)]

    input_ids = prompt_ids + cot_ids + final_ids
    exceed_max_length = len(input_ids) >= args.max_length
    if not exceed_max_length:
        input_ids.append(tokenizer.eos_token_id)

    # Use -100 directly as the ignore index — pad_token == eos_token for LLaMA,
    # so using pad_token_id would also mask genuine EOS tokens in the labels.
    if args.mask_cot_loss:
        supervised_start = len(prompt_ids) + len(cot_ids)
        label_ids = [IGNORE_INDEX] * supervised_start + input_ids[supervised_start:]
    else:
        label_ids = [IGNORE_INDEX] * len(prompt_ids) + input_ids[len(prompt_ids):]

    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "exceed_max_length": exceed_max_length,
    }


def _is_peft_adapter_dir(path: str) -> bool:
    """True if path is a saved PEFT adapter (contains adapter_config.json)."""
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


class SavePeftModelCallback(TrainerCallback):
    """Save PEFT adapter format alongside each Trainer/DeepSpeed ZeRO checkpoint.

    With ZeRO-3, save_pretrained() triggers a cross-rank param gather
    (requires gather_16bit_weights_on_model_save=true in the DS config).
    Must be called on ALL ranks; only rank 0 writes files.
    """

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_dir)
        return control


class GenerationEvalCallback(TrainerCallback):

    def __init__(self, eval_dataset, max_length=8192, eval_batch_size=4):
        # Materialize once — avoids re-converting the Arrow table every eval step.
        self._dataset_list = list(eval_dataset)
        self.max_length = max_length
        self.eval_batch_size = eval_batch_size

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):

        model = kwargs['model']
        tokenizer = kwargs['tokenizer']

        global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        local_rank  = int(os.environ.get("LOCAL_RANK", 0))

        generated_texts, reference_texts = [], []

        orig_padding_side = tokenizer.padding_side
        tokenizer.padding_side = 'left'

        n_batches = (len(self._dataset_list) + self.eval_batch_size - 1) // self.eval_batch_size

        # Ensure all ranks enter the eval loop together after training completes.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        for batch_idx in tqdm(range(n_batches), disable=global_rank != 0):
            batch = self._dataset_list[batch_idx * self.eval_batch_size:(batch_idx + 1) * self.eval_batch_size]
            prompts = [build_llama3_prompt(tokenizer, f['prompt']) for f in batch]
            gts = [f['answer'] for f in batch]

            # All ranks receive identical input — ZeRO-3 allgather shapes stay consistent.
            inputs = tokenizer(
                prompts, return_tensors='pt',
                padding=True, truncation=True, max_length=self.max_length,
            )
            padded_input_len = inputs['input_ids'].shape[1]
            inputs = {k: v.to(f'cuda:{local_rank}') for k, v in inputs.items()}

            with torch.no_grad():
                res = model.generate(
                    **inputs,
                    use_cache=True,
                    max_new_tokens=2048,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                )

            if global_rank == 0:
                for i, gt in enumerate(gts):
                    new_tokens = res[i][padded_input_len:]
                    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    answer = extract_cot_answer(answer)
                    gt_answer = extract_cot_answer(gt) if COT_END in gt else gt
                    if batch_idx == 0 and i < 3:
                        print(f"\nGenerated Answer:\n{answer}\n")
                        print(f"Ground Truth Answer:\n{gt_answer}\n")
                    generated_texts.append(answer)
                    reference_texts.append(gt_answer)

            # Decode happens above before this point; drop tensors so empty_cache
            # actually reclaims VRAM rather than being a no-op.
            del res, inputs
            torch.cuda.empty_cache()

        tokenizer.padding_side = orig_padding_side

        # Wait for all ranks to finish generate before rank 0 logs metrics.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if global_rank == 0:
            metrics = calc_metrics(reference_texts, generated_texts)
            print(f"Step {state.global_step} evaluation metrics: {metrics}")
            if wandb.run is None:
                wandb.init()
            wandb.log(metrics, step=state.global_step)
            torch.cuda.empty_cache()


def load_local_dataset(path):
    ds = datasets.load_from_disk(path)
    if 'train' not in ds or 'test' not in ds:
        raise ValueError(f"{path} must contain 'train' and 'test' splits, found: {list(ds.keys())}")
    return ds


def main(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    global_rank = torch.distributed.get_rank()

    if global_rank == 0:
        run = wandb.init(project=os.environ['WANDB_PROJECT'], name=args.run_name)
        args.run_name = f"{args.run_name}-{run.id}"

    # Broadcast run_name so every rank uses the same output_dir.
    _name = [args.run_name]
    torch.distributed.broadcast_object_list(_name, src=0)
    args.run_name = _name[0]

    model_name = parse_model_name('llama3')
    token = args.hf_token or os.environ.get('HF_TOKEN') or os.environ.get('HF_USER_ACCESS_TOKEN')

    # Stagger loading: rank 0 loads first, then releases barrier for other ranks.
    if local_rank != 0:
        torch.distributed.barrier()

    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
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
            torch_dtype=torch.bfloat16,
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
        print(f"No CoT dataset found at {args.cot_dataset}. Provide a path saved with save_to_disk.")
        sys.exit(1)

    cot_dataset_train = extract_cot_reasoning(cot_dataset['train'])
    cot_dataset_test  = extract_cot_reasoning(cot_dataset['test'])

    if args.dataset:
        dataset_ = load_dataset(args.dataset, True)[0]

    cot_subset_train = cot_dataset_train.select_columns(['cot_reasoning'])
    cot_subset_test  = cot_dataset_test.select_columns(['cot_reasoning'])

    dataset_train_untoken = datasets.concatenate_datasets(
        [dataset_['train'], cot_subset_train], axis=1
    ).shuffle(seed=42)
    dataset_test_untoken = datasets.concatenate_datasets(
        [dataset_['test'], cot_subset_test], axis=1
    )

    tokenize_fn = partial(tokenize_cot, args, tokenizer)

    def build_dataset(source):
        def _tok(feature):
            t = tokenize_fn(feature)
            return {'input_ids': t['input_ids'], 'labels': t['labels']}
        return source.map(_tok, remove_columns=source.column_names)

    dataset_train = build_dataset(dataset_train_untoken)
    dataset_test  = build_dataset(dataset_test_untoken)

    # Free raw text datasets — tokenized versions are Arrow-backed and no longer
    # need the originals. dataset_test_untoken is kept for GenerationEvalCallback.
    del cot_dataset, cot_dataset_train, cot_dataset_test, cot_subset_train, cot_subset_test
    del dataset_train_untoken, dataset_
    gc.collect()

    # Compute timestamp on rank 0 and broadcast so all ranks get the same output_dir.
    _ts = [datetime.now().strftime('%Y%m%d%H%M') if global_rank == 0 else '']
    torch.distributed.broadcast_object_list(_ts, src=0)
    formatted_time = _ts[0]

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
        bf16=True,
        evaluation_strategy='no',
        group_by_length=True,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to='wandb',
        run_name=args.run_name,
        ddp_find_unused_parameters=False,
        deepspeed=args.ds_config if args.ds_config else None,
    )

    if not args.load_in_4bit:
        # prepare_model_for_kbit_training already enables gradient checkpointing for 4-bit
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.model.config.use_cache = False

    if args.resume_from_checkpoint and _is_peft_adapter_dir(args.resume_from_checkpoint):
        # Path is a saved PEFT adapter — load weights directly.
        model = PeftModel.from_pretrained(
            model, args.resume_from_checkpoint, is_trainable=True,
        )
    else:
        # Fresh LoRA config — either first run or resuming from a DeepSpeed ZeRO
        # checkpoint (no adapter_config.json). Trainer handles ZeRO state loading.
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=lora_module_dict['llama3'],
            bias='none',
        )
        model = get_peft_model(model, peft_config)
    if not args.load_in_4bit:
        # Cast trainable LoRA params to fp32 for gradient stability (base stays bf16).
        # 4-bit: bitsandbytes upcasts automatically during compute, no manual cast needed.
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)

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
                eval_dataset=dataset_test_untoken,
                max_length=args.max_length,
                eval_batch_size=args.gen_eval_batch_size,
            ),
            SavePeftModelCallback(),
        ],
    )

    torch.cuda.empty_cache()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--run_name", default='local-test-llama3-cot', type=str)
    parser.add_argument("--cot_dataset", required=True, type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--max_length", default=8192, type=int,
                        help="Total sequence length (prompt + CoT + final answer). LLaMA-3 native context is 8192.")
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
    parser.add_argument("--eval_steps", default=0.1, type=float)
    parser.add_argument("--hf_token", default=None, type=str,
                        help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--gen_eval_batch_size", default=4, type=int,
                        help="Batch size for GenerationEvalCallback — all GPUs process the same padded batch")
    parser.add_argument("--mask_cot_loss", action="store_true",
                        help="Mask loss on CoT reasoning tokens; supervise only on final answer")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="QLoRA: load base model in 4-bit NF4 to cut VRAM ~50%% and allow larger batches")
    parser.add_argument("--resume_from_checkpoint", default=None, type=str,
                        help="Resume training from a checkpoint. Accepts either a PEFT adapter "
                             "directory (must contain adapter_config.json) or a Trainer output "
                             "directory whose checkpoint-N subdirs hold DeepSpeed ZeRO state.")
    args = parser.parse_args()

    wandb.login()
    main(args)
