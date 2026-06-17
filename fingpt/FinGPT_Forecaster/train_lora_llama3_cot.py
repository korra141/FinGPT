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

# Trigger phrase inserted between the prompt and CoT reasoning.
# Teaches the model to begin reasoning before answering, so that at inference
# time (no CoT provided) it still starts with this phrase and reasons through.
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
    """Insert REASONING_INSTRUCTION before [/INST] in a LLaMA-2 chat prompt.
    build_llama3_prompt is called after injection so the instruction lands in
    the user turn of the LLaMA-3 chat template."""
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
    add_generation_prompt=True appends the header for the next assistant
    turn (the final structured answer).
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


def extract_cot_reasoning(dataset):
    """Extract only the CoT reasoning portion of a CoT-format answer.

    Strips the trailing COT_END ('assistantfinal') marker so cot_reasoning
    contains pure reasoning text. COT_END is appended by build_cot_assistant_content
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


def tokenize_cot(args, tokenizer, feature):
    """
    Tokenize a CoT-format sample for Llama3.

    When cot_reasoning is available the CoT block (COT_TRIGGER + cot body +
    COT_END) is passed to build_llama3_prompt as an assistant-role
    message so training matches the two-turn inference format. The final
    structured answer then follows as the next assistant generation target.

    When args.mask_cot_loss is True, only the final answer tokens are
    supervised; otherwise supervision starts after the base prompt (system +
    user) so the model also learns the CoT turn.

    Falls back to plain prompt + answer tokenization when no cot_reasoning is
    available (e.g. test split) so eval loss stays meaningful.
    """
    llama3_prompt = build_llama3_prompt(tokenizer, inject_reasoning_instruction(feature['prompt']))
    prompt_ids = tokenizer.encode(
        llama3_prompt, padding=False,
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
        label_ids    = [IGNORE_INDEX] * len(prompt_ids) + input_ids[len(prompt_ids):]
        ignore_all   = [IGNORE_INDEX] * len(input_ids)
        return {
            "input_ids":      input_ids,
            "labels":         label_ids,
            "cot_labels":     ignore_all,
            "answer_labels":  ignore_all,
            "exceed_max_length": exceed_max_length,
        }

    # Build the full prefix (prompt + CoT as assistant role) through the chat
    # template so training matches the inference prompt structure.
    cot_assistant_content = build_cot_assistant_content(cot_text)
    full_prefix_str = build_llama3_prompt(
        tokenizer, inject_reasoning_instruction(feature['prompt']), cot_assistant_content
    )
    prefix_ids = tokenizer.encode(
        full_prefix_str, padding=False,
        max_length=args.max_length, truncation=True,
    )

    # Guarantee total sequence fits within max_length (reserve 1 slot for EOS).
    # Trim prefix first (cot body is context), then final as last resort.
    budget = args.max_length - len(final_ids) - 1
    if budget < 0:
        final_ids = final_ids[:args.max_length - 1]
        prefix_ids = []
    else:
        prefix_ids = prefix_ids[:budget]

    input_ids = prefix_ids + final_ids
    exceed_max_length = len(input_ids) >= args.max_length
    if not exceed_max_length:
        input_ids.append(tokenizer.eos_token_id)

    n_prompt = len(prompt_ids)   # system + user + first assistant header (no CoT)
    n_prefix = len(prefix_ids)  # everything up to and including the CoT assistant turn

    # Default: loss on CoT reasoning AND final answer (mask only the system+user prompt).
    # --mask_cot_loss: loss on final answer only (mask the entire prefix including CoT).
    if args.mask_cot_loss:
        label_ids = [IGNORE_INDEX] * n_prefix + input_ids[n_prefix:]
    else:
        label_ids = [IGNORE_INDEX] * n_prompt + input_ids[n_prompt:]

    # Segment masks for CotTrainer.compute_loss monitoring.
    cot_labels = (
        [IGNORE_INDEX] * n_prompt
        + input_ids[n_prompt:n_prefix]                        # CoT assistant turn tokens
        + [IGNORE_INDEX] * (len(input_ids) - n_prefix)
    )
    answer_labels = [IGNORE_INDEX] * n_prefix + input_ids[n_prefix:]

    return {
        "input_ids":      input_ids,
        "labels":         label_ids,
        "cot_labels":     cot_labels,
        "answer_labels":  answer_labels,
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


class CotDataCollator:
    """Wraps DataCollatorForSeq2Seq and pads cot_labels/answer_labels tensors
    with IGNORE_INDEX so they match the padded input_ids length."""

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

    def __init__(self, eval_dataset, max_length=8192, eval_batch_size=4, cot_budget=400):
        self._dataset_list = list(eval_dataset)
        self.max_length = max_length
        self.eval_batch_size = eval_batch_size
        self.cot_budget = cot_budget

    def _run_generation_metrics(self, state: TrainerState, model, tokenizer):
        global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        local_rank  = int(os.environ.get("LOCAL_RANK", 0))
        device = f'cuda:{local_rank}'

        dataset_list = self._dataset_list[:3]
        gts = [f['answer'] for f in dataset_list]

        generated_texts, reference_texts, example_rows = [], [], []

        orig_padding_side = tokenizer.padding_side
        tokenizer.padding_side = 'left'

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        torch.cuda.empty_cache()

        for i, (f, gt) in enumerate(zip(dataset_list, gts)):
            # Phase 1: prime the model with the CoT trigger, give it cot_budget tokens
            phase1_prompt = (
                build_llama3_prompt(
                    tokenizer, inject_reasoning_instruction(f['prompt'].strip())
                ) + ' ' + COT_TRIGGER
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
            cot_text = tokenizer.decode(out1[0][in_len1:], skip_special_tokens=True).strip()

            # If the model echoes its own context (common in early training), cut off
            # at the first prompt-structure marker before it contaminates extraction.
            for _stop in ('<|eot_id|>', '<|start_header_id|>', '<|end_header_id|>'):
                _idx = cot_text.find(_stop)
                if _idx != -1:
                    cot_text = cot_text[:_idx].strip()

            # Phase 2: if model didn't close CoT, force COT_END then answer.
            # Synchronize this decision across all ranks to avoid asymmetric
            # NCCL op counts under ZeRO-3 (fp16 can yield different outputs per rank).
            run_phase2 = torch.tensor(
                int(COT_END.lower() not in cot_text.lower()), device=device
            )
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(run_phase2, op=torch.distributed.ReduceOp.MAX)
            if run_phase2.item():
                # Build the phase-2 prompt via build_llama3_prompt so the completed
                # CoT block appears as the assistant role ending with COT_END,
                # matching the two-turn format used during training.
                phase2_prompt = build_llama3_prompt(
                    tokenizer,
                    inject_reasoning_instruction(f['prompt'].strip()),
                    assistant_content=build_cot_assistant_content(cot_text),
                )
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
                for _stop in ('<|eot_id|>', '<|start_header_id|>', '<|end_header_id|>'):
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
        torch.cuda.empty_cache()

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

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self._run_generation_metrics(state, kwargs['model'], kwargs['tokenizer'])


def load_local_dataset(path):
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
        # use_gradient_checkpointing=False skips the OOM-inducing float32 cast of
        # layer norms / embeddings; we re-enable GC manually below.
        # enable_input_require_grads() is normally called inside
        # prepare_model_for_kbit_training when use_gradient_checkpointing=True;
        # since we pass False here, we must call it ourselves so that gradients
        # can flow back through the frozen base model to the LoRA adapters.
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        torch.cuda.empty_cache()
        gc.collect()
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
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
        tokenized = [tokenize_fn(f) for f in source]
        return datasets.Dataset.from_dict({
            'input_ids':      [t['input_ids']      for t in tokenized],
            'labels':         [t['labels']          for t in tokenized],
            'cot_labels':     [t['cot_labels']      for t in tokenized],
            'answer_labels':  [t['answer_labels']   for t in tokenized],
        })

    dataset_train = build_dataset(dataset_train_untoken)
    dataset_test  = build_dataset(dataset_test_untoken)

    if global_rank == 0:
        log_sample_inputs(tokenizer, dataset_train)

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
        eval_steps=args.eval_steps,
        bf16=True,
        evaluation_strategy=args.evaluation_strategy,
        group_by_length=True,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to='wandb',
        run_name=args.run_name,
        ddp_find_unused_parameters=False,
        deepspeed=args.ds_config if args.ds_config else None,
    )

    if not args.load_in_4bit:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.model.config.use_cache = False

    # If checkpoint is a PEFT adapter dir, load adapter weights directly.
    # Do NOT pass the path to trainer.train() in this case — DeepSpeed would
    # also try to load model weights from the ZeRO-3 per-rank shard files
    # (global_stepN/zero_pp_rank_X_model_states.pt), which store torch.Size([0])
    # placeholders for params not owned by that rank, causing a size mismatch
    # against the full LoRA weight shapes.
    peft_resume_path = None
    if args.resume_from_checkpoint and _is_peft_adapter_dir(args.resume_from_checkpoint):
        model = PeftModel.from_pretrained(
            model, args.resume_from_checkpoint, is_trainable=True,
        )
        # Model weights loaded; don't let Trainer/DeepSpeed reload from ZeRO shards.
        peft_resume_path = None
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=lora_module_dict['llama3'],
            bias='none',
        )
        model = get_peft_model(model, peft_config)
        peft_resume_path = args.resume_from_checkpoint or None
    if not args.load_in_4bit:
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
            ),
            SavePeftModelCallback(),
        ],
    )

    torch.cuda.empty_cache()
    trainer.train(resume_from_checkpoint=peft_resume_path)
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
    parser.add_argument("--evaluation_strategy", default='steps', type=str)
    parser.add_argument("--eval_steps", default=0.1, type=float)
    parser.add_argument("--hf_token", default=None, type=str,
                        help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--gen_eval_batch_size", default=4, type=int,
                        help="Batch size for GenerationEvalCallback — all GPUs process the same padded batch")
    parser.add_argument("--mask_cot_loss", action="store_true",
                        help="Supervise only the final answer tokens; by default loss is on both CoT reasoning and final answer")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="QLoRA: load base model in 4-bit NF4 to cut VRAM ~50%% and allow larger batches")
    parser.add_argument("--resume_from_checkpoint", default=None, type=str,
                        help="Resume training from a checkpoint. Accepts either a PEFT adapter "
                             "directory (must contain adapter_config.json) or a Trainer output "
                             "directory whose checkpoint-N subdirs hold DeepSpeed ZeRO state.")
    parser.add_argument("--cot_budget", default=700, type=int,
                        help="Max new tokens for CoT reasoning in phase-1 generation. "
                             "If model doesn't emit assistantfinal within this budget, "
                             "it is forcibly appended and answer generation continues.")
    args = parser.parse_args()

    wandb.login()
    main(args)
