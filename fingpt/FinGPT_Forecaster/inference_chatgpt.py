from transformers import pipeline
import torch

model_id = "openai/gpt-oss-20b"

import torch
from transformers.utils.import_utils import is_triton_available
from huggingface_hub import login
import os
print(f"GPU Detected: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"Triton Available: {is_triton_available()}")

hf_token = "REDACTED_HF_TOKEN"
if hf_token:
    login(token=hf_token)

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
