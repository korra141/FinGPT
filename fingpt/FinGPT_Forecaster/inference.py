from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizerFast
from peft import PeftModel  # 0.5.0
import torch
import os
from huggingface_hub import login

# Hugging Face Token
# hf_token = os.getenv('HF_TOKEN')  # Set HF_TOKEN environment variable
hf_token = "REDACTED_HF_TOKEN"
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN environment variable not set. Some models may require authentication.")

base_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-chat-hf',
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,   # optional if you have enough VRAM
)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')

model = PeftModel.from_pretrained(base_model, 'FinGPT/fingpt-forecaster_dow30_llama2-7b_lora')
model = model.eval()

# Load Models
#base_model = "meta-llama/Meta-Llama-3-8B" 
#peft_model = "FinGPT/fingpt-mt_llama3-8b_lora"
#tokenizer = LlamaTokenizerFast.from_pretrained(base_model, trust_remote_code=True, token=hf_token)
#tokenizer.pad_token = tokenizer.eos_token
#model = LlamaForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map="cuda:0", token=hf_token)
#model = PeftModel.from_pretrained(model, peft_model, token=hf_token)
#model = model.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
tokens = tokenizer(prompt, return_tensors='pt', padding=True, max_length=512).to(device)
res = model.generate(**tokens, max_length=512)
res_sentences = [tokenizer.decode(i) for i in res]
out_text = [o.split("Answer: ")[1] for o in res_sentences]

# Make prompts
prompt = [
'''Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
Input: FINANCING OF ASPOCOMP 'S GROWTH Aspocomp is aggressively pursuing its growth strategy by increasingly focusing on technologically more demanding HDI printed circuit boards PCBs .
Answer: ''',
'''Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
Input: According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .
Answer: '''
]


# Show results
for sentiment in out_text:
    print(sentiment)
