from   contextlib import contextmanager
import time
import torch
from   transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer


@contextmanager
def time_block(label):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"{label} took {end - start} seconds")

    # Some code to time
    time.sleep(2)

model_name = "augmxnt/shisa-7b-v1"


with time_block("Load Tokenizer"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


with time_block("Load Model"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )

# Read prompts
with open('ja_512.txt') as file:
    ja_text = file.read()
with open('en_512.txt') as file:
    en_text = file.read()


def generate(label, text):
    print(label)
    print('---')
    with time_block("Prompt Processing 512 tokens"):
        inputs = tokenizer.encode(text, return_tensors="pt")
    with time_block("Copy to GPU"):
        # For multi-GPU, find the device of the first parameter of the model
        first_param_device = next(model.parameters()).device
        inputs = inputs.to(first_param_device)
    with time_block("Inference 512 tokens"):
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                early_stopping=False,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                temperature=0.5,
                repetition_penalty=1.15,
                top_p=0.95,
                do_sample=True,
            )
    with time_block("Decode new output"):
        new_tokens = outputs[0, inputs.size(1):]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

generate('JA', ja_text)
generate('EN', en_text)
