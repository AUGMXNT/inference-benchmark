from   contextlib import contextmanager
import time
import torch
from   transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


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
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        # torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        torch_dtype=torch.float16,
        device_map="auto",
        use_flash_attention_2=True,
        quantization_config=quantization_config,
    )
    # https://pytorch.org/get-started/pytorch-2.0/#user-experience
    model = torch.compile(model)

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
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
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
