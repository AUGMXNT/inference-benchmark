from   contextlib import contextmanager
import time
import torch
from   transformers import AutoTokenizer
from   vllm import LLM, SamplingParams


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
    model = LLM(model=model_name, tensor_parallel_size=2)

# Read prompts
with open('ja_512.txt') as file:
    ja_text = file.read()
with open('en_512.txt') as file:
    en_text = file.read()

def generate(label, text):
    print(label)
    print('---')
    with time_block("Inference 512 tokens"):
        sampling_params = SamplingParams(
            early_stopping=False,
            max_tokens=512,
            temperature=0.4,
            min_p=0.05,
            repetition_penalty=1.05,
            skip_special_tokens=True,
        )
        outputs = model.generate([text], sampling_params=sampling_params, use_tqdm=True)


generate('JA', ja_text)
generate('EN', en_text)
