from   contextlib import contextmanager
import time
import torch
from   transformers import AutoTokenizer
import ctranslate2

# ct2-transformers-converter --model augmxnt/shisa-7b-v1 --output_dir /models/shisa-7b-v1-ct2


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

model_name = "/models/shisa-7b-v1-ct2"


with time_block("Load Tokenizer"):
    tokenizer = AutoTokenizer.from_pretrained('augmxnt/shisa-7b-v1', use_fast=True)


with time_block("Load Model"):
    model = ctranslate2.Generator(model_name, device='cuda')

# Read prompts
with open('ja_512.txt') as file:
    ja_text = file.read()
with open('en_512.txt') as file:
    en_text = file.read()

def generate(label, text):
    print(label)
    print('---')
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))

    '''
    TypeError: generate_batch(): incompatible function arguments. The following argument types are supported:
    1. (self: ctranslate2._ext.Generator, start_tokens: List[List[str]], *, max_batch_size: int = 0, batch_type: str = 'examples', asynchronous: bool = False, beam_size: int = 1, patience: float = 1, num_hypotheses: int = 1, length_penalty: float = 1, repetition_penalty: float = 1, no_repeat_ngram_size: int = 0, disable_unk: bool = False, suppress_sequences: Optional[List[List[str]]] = None, end_token: Optional[Union[str, List[str], List[int]]] = None, return_end_token: bool = False, max_length: int = 512, min_length: int = 0, static_prompt: Optional[List[str]] = None, cache_static_prompt: bool = True, include_prompt_in_result: bool = True, return_scores: bool = False, return_alternatives: bool = False, min_alternative_expansion_prob: float = 0, sampling_topk: int = 1, sampling_topp: float = 1, sampling_temperature: float = 1, callback: Callable[[ctranslate2._ext.GenerationStepResult], bool] = None) -> Union[List[ctranslate2._ext.GenerationResult], List[ctranslate2._ext.AsyncGenerationResult]]
    '''

    with time_block("Inference 512 tokens"):
        outputs = model.generate_batch([tokens], 
            max_length=512,
            min_length=512,
            sampling_temperature=0.4,
            sampling_topp=0.95,
            repetition_penalty=1.05,
        )


generate('JA', ja_text)
generate('EN', en_text)
