# Convert Model
# time python convert_shisa.py /models/shisa-7b-v1 --outtype f16 --vocabtype spm

# llama-bench
# ❯ time ~/llm/llama.cpp/llama-bench -m /models/shisa-7b-v1/ggml-model-f16.gguf -p 512 -n 512

# JA
time ~/llm/llama.cpp/main -m /models/shisa-7b-v1/ggml-model-f16.gguf -ngl 99 -f ja_512.txt -n 512 -c 4096

# EN
time ~/llm/llama.cpp/main -m /models/shisa-7b-v1/ggml-model-f16.gguf -ngl 99 -f en_512.txt -n 512 -c 4096
