# Quantize Model
# time ~/llm/llama.cpp/quantize /models/shisa-7b-v1/ggml-model-f16.gguf /models/shisa-7b-v1/q8.gguf q8_0

# llama-bench
time ~/llm/llama.cpp/llama-bench -m /models/shisa-7b-v1/q8.gguf -p 512 -n 512

# JA
time ~/llm/llama.cpp/main -m /models/shisa-7b-v1/q8.gguf -ngl 99 -f ja_512.txt -n 512 -c 4096

# EN
time ~/llm/llama.cpp/main -m /models/shisa-7b-v1/q8.gguf -ngl 99 -f en_512.txt -n 512 -c 4096
