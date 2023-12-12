We will use `shisa-7b` to test inference performance

Full spreadsheet:
https://docs.google.com/spreadsheets/d/19YaxXkMJu7VweJihBMxQfMuz290Q3VxpqeG2DYCdRws/edit?usp=sharing

All tests run on a Ryzen 5950X workstation w/ an RTX 4090 and RTX 3090 w/ CUDA 12.3.1 ~ 2023-12-10

* Python 3.11.5
* HF Transfromers 4.35.2
* vLLM 0.2.3
* cTranslate2 3.23.0
* llama.cpp fe680e3 (1620)
* ExLlamaV2 0.0.10

| Software        | Settings                              | Avg Tok/s   | Max Mem   | Speed X   | Max mem %          | Notes                                                                                                                                          |
|:----------------|:--------------------------------------|:------------|:----------|:----------|:-------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------|
| HF Transformers | Baseline (FP32)                       | 1.48        | 47677.0   | 1.0       | 100.0              |                                                                                                                                                |
| HF Transformers | BF16                                  | 3.88        | 46211.0   | 2.63      | 97.0               |                                                                                                                                                |
| HF Transformers | BF16                                  | 3.88        | 46211.0   | 2.63      | 97.0               |                                                                                                                                                |
|                 | torch.no_grad()                       |             |           |           |                    |                                                                                                                                                |
| HF Transformers | BF16                                  | 3.89        | 45495.0   | 2.63      | 95.0               |                                                                                                                                                |
|                 | torch.inference_mode()                |             |           |           |                    |                                                                                                                                                |
| HF Transformers | BF16                                  | 4.32        | 47191.0   | 2.93      | 99.0               |                                                                                                                                                |
|                 | torch.inference_mode()                |             |           |           |                    |                                                                                                                                                |
|                 | use_flash_attention_2=True            |             |           |           |                    |                                                                                                                                                |
| HF Transformers | BF16                                  |      |           |           |                           | BetterTransformers doesn't support Mistral                                                                                                                                               |
|                 | torch.inference_mode()                |             |           |           |                    |                                                                                                                                                |
|                 | use_flash_attention_2=True            |             |           |           |                    |                                                                                                                                                |
|                 | Optimum BetterTransformer             |             |           |           |                    |                                                                                                                                                |
| HF Transformers | BF16                                  | 4.3         | 46851.0   | 2.91      | 98.0               |                                                                                                                                                |
|                 | torch.inference_mode()                |             |           |           |                    |                                                                                                                                                |
|                 | use_flash_attention_2=True            |             |           |           |                    |                                                                                                                                                |
|                 | SDPA flash attention                  |             |           |           |                    |                                                                                                                                                |
| HF Transformers | BF16                                  | 2.07        | 42623.0   | 1.4       | 89.0               | UserWarning: MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization                                              |
|                 | load_in_8bit=True                     |             |           |           |                    |                                                                                                                                                |
|                 | torch.inference_mode()                |             |           |           |                    |                                                                                                                                                |
|                 | use_flash_attention_2=True            |             |           |           |                    |                                                                                                                                                |
|                 | SDPA flash attention                  |             |           |           |                    |                                                                                                                                                |
| HF Transformers | BF16                                  | 2.06        | 45127.0   | 1.4       | 95.0               | UserWarning: MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization                                              |
|                 | load_in_8bit=True                     |             |           |           |                    |                                                                                                                                                |
|                 | bnb_8bit_compute_dtype=torch.bfloat16 |             |           |           |                    |                                                                                                                                                |
|                 | torch.inference_mode()                |             |           |           |                    |                                                                                                                                                |
|                 | use_flash_attention_2=True            |             |           |           |                    |                                                                                                                                                |
|                 | SDPA flash attention                  |             |           |           |                    |                                                                                                                                                |
| HF Transformers | FP16                                  | 2.13        | 45511.0   | 1.44      | 95.0               | https://github.com/TimDettmers/bitsandbytes/issues/490                                                                                         |
|                 | load_in_8bit=True                     |             |           |           |                    |                                                                                                                                                |
|                 | bnb_4bit_compute_dtype=torch.float16  |             |           |           |                    |                                                                                                                                                |
|                 | torch.inference_mode()                |             |           |           |                    |                                                                                                                                                |
|                 | use_flash_attention_2=True            |             |           |           |                    |                                                                                                                                                |
|                 | SDPA flash attention                  |             |           |           |                    |                                                                                                                                                |
| HF Transformers | FP16                                  | 2.11        | 45509.0   | 1.43      | 95.0               |                                                                                                                                                |
|                 | load_in_8bit=True                     |             |           |           |                    |                                                                                                                                                |
|                 | bnb_4bit_compute_dtype=torch.float16  |             |           |           |                    |                                                                                                                                                |
|                 | torch.compile()                       |             |           |           |                    |                                                                                                                                                |
|                 | torch.inference_mode()                |             |           |           |                    |                                                                                                                                                |
|                 | use_flash_attention_2=True            |             |           |           |                    |                                                                                                                                                |
|                 | SDPA flash attention                  |             |           |           |                    |                                                                                                                                                |
| HF Transformers | FP16                                  | 3.51        | 44101.0   | 2.37      | 92.0               |                                                                                                                                                |
|                 | load_in_4bit=True                     |             |           |           |                    |                                                                                                                                                |
|                 | bnb_4bit_compute_dtype=torch.float16  |             |           |           |                    |                                                                                                                                                |
|                 | torch.compile()                       |             |           |           |                    |                                                                                                                                                |
|                 | torch.inference_mode()                |             |           |           |                    |                                                                                                                                                |
|                 | use_flash_attention_2=True            |             |           |           |                    |                                                                                                                                                |
|                 | SDPA flash attention                  |             |           |           |                    |                                                                                                                                                |
| vLLM            | tensor_parallel_size=1                | 55.28       | 19958.0   | 37.44     | 42.0               | vLLM is fast even for batch=1 but you need to batch by SamplerSettings and also you can't batch w/ multiple seeds                              |
| vLLM            | tensor_parallel_size=2                | 68.31       | 47843.0   | 46.27     | 100.0              | A copy on each GPU                                                                                                                             |
| vLLM            | tensor_parallel_size=2                | 86.81       | 47175.0   | 58.8      | 99.0               |                                                                                                                                                |
|                 | quantization='awq'                    |             |           |           |                    |                                                                                                                                                |
| vLLM            | tensor_parallel_size=2                |             |           |           |                    | NotImplementedError: Pipeline parallelism is not supported yet.                                                                                |
|                 | pipeline_parallel_size=2              |             |           |           |                    |                                                                                                                                                |
|                 | quantization='awq'                    |             |           |           |                    |                                                                                                                                                |
| cTranslate2     |                                       | 55.86       | 16996.0   | 37.84     | 36.0               | requires model conversion: https://opennmt.net/CTranslate2/conversion.html                                                                     |
|                 |                                       |             |           |           |                    |                                                                                                                                                |
|                 |                                       |             |           |           |                    | Missing some of the usual generation parameters, 4090 only                                                                                     |
| llama.cpp       | fp16                                  | 40.93       | 17987.0   | 27.72     | 38.0               | convert_shisa.py                                                                                                                               |
|                 |                                       |             |           |           |                    | 4090+3090                                                                                                                                      |
| llama.cpp       | fp16                                  | 54.6        | 15873.0   | 36.98     | 33.0               | 4090 only                                                                                                                                      |
| llama.cpp       | q8                                    | 48.95       | 11541.0   | 33.15     | 24.0               | 4090+3090                                                                                                                                      |
| llama.cpp       | q8                                    | 87.85       | 9919.0    | 59.49     | 21.0               | 4090 only                                                                                                                                      |
| llama.cpp       | q4_k_m                                | 53.08       | 8271.0    | 35.94     | 17.0               | 4.63 BPW                                                                                                                                       |
|                 |                                       |             |           |           |                    | 4090+3090                                                                                                                                      |
| llama.cpp       | q4_k_m                                | 126.67      | 6701.0    | 85.78     | 14.0               | 4090 only                                                                                                                                      |
| ExLLamaV2       | EXLV2 8 BPW                           | 92.94       | 13688.0   | 62.96     | 29.0               | 4090 only                                                                                                                                      |
| ExLLamaV2       | EXLV2 4.63 BPW                        | 134.4       | 10856.0   | 91.06     | 23.0               | 4090 only                                                                                                                                      |
| ExLLamaV2       | GPTQ Q4 GS128 actorder                | 131.57      | 10938.0   | 89.12     | 23.0               | 4090 only                                                                                                                                      |
| MLC LLM         | q0f16                                 |             |           |           |                    |                                                                                                                                                |
| MLC LLM         | q8f16_1                               |             |           |           |                    |                                                                                                                                                |
| MLC LLM         | q4f16_1                               |             |           |           |                    | mlc_chat_cli: symbol lookup error: ... mlc-llm/dist/shisa-7b-v1-q4f16_1/shisa-7b-v1-q4f16_1-cuda.so: undefined symbol: __cudaRegisterFatBinary |
| MLC LLM         | autogptq_llama_q4f16_1                |             |           |           |                    |                                                                                                                                                |
| gpt-fast        |                                       |             |           |           |                    | many issues...                                                                                                                                               |
