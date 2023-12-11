# Quantize Model
# mkdir /models/shisa-7b-v1-exl2
# cd exllamav2
# time python convert.py -i /models/shisa-7b-v1 -o /models/shisa-7b-v1-exl2 -c /models/llm/datasets/augmxnt_shisa-en-ja-dpo-v1/dataset.parquet -b 8.0 -hb 8

### Slightly customized test_inference.py

# JA
python exllama-test.py -m /models/shisa-7b-v1-exl2 -lang ja

# EN
python exllama-test.py -m /models/shisa-7b-v1-exl2 -lang en
