# Quantize Model
# mkdir /models/shisa-7b-v1-exl2-4.63bpw
# cd exllamav2
# time python convert.py -i /models/shisa-7b-v1 -o /models/shisa-7b-v1-exl2-4.63bpw -b 4.63 -m /models/shisa-7b-v1-exl2-8.30bpw/measurement.json -c /models/llm/datasets/augmxnt_shisa-en-ja-dpo-v1/dataset.parquet

### Slightly customized test_inference.py

# JA
python exllama-test.py -m /models/shisa-7b-v1-exl2-4.63bpw -lang ja

# EN
python exllama-test.py -m /models/shisa-7b-v1-exl2-4.63bpw -lang en
