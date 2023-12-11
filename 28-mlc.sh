```
# MLC sucks to configure...

# env
mamba create -n mlc
mamba activate mlc
mamba install -c "nvidia/label/cuda-12.3.1" cuda-toolkit cuda-cudart
mamba install gxx=12.2 ninja cmake
conda env config vars set CUDA_PATH="$CONDA_PREFIX"
conda env config vars set CUDA_HOME="$CONDA_PREFIX"
mamba activate mlc

# CLI
 mamba install -c mlc-ai mlc-chat-cli-nightly

# TVM
# https://llm.mlc.ai/docs/install/tvm.html#install-tvm-unity
python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-chat-nightly-cu122 mlc-ai-nightly-cu122 

# Then install package from source
# https://llm.mlc.ai/docs/compilation/compile_models.html#install-mlc-llm-package
git clone https://github.com/mlc-ai/mlc-llm.git --recursive
cd mlc-llm
pip install .
python3 -m mlc_llm.build --help
python -c "import tvm; print(tvm.cuda().exist)"


# Compile Quant - needs --use-safetensors

time python3 -m mlc_llm.build --model /models/shisa-7b-v1 --target cuda --quantization q4f16_1 --use-safetensors


# Fail...
mlc_chat_cli --model shisa-7b-v1-q4f16_1 --device cuda
mlc_chat_cli: symbol lookup error: /home/local/llm/mlc/mlc-llm/dist/shisa-7b-v1-q4f16_1/shisa-7b-v1-q4f16_1-cuda.so: undefined symbol: __cudaRegisterFatBinary

```
