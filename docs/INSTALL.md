# Installation

## Recommended environment

- Linux x86_64 with NVIDIA Ampere or newer GPUs (A100 was used for training)
- Python 3.12 (the released experiments used Python 3.12.0)
- CUDA Toolkit 12.8 and a compatible NVIDIA driver; PyTorch 2.8.0 / vLLM 0.11.0
- A working C++ compiler, Python development headers, and `nvcc` for FlashAttention
- `ffmpeg` for StreamingBench clip extraction

On Ubuntu/Debian, install `build-essential`, the development and venv packages
for your Python 3.12 installation, and `ffmpeg` with your system package manager.
Install CUDA Toolkit 12.8 separately if it is not already available. For a
standard toolkit installation, expose it before building FlashAttention:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
ffmpeg -version
nvidia-smi
```

Adjust `CUDA_HOME` to your actual toolkit path. The CUDA version shown by
`nvidia-smi` describes driver support, not whether the toolkit/compiler is
installed. The PyTorch wheel alone does not supply `nvcc`.

Create an isolated environment and install the local fork:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel packaging ninja
pip install -r requirements.txt
MAX_JOBS=4 pip install flash-attn==2.8.1 --no-build-isolation

pip check
python scripts/utils/validate_setup.py --mode runtime
```

`requirements.txt` installs the local training backend and the packages used by
EventMemAgent, including video decoding and tool services. Core versions are
constrained by `constraints-training.txt`; FlashAttention is installed afterward
because its build requires PyTorch.

The constraints also pin Accelerate, PEFT, Datasets, Hugging Face Hub, image/data
processing packages, and tool-service packages to the versions used in the
verified server environment. Do not upgrade PyTorch or
Transformers independently of this stack. `MAX_JOBS=4` limits host RAM use if
FlashAttention needs to compile from source. See the upstream
[vLLM 0.11.0 CUDA dependencies](https://github.com/vllm-project/vllm/blob/v0.11.0/requirements/cuda.txt)
and [FlashAttention 2.8.1 installation requirements](https://github.com/Dao-AILab/flash-attention/tree/v2.8.1#installation-and-features).

Copy the environment template:

```bash
cp .env.example .env
```

All stable launcher scripts load `.env` automatically. Relative paths are resolved from the repository root.

## Tool services

Memory retrieval runs in the agent process. OCR and object detection are HTTP services and should be started separately when those tools are enabled:

```bash
bash scripts/services/ocr.sh
bash scripts/services/detection.sh
```

Health endpoints are available at `http://localhost:8089/health` and `http://localhost:8090/health`. GPU IDs, ports, devices, and model paths can be overridden through `.env` or command-line flags.

The historical launcher uses the compatibility-updated DeepSeek-OCR artifact
described in `checkpoints/README.md`. The original OCR custom code is incompatible
with Transformers 4.57.3; changing the policy environment to make it import would
also change the Qwen3-VL training stack. Use the documented OCR artifact, or keep
an original-OCR deployment in a separate environment and label it as a different
tool configuration.

## Validate without loading models

```bash
python scripts/utils/validate_setup.py --mode repo
```

Before training, also run `python scripts/utils/validate_setup.py --mode runtime`.
It imports the compiled FlashAttention extension as well as PyTorch and vLLM, so
a PyTorch/FlashAttention ABI mismatch fails before distributed workers launch.
Rebuild FlashAttention whenever the PyTorch version changes; a package being
installed does not mean its compiled extension is compatible.

The `repo` mode checks required repository files, selected YAML configurations,
unresolved Git LFS pointers, legacy personal paths, and tracked generated
artifacts without importing CUDA-heavy packages. It is a lightweight hygiene
check, not a comprehensive secret scanner.
