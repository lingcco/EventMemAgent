# Training

The reference recipe uses GRPO with asynchronous vLLM rollouts, a Qwen3-VL 8B base model, multi-turn tool calls, and the EventMemAgent reward function.

Memory preprocessing uses a separate frozen Qwen3-VL-4B-Instruct model for event captions and Qwen3-Embedding-0.6B for event/query embeddings. The 4B caption model is not the policy optimized by GRPO.

The paper setting samples videos at 1 FPS, keeps at most 32 STM frames, enables boundary detection after at least 8 frames in the current event, and starts a new event when grayscale-histogram correlation falls below 0.2. Caption generation samples at most 32 event frames, uses temperature 0.7 and at most 256 new tokens. Memory search returns up to three events with cosine similarity at least 0.3.

## Inputs

Set at least these values in `.env`:

```dotenv
BASE_MODEL_PATH=checkpoints/pretrained/Qwen3-VL-8B-Instruct
MEMORY_CAPTION_MODEL_PATH=checkpoints/pretrained/Qwen3-VL-4B-Instruct
EMBEDDING_MODEL_PATH=checkpoints/pretrained/Qwen3-Embedding-0.6B
TRAIN_DATA=data/train/eventmemagent_train.parquet
VAL_DATA=
TRAIN_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8
TRAIN_TENSOR_PARALLEL_SIZE=8
```

### Train with the released data

No additional validation dataset is required for this command:

```bash
bash scripts/train/train.sh --dry-run
bash scripts/train/train.sh
```

Leave `VAL_DATA` empty when training without validation. The trainer does not
load a validation dataset or create a validation loader in this case. No training
row or benchmark input is substituted as validation data. Training rows, batch
sizes, reward, optimizer, rollout sampling and checkpoint frequency are unchanged.
The backend compatibility change is limited to making validation data optional.

If you want validation, supply your own held-out memory parquet through
`VAL_DATA`. The optional `data/val/` path is not a
released split. Configure the desired `trainer.test_freq` explicitly.
Use `--no-validation` to explicitly ignore `VAL_DATA` from an existing `.env`.

OCR and detection are part of the rollout environment, so their services must be healthy before training starts. Run them on separate GPUs or separate hosts and set `OCR_API_URL`, `DETECTION_API_URL`, `OCR_HEALTH_URL`, and `DETECTION_HEALTH_URL` in `.env`. The launcher intentionally does not start these services on the training GPUs.

The canonical Hydra configuration is `configs/train.yaml`. Paths use OmegaConf environment interpolation, so the config remains portable.

## Overrides

Any additional argument is forwarded to Hydra:

```bash
bash scripts/train/train.sh --no-validation \
  data.train_batch_size=32 \
  actor_rollout_ref.rollout.n=4 \
  trainer.save_freq=20
```

Checkpoints, validation generations, and TensorBoard logs are written under `TRAIN_OUTPUT_DIR`. These outputs are ignored by Git.

Training resumes from the latest checkpoint in `TRAIN_OUTPUT_DIR` by default. To start a clean run, set `RESUME_MODE=disable` and use an empty output directory. To resume an explicit checkpoint:

```bash
RESUME_MODE=resume_path \
RESUME_FROM_PATH=runs/train/eventmemagent_qwen3vl_8b/global_step_90 \
  bash scripts/train/train.sh --no-validation
```

verl stores sharded FSDP training checkpoints. Convert the latest saved actor to standard Hugging Face format with:

```bash
bash scripts/train/merge_checkpoint.sh
```

Conversion only writes a local checkpoint. To upload as well, set a real `HF_MODEL_REPO` and explicitly pass `UPLOAD_TO_HF=1`. Merely configuring the destination never publishes files.

## Paper-to-code mapping

The archived 8B recipe uses `ppo_mini_batch_size=32` and
`ppo_micro_batch_size_per_gpu=1`. The appendix calls the value 32 “Micro Batch
Size”; this release keeps the actual code's distinction instead of setting
32 samples per GPU. Prompt/response limits are 8192/8192, the group size is 8,
and the policy makes at most 10 assistant turns. The reward implementation is
the archived 8B version: answer reward, zero tool bonus, and a 0.1 penalty for
multiple boxed answers. Do not substitute the earlier 4B tool-reward experiment.

Pass an explicit actor directory to `merge_checkpoint.sh` when selecting a
checkpoint for release; the latest saved step is not necessarily the step you
intend to publish.

## Preprocessing

The original memory construction pipeline is retained under `eventmemagent/memory/preprocess.py`. It builds event boundaries, captions, event embeddings, representative frames, short-term memory, and tool arguments. Before a public data release, record the captioning model revision and preprocessing parameters in the Hugging Face dataset card.

For already processed release parquet, the 4B caption server is not needed during
GRPO: the memories are stored in each row. For raw videos, start
`scripts/services/memory.sh` and use the preprocessing command
below (the input JSON follows the original pipeline's annotation schema):

```bash
INPUT_JSON=data/raw/input.json VIDEO_DIR=data/raw/videos \
  bash scripts/data/preprocess.sh
```
