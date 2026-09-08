# Paper settings

The standard recipe is `configs/train.yaml`. Tool settings are in
`configs/tools.yaml`.

| Component | Setting |
| --- | --- |
| Trainable policy | Qwen3-VL-8B-Instruct |
| Frozen event-caption model | Qwen3-VL-4B-Instruct |
| Event/query embeddings | Qwen3-Embedding-0.6B |
| Perception tools | DeepSeek-OCR, Grounding DINO |
| Video sampling | 1 FPS |
| Short-term memory | At most 32 frames |
| Event segmentation | Minimum event length 8, content threshold 0.2 |
| Memory retrieval | Top 3 events, similarity threshold 0.3 |
| Training | Multi-turn GRPO, one epoch |
| Optimizer | AdamW, learning rate 1e-6, five warmup steps |
| Global batch / rollout group | 64 / 8 |
| PPO minibatch / per-GPU microbatch | 32 / 1 |
| KL loss / KL reward | Disabled |
| Prompt / response training budget | 8192 / 8192 tokens |
| Maximum training assistant turns | 10 |
| Checkpoint interval | 30 steps |
| Paper hardware | Eight 80GB A100 GPUs |

The appendix's batch-size entry of 32 corresponds to the PPO minibatch in the
training implementation, not 32 samples per GPU. Keep
`ppo_micro_batch_size_per_gpu=1` when using the standard recipe.

## Memory and prompts

Training consumes the prompts and precomputed memories in the training parquet.
The frozen 4B caption model is used when building those memories; it is not the
trainable policy. Evaluation on prepared parquet also uses precomputed memories.
Raw-video inference requires the separate caption service.

Standalone inference constructs its system prompt in
`eventmemagent/agent/video_agent.py`. Preserve each entry point's prompt behavior when
comparing experiments. Evaluation and inference parameters can be set explicitly
through the launch scripts; see [Evaluation](EVALUATION.md).

## Reproducing a run

1. Install the versions in `constraints-training.txt` and a compatible CUDA stack.
2. Configure model, data and service paths in `.env`.
3. Use `--no-validation` with the public training data. If validation is desired,
   provide a held-out split; never substitute benchmark test data.
4. Inspect the fully resolved configuration before starting training:

   ```bash
   bash scripts/train/train.sh --no-validation --dry-run
   ```

5. Retain your chosen checkpoint step, resolved configuration, tool versions and
   decoding settings with local experiment records.

The `data/val/` location is a placeholder, not a bundled validation dataset.
Changing hardware, precision, batch size, prompts, memory construction or sampling
settings can affect results. A small runtime check is not a full training or
benchmark reproduction.
