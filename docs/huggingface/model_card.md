---
license: apache-2.0
base_model: Qwen/Qwen3-VL-8B-Instruct
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - video-understanding
  - reinforcement-learning
  - eventmemagent
---

# EventMemAgent-8B

Model for *EventMemAgent: Hierarchical Event-Centric Memory for Online Video
Understanding with Adaptive Tool Use* (ECCV 2026).

- Code: https://github.com/lingcco/EventMemAgent
- Paper: https://arxiv.org/abs/2602.15329
- Dataset: [lingcco/EventMemAgent](https://huggingface.co/datasets/lingcco/EventMemAgent).
- Checkpoint: merged actor from run `1-30_8B`, step 150.
- License: Apache 2.0, following Qwen3-VL-8B-Instruct.

Only model weights and loading assets are included. Optimizer states, intermediate
checkpoints, training logs and evaluation predictions are not included.

```bash
hf download lingcco/EventMemAgent-8B --local-dir checkpoints/released/EventMemAgent-8B
```

## Training

Qwen3-VL-8B-Instruct policy, 10K MovieChat samples annotated by VideoMarathon,
GRPO group size 8, global batch 64, PPO minibatch 32, per-GPU microbatch 1,
AdamW at 1e-6 with five warmup steps, one epoch, no KL loss. Event captions are
generated separately by frozen Qwen3-VL-4B-Instruct.

## Evaluation

The paper reports OVO-Bench 60.75 and StreamingBench 77.00.
Use the evaluation and scoring scripts in `scripts/eval/` to evaluate this model.

## Use and limitations

Use the repository's evaluation scripts with processed memory parquet and the
OCR/detection services. Loading the checkpoint alone does not construct memory
or provide external tools. Answers and OCR/object detections can be incorrect;
this research model is not intended for safety-critical decisions.
