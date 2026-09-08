---
task_categories:
  - visual-question-answering
language:
  - en
tags:
  - video
  - eventmemagent
configs:
  - config_name: training
    data_files:
      - split: train
        path: train/eventmemagent_train.parquet
  - config_name: ovobench
    data_files:
      - split: test
        path: eval/ovobench/eventmemagent_ovobench.parquet
  - config_name: streamingbench
    data_files:
      - split: test
        path: eval/streamingbench/eventmemagent_streamingbench.parquet
---

# EventMemAgent processed data

Prepared memories and question annotations for the EventMemAgent training and
evaluation pipeline, hosted at
[lingcco/EventMemAgent](https://huggingface.co/datasets/lingcco/EventMemAgent).

| File | Rows | Role |
| --- | ---: | --- |
| train/eventmemagent_train.parquet | 10,000 | MovieChat / VideoMarathon training data |
| eval/ovobench/eventmemagent_ovobench.parquet | 3,035 | OVO-Bench evaluation |
| eval/streamingbench/eventmemagent_streamingbench.parquet | 2,500 | StreamingBench evaluation |

Include `eval/ovobench/ovo_bench_new.json` and
`eval/streamingbench/questions_real.json` for scoring.

## Construction

1 FPS, at most 32 STM frames, minimum event length 8, histogram-correlation
threshold 0.2; frozen Qwen3-VL-4B-Instruct event captions and
Qwen3-Embedding-0.6B event embeddings. The released files preserve the processed
memories and prompts used by the original pipeline, without rebuilding them.

## Schema

The nested parquet fields and stable local layout are documented in the GitHub
repository's `data/README.md`. Preserve `prompt`, `agent_name=tool_agent`,
`short_term_memory`, `long_term_memory`, `reward_model`, and
`extra_info.tools_kwargs`. The images are embedded in the memory records.

## Rights and limitations

The inputs derive from MovieChat, VideoMarathon, OVO-Bench and StreamingBench.
Original video frames and annotations remain subject to their upstream licenses
and usage terms. This release grants no additional rights to third-party content;
the code license does not automatically license the datasets. Keep evaluation
splits out of policy optimization and checkpoint selection.

Full video files, generated predictions, scoring outputs and training logs are
not included. The embedded memory images are required by the agent and its tools.

```python
from datasets import load_dataset

train = load_dataset("lingcco/EventMemAgent", "training", split="train", streaming=True)
ovo = load_dataset("lingcco/EventMemAgent", "ovobench", split="test", streaming=True)
streaming = load_dataset("lingcco/EventMemAgent", "streamingbench", split="test", streaming=True)
```
