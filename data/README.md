# Data

Dataset files are intentionally excluded from Git. This directory reserves stable paths for the public EventMemAgent release.

## Expected layout

```text
data/
├── train/
│   └── eventmemagent_train.parquet
├── val/
│   └── eventmemagent_validation.parquet  # optional, user-provided
├── eval/
│   ├── ovobench/
│   │   ├── eventmemagent_ovobench.parquet
│   │   └── ovo_bench_new.json
│   └── streamingbench/
│       ├── eventmemagent_streamingbench.parquet
│       └── questions_real.json
└── raw/
    └── ... video files referenced by the metadata ...
```

The training/evaluation Parquet files used by the offline agent contain the following core fields:

| Field | Description |
| --- | --- |
| `prompt` | Chat-format model input used by verl. |
| `question` | Human-readable question used by inference/evaluation. |
| `ground_truth` | Reference answer. |
| `agent_name` | `tool_agent`, selecting the multi-turn tool loop. |
| `reward_model` | Rule-based reward metadata and ground truth. |
| `data_source` | Reward routing identifier. |
| `short_term_memory` | Recent video frames and their metadata. |
| `long_term_memory` | Event records with time spans, captions, embeddings, and representative frames. |
| `extra_info` | Tool arguments, question type, source index, and reward metadata. |

Exact preprocessing code is under `eventmemagent/memory/preprocess.py`. The public Hugging Face dataset card documents the upstream datasets, preprocessing steps, schema, and redistribution constraints. Do not redistribute videos whose licenses do not allow it.

The inspected artifacts contain 10,000 training rows, 3,035 OVO-Bench rows, and
2,500 StreamingBench rows. STM entries contain `frame_index`, `timestamp`, and
an `image` payload. LTM entries contain event times, caption, embedding,
`first_frame_base64`, and change descriptions. Keep these nested structures intact.

No additional validation split is distributed or required for
`bash scripts/train/train.sh --no-validation`. The optional `data/val/` path is
for users who want held-out validation. Do not use benchmark test inputs there.

The [Hugging Face dataset](https://huggingface.co/datasets/lingcco/EventMemAgent)
is available. Download on your compute server:

```bash
hf download lingcco/EventMemAgent --repo-type dataset --local-dir data --include 'train/*' 'eval/*'
```

The three parquet files are approximately 28.5 GB (training), 4.8 GB (OVO-Bench)
and 9.2 GB (StreamingBench). They contain embedded memory images, not full videos.
`video_path` is source metadata; the path need not exist for memory-based runs.
