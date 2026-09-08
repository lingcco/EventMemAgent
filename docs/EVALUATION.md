# Evaluation

## OVO-Bench

Set `MODEL_PATH`, `EMBEDDING_MODEL_PATH`, `OVOBENCH_DATA`, and `OVOBENCH_METADATA` in `.env`.

```bash
bash scripts/eval/ovobench.sh
bash scripts/eval/score_ovobench.sh runs/eval/ovobench/<run>/eval_results.json
```

The evaluator writes `eval_results.json`, a CSV summary, intermediate checkpoints, and aggregate statistics. The scorer reports Backward Tracing, Real-time Visual Perception, Forward Active Responding, and the arithmetic mean used by the project.

## StreamingBench

For precomputed-memory evaluation, `STREAMINGBENCH_PARQUET` points to
the 4B-memory parquet and `STREAMINGBENCH_DATA` points to `questions_real.json` for
scoring. The parquet contains the frame payloads; raw-video paths are used as
sample identifiers, not opened during this mode.

```bash
bash scripts/eval/streamingbench.sh
bash scripts/eval/score_streamingbench.sh \
  runs/eval/streamingbench/<run>/eval_results.json
```

StreamingBench reports both weighted accuracy and task-macro accuracy. The
paper's overall column uses weighted accuracy. Do not interchange them.
Unmatched, missing, and duplicated predictions are checked; use `--allow-partial`
only when scoring diagnostic subsets.

For an alternative raw-video evaluation, start the independent frozen 4B caption
server, make the video paths in `questions_real.json` accessible, install `ffmpeg`,
and run `INPUT_MODE=online bash scripts/eval/streamingbench.sh`.
This regenerates memory and is not the same as reusing archived parquet.

`EVAL_TENSOR_PARALLEL_SIZE` defaults to 1 and is separate from training TP=8.
Set `EVAL_CUDA_VISIBLE_DEVICES`, `EVAL_GPU_MEMORY_UTILIZATION`, and
`EVAL_MAX_MODEL_LEN` for the evaluation machine. The defaults use greedy decoding;
set `TEMPERATURE`, `MAX_TURNS`, and `MAX_NEW_TOKENS` explicitly when comparing
different configurations. Benchmark launchers default to 10 assistant turns and
16,384 new tokens per turn; standalone inference defaults to 5 turns and 2,048
tokens. The raw-video StreamingBench entry point defaults to 5 turns.

## Reproducibility metadata

For reported numbers, retain:

- Git commit and checkpoint revision.
- Hugging Face dataset revision or local dataset path.
- GPU model/count, CUDA, PyTorch, Transformers, and vLLM versions.
- Decoding parameters, maximum turns, context time, and tool service revisions.
- Raw per-sample predictions in addition to aggregate metrics.

These are local experiment records, not files to include in the public dataset
release. Hugging Face evaluation datasets contain inputs and reference
annotations only; GitHub contains the evaluation and scoring scripts.
