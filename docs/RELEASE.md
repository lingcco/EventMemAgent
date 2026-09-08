# Publishing Models and Data

Large datasets and model weights are hosted on Hugging Face. GitHub contains the source code, configuration, documentation, and lightweight examples only.

## Dataset release

Create a Hugging Face dataset repository containing the training and evaluation files. Include a dataset card covering:

- dataset sources and licenses;
- preprocessing and memory construction;
- train/evaluation split names;
- parquet schema and a minimal loading example;
- known limitations and redistribution restrictions.

The dataset destination is `lingcco/EventMemAgent`; the model destination is
`lingcco/EventMemAgent-8B`. Mark them as available in the main README only after
upload completion and remote file verification.

## Checkpoint release

Convert a saved verl actor checkpoint to standard Hugging Face format:

```bash
bash scripts/train/merge_checkpoint.sh
```

Set `HF_MODEL_REPO` in `.env` and explicitly pass `UPLOAD_TO_HF=1` to upload during conversion. Merely setting the repository ID does not upload. The Hugging Face model card should identify Qwen3-VL-8B-Instruct as the base model and describe the training data, evaluation results, intended use, limitations, and license.

This release excludes optimizer states, training states, intermediate checkpoints,
predictions and scores. Do not commit tokens, private URLs, datasets, weights, or
experiment logs to GitHub. The merged model is for inference and further fine-tuning,
not exact resumption of the original optimizer state.

Model and dataset cards are in `docs/huggingface/`. Update their training step and
artifact details when publishing a different run. The release includes evaluation inputs and
scoring code, not generated predictions, scoring outputs or experiment logs.

For an existing merged checkpoint or prepared dataset folder, use the standard
Hugging Face CLI after logging in locally:

```bash
hf auth login
hf upload YOUR_ORG/EventMemAgent-8B checkpoints/released/EventMemAgent-8B .
hf upload YOUR_ORG/EventMemAgent data/train/eventmemagent_train.parquet train/eventmemagent_train.parquet --repo-type dataset
hf upload YOUR_ORG/EventMemAgent data/eval/ovobench/eventmemagent_ovobench.parquet eval/ovobench/eventmemagent_ovobench.parquet --repo-type dataset
hf upload YOUR_ORG/EventMemAgent data/eval/ovobench/ovo_bench_new.json eval/ovobench/ovo_bench_new.json --repo-type dataset
hf upload YOUR_ORG/EventMemAgent data/eval/streamingbench/eventmemagent_streamingbench.parquet eval/streamingbench/eventmemagent_streamingbench.parquet --repo-type dataset
hf upload YOUR_ORG/EventMemAgent data/eval/streamingbench/questions_real.json eval/streamingbench/questions_real.json --repo-type dataset
```

Upload only these selected dataset files and the completed dataset card as
`README.md`. Do not upload the whole working directory or `runs/`. The `val/`
placeholder is not a released validation split.
