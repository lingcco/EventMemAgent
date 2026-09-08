# Contributing to EventMemAgent

Contributions that improve reproducibility, training stability, benchmark coverage, documentation, and tool integrations are welcome.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-build-isolation flash-attn==2.8.1
```

## Before opening a pull request

Run the portable checks:

```bash
python scripts/utils/validate_setup.py --mode repo
find scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
python -m compileall -q eventmemagent verl/tools verl/utils/reward_score/video_agent.py scripts
git diff --check
```

For training or evaluation changes, report the model revision, dataset manifest, hardware, configuration overrides, and a small before/after result. Never include private datasets, model weights, API keys, personal absolute paths, or generated experiment logs in a pull request.

## Scope and upstream changes

The `verl/` directory is a modified vendored dependency. Keep EventMemAgent-specific changes focused and preserve upstream copyright notices. If a change belongs generally in verl, consider proposing it upstream as well and record the relevant commit or pull request in the EventMemAgent change description.

## Style

- Keep stable user-facing commands under `scripts/`.
- Make local paths configurable through `.env`, environment variables, or CLI arguments.
- Add or update documentation for user-visible behavior.
- Prefer deterministic tests that do not require downloading models in CI.

All contributions are provided under the repository's Apache License 2.0.
