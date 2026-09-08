# Upstream provenance

EventMemAgent vendors a modified copy of [verl](https://github.com/volcengine/verl), reported by this checkout as version `0.7.0.dev`.

The framework is retained in-tree because the project integrates custom online-video data handling, multi-turn agent behavior, memory/OCR/detection tools, and reward calculation with verl internals. Removing it or replacing it with an unpinned external dependency would not reproduce the original training pipeline.

Upstream copyright headers and the Apache License 2.0 are preserved. EventMemAgent-specific files carry their own contributor notice. Before updating the vendored framework, record the selected upstream commit, review the local integrations, and verify training and evaluation end to end.

Unrelated upstream recipes, Docker builds, documentation-site sources, example
applications and CI workflows are not included in this release. The vendored
backend remains intact because runtime-selected modules and shared Hydra
configurations are intertwined; it is not an additional user-facing workflow.
Project CI covers portable source/configuration checks. GPU training and
benchmark runs require dedicated infrastructure.

Validation input is optional in this release: when `data.val_files` is empty,
the PPO runner skips validation dataset/loader creation and validation rewards.
When supplied, validation follows the existing path. This compatibility change
does not modify PPO/GRPO updates, the training sampler or optimizer settings.
