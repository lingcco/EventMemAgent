#!/usr/bin/env python3
"""Lightweight repository and release validation without loading ML models."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LFS_HEADER = "version https://git-lfs.github.com/spec/v1"
PERSONAL_PATH_PATTERN = re.compile(
    "/mnt/" + "data/" + r"(?:i-" + "wensiwei" + r"|[^/]*" + "wensiwei" + r")"
)
SCAN_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".json", ".md", ".toml", ".txt"}
IGNORED_SCAN_DIRS = {".git", ".venv", "data", "checkpoints", "runs"}


def load_dotenv() -> dict[str, str]:
    values = dict(os.environ)
    path = ROOT / ".env"
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values.setdefault(key.strip(), value.strip().strip("'\""))
    return values


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def check_repo(errors: list[str]) -> None:
    required = [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "configs/train.yaml",
        "configs/tools.yaml",
    ]
    for relative in required:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required repository file: {relative}")

    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in IGNORED_SCAN_DIRS or part.startswith('.venv-') for part in path.relative_to(ROOT).parts):
            continue
        if path.suffix not in SCAN_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(ROOT)
        if text.startswith(LFS_HEADER):
            errors.append(f"unresolved Git LFS pointer: {relative}")
        if PERSONAL_PATH_PATTERN.search(text):
            errors.append(f"personal absolute path remains in: {relative}")

    tracked_artifact_patterns = ("__pycache__", ".pyc", ".egg-info", "tensorboard_log")
    # GitHub's source ZIP has no .git directory and must remain runnable.
    try:
        result = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, check=False, capture_output=True, text=True
        )
        tracked = result.stdout.splitlines() if result.returncode == 0 else []
    except FileNotFoundError:
        tracked = []
    for relative in tracked:
        if any(pattern in relative for pattern in tracked_artifact_patterns):
            errors.append(f"generated artifact is tracked by Git: {relative}")
            break

    try:
        import yaml
    except ImportError:
        errors.append("PyYAML is required for configuration validation")
    else:
        for relative in (
            "configs/train.yaml",
            "configs/tools.yaml",
            "CITATION.cff",
        ):
            try:
                yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError) as exc:
                errors.append(f"invalid YAML in {relative}: {exc}")


def check_train(values: dict[str, str], errors: list[str]) -> None:
    defaults = {
        "BASE_MODEL_PATH": "checkpoints/pretrained/Qwen3-VL-8B-Instruct",
        "EMBEDDING_MODEL_PATH": "checkpoints/pretrained/Qwen3-Embedding-0.6B",
        "TRAIN_DATA": "data/train/eventmemagent_train.parquet",
        "VAL_DATA": "",
        "TOOL_CONFIG_PATH": "configs/tools.yaml",
    }
    for key, default in defaults.items():
        if key == "VAL_DATA" and not values.get(key, default):
            continue
        path = resolve(values.get(key, default))
        if not path.exists():
            errors.append(f"{key} does not exist: {path}")

    train_path = resolve(values.get("TRAIN_DATA", defaults["TRAIN_DATA"]))
    val_path = resolve(values.get("VAL_DATA", defaults["VAL_DATA"]))
    for label, path in (("TRAIN_DATA", train_path), ("VAL_DATA", val_path)):
        if label == "VAL_DATA" and not values.get("VAL_DATA", ""):
            continue
        if not path.is_file():
            continue
        try:
            import pyarrow.parquet as pq
        except ImportError:
            errors.append("pyarrow is required to validate training parquet files")
            break

        try:
            parquet = pq.ParquetFile(path)
        except Exception as exc:
            errors.append(f"{label} is not a readable parquet file: {path} ({exc})")
            continue

        required_columns = {
            "prompt",
            "short_term_memory",
            "long_term_memory",
            "data_source",
            "reward_model",
            "extra_info",
            "agent_name",
        }
        missing = sorted(required_columns - set(parquet.schema_arrow.names))
        if missing:
            errors.append(f"{label} is missing columns: {', '.join(missing)}")
        if parquet.metadata.num_rows == 0:
            errors.append(f"{label} contains no samples: {path}")
        elif not missing:
            # Inspect only small routing columns, never materialize image data.
            batch = next(parquet.iter_batches(batch_size=8, columns=['agent_name', 'reward_model', 'extra_info']))
            for index, row in enumerate(batch.to_pylist()):
                if row['agent_name'] != 'tool_agent':
                    errors.append(f'{label} row {index}: agent_name must be tool_agent')
                reward = row['reward_model'] or {}
                if not reward.get('ground_truth'):
                    errors.append(f'{label} row {index}: missing reward ground truth')
                tool_kwargs = (row['extra_info'] or {}).get('tools_kwargs') or {}
                if set(tool_kwargs) != {'search_memory', 'ocr', 'detect_objects'}:
                    errors.append(f'{label} row {index}: missing EventMemAgent tool routing')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("repo", "train", "runtime"), default="repo")
    args = parser.parse_args()

    errors: list[str] = []
    check_repo(errors)
    if args.mode == "train":
        check_train(load_dotenv(), errors)
    if args.mode == "runtime":
        from importlib import import_module
        from importlib.metadata import version
        for module, package in [('torch', 'torch'), ('flash_attn', 'flash-attn'), ('transformers', 'transformers'), ('vllm', 'vllm')]:
            try:
                import_module(module)
                print(f'{package}: {version(package)}')
            except Exception as exc:
                errors.append(f'{package} cannot be imported: {exc}. Use an isolated environment matching constraints-training.txt; rebuild FlashAttention after changing PyTorch.')

    if errors:
        print(f"Validation failed ({len(errors)} issue(s)):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"Validation passed: {args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
