"""Small, readable evaluation metadata; no credentials or environment dumps."""
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess


def save_run_config(output_dir, arguments):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'eval_results.json').exists():
        raise FileExistsError(f'{directory}/eval_results.json already exists; choose a new output directory')
    packages = {}
    for name in ('torch', 'transformers', 'vllm', 'pyarrow'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True)
    status = subprocess.run(['git', 'status', '--porcelain'], cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True)
    record = {
        'started_at': datetime.now(timezone.utc).isoformat(),
        'git_commit': result.stdout.strip() if result.returncode == 0 else None,
        'working_tree_modified': bool(status.stdout.strip()) if status.returncode == 0 else None,
        'arguments': arguments,
        'packages': packages,
    }
    (directory / 'run_config.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
