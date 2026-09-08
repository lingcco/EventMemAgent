#!/usr/bin/env python3
"""Calculate StreamingBench metrics from EventMemAgent evaluator output."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def calculate_metrics(eval_data, model_name: str):
    """Score the list-form JSON written by ``run_streamingbench``."""
    stats = defaultdict(lambda: {"total": 0, "correct": 0})
    missing = []

    if not isinstance(eval_data, list):
        raise ValueError(
            "Expected the list-form JSON written by eventmemagent.evaluation.streamingbench. "
            "Re-run evaluation with the standardized entry point."
        )

    for subset in eval_data:
        video_path = subset.get("video_path", "")
        for question in subset.get("questions", []):
            task_type = question.get("task_type", "unknown")
            ground_truth = str(question.get("answer", "")).strip().upper()
            prediction = str(
                question.get(f"{model_name}_extracted") or question.get(model_name, "")
            ).strip().upper()

            stats[task_type]["total"] += 1
            if prediction and ground_truth and prediction == ground_truth:
                stats[task_type]["correct"] += 1
            if not prediction:
                missing.append({"video_path": video_path, "question": question.get("question", "")})

    task_metrics = {}
    for task_type, counts in sorted(stats.items()):
        total = counts["total"]
        task_metrics[task_type] = {
            **counts,
            "accuracy": counts["correct"] / total * 100 if total else 0.0,
        }

    total_samples = sum(item["total"] for item in task_metrics.values())
    total_correct = sum(item["correct"] for item in task_metrics.values())
    arithmetic_mean = (
        sum(item["accuracy"] for item in task_metrics.values()) / len(task_metrics)
        if task_metrics
        else 0.0
    )
    weighted_accuracy = total_correct / total_samples * 100 if total_samples else 0.0

    return {
        "model_name": model_name,
        "task_metrics": task_metrics,
        "all_accuracy_arithmetic_mean": arithmetic_mean,
        "overall_accuracy_weighted": weighted_accuracy,
        "total_samples": total_samples,
        "total_correct": total_correct,
        "missing_prediction_count": len(missing),
    }


def calculate_parquet_metrics(eval_data, questions_data):
    """Score archived batch-evaluator output using benchmark question metadata.

    Preserve the historical evaluator's ``correct``/``score`` semantics. Do not
    substitute the training reward average for benchmark accuracy.
    """
    index = {}
    for video in questions_data:
        name = Path(video['video_path']).stem
        for question in video['questions']:
            timestamp = sum(int(part) * 60 ** i for i, part in enumerate(reversed(question['time_stamp'].split(':'))))
            key = (name, timestamp, question['question'].strip())
            if key in index and index[key]['task_type'] != question['task_type']:
                raise ValueError(f'Conflicting task labels for benchmark question: {key}')
            entry = index.setdefault(key, {'task_type': question['task_type'], 'count': 0})
            entry['count'] += 1
    stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    unmatched = 0
    seen = defaultdict(int)
    for result in eval_data.get('results', []):
        match = re.fullmatch(r'(.+)_(\d+)_(\d+)', Path(result.get('video_path', '')).stem)
        question = result.get('question', '').split('\n\nOptions:')[0].strip()
        candidates = []
        if match:
            name, _, end = match.groups()
            exact = (name, int(end), question)
            if exact in index:
                candidates = [exact]
            else:
                # The archived pipeline sometimes changes option formatting.
                # Match its 50-character question prefix only if unambiguous.
                prefix = question[:50]
                candidates = [key for key in index if key[:2] == (name, int(end))
                              and prefix and (prefix in key[2][:50] or key[2][:50] in prefix)]
        if len(candidates) != 1:
            unmatched += 1
            continue
        key = candidates[0]
        if seen[key] >= index[key]['count']:
            raise ValueError(f'Duplicate prediction for benchmark question: {key}')
        seen[key] += 1
        values = stats[index[key]['task_type']]
        values['total'] += 1
        values['correct'] += int(bool(result.get('correct')) or result.get('score', 0) >= 1.0)
    task_metrics = {
        task: {**values, 'accuracy': 100 * values['correct'] / values['total']}
        for task, values in sorted(stats.items())
    }
    total = sum(seen.values())
    expected = sum(entry['count'] for entry in index.values())
    correct = sum(values['correct'] for values in stats.values())
    return {
        'input_mode': 'parquet',
        'task_metrics': task_metrics,
        'all_accuracy_arithmetic_mean': sum(v['accuracy'] for v in task_metrics.values()) / len(task_metrics) if task_metrics else 0.0,
        'overall_accuracy_weighted': 100 * correct / total if total else 0.0,
        'total_samples': total,
        'total_correct': correct,
        'expected_samples': expected,
        'unmatched_count': unmatched,
        'missing_prediction_count': expected - total,
    }


def print_metrics(metrics):
    print("\n" + "=" * 78)
    print("StreamingBench Evaluation Metrics")
    print("=" * 78)
    print(f"{'Task Type':<40} {'Total':>8} {'Correct':>8} {'Accuracy':>10}")
    print("-" * 78)
    for task_type, values in metrics["task_metrics"].items():
        print(
            f"{task_type:<40} {values['total']:>8} {values['correct']:>8} "
            f"{values['accuracy']:>9.2f}%"
        )
    print("-" * 78)
    print(f"Overall (weighted): {metrics['overall_accuracy_weighted']:.2f}%")
    print(f"All tasks (arithmetic mean): {metrics['all_accuracy_arithmetic_mean']:.2f}%")
    print(f"Missing predictions: {metrics['missing_prediction_count']}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Calculate StreamingBench metrics")
    parser.add_argument("--eval-results", required=True, help="JSON written by run_streamingbench")
    parser.add_argument("--model-name", default="EventMemAgent", help="Prediction field name")
    parser.add_argument("--questions-json", help="questions_real.json, required for parquet predictions")
    parser.add_argument("--allow-partial", action="store_true", help="Allow scoring an incomplete evaluation run")
    parser.add_argument("--output", help="Optional metrics JSON output path")
    args = parser.parse_args()

    with open(args.eval_results, "r", encoding="utf-8") as handle:
        eval_data = json.load(handle)

    if isinstance(eval_data, dict):
        if not args.questions_json:
            parser.error('--questions-json is required for parquet predictions')
        with open(args.questions_json, encoding='utf-8') as handle:
            metrics = calculate_parquet_metrics(eval_data, json.load(handle))
    else:
        metrics = calculate_metrics(eval_data, args.model_name)
    if not args.allow_partial and (
        not metrics['total_samples'] or metrics.get('unmatched_count', 0)
        or metrics['missing_prediction_count']
    ):
        parser.error('Incomplete or unmatched predictions; use --allow-partial only for diagnostic subsets')
    print_metrics(metrics)

    output_path = args.output or str(Path(args.eval_results).with_suffix("")) + "_metrics.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    main()
