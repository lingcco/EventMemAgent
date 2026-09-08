#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算 OVO-Bench 评测结果的指标

根据 OVO-Bench 的评分规则计算各任务的准确率：
- Backward Tracing: EPM, ASI, HLD
- Real-time Visual Perception: STU, OJR, ATR, ACR, OCR, FPD
- Forward Active Responding: REC, SSR, CRR
"""

import json
import os
import re
import argparse
from collections import defaultdict
from pathlib import Path

# 任务分类
BACKWARD_TASKS = ["EPM", "ASI", "HLD"]
REALTIME_TASKS = ["STU", "OJR", "ATR", "ACR", "OCR", "FPD"]
FORWARD_TASKS = ["REC", "SSR", "CRR"]


def extract_sample_id_from_video_path(video_path: str):
    """
    从视频路径提取样本ID
    例如: /path/to/123.mp4 -> 123
         /path/to/456_0.mp4 -> 456_0
    """
    basename = os.path.basename(video_path)
    name_without_ext = os.path.splitext(basename)[0]
    return name_without_ext


def build_ovo_bench_index(ovo_bench_data):
    """
    构建 OVO-Bench 数据索引: sample_id -> sample_info
    """
    index = {}
    
    for sample in ovo_bench_data:
        sample_id = sample.get("id")
        task = sample.get("task", "unknown")
        
        if task in BACKWARD_TASKS or task in REALTIME_TASKS:
            # Backward/Realtime 任务
            index[str(sample_id)] = {
                "task": task,
                "gt": sample.get("gt"),
                "answer": sample.get("answer", ""),
                "options": sample.get("options", [])
            }
        elif task in FORWARD_TASKS:
            # Forward 任务：每个 test_info 对应一个样本
            test_info = sample.get("test_info", [])
            for test_idx, test_item in enumerate(test_info):
                key = f"{sample_id}_{test_idx}"
                index[key] = {
                    "task": task,
                    "test_info": test_item,
                    "activity": sample.get("activity", ""),
                    "question": sample.get("question", "")
                }
    
    return index


def extract_answer_letter(response: str):
    """
    从响应中提取答案字母 (A, B, C, D)
    """
    if not response:
        return None
    
    # 查找单个字母答案
    match = re.search(r'\b([A-D])\b', response.upper())
    if match:
        return match.group(1)
    
    return None


def extract_number(response: str):
    """
    从响应中提取数字
    """
    if not response:
        return None
    
    # 提取所有数字并连接
    numbers = re.findall(r'\d+', response)
    if numbers:
        return int("".join(numbers))
    
    return None


def extract_yes_no(response: str):
    """
    从响应中提取 Yes/No
    """
    if not response:
        return None
    
    response_upper = response.upper()
    
    # 检查简写
    if response_upper.strip() in ["Y", "N"]:
        return "Yes" if response_upper.strip() == "Y" else "No"
    
    # 检查完整单词
    if "YES" in response_upper:
        return "Yes"
    if "NO" in response_upper:
        return "No"
    
    return None


def score_backward_realtime(result, ovo_info):
    """
    评分 Backward/Realtime 任务（选择题）
    """
    # 优先使用 extracted_answer，如果没有则从 response 提取
    extracted = result.get("extracted_answer", "")
    if not extracted:
        response = result.get("response", "")
        extracted = extract_answer_letter(response) or ""
    
    gt_idx = ovo_info.get("gt")
    
    if gt_idx is None:
        return 0
    
    # 正确答案字母
    gt_letter = chr(65 + gt_idx)  # 0->A, 1->B, 2->C, 3->D
    
    # 提取答案字母
    answer_letter = extract_answer_letter(extracted)
    
    if answer_letter == gt_letter:
        return 1
    
    return 0


def score_rec(result, ovo_info):
    """
    评分 REC 任务（计数）
    """
    # 优先使用 extracted_answer，如果没有则从 response 提取
    extracted = result.get("extracted_answer", "")
    if not extracted:
        response = result.get("response", "")
        extracted = str(extract_number(response) or "")
    
    gt_count = ovo_info.get("test_info", {}).get("count")
    
    if gt_count is None:
        return 0
    
    # 从提取的答案中获取数字
    response_number = extract_number(extracted)
    
    if response_number == gt_count:
        return 1
    
    return 0


def score_ssr_crr(result, ovo_info):
    """
    评分 SSR/CRR 任务（Yes/No）
    """
    # 优先使用 extracted_answer，如果没有则从 response 提取
    extracted = result.get("extracted_answer", "")
    if not extracted:
        response = result.get("response", "")
        extracted = extract_yes_no(response) or ""
    
    test_info = ovo_info.get("test_info", {})
    gt_type = test_info.get("type")  # 0: No, 1: Yes
    
    if gt_type is None:
        return 0
    
    # 正确答案
    gt_answer = "Yes" if gt_type == 1 else "No"
    
    # 从提取的答案中获取 Yes/No
    response_answer = extract_yes_no(extracted)
    
    # 检查简写
    if extracted.strip().upper() in ["Y", "N"]:
        response_answer = "Yes" if extracted.strip().upper() == "Y" else "No"
    
    if response_answer == gt_answer:
        return 1
    
    return 0


def calculate_metrics(eval_results, ovo_bench_data):
    """
    计算指标
    """
    # 构建 OVO-Bench 索引
    ovo_index = build_ovo_bench_index(ovo_bench_data)
    
    # 按任务类型统计
    stats = defaultdict(lambda: {"total": 0, "correct": 0, "scores": []})
    unmatched = []
    
    results = eval_results.get("results", [])
    
    for result in results:
        video_path = result.get("video_path", "")
        sample_id = extract_sample_id_from_video_path(video_path)
        
        if sample_id not in ovo_index:
            unmatched.append(result)
            continue
        
        ovo_info = ovo_index[sample_id]
        task = ovo_info["task"]
        
        # 根据任务类型评分
        if task in BACKWARD_TASKS or task in REALTIME_TASKS:
            score = score_backward_realtime(result, ovo_info)
        elif task == "REC":
            score = score_rec(result, ovo_info)
        elif task in ["SSR", "CRR"]:
            score = score_ssr_crr(result, ovo_info)
        else:
            score = 0
        
        stats[task]["total"] += 1
        stats[task]["correct"] += score
        stats[task]["scores"].append(score)
    
    # 计算每个任务的准确率
    task_accuracies = {}
    for task, counts in stats.items():
        if counts["total"] > 0:
            accuracy = counts["correct"] / counts["total"] * 100
            task_accuracies[task] = {
                "total": counts["total"],
                "correct": counts["correct"],
                "accuracy": accuracy
            }
    
    # 按类别计算平均准确率（算术平均）
    backward_tasks_present = [t for t in BACKWARD_TASKS if t in task_accuracies]
    realtime_tasks_present = [t for t in REALTIME_TASKS if t in task_accuracies]
    forward_tasks_present = [t for t in FORWARD_TASKS if t in task_accuracies]
    
    backward_avg = sum(task_accuracies[t]["accuracy"] for t in backward_tasks_present) / len(backward_tasks_present) if backward_tasks_present else 0
    realtime_avg = sum(task_accuracies[t]["accuracy"] for t in realtime_tasks_present) / len(realtime_tasks_present) if realtime_tasks_present else 0
    forward_avg = sum(task_accuracies[t]["accuracy"] for t in forward_tasks_present) / len(forward_tasks_present) if forward_tasks_present else 0
    
    # 总平均（三个类别的算术平均）
    category_avgs = []
    if backward_tasks_present:
        category_avgs.append(backward_avg)
    if realtime_tasks_present:
        category_avgs.append(realtime_avg)
    if forward_tasks_present:
        category_avgs.append(forward_avg)
    total_avg = sum(category_avgs) / len(category_avgs) if category_avgs else 0
    
    return task_accuracies, backward_avg, realtime_avg, forward_avg, total_avg, unmatched


def main():
    parser = argparse.ArgumentParser(description="Calculate OVO-Bench metrics")
    parser.add_argument("--eval-results", required=True, help="Path to eval_results.json")
    parser.add_argument("--ovo-bench-json", required=True, help="Path to ovo_bench_new.json")
    parser.add_argument("--output", help="Optional metrics JSON output path")
    args = parser.parse_args()

    # 加载评测结果
    print(f"Loading eval results from: {args.eval_results}")
    with open(args.eval_results, "r", encoding="utf-8") as f:
        eval_results = json.load(f)
    
    # 加载 OVO-Bench 数据
    print(f"Loading OVO-Bench data from: {args.ovo_bench_json}")
    with open(args.ovo_bench_json, "r", encoding="utf-8") as f:
        ovo_bench_data = json.load(f)
    
    # 计算指标
    task_accuracies, backward_avg, realtime_avg, forward_avg, total_avg, unmatched = calculate_metrics(
        eval_results, ovo_bench_data
    )
    
    # 打印结果
    print("\n" + "=" * 70)
    print("OVO-Bench Evaluation Metrics")
    print("=" * 70)
    
    # Backward Tracing
    print("\n[Backward Tracing]")
    print(f"{'Task':<10} {'Total':>8} {'Correct':>8} {'Accuracy':>10}")
    print("-" * 40)
    for task in BACKWARD_TASKS:
        if task in task_accuracies:
            metrics = task_accuracies[task]
            print(f"{task:<10} {metrics['total']:>8} {metrics['correct']:>8} {metrics['accuracy']:>9.2f}%")
    print(f"{'Avg.':<10} {'-':>8} {'-':>8} {backward_avg:>9.2f}%")
    
    # Real-time Visual Perception
    print("\n[Real-time Visual Perception]")
    print(f"{'Task':<10} {'Total':>8} {'Correct':>8} {'Accuracy':>10}")
    print("-" * 40)
    for task in REALTIME_TASKS:
        if task in task_accuracies:
            metrics = task_accuracies[task]
            print(f"{task:<10} {metrics['total']:>8} {metrics['correct']:>8} {metrics['accuracy']:>9.2f}%")
    print(f"{'Avg.':<10} {'-':>8} {'-':>8} {realtime_avg:>9.2f}%")
    
    # Forward Active Responding
    print("\n[Forward Active Responding]")
    print(f"{'Task':<10} {'Total':>8} {'Correct':>8} {'Accuracy':>10}")
    print("-" * 40)
    for task in FORWARD_TASKS:
        if task in task_accuracies:
            metrics = task_accuracies[task]
            print(f"{task:<10} {metrics['total']:>8} {metrics['correct']:>8} {metrics['accuracy']:>9.2f}%")
    print(f"{'Avg.':<10} {'-':>8} {'-':>8} {forward_avg:>9.2f}%")
    
    # Total
    print("\n" + "=" * 70)
    print(f"Total Average (arithmetic mean of 3 categories): {total_avg:.2f}%")
    print("=" * 70)
    
    if unmatched:
        print(f"\n[Warning] {len(unmatched)} samples could not be matched")
        print("First 3 unmatched samples:")
        for i, sample in enumerate(unmatched[:3]):
            print(f"  {i+1}. video_path: {sample.get('video_path', 'N/A')}")
    
    # 保存结果
    output = {
        "task_metrics": task_accuracies,
        "category_averages": {
            "backward_tracing": backward_avg,
            "realtime_perception": realtime_avg,
            "forward_responding": forward_avg
        },
        "total_average": total_avg,
        "total_samples": sum(t["total"] for t in task_accuracies.values()),
        "total_correct": sum(t["correct"] for t in task_accuracies.values()),
        "unmatched_count": len(unmatched)
    }
    
    output_path = args.output or str(Path(args.eval_results).with_suffix("")) + "_ovobench_metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    main()
