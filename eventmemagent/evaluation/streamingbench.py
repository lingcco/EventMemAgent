#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StreamingBench 评测脚本

对 VideoAgent 在 StreamingBench 上进行评测。

使用方法：
    python -m eventmemagent.evaluation.streamingbench \
        --data_file /path/to/questions_real.json \
        --output_file ./streamingbench_results.json \
        --context_time -1

    # 多进程评测
    python -m eventmemagent.evaluation.streamingbench \
        --data_file /path/to/questions_real.json \
        --num_workers 4 \
        --gpus "0,1,2,3"
"""

import argparse
import json
import os
import sys
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from multiprocessing import Process, Queue
import queue

from tqdm import tqdm

# 添加项目路径
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(ROOT, "checkpoints/released/EventMemAgent-8B"))
DEFAULT_EMBEDDING_PATH = os.environ.get(
    "EMBEDDING_MODEL_PATH", os.path.join(ROOT, "checkpoints/pretrained/Qwen3-Embedding-0.6B")
)
DEFAULT_TOOL_CONFIG = os.environ.get(
    "TOOL_CONFIG_PATH", os.path.join(ROOT, "configs/tools.yaml")
)
sys.path.insert(0, ROOT)


# ============== Video Processing ==============

def split_video(video_file: str, start_time: int, end_time: int) -> str:
    """
    使用 OpenCV 切分视频
    """
    import cv2
    
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_dir = os.path.join(os.path.dirname(video_file), "tmp_60")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{video_name}_{start_time}_{end_time}.mp4")

    if os.path.exists(output_file):
        return output_file

    try:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_file}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if fps <= 0 or width <= 0 or height <= 0:
            raise Exception(f"Invalid video properties: fps={fps}, width={width}, height={height}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise Exception(f"Cannot create output video: {output_file}")
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        current_frame = start_frame
        frames_written = 0
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            current_frame += 1
            frames_written += 1
        
        cap.release()
        out.release()
        
        if frames_written == 0:
            raise Exception("No frames written to output video")
        
        # 验证输出文件存在
        if not os.path.exists(output_file):
            raise Exception(f"Output file not created: {output_file}")
            
    except Exception as e:
        print(f"Video split error: {e}")
        raise
        
    return output_file


def parse_timestamp(timestamp: str) -> int:
    """将时间戳字符串转换为秒数"""
    # 支持 "00:03:10" 或 "0:03:10" 格式
    parts = timestamp.split(":")
    return sum(int(x) * 60 ** i for i, x in enumerate(reversed(parts)))


# ============== Answer Extraction ==============

def extract_answer_letter(response: str) -> str:
    """
    从模型响应中提取答案字母 (A/B/C/D)
    """
    # 先尝试从 \boxed{} 中提取
    boxed_match = re.search(r'\\boxed\{([A-Da-d])\}', response)
    if boxed_match:
        return boxed_match.group(1).upper()
    
    # 尝试匹配 "The answer is X" 或 "Answer: X" 格式
    answer_patterns = [
        r'(?:the\s+)?(?:best\s+)?(?:answer|option)\s*(?:is|:)\s*([A-Da-d])',
        r'^([A-Da-d])[\.\s\)]',  # 开头的 A. 或 A) 或 A
        r'\b([A-Da-d])\s*$',  # 结尾的单独字母
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # 如果响应很短，直接检查是否是单个字母
    response_clean = response.strip().upper()
    if len(response_clean) == 1 and response_clean in 'ABCD':
        return response_clean
    
    # 最后尝试找到任何 A/B/C/D
    for letter in ['A', 'B', 'C', 'D']:
        if letter in response.upper():
            return letter
    
    return ""


# ============== Evaluation Logic ==============

def evaluate_single_sample(
    agent,
    video_path: str,
    question_data: Dict,
    context_time: int,
) -> Tuple[str, str]:
    """
    评估单个问题
    
    Returns:
        (model_response, extracted_answer)
    """
    timestamp_str = question_data["time_stamp"]
    timestamp = parse_timestamp(timestamp_str)
    
    # 计算视频起始时间
    if context_time > 0:
        time_start = max(0, timestamp - context_time)
    else:
        time_start = 0
    
    # 切分视频
    video_clip = split_video(video_path, time_start, timestamp)
    
    # 构建问题（只用 question + options，不用 benchmark 的 prompt template）
    ques = question_data["question"]
    if "options" in question_data:
        options = question_data["options"]
        # 确保选项有 A./B./C./D. 前缀
        if not options[0].startswith("A."):
            options = [f"A. {options[0]}", f"B. {options[1]}", f"C. {options[2]}", f"D. {options[3]}"]
        
        # 组合成完整问题
        question_text = f"{ques}\n\nOptions:\n" + "\n".join(options)
    else:
        question_text = ques
    
    # 调用模型推理（使用你自己的 system prompt 和 memory 系统）
    result = agent.inference_online(video_clip, question_text)
    response = result.answer
    
    # 提取答案字母
    extracted = extract_answer_letter(response)
    
    return response, extracted


def evaluate_dataset(
    agent,
    data: List[Dict],
    output_file: str,
    context_time: int,
    model_name: str = "VideoAgent",
) -> Dict[str, Any]:
    """
    评估整个数据集
    """
    total_questions = sum(len(subset["questions"]) for subset in data)
    correct = 0
    total = 0
    
    print("\n" + "=" * 60)
    print("📊 StreamingBench Evaluation")
    print("=" * 60)
    print(f"Total subsets: {len(data)}")
    print(f"Total questions: {total_questions}")
    print(f"Context time: {context_time}")
    print(f"Output file: {output_file}")
    print("=" * 60 + "\n")
    
    start_time = time.time()
    
    for subset in tqdm(data, desc="Evaluating subsets"):
        video_path = subset["video_path"]
        
        for question in subset["questions"]:
            # 跳过已评估的
            if model_name in question and question[model_name]:
                # 统计已有结果
                total += 1
                if "answer" in question:
                    extracted = extract_answer_letter(question[model_name])
                    if extracted == question["answer"]:
                        correct += 1
                continue
            
            try:
                response, extracted = evaluate_single_sample(
                    agent, video_path, question, context_time
                )
                
                question[model_name] = response
                question[f"{model_name}_extracted"] = extracted
                
                total += 1
                if "answer" in question and extracted == question["answer"]:
                    correct += 1
                
                # 打印进度
                print(f"\n[Q{total}] GT: {question.get('answer', 'N/A')} | Pred: {extracted} | {'✓' if extracted == question.get('answer') else '✗'}")
                print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")
                
            except Exception as e:
                print(f"\n[Error] Failed to evaluate question: {e}")
                question[model_name] = f"ERROR: {str(e)}"
                question[f"{model_name}_extracted"] = ""
                total += 1
            
            # 保存中间结果
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
    
    elapsed_time = time.time() - start_time
    accuracy = correct / total * 100 if total > 0 else 0
    
    # 打印结果
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Questions: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Time Elapsed: {elapsed_time:.2f}s")
    print("=" * 60)
    
    # 保存最终结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    # 保存统计信息
    stats = {
        "model_name": model_name,
        "total_questions": total,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_time": elapsed_time,
        "context_time": context_time,
        "timestamp": datetime.now().isoformat(),
    }
    
    stats_file = output_file.replace(".json", "_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    
    return stats


# ============== Multi-process Evaluation ==============

def worker_evaluate(
    worker_id: int,
    gpu_id: str,
    subset_indices: List[int],
    data: List[Dict],
    result_queue: Queue,
    context_time: int,
    args,
):
    """
    Worker 进程：评估分配的 subset
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    print(f"\n[Worker {worker_id}] Starting on GPU {gpu_id}, subsets: {len(subset_indices)}")
    
    from eventmemagent.agent.video_agent import VideoAgent
    
    # 初始化 Agent
    agent = VideoAgent(
        model_path=args.model_path,
        tool_config_path=args.tool_config_path,
        embedding_model_path=args.embedding_model_path,
        max_assistant_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        tensor_parallel_size=1,  # 单卡
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    
    model_name = args.model_name
    
    for subset_idx in subset_indices:
        subset = data[subset_idx]
        video_path = subset["video_path"]
        
        for q_idx, question in enumerate(subset["questions"]):
            if model_name in question and question[model_name]:
                result_queue.put(("skip", worker_id, subset_idx, q_idx, None, None))
                continue
            
            try:
                response, extracted = evaluate_single_sample(
                    agent, video_path, question, context_time
                )
                result_queue.put(("result", worker_id, subset_idx, q_idx, response, extracted))
            except Exception as e:
                print(f"\n[Worker {worker_id}] Error: {e}")
                result_queue.put(("error", worker_id, subset_idx, q_idx, str(e), ""))
    
    result_queue.put(("done", worker_id, len(subset_indices), None, None, None))
    print(f"\n[Worker {worker_id}] Finished")


def evaluate_multiprocess(
    data: List[Dict],
    output_file: str,
    context_time: int,
    num_workers: int,
    gpus: str,
    args,
) -> Dict[str, Any]:
    """
    多进程评估
    """
    gpu_list = [g.strip() for g in gpus.split(",")]
    if len(gpu_list) < num_workers:
        gpu_list = gpu_list * ((num_workers // len(gpu_list)) + 1)
    gpu_list = gpu_list[:num_workers]
    
    model_name = args.model_name
    total_questions = sum(len(subset["questions"]) for subset in data)
    
    print("\n" + "=" * 60)
    print("📊 StreamingBench Multi-Process Evaluation")
    print("=" * 60)
    print(f"Total subsets: {len(data)}")
    print(f"Total questions: {total_questions}")
    print(f"Num workers: {num_workers}")
    print(f"GPUs: {gpus}")
    print(f"Context time: {context_time}")
    print("=" * 60 + "\n")
    
    # 分配 subset 给各个 worker
    subset_indices = list(range(len(data)))
    subsets_per_worker = len(subset_indices) // num_workers
    worker_subsets = []
    for i in range(num_workers):
        start = i * subsets_per_worker
        if i == num_workers - 1:
            end = len(subset_indices)
        else:
            end = start + subsets_per_worker
        worker_subsets.append(subset_indices[start:end])
    
    result_queue = Queue()
    processes = []
    start_time = time.time()
    
    for i in range(num_workers):
        p = Process(
            target=worker_evaluate,
            args=(i, gpu_list[i], worker_subsets[i], data, result_queue, context_time, args)
        )
        p.start()
        processes.append(p)
    
    # 收集结果
    workers_done = 0
    processed = 0
    correct = 0
    total = 0
    
    pbar = tqdm(total=total_questions, desc="Evaluating")
    
    while workers_done < num_workers:
        try:
            msg = result_queue.get(timeout=1)
            msg_type, worker_id, subset_idx, q_idx, response, extracted = msg
            
            if msg_type == "result":
                data[subset_idx]["questions"][q_idx][model_name] = response
                data[subset_idx]["questions"][q_idx][f"{model_name}_extracted"] = extracted
                
                gt = data[subset_idx]["questions"][q_idx].get("answer", "")
                if extracted == gt:
                    correct += 1
                total += 1
                pbar.update(1)
                
                # 定期保存
                if total % 10 == 0:
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)
                
            elif msg_type == "skip":
                # 统计已有结果
                gt = data[subset_idx]["questions"][q_idx].get("answer", "")
                existing = data[subset_idx]["questions"][q_idx].get(model_name, "")
                extracted = extract_answer_letter(existing)
                if extracted == gt:
                    correct += 1
                total += 1
                pbar.update(1)
                
            elif msg_type == "error":
                data[subset_idx]["questions"][q_idx][model_name] = f"ERROR: {response}"
                data[subset_idx]["questions"][q_idx][f"{model_name}_extracted"] = ""
                total += 1
                pbar.update(1)
                
            elif msg_type == "done":
                workers_done += 1
                
        except queue.Empty:
            continue
    
    pbar.close()
    
    for p in processes:
        p.join()
    
    elapsed_time = time.time() - start_time
    accuracy = correct / total * 100 if total > 0 else 0
    
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Questions: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Time Elapsed: {elapsed_time:.2f}s")
    print(f"Workers: {num_workers}")
    print("=" * 60)
    
    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    stats = {
        "model_name": model_name,
        "total_questions": total,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_time": elapsed_time,
        "context_time": context_time,
        "num_workers": num_workers,
        "timestamp": datetime.now().isoformat(),
    }
    
    stats_file = output_file.replace(".json", "_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    
    return stats


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description="StreamingBench Evaluation for VideoAgent")
    
    # 数据配置
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to questions JSON file (e.g., questions_real.json)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to output file (default: auto-generated)")
    parser.add_argument("--context_time", type=int, default=-1,
                        help="Context time before query (-1 for all context)")
    
    # 多进程配置
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of worker processes")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs")
    
    # 模型配置
    parser.add_argument("--model_name", type=str, default="VideoAgent",
                        help="Model name for output")
    parser.add_argument("--model_path", type=str, 
                        default=DEFAULT_MODEL_PATH,
                        help="Path to the model")
    parser.add_argument("--tool_config_path", type=str,
                        default=DEFAULT_TOOL_CONFIG,
                        help="Path to tool config")
    parser.add_argument("--embedding_model_path", type=str,
                        default=DEFAULT_EMBEDDING_PATH,
                        help="Path to embedding model")
    
    # vLLM 参数
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7,
                        help="GPU memory utilization")
    parser.add_argument("--max_model_len", type=int, default=32768,
                        help="Max model length")
    
    # 推理参数
    parser.add_argument("--max_turns", type=int, default=5, help="Max assistant turns")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    
    args = parser.parse_args()
    
    # 自动生成输出文件名
    if args.output_file is None:
        data_name = os.path.splitext(os.path.basename(args.data_file))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = f"./streamingbench_results/{data_name}_{args.model_name}_{timestamp}.json"
    
    # 创建输出目录
    output_parent = os.path.dirname(args.output_file)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    
    # 加载数据
    print(f"Loading data from {args.data_file}...")
    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if args.num_workers > 1:
        evaluate_multiprocess(
            data=data,
            output_file=args.output_file,
            context_time=args.context_time,
            num_workers=args.num_workers,
            gpus=args.gpus,
            args=args,
        )
    else:
        # 单进程
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus.split(",")[0]
        
        from eventmemagent.agent.video_agent import VideoAgent
        
        print("Initializing VideoAgent...")
        agent = VideoAgent(
            model_path=args.model_path,
            tool_config_path=args.tool_config_path,
            embedding_model_path=args.embedding_model_path,
            max_assistant_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
        
        evaluate_dataset(
            agent=agent,
            data=data,
            output_file=args.output_file,
            context_time=args.context_time,
            model_name=args.model_name,
        )


if __name__ == "__main__":
    main()
