#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估脚本 - 对整个 parquet 数据集进行推理并计算得分

与训练时的 evaluation 保持一致：
1. 遍历数据集中的所有样本
2. 对每个样本进行推理
3. 使用 compute_score 计算得分
4. 输出统计结果并保存详细结果

使用方法：
    python -m eventmemagent.evaluation.run \
        --parquet_path /path/to/data.parquet \
        --output_dir ./eval_results
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any

import pandas as pd
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

from eventmemagent.agent.video_agent import VideoAgent

# 导入评分函数（与训练时一致）
from verl.utils.reward_score.video_agent import compute_score, extract_boxed_answer


def _load_parquet_with_nested_data(parquet_path: str) -> pd.DataFrame:
    """
    加载包含嵌套数据的 parquet 文件
    
    使用 ParquetFile 逐批读取，避免 chunked array 转换问题
    """
    import pyarrow.parquet as pq
    
    try:
        parquet_file = pq.ParquetFile(parquet_path)
        print(f"[Loading] ParquetFile has {parquet_file.metadata.num_rows} rows")
        
        # 逐批读取并转换为 Python 对象
        all_rows = []
        for batch in parquet_file.iter_batches(batch_size=100):
            batch_dict = batch.to_pydict()
            num_rows = len(batch_dict[list(batch_dict.keys())[0]])
            for i in range(num_rows):
                row = {col: batch_dict[col][i] for col in batch_dict}
                all_rows.append(row)
        
        df = pd.DataFrame(all_rows)
        print(f"[Loading] Successfully loaded {len(df)} samples with ParquetFile iter_batches")
        return df
    except Exception as e:
        print(f"[Warning] Failed to read with ParquetFile iter_batches: {e}")
    
    # 备选方案：使用 pyarrow 逐元素转换
    try:
        table = pq.read_table(parquet_path)
        print(f"[Loading] Table has {table.num_rows} rows, {len(table.column_names)} columns")
        
        data_dict = {}
        for col_name in table.column_names:
            chunked_col = table.column(col_name)
            col_values = []
            for chunk in chunked_col.chunks:
                for i in range(len(chunk)):
                    col_values.append(chunk[i].as_py())
            data_dict[col_name] = col_values
        
        df = pd.DataFrame(data_dict)
        print(f"[Loading] Successfully loaded with pyarrow element-by-element conversion")
        return df
    except Exception as e:
        print(f"[Warning] Failed to read with pyarrow element-by-element: {e}")
    
    # 最后尝试 pandas 默认方式
    df = pd.read_parquet(parquet_path)
    print(f"[Loading] Successfully loaded with pandas default")
    return df


def _serialize_message_content(content):
    """序列化消息内容，只保留文本，过滤图片"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # 只保留文本内容，跳过图片
        texts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
                # 跳过 type=image
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(texts) if texts else ""
    else:
        return str(content)


def _serialize_messages(messages: List[Dict]) -> List[Dict]:
    """序列化消息列表，只保留文本交互"""
    serialized = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        text_content = _serialize_message_content(content)
        
        # 跳过空内容（比如只有图片的消息）
        if not text_content and role == "user":
            continue
            
        serialized.append({
            "role": role,
            "content": text_content
        })
    return serialized


def evaluate_dataset(
    agent: VideoAgent,
    parquet_path: str,
    output_dir: str = None,
    start_idx: int = 0,
    end_idx: int = None,
    save_interval: int = 10,
) -> Dict[str, Any]:
    """
    对数据集进行评估
    
    Args:
        agent: VideoAgent 实例
        parquet_path: parquet 文件路径
        output_dir: 结果保存目录
        start_idx: 起始样本索引
        end_idx: 结束样本索引（不包含），None 表示到末尾
        save_interval: 每隔多少样本保存一次结果
    
    Returns:
        评估结果统计
    """
    # 加载数据（使用支持嵌套数据的方式）
    df = _load_parquet_with_nested_data(parquet_path)
    total_samples = len(df)
    
    if end_idx is None:
        end_idx = total_samples
    end_idx = min(end_idx, total_samples)
    
    print("\n" + "=" * 60)
    print("📊 EVALUATION START")
    print("=" * 60)
    print(f"Parquet: {parquet_path}")
    print(f"Total samples in file: {total_samples}")
    print(f"Evaluating: [{start_idx}, {end_idx})")
    print(f"Samples to evaluate: {end_idx - start_idx}")
    if output_dir:
        print(f"Output dir: {output_dir}")
    print("=" * 60 + "\n")
    
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 结果文件路径（固定名称，不断更新）
    results_path = os.path.join(output_dir, "eval_results.json") if output_dir else None
    
    # 评估结果
    results = []
    scores = []
    correct_count = 0
    all_input_tokens = []
    all_output_tokens = []
    
    # 开始时间
    start_time = time.time()
    
    # 遍历样本
    for idx in tqdm(range(start_idx, end_idx), desc="Evaluating"):
        sample = df.iloc[idx]
        
        # 获取 ground truth
        ground_truth = sample.get("ground_truth", None)
        if ground_truth is None:
            ground_truth = sample.get("answer", "")
        # 处理可能是 numpy 数组的情况
        if hasattr(ground_truth, 'item'):
            ground_truth = ground_truth.item()
        elif isinstance(ground_truth, (list, tuple)) and len(ground_truth) > 0:
            ground_truth = ground_truth[0]
        ground_truth = str(ground_truth) if ground_truth else ""
        
        question = sample.get("question", "")
        if hasattr(question, 'item'):
            question = question.item()
        question = str(question) if question else ""
        
        video_id = sample.get("video_id", None)
        if video_id is None:
            video_id = sample.get("id", f"sample_{idx}")
        if hasattr(video_id, 'item'):
            video_id = video_id.item()
        video_id = str(video_id)
        
        # 获取 video_path
        video_path = sample.get("video_path", None)
        if video_path is None:
            video_path = sample.get("video", sample.get("path", ""))
        if hasattr(video_path, 'item'):
            video_path = video_path.item()
        video_path = str(video_path) if video_path else ""
        
        try:
            # 推理
            inference_result = agent.inference_offline(parquet_path, idx)
            
            # 获取最后一条 assistant 消息的完整内容（用于 compute_score）
            full_response = ""
            for msg in reversed(inference_result.messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        full_response = content
                    break
            
            # 提取答案（用于展示）
            extracted_answer = extract_boxed_answer(full_response)
            if isinstance(extracted_answer, tuple):
                extracted_answer = extracted_answer[0]
            # 计算得分（与训练时一致，传入完整的模型输出）
            score = compute_score(
                data_source="evaluation",
                solution_str=full_response,
                ground_truth=ground_truth,
                extra_info={}
            )
            
            # 记录结果（包含完整的多轮对话）
            result = {
                "idx": idx,
                "video_id": video_id,
                "video_path": video_path,
                "question": question,
                "ground_truth": ground_truth,
                "extracted_answer": extracted_answer,
                "score": score,
                "num_turns": inference_result.num_turns,
                "num_tool_calls": len(inference_result.tool_calls),
                "tool_calls": inference_result.tool_calls,
                "correct": score >= 1.0,
                "messages": _serialize_messages(inference_result.messages),  # 完整的多轮对话
                "input_tokens": inference_result.input_tokens,
                "output_tokens": inference_result.output_tokens,
            }
            
            results.append(result)
            scores.append(score)
            all_input_tokens.append(inference_result.input_tokens)
            all_output_tokens.append(inference_result.output_tokens)
            
            if score >= 1.0:
                correct_count += 1
            
            # 打印进度
            if (idx - start_idx + 1) % 10 == 0:
                current_acc = correct_count / (idx - start_idx + 1) * 100
                # 计算当前 batch 的 token 统计
                batch_input = all_input_tokens[-10:] if len(all_input_tokens) >= 10 else all_input_tokens
                batch_output = all_output_tokens[-10:] if len(all_output_tokens) >= 10 else all_output_tokens
                print(f"\n[Progress] {idx - start_idx + 1}/{end_idx - start_idx} | "
                      f"Acc: {current_acc:.2f}% ({correct_count}/{idx - start_idx + 1})")
                print(f"[Tokens] Input: min={min(batch_input)}, max={max(batch_input)}, avg={sum(batch_input)/len(batch_input):.0f} | "
                      f"Output: min={min(batch_output)}, max={max(batch_output)}, avg={sum(batch_output)/len(batch_output):.0f}")
            
        except Exception as e:
            print(f"\n[Error] Sample {idx} failed: {e}")
            result = {
                "idx": idx,
                "video_id": video_id,
                "video_path": video_path,
                "question": question,
                "ground_truth": ground_truth,
                "extracted_answer": "",
                "score": 0.0,
                "num_turns": 0,
                "num_tool_calls": 0,
                "tool_calls": [],
                "correct": False,
                "messages": [],
                "error": str(e),
                "input_tokens": 0,
                "output_tokens": 0,
            }
            results.append(result)
            scores.append(0.0)
            all_input_tokens.append(0)
            all_output_tokens.append(0)
        
        # 定期保存结果（更新同一个文件）
        if results_path and (idx - start_idx + 1) % save_interval == 0:
            _save_results(results, scores, correct_count, start_time, results_path)
    
    # 计算统计
    elapsed_time = time.time() - start_time
    num_evaluated = len(results)
    accuracy = correct_count / num_evaluated * 100 if num_evaluated > 0 else 0
    avg_score = sum(scores) / num_evaluated if num_evaluated > 0 else 0
    avg_turns = sum(r["num_turns"] for r in results) / num_evaluated if num_evaluated > 0 else 0
    avg_tool_calls = sum(r["num_tool_calls"] for r in results) / num_evaluated if num_evaluated > 0 else 0
    
    # Token 统计
    valid_input_tokens = [t for t in all_input_tokens if t > 0]
    valid_output_tokens = [t for t in all_output_tokens if t > 0]
    avg_input_tokens = sum(valid_input_tokens) / len(valid_input_tokens) if valid_input_tokens else 0
    avg_output_tokens = sum(valid_output_tokens) / len(valid_output_tokens) if valid_output_tokens else 0
    max_input_tokens = max(valid_input_tokens) if valid_input_tokens else 0
    max_output_tokens = max(valid_output_tokens) if valid_output_tokens else 0
    
    # 统计结果
    stats = {
        "parquet_path": parquet_path,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "num_evaluated": num_evaluated,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "avg_score": avg_score,
        "avg_turns": avg_turns,
        "avg_tool_calls": avg_tool_calls,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "max_input_tokens": max_input_tokens,
        "max_output_tokens": max_output_tokens,
        "elapsed_time_seconds": elapsed_time,
        "samples_per_second": num_evaluated / elapsed_time if elapsed_time > 0 else 0,
        "timestamp": datetime.now().isoformat(),
    }
    
    # 打印结果
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Samples Evaluated: {num_evaluated}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Score: {avg_score:.4f}")
    print(f"Average Turns: {avg_turns:.2f}")
    print(f"Average Tool Calls: {avg_tool_calls:.2f}")
    print(f"Avg Input Tokens: {avg_input_tokens:.0f} (max: {max_input_tokens})")
    print(f"Avg Output Tokens: {avg_output_tokens:.0f} (max: {max_output_tokens})")
    print(f"Time Elapsed: {elapsed_time:.2f}s ({num_evaluated/elapsed_time:.2f} samples/s)")
    print("=" * 60)
    
    # 保存最终结果
    if output_dir:
        _save_final_results(results, stats, output_dir)
    
    return {
        "stats": stats,
        "results": results,
    }


def _save_results(results: List[Dict], scores: List[float], correct_count: int, start_time: float, results_path: str):
    """保存结果到单个文件（不断更新）"""
    elapsed_time = time.time() - start_time
    num_evaluated = len(results)
    
    data = {
        "num_evaluated": num_evaluated,
        "correct_count": correct_count,
        "current_accuracy": correct_count / num_evaluated * 100 if num_evaluated > 0 else 0,
        "elapsed_time_seconds": elapsed_time,
        "results": results,
    }
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _save_final_results(results: List[Dict], stats: Dict, output_dir: str):
    """保存最终结果"""
    # 保存完整结果（覆盖之前的中间结果）
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "stats": stats,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] Results to {results_path}")
    
    # 保存为 CSV（方便查看摘要）
    csv_path = os.path.join(output_dir, "eval_results.csv")
    df_results = pd.DataFrame([{
        "idx": r["idx"],
        "video_id": r["video_id"],
        "video_path": r["video_path"],
        "question": r["question"][:100] + "..." if len(r["question"]) > 100 else r["question"],
        "ground_truth": r["ground_truth"],
        "extracted_answer": r["extracted_answer"],
        "score": r["score"],
        "correct": r["correct"],
        "num_turns": r["num_turns"],
        "num_tool_calls": r["num_tool_calls"],
    } for r in results])
    df_results.to_csv(csv_path, index=False)
    print(f"[Saved] CSV to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Video Agent Evaluation")
    
    # 数据配置
    parser.add_argument("--parquet_path", type=str, required=True,
                        help="Path to parquet file")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Directory to save results")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start sample index")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End sample index (exclusive), None for all")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save intermediate results every N samples")
    
    # 模型配置
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
                        help="GPU memory utilization for vLLM")
    parser.add_argument("--max_model_len", type=int, default=32768,
                        help="Max model length for vLLM")
    
    # 推理参数
    parser.add_argument("--max_turns", type=int, default=5, help="Max assistant turns")
    parser.add_argument("--max_new_tokens", type=int, default=16384, help="Max new tokens per turn")
    parser.add_argument("--temperature", type=float, default=0.0, 
                        help="Sampling temperature (0 for greedy, like validation)")
    
    args = parser.parse_args()
    from eventmemagent.utils.run_metadata import save_run_config
    save_run_config(args.output_dir, vars(args))
    
    # 初始化 Agent
    print("=" * 60)
    print("Initializing VideoAgent...")
    print("=" * 60)
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
    
    # 执行评估
    evaluate_dataset(
        agent=agent,
        parquet_path=args.parquet_path,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main()
