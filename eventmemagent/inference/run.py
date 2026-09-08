#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理脚本（vLLM Python API 版本）

支持两种模式：
1. 单次推理：指定 --sample_idx，推理完退出
2. 交互式推理：不指定 --sample_idx，模型加载后可连续推理多个样本

使用方法：
    # 单次推理
    python -m eventmemagent.inference.run --mode offline \
        --parquet_path /path/to/data.parquet \
        --sample_idx 0
    
    # 交互式推理（模型只加载一次）
    python -m eventmemagent.inference.run --mode offline \
        --parquet_path /path/to/data.parquet
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(ROOT, "checkpoints/released/EventMemAgent-8B"))
DEFAULT_EMBEDDING_PATH = os.environ.get(
    "EMBEDDING_MODEL_PATH", os.path.join(ROOT, "checkpoints/pretrained/Qwen3-Embedding-0.6B")
)
DEFAULT_TOOL_CONFIG = os.environ.get(
    "TOOL_CONFIG_PATH", os.path.join(ROOT, "configs/tools.yaml")
)

# 添加项目路径
sys.path.insert(0, ROOT)

from eventmemagent.agent.video_agent import VideoAgent


def print_result(result):
    """打印推理结果"""
    print("\n" + "=" * 60)
    print("INFERENCE RESULT")
    print("=" * 60)
    print(f"Answer: {result.answer}")
    print("-" * 60)
    print(f"Num turns: {result.num_turns}")
    print(f"Tool calls: {len(result.tool_calls)}")
    
    if result.tool_calls:
        print("\nTool call history:")
        for i, tc in enumerate(result.tool_calls):
            print(f"  {i+1}. {tc.get('name', 'unknown')}: {tc.get('arguments', {})}")
    
    print("=" * 60)


def interactive_mode(agent, parquet_path):
    """交互式推理模式"""
    import pandas as pd
    
    df = pd.read_parquet(parquet_path)
    total_samples = len(df)
    
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print(f"Parquet: {parquet_path}")
    print(f"Total samples: {total_samples}")
    print("-" * 60)
    print("Commands:")
    print("  <number>  - Inference on sample at index")
    print("  q         - Quit")
    print("  info      - Show parquet info")
    print("  show <n>  - Show question of sample n")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n>>> Enter sample index (or command): ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'q':
                print("Bye!")
                break
            
            if user_input.lower() == 'info':
                print(f"Total samples: {total_samples}")
                print(f"Columns: {list(df.columns)}")
                continue
            
            if user_input.lower().startswith('show '):
                try:
                    idx = int(user_input.split()[1])
                    if 0 <= idx < total_samples:
                        sample = df.iloc[idx]
                        print(f"\n[Sample {idx}]")
                        print(f"Question: {sample.get('question', 'N/A')}")
                        print(f"LTM events: {len(sample.get('long_term_memory', []))}")
                        print(f"STM frames: {len(sample.get('short_term_memory', []))}")
                    else:
                        print(f"Index out of range. Valid: 0-{total_samples-1}")
                except (ValueError, IndexError):
                    print("Usage: show <index>")
                continue
            
            # 尝试解析为数字
            try:
                sample_idx = int(user_input)
            except ValueError:
                print(f"Unknown command: {user_input}")
                continue
            
            if sample_idx < 0 or sample_idx >= total_samples:
                print(f"Index out of range. Valid: 0-{total_samples-1}")
                continue
            
            print(f"\n[Inferencing sample {sample_idx}...]")
            result = agent.inference_offline(parquet_path, sample_idx)
            print_result(result)
            
        except KeyboardInterrupt:
            print("\nInterrupted. Type 'q' to quit.")
        except Exception as e:
            print(f"Error: {e}")


def interactive_online_mode(agent):
    """在线推理交互模式"""
    print("\n" + "=" * 60)
    print("ONLINE INTERACTIVE MODE")
    print("=" * 60)
    print("Commands:")
    print("  video <path> <question>  - Inference on video")
    print("  q                        - Quit")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n>>> Enter command: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'q':
                print("Bye!")
                break
            
            if user_input.lower().startswith('video '):
                parts = user_input[6:].strip().split(' ', 1)
                if len(parts) < 2:
                    print("Usage: video <path> <question>")
                    continue
                
                video_path = parts[0]
                question = parts[1]
                
                print(f"\n[Processing video: {video_path}]")
                print(f"[Question: {question}]")
                
                result = agent.inference_online(video_path, question)
                print_result(result)
            else:
                print(f"Unknown command. Use 'video <path> <question>' or 'q' to quit.")
            
        except KeyboardInterrupt:
            print("\nInterrupted. Type 'q' to quit.")
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Video Agent Inference (vLLM Python API)")
    parser.add_argument("--mode", type=str, required=True, choices=["offline", "online"],
                        help="Inference mode: offline (use parquet) or online (use video)")
    
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
    
    # 离线推理参数
    parser.add_argument("--parquet_path", type=str, help="Path to parquet file (offline mode)")
    parser.add_argument("--sample_idx", type=int, default=None, 
                        help="Sample index. If not specified, enters interactive mode")
    
    # 在线推理参数
    parser.add_argument("--video_path", type=str, help="Path to video file (online mode)")
    parser.add_argument("--question", type=str, help="Question to ask (online mode)")
    
    # 推理参数
    parser.add_argument("--max_turns", type=int, default=5, help="Max assistant turns")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max new tokens per turn")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    
    args = parser.parse_args()
    
    # 初始化 Agent（模型只加载一次）
    print("=" * 60)
    print("Initializing VideoAgent (vLLM Python API mode)...")
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
    
    # 执行推理
    if args.mode == "offline":
        if not args.parquet_path:
            parser.error("--parquet_path is required for offline mode")
        
        if args.sample_idx is not None:
            # 单次推理模式
            print(f"\n[Offline Inference - Single Sample]")
            print(f"  Parquet: {args.parquet_path}")
            print(f"  Sample Index: {args.sample_idx}")
            print("-" * 60)
            result = agent.inference_offline(args.parquet_path, args.sample_idx)
            print_result(result)
        else:
            # 交互式模式
            interactive_mode(agent, args.parquet_path)
    
    else:  # online
        if args.video_path and args.question:
            # 单次在线推理
            print(f"\n[Online Inference - Single Video]")
            print(f"  Video: {args.video_path}")
            print(f"  Question: {args.question}")
            print("-" * 60)
            result = agent.inference_online(args.video_path, args.question)
            print_result(result)
        else:
            # 在线交互模式
            interactive_online_mode(agent)


if __name__ == "__main__":
    main()
