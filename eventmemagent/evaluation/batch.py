#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量评估脚本 - 使用 vLLM 批量推理提升速度

核心思路：将多个样本的同一轮推理批量化处理
- Round 1: [Sample1, Sample2, ...] → vLLM batch generate
- 分别处理工具调用
- Round 2: [还需要继续的样本] → vLLM batch generate
- ...

使用方法：
    python -m eventmemagent.evaluation.batch \
        --parquet_path /path/to/data.parquet \
        --output_dir ./eval_results \
        --batch_size 8
"""

import argparse
import json
import os
import sys
import time
import asyncio
import copy
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(ROOT, "checkpoints/released/EventMemAgent-8B"))
DEFAULT_EMBEDDING_PATH = os.environ.get(
    "EMBEDDING_MODEL_PATH", os.path.join(ROOT, "checkpoints/pretrained/Qwen3-Embedding-0.6B")
)
DEFAULT_TOOL_CONFIG = os.environ.get(
    "TOOL_CONFIG_PATH", os.path.join(ROOT, "configs/tools.yaml")
)
sys.path.insert(0, ROOT)

from verl.utils.reward_score.video_agent import compute_score, extract_boxed_answer
from eventmemagent.agent.video_agent import (
    VideoAgent, InferenceResult, _load_parquet_with_nested_data,
    _qwen2_5_vl_dedup_image_tokens
)
from eventmemagent.agent.tools import ToolManager


@dataclass
class BatchSampleState:
    """批量推理中每个样本的状态"""
    idx: int
    messages: List[Dict]
    prompt_ids: List[int]
    image_data: List[Any]  # PIL Images
    long_term_memory: List[Dict]
    short_term_memory_raw: List[Dict]
    tool_calls_history: List[Dict] = field(default_factory=list)
    assistant_turns: int = 0
    user_turns: int = 0
    finished: bool = False
    result: Optional[InferenceResult] = None
    error: Optional[str] = None
    # 元数据
    ground_truth: str = ""
    question: str = ""
    video_id: str = ""
    video_path: str = ""
    # Token 统计
    initial_input_tokens: int = 0
    total_output_tokens: int = 0


class BatchVideoAgent:
    """支持批量推理的 VideoAgent"""
    
    def __init__(self, base_agent: VideoAgent):
        """复用已有的 VideoAgent"""
        self.agent = base_agent
        self.processor = base_agent.processor
        self.llm = base_agent.llm
        self.SamplingParams = base_agent.SamplingParams
        self.tool_schemas = base_agent.tool_schemas
        self.max_assistant_turns = base_agent.max_assistant_turns
        self.max_new_tokens = base_agent.max_new_tokens
        self.temperature = base_agent.temperature
    
    def _batch_generate(
        self,
        batch_prompt_ids: List[List[int]],
        batch_image_data: List[List[Any]]
    ) -> List[Tuple[str, List[int]]]:
        """批量生成"""
        from vllm import TokensPrompt
        
        sampling_params = self.SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        
        # 构建批量 prompts
        prompts = []
        for prompt_ids, image_data in zip(batch_prompt_ids, batch_image_data):
            prompt_ids_deduped = _qwen2_5_vl_dedup_image_tokens(prompt_ids, self.processor)
            prompt = TokensPrompt(
                prompt_token_ids=prompt_ids_deduped,
                multi_modal_data={"image": image_data} if image_data else None
            )
            prompts.append(prompt)
        
        # 批量推理
        outputs = self.llm.generate(prompts=prompts, sampling_params=sampling_params)
        
        # 提取结果
        results = []
        for output in outputs:
            response_text = output.outputs[0].text.strip()
            response_ids = list(output.outputs[0].token_ids)
            results.append((response_text, response_ids))
        
        return results
    
    def _prepare_sample(
        self,
        df: pd.DataFrame,
        idx: int,
        parquet_path: str
    ) -> BatchSampleState:
        """准备单个样本的初始状态"""
        sample = df.iloc[idx]
        
        # 提取字段
        long_term_memory = sample.get("long_term_memory", [])
        if hasattr(long_term_memory, 'tolist'):
            long_term_memory = long_term_memory.tolist()
        if not isinstance(long_term_memory, list):
            long_term_memory = list(long_term_memory) if long_term_memory is not None else []

        short_term_memory_raw = sample.get("short_term_memory", [])
        if hasattr(short_term_memory_raw, 'tolist'):
            short_term_memory_raw = short_term_memory_raw.tolist()
        if not isinstance(short_term_memory_raw, list):
            short_term_memory_raw = list(short_term_memory_raw) if short_term_memory_raw is not None else []
        
        question = sample.get("question", "")
        if hasattr(question, 'item'):
            question = question.item()
        question = str(question) if question else ""
        
        ground_truth = sample.get("ground_truth", None)
        if ground_truth is None:
            ground_truth = sample.get("answer", "")
        if hasattr(ground_truth, 'item'):
            ground_truth = ground_truth.item()
        elif isinstance(ground_truth, (list, tuple)) and len(ground_truth) > 0:
            ground_truth = ground_truth[0]
        ground_truth = str(ground_truth) if ground_truth else ""
        
        video_id = sample.get("video_id", None)
        if video_id is None:
            video_id = sample.get("id", f"sample_{idx}")
        if hasattr(video_id, 'item'):
            video_id = video_id.item()
        video_id = str(video_id)
        
        video_path = sample.get("video_path", None)
        if video_path is None:
            video_path = sample.get("video", sample.get("path", ""))
        if hasattr(video_path, 'item'):
            video_path = video_path.item()
        video_path = str(video_path) if video_path else ""
        
        # 加载图片
        image_data = self.agent._load_images_from_short_term_memory(short_term_memory_raw)
        
        # 构建消息
        system_prompt = self.agent._create_system_prompt()
        user_content = self.agent._build_user_message(question, long_term_memory, short_term_memory_raw)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        
        # 应用 chat template
        _, prompt_ids = self.agent._apply_chat_template(messages, image_data)
        
        return BatchSampleState(
            idx=idx,
            messages=messages,
            prompt_ids=prompt_ids,
            image_data=image_data,
            long_term_memory=long_term_memory,
            short_term_memory_raw=short_term_memory_raw,
            ground_truth=ground_truth,
            question=question,
            video_id=video_id,
            video_path=video_path,
            initial_input_tokens=len(prompt_ids),
        )

    def _process_tool_calls_for_sample(
        self,
        state: BatchSampleState,
        response_text: str,
        tool_manager: ToolManager
    ) -> Tuple[List[Dict], List[Any]]:
        """处理单个样本的工具调用"""
        tool_calls = self.agent._extract_tool_calls(response_text)
        
        add_messages = []
        new_images = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("arguments", {})
            state.tool_calls_history.append(tool_call)
            
            # 初始化工具管理器（每个样本独立）
            tool_manager.initialize(
                state.long_term_memory,
                state.short_term_memory_raw,
                state.image_data
            )
            
            tool_response, tool_images = asyncio.run(
                tool_manager.execute_tool(tool_name, tool_args)
            )
            
            if tool_images:
                content = []
                for img in tool_images:
                    content.append({"type": "image"})
                    new_images.append(img)
                content.append({"type": "text", "text": tool_response})
                add_messages.append({"role": "tool", "content": content})
            else:
                add_messages.append({"role": "tool", "content": tool_response})
        
        return add_messages, new_images
    
    def inference_batch(
        self,
        df: pd.DataFrame,
        indices: List[int],
        tool_config_path: str,
        embedding_model_path: str = None,
        verbose: bool = False
    ) -> List[BatchSampleState]:
        """批量推理多个样本"""
        
        # 准备所有样本
        states = []
        for idx in indices:
            try:
                state = self._prepare_sample(df, idx, "")
                states.append(state)
            except Exception as e:
                # 创建错误状态
                error_state = BatchSampleState(
                    idx=idx, messages=[], prompt_ids=[], image_data=[],
                    long_term_memory=[], short_term_memory_raw=[],
                    finished=True, error=str(e)
                )
                states.append(error_state)
        
        if verbose:
            print(f"\n[Batch] Processing {len(states)} samples")

        # 创建工具管理器（每个样本需要独立的实例）
        tool_managers = [
            ToolManager(tool_config_path, embedding_model_path)
            for _ in range(len(states))
        ]
        
        # 多轮推理循环
        round_num = 0
        while True:
            round_num += 1
            
            # 找出未完成的样本
            active_indices = [i for i, s in enumerate(states) if not s.finished]
            if not active_indices:
                break
            
            if verbose:
                print(f"[Round {round_num}] Active samples: {len(active_indices)}")
            
            # 批量生成
            batch_prompt_ids = [states[i].prompt_ids for i in active_indices]
            batch_image_data = [states[i].image_data for i in active_indices]
            
            try:
                batch_results = self._batch_generate(batch_prompt_ids, batch_image_data)
            except Exception as e:
                # 批量生成失败，标记所有活跃样本为错误
                for i in active_indices:
                    states[i].finished = True
                    states[i].error = f"Batch generate error: {e}"
                break
            
            # 处理每个样本的结果
            for local_idx, global_idx in enumerate(active_indices):
                state = states[global_idx]
                response_text, response_ids = batch_results[local_idx]
                
                state.assistant_turns += 1
                state.total_output_tokens += len(response_ids)
                state.prompt_ids = state.prompt_ids + response_ids
                
                # 检查最终答案
                final_answer = self.agent._extract_final_answer(response_text)
                if final_answer:
                    state.messages.append({"role": "assistant", "content": response_text})
                    state.finished = True
                    state.result = InferenceResult(
                        answer=final_answer,
                        messages=state.messages,
                        tool_calls=state.tool_calls_history,
                        num_turns=state.assistant_turns + state.user_turns,
                        input_tokens=state.initial_input_tokens,
                        output_tokens=state.total_output_tokens,
                    )
                    continue
                
                # 检查工具调用
                tool_calls = self.agent._extract_tool_calls(response_text)
                
                if not tool_calls:
                    # 没有工具调用也没有最终答案
                    state.messages.append({"role": "assistant", "content": response_text})
                    state.finished = True
                    state.result = InferenceResult(
                        answer=response_text,
                        messages=state.messages,
                        tool_calls=state.tool_calls_history,
                        num_turns=state.assistant_turns + state.user_turns,
                        input_tokens=state.initial_input_tokens,
                        output_tokens=state.total_output_tokens,
                    )
                    continue

                # 处理工具调用
                state.messages.append({"role": "assistant", "content": response_text})
                
                try:
                    add_messages, new_images = self._process_tool_calls_for_sample(
                        state, response_text, tool_managers[global_idx]
                    )
                    
                    state.messages.extend(add_messages)
                    
                    # 更新 prompt_ids
                    tool_response_ids = self.agent._apply_chat_template_for_tool_response(
                        add_messages, new_images
                    )
                    state.prompt_ids = state.prompt_ids + tool_response_ids
                    
                    # 更新图片
                    if new_images:
                        state.image_data.extend(new_images)
                    
                    state.user_turns += 1
                    
                except Exception as e:
                    state.finished = True
                    state.error = f"Tool execution error: {e}"
                    state.result = InferenceResult(
                        answer="",
                        messages=state.messages,
                        tool_calls=state.tool_calls_history,
                        num_turns=state.assistant_turns + state.user_turns,
                        input_tokens=state.initial_input_tokens,
                        output_tokens=state.total_output_tokens,
                    )
                
                # 检查是否达到最大轮次
                if state.assistant_turns >= self.max_assistant_turns:
                    state.finished = True
                    if state.result is None:
                        state.result = InferenceResult(
                            answer="Max turns reached",
                            messages=state.messages,
                            tool_calls=state.tool_calls_history,
                            num_turns=state.assistant_turns + state.user_turns,
                            input_tokens=state.initial_input_tokens,
                            output_tokens=state.total_output_tokens,
                        )
        
        return states


def _serialize_message_content(content):
    """序列化消息内容，只保留文本"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(texts) if texts else ""
    else:
        return str(content)


def _serialize_messages(messages: List[Dict]) -> List[Dict]:
    """序列化消息列表"""
    serialized = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        text_content = _serialize_message_content(content)
        if not text_content and role == "user":
            continue
        serialized.append({"role": role, "content": text_content})
    return serialized


def evaluate_dataset_batch(
    agent: VideoAgent,
    parquet_path: str,
    output_dir: str = None,
    start_idx: int = 0,
    end_idx: int = None,
    batch_size: int = 8,
    save_interval: int = 10,
    verbose: bool = False,
) -> Dict[str, Any]:
    """批量评估数据集"""
    
    # 加载数据
    df = _load_parquet_with_nested_data(parquet_path)
    total_samples = len(df)
    
    if end_idx is None:
        end_idx = total_samples
    end_idx = min(end_idx, total_samples)
    
    print("\n" + "=" * 60)
    print("📊 BATCH EVALUATION START")
    print("=" * 60)
    print(f"Parquet: {parquet_path}")
    print(f"Total samples: {total_samples}")
    print(f"Evaluating: [{start_idx}, {end_idx})")
    print(f"Batch size: {batch_size}")
    if output_dir:
        print(f"Output dir: {output_dir}")
    print("=" * 60 + "\n")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 创建批量 Agent
    batch_agent = BatchVideoAgent(agent)
    
    results = []
    scores = []
    correct_count = 0
    all_input_tokens = []
    all_output_tokens = []
    start_time = time.time()
    
    # 按批次处理
    all_indices = list(range(start_idx, end_idx))
    num_batches = (len(all_indices) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Batch Evaluating"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(all_indices))
        batch_indices = all_indices[batch_start:batch_end]
        
        if verbose:
            print(f"\n[Batch {batch_idx + 1}/{num_batches}] Processing indices {batch_indices}")
        
        # 批量推理
        states = batch_agent.inference_batch(
            df=df,
            indices=batch_indices,
            tool_config_path=agent.tool_manager.tool_config_path,
            embedding_model_path=agent.tool_manager.embedding_model_path,
            verbose=verbose
        )

        # 处理结果
        batch_input_tokens = []
        batch_output_tokens = []
        for state in states:
            if state.error:
                result = {
                    "idx": state.idx,
                    "video_id": state.video_id,
                    "video_path": state.video_path,
                    "question": state.question,
                    "ground_truth": state.ground_truth,
                    "extracted_answer": "",
                    "score": 0.0,
                    "num_turns": 0,
                    "num_tool_calls": 0,
                    "tool_calls": [],
                    "correct": False,
                    "messages": [],
                    "error": state.error,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
                results.append(result)
                scores.append(0.0)
                all_input_tokens.append(0)
                all_output_tokens.append(0)
                batch_input_tokens.append(0)
                batch_output_tokens.append(0)
                continue
            
            # 获取完整响应
            full_response = ""
            if state.result:
                for msg in reversed(state.result.messages):
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            full_response = content
                        break
            
            extracted_answer = extract_boxed_answer(full_response)
            if isinstance(extracted_answer, tuple):
                extracted_answer = extracted_answer[0]
              
            score = compute_score(
                data_source="evaluation",
                solution_str=full_response,
                ground_truth=state.ground_truth,
                extra_info={}
            )
            
            input_tokens = state.result.input_tokens if state.result else 0
            output_tokens = state.result.output_tokens if state.result else 0
            
            result = {
                "idx": state.idx,
                "video_id": state.video_id,
                "video_path": state.video_path,
                "question": state.question,
                "ground_truth": state.ground_truth,
                "extracted_answer": extracted_answer,
                "score": score,
                "num_turns": state.result.num_turns if state.result else 0,
                "num_tool_calls": len(state.tool_calls_history),
                "tool_calls": state.tool_calls_history,
                "correct": score >= 1.0,
                "messages": _serialize_messages(state.result.messages) if state.result else [],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            
            results.append(result)
            scores.append(score)
            all_input_tokens.append(input_tokens)
            all_output_tokens.append(output_tokens)
            batch_input_tokens.append(input_tokens)
            batch_output_tokens.append(output_tokens)
            
            if score >= 1.0:
                correct_count += 1
        
        # 打印进度
        processed = len(results)
        if processed > 0:
            current_acc = correct_count / processed * 100
            elapsed = time.time() - start_time
            speed = processed / elapsed if elapsed > 0 else 0
            # 当前 batch 的 token 统计
            valid_batch_input = [t for t in batch_input_tokens if t > 0]
            valid_batch_output = [t for t in batch_output_tokens if t > 0]
            print(f"[Progress] {processed}/{end_idx - start_idx} | "
                  f"Acc: {current_acc:.2f}% | Speed: {speed:.2f} samples/s")
            if valid_batch_input and valid_batch_output:
                print(f"[Tokens] Input: min={min(valid_batch_input)}, max={max(valid_batch_input)}, avg={sum(valid_batch_input)/len(valid_batch_input):.0f} | "
                      f"Output: min={min(valid_batch_output)}, max={max(valid_batch_output)}, avg={sum(valid_batch_output)/len(valid_batch_output):.0f}")
        
        # 定期保存
        if output_dir and (batch_idx + 1) % max(1, save_interval // batch_size) == 0:
            _save_intermediate_results(results, scores, correct_count, start_time, output_dir)

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
    
    stats = {
        "parquet_path": parquet_path,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "batch_size": batch_size,
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
    
    print("\n" + "=" * 60)
    print("📊 BATCH EVALUATION RESULTS")
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
    
    if output_dir:
        _save_final_results(results, stats, output_dir)
    
    return {"stats": stats, "results": results}


def _save_intermediate_results(results, scores, correct_count, start_time, output_dir):
    """保存中间结果"""
    elapsed_time = time.time() - start_time
    num_evaluated = len(results)
    
    data = {
        "num_evaluated": num_evaluated,
        "correct_count": correct_count,
        "current_accuracy": correct_count / num_evaluated * 100 if num_evaluated > 0 else 0,
        "elapsed_time_seconds": elapsed_time,
        "results": results,
    }
    
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _save_final_results(results, stats, output_dir):
    """保存最终结果"""
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] Results to {results_path}")
    
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
    parser = argparse.ArgumentParser(description="Video Agent Batch Evaluation")
    
    # 数据配置
    parser.add_argument("--parquet_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./eval_results_batch")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of samples to process in parallel")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    
    # 模型配置
    parser.add_argument("--model_path", type=str,
                        default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tool_config_path", type=str,
                        default=DEFAULT_TOOL_CONFIG)
    parser.add_argument("--embedding_model_path", type=str,
                        default=DEFAULT_EMBEDDING_PATH)
    
    # vLLM 参数
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--max_model_len", type=int, default=32768)
    
    # 推理参数
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    
    args = parser.parse_args()
    from eventmemagent.utils.run_metadata import save_run_config
    save_run_config(args.output_dir, vars(args))
    
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
    
    evaluate_dataset_batch(
        agent=agent,
        parquet_path=args.parquet_path,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        batch_size=args.batch_size,
        save_interval=args.save_interval,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
