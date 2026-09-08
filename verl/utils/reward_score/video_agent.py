# Copyright 2025 Online Video Agent Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
视频Agent的Reward计算模块

从模型输出中解析 \\boxed{} 格式的答案，与 ground_truth 比较计算分数。
支持两种问题类型：
1. 选择题（choice）：精确匹配
2. OCR问题（ocr）：使用文本相似度（编辑距离）
"""

import random
import re
from difflib import SequenceMatcher
from typing import Tuple


def text_similarity(text1: str, text2: str) -> float:
    """计算两个文本的相似度（使用SequenceMatcher）
    
    Args:
        text1: 第一个文本
        text2: 第二个文本
        
    Returns:
        相似度分数 (0.0-1.0)
    """
    if not text1 or not text2:
        return 0.0
    
    # 标准化文本
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()
    
    # 使用SequenceMatcher计算相似度
    return SequenceMatcher(None, text1, text2).ratio()


def extract_boxed_answer(text: str) -> Tuple[str, int]:
    """从文本中提取 \\boxed{} 格式的答案及其数量
    
    Args:
        text: 模型输出文本
        
    Returns:
        tuple: (提取的答案字符串, 找到的boxed数量)
               如果没有找到返回 ("", 0)
    """
    # 匹配 \boxed{...} 格式，支持嵌套大括号
    # 使用贪婪匹配找到所有 boxed
    matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if matches:
        # 返回 (最后一个匹配作为最终答案, 匹配的总数量)
        return matches[-1].strip(), len(matches)
    return "", 0


def normalize_answer(answer: str) -> str:
    """标准化答案用于比较
    
    Args:
        answer: 原始答案
        
    Returns:
        标准化后的答案
    """
    # 转小写，去除首尾空白
    answer = answer.lower().strip()
    # 去除标点符号
    answer = re.sub(r'[^\w\s]', '', answer)
    # 合并多个空格
    answer = re.sub(r'\s+', ' ', answer)
    return answer


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs
) -> float:
    """
    计算视频Agent的reward分数
    
    从模型输出中解析 \\boxed{} 格式的答案，与 ground_truth 比较。
    支持两种问题类型：
    1. 选择题（choice）：精确匹配
    2. OCR问题（ocr）：使用文本相似度，超过阈值给1分
    
    Args:
        data_source: 数据源标识
        solution_str: 模型的完整输出
        ground_truth: 标准答案
        extra_info: 额外信息，包含 question_type 和 ground_truth_text
    
    Returns:
        float: 最终的 reward 分数
    """
    extra_info = extra_info or {}
    
    # 从模型输出中提取答案和数量
    extracted_answer, match_count = extract_boxed_answer(solution_str)
    
    # 获取问题类型
    question_type = extra_info.get("question_type", "choice")
    
    # 根据问题类型计算答案分数
    if question_type == "ocr":
        # OCR问题：使用文本相似度
        ground_truth_text = extra_info.get("ground_truth_text", ground_truth)
        
        if extracted_answer and ground_truth_text:
            similarity = text_similarity(extracted_answer, ground_truth_text)
            # 相似度阈值：0.6
            ocr_threshold = 0.6
            answer_reward = 1.0 if similarity >= ocr_threshold else 0.0
            
            # 采样打印OCR评估详情
            if random.randint(1, 32) == 1:
                print("=" * 50)
                print(f"[OCR Reward]")
                print(f"  ground_truth_text: {ground_truth_text}")
                print(f"  extracted_answer: {extracted_answer}")
                print(f"  similarity: {similarity:.3f}")
                print(f"  threshold: {ocr_threshold}")
                print(f"  answer_reward: {answer_reward}")
                print("=" * 50)
        else:
            answer_reward = 0.0
    else:
        # 选择题：精确匹配
        normalized_extracted = normalize_answer(extracted_answer)
        normalized_ground_truth = normalize_answer(ground_truth)
        
        if normalized_extracted and normalized_ground_truth:
            # 完全匹配
            if normalized_extracted == normalized_ground_truth:
                answer_reward = 1.0
            # 选择题特殊处理：只比较首字母
            elif len(normalized_ground_truth) == 1 and normalized_extracted.startswith(normalized_ground_truth):
                answer_reward = 1.0
            elif len(normalized_extracted) == 1 and normalized_ground_truth.startswith(normalized_extracted):
                answer_reward = 1.0
            else:
                answer_reward = 0.0
        else:
            # 没有提取到答案
            answer_reward = 0.0
    
    # 获取工具 reward（如 search_memory 的惩罚）
    tool_rewards = extra_info.get("tool_rewards", [])

    # if len(tool_rewards) > 0 and answer_reward == 0:
    #     tool_reward = 0.2
    # else:
    #     tool_reward = 0.0

    tool_reward = 0.0

    # 格式惩罚：如果出现多个 \boxed{}，扣除 0.1 分
    format_penalty = 0.0
    if match_count > 1:
        format_penalty = 0.1

    # 总分 = 答案分数 + 工具分数 - 格式惩罚
    total_reward = answer_reward + tool_reward - format_penalty
    
    # 日志（采样打印）
    do_print = random.randint(1, 32) == 1
    if do_print:
        print("=" * 50)
        print(f"[VideoAgent Reward]")
        print(f"  question_type: {question_type}")
        print(f"  data_source: {data_source}")
        print(f"  ground_truth: {ground_truth}")
        print(f"  extracted_answer: {extracted_answer} (Count: {match_count})")
        print(f"  answer_reward: {answer_reward}")
        print(f"  format_penalty: {format_penalty}")
        print(f"  total_reward: {total_reward}")
        print(f"  solution: {solution_str}")
        print("=" * 50)
    
    return total_reward