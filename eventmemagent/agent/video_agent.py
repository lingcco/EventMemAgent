#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Online Video Agent - 统一推理接口（vLLM 直接推理版本）

与训练保持完全一致：
1. 使用 processor.apply_chat_template 生成 prompt 文本
2. 使用 processor 处理图片
3. 直接使用 vLLM 的 Python API（和训练时一样）

使用方法：
    from eventmemagent.agent.video_agent import VideoAgent
    
    agent = VideoAgent(
        model_path="...", 
        tool_config_path="..."
    )
    
    # 离线推理
    result = agent.inference_offline(parquet_path, sample_idx)
    
    # 在线推理
    result = agent.inference_online(video_path, question)
"""

import json
import os
import base64
import re
import time
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _load_parquet_with_nested_data(parquet_path: str) -> pd.DataFrame:
    """
    加载包含嵌套数据的 parquet 文件
    
    使用 ParquetFile 逐批读取，避免 chunked array 转换问题
    """
    import pyarrow.parquet as pq
    
    try:
        parquet_file = pq.ParquetFile(parquet_path)
        
        # 逐批读取并转换为 Python 对象
        all_rows = []
        for batch in parquet_file.iter_batches(batch_size=100):
            batch_dict = batch.to_pydict()
            num_rows = len(batch_dict[list(batch_dict.keys())[0]])
            for i in range(num_rows):
                row = {col: batch_dict[col][i] for col in batch_dict}
                all_rows.append(row)
        
        return pd.DataFrame(all_rows)
    except Exception as e:
        print(f"[Warning] Failed to read with ParquetFile iter_batches: {e}")
    
    # 备选方案：使用 pyarrow 逐元素转换
    try:
        table = pq.read_table(parquet_path)
        
        data_dict = {}
        for col_name in table.column_names:
            chunked_col = table.column(col_name)
            col_values = []
            for chunk in chunked_col.chunks:
                for i in range(len(chunk)):
                    col_values.append(chunk[i].as_py())
            data_dict[col_name] = col_values
        
        return pd.DataFrame(data_dict)
    except Exception as e:
        print(f"[Warning] Failed to read with pyarrow element-by-element: {e}")
    
    # 最后尝试 pandas 默认方式
    return pd.read_parquet(parquet_path)
from PIL import Image
from transformers import AutoProcessor

from .tools import ToolManager
from eventmemagent.memory.builder import OnlineMemoryBuilder


def encode_pil_image(pil_image: Image.Image, format: str = "JPEG") -> str:
    """将 PIL Image 编码为 base64"""
    buffered = BytesIO()
    pil_image.save(buffered, format=format, quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _qwen2_5_vl_dedup_image_tokens(prompt_ids: List[int], processor) -> List[int]:
    """
    对 Qwen2.5-VL / Qwen3-VL 的 image tokens 进行去重（与训练时一致）
    
    vLLM 会根据 image_data 自动扩展 <|image_pad|> token，
    所以需要先去重，只保留一个 <|image_pad|>
    """
    if processor is None:
        return prompt_ids
    
    # 尝试获取 image_token_id
    image_token_id = None
    if hasattr(processor, 'image_token_id'):
        image_token_id = processor.image_token_id
    elif hasattr(processor, 'tokenizer'):
        tokenizer = processor.tokenizer
        if hasattr(tokenizer, 'image_token_id'):
            image_token_id = tokenizer.image_token_id
        else:
            for token_name in ['<|image_pad|>', '<|IMAGE|>', '<image>']:
                try:
                    token_id = tokenizer.convert_tokens_to_ids(token_name)
                    if token_id != tokenizer.unk_token_id:
                        image_token_id = token_id
                        break
                except:
                    pass
    
    if image_token_id is None:
        return prompt_ids
    
    prompt_ids_arr = np.array(prompt_ids)
    mask = np.ones(len(prompt_ids_arr), dtype=bool)
    is_value = prompt_ids_arr == image_token_id
    mask[1:] &= ~(is_value[1:] & is_value[:-1])
    
    return prompt_ids_arr[mask].tolist()


@dataclass
class InferenceResult:
    """推理结果"""
    answer: str
    messages: List[Dict]
    tool_calls: List[Dict]
    num_turns: int
    input_tokens: int = 0  # 初始输入 token 数
    output_tokens: int = 0  # 总输出 token 数


class VideoAgent:
    """视频理解 Agent（使用 vLLM Python API 推理，与训练保持完全一致）"""
    
    def __init__(
        self,
        model_path: str,
        tool_config_path: str,
        embedding_model_path: str = None,
        max_assistant_turns: int = 5,
        max_new_tokens: int = 16384,
        temperature: float = 1.0,  # 与训练时一致，训练用 1.0，validation 用 0
        # 在线推理时的视频处理参数
        sample_fps: float = 1.0,
        change_threshold: float = 0.2,
        short_term_max_frames: int = 32,
        short_term_min_frames: int = 8,
        # vLLM 参数
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = None,
    ):
        self.model_path = model_path
        self.max_assistant_turns = max_assistant_turns
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # 加载 processor（与训练保持一致）
        print(f"[VideoAgent] Loading processor from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        # 初始化工具管理器
        self.tool_manager = ToolManager(
            tool_config_path=tool_config_path,
            embedding_model_path=embedding_model_path,
        )
        
        # 获取工具 schema（与训练保持一致）
        self.tool_schemas = self.tool_manager.get_tool_schemas()
        
        # 在线推理参数
        self.sample_fps = sample_fps
        self.change_threshold = change_threshold
        self.short_term_max_frames = short_term_max_frames
        self.short_term_min_frames = short_term_min_frames
        
        # 在线记忆构建器（懒加载）
        self._memory_builder = None
        
        # 初始化 vLLM（与训练时一致，直接用 Python API）
        print(f"[VideoAgent] Loading vLLM model from {model_path}...")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from vllm import LLM, SamplingParams
        
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 64, "video": 0},
            max_num_seqs=int(os.environ.get("EVAL_MAX_NUM_SEQS", "16")),
            mm_processor_cache_gb=float(os.environ.get("EVAL_MM_PROCESSOR_CACHE_GB", "0"))
        )
        self.SamplingParams = SamplingParams
        
        print("[VideoAgent] Initialization complete.")
    
    def _get_memory_builder(self) -> OnlineMemoryBuilder:
        """Build memory with the frozen 4B caption service used in the paper."""
        if self._memory_builder is None:
            mode = os.environ.get("MEMORY_CAPTION_MODE", "http")
            if mode not in ("http", "shared"):
                raise ValueError("MEMORY_CAPTION_MODE must be http or shared")
            if mode == "shared":
                print("[Warning] Shared policy captions are an ablation, not the paper's frozen 4B memory.")
            self._memory_builder = OnlineMemoryBuilder(
                llm=self.llm if mode == "shared" else None,
                processor=self.processor if mode == "shared" else None,
                sampling_params_class=self.SamplingParams if mode == "shared" else None,
                api_base=os.environ.get("MEMORY_API_BASE", "http://localhost:8000/v1"),
                api_key=os.environ.get("MEMORY_API_KEY", "EMPTY"),
                model_name=os.environ.get("MEMORY_CAPTION_MODEL_NAME", "Qwen3-VL-4B-Instruct"),
                # 通用参数
                embedding_model_name=self.tool_manager.embedding_model_path,
                sample_fps=self.sample_fps,
                change_threshold=self.change_threshold,
                short_term_max_frames=self.short_term_max_frames,
                short_term_min_frames=self.short_term_min_frames,
            )
        return self._memory_builder
    
    def _create_system_prompt(self) -> str:
        """Archived inference prompt; training uses the prompt stored in parquet.

        See docs/REPRODUCIBILITY.md for the recovered prompt differences.
        """
        return """You are an intelligent video analysis agent with access to a memory system containing events from a video.

## Your Capabilities:
- You can directly see all short-term memory (recent video frames) that are provided to you
- You can search through long-term memory to retrieve relevant events (each event has an image, caption, and change_log)
- You can perform OCR (text recognition) on any image from long-term memory or short-term memory
- You can detect and locate specific objects in video frames using zero-shot object detection
- You can reason about temporal relationships and visual details

## Memory System Details:

### Short-Term Memory:
- Each frame in short-term memory has a **frame number and timestamp label** in the top-left corner (e.g., "Frame 0 | 12.5s", "Frame 1 | 13.0s", etc.)
- The frame number indicates the temporal order: Frame 0 is the earliest, and higher numbers are more recent
- The timestamp (in seconds) shows the exact time point in the video when this frame was captured
- Use both frame numbers and timestamps to understand the sequence and timing of events in the recent video

### Long-Term Memory:
- Each event in long-term memory contains:
  - An image (first frame of the event)
  - A caption describing what happens in the event
  - Change_log: Information about transitions between events
    - `change_from_previous`: Describes how the current event differs from the previous event
    - `change_to_next`: Describes how the current event transitions to the next event
- The change_log helps you understand temporal relationships and causal connections between events
- When searching long-term memory, pay attention to change_log to understand event sequences

## Available Tools:
1. **search_memory**: Search long-term memory to find relevant events. Use when you need historical context.
   - Two search modes:
     (1) Text query: Use `query` parameter to search by semantic similarity with event captions
     (2) Time range: Use `start_time` and `end_time` parameters to find events overlapping with a specific time range
   - You can use either mode, but not both at the same time
   - The search results will include events with their captions and change_log information
   - Use change_log to understand how events relate to each other temporally
2. **ocr**: Extract text from images. Use when you need to read text visible in video frames.
   - Use `event_id` parameter to OCR an image from long-term memory (after search_memory returns events)
   - Use `frame_index` parameter to OCR a frame from short-term memory (0-indexed based on the frames shown to you)
3. **detect_objects**: Detect and locate specific objects in images. Use when you need to find, count, or verify presence of objects.
   - `labels`: List of object names to detect (e.g., ["dog", "cat", "person"])
   - Choose one of the following parameters to specify the image to detect objects in:
    (1) `event_id`: Detect in long-term memory event image
    (2) `frame_index`: Detect in short-term memory frame. If not specified, uses the last frame.
    
## Response Guidelines:

For each step, follow this format:
1) Thought: one concise sentence explaining what you're thinking or planning to do.
2) Action: a short description of what action you're taking (search memory, perform OCR, detect objects, or provide final answer).
3) If using a tool, make a tool call. If answering, output \\boxed{your answer}.

Rules:
- Be brief: one sentence for Thought, one for Action.
- Only search when current information is insufficient.
- Use OCR when you need to read text that appears in images (signs, documents, screens, etc.)
- Use detect_objects when you need to find, count, or locate specific objects in images.
- Pay attention to frame numbers in short-term memory to understand temporal order.
- If the question doesn't mention a specific time, then only search long-term memory if short-term memory is unrelated to the question; otherwise, only refer to short-term memory for the answer.
- When you have enough information, output your final answer using \\boxed{} format. For example: \\boxed{A} or \\boxed{The cat is sleeping}."""


    def _load_images_from_short_term_memory(self, short_term_memory: List[Dict]) -> List[Image.Image]:
        """从短期记忆加载图片"""
        images = []
        for frame in short_term_memory:
            img_data = frame.get("image", "")
            if isinstance(img_data, Image.Image):
                images.append(img_data)
            elif isinstance(img_data, str) and img_data.startswith("data:image"):
                img_base64 = img_data.split(",")[1]
                img_bytes = base64.b64decode(img_base64)
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                images.append(img)
            elif isinstance(img_data, str) and img_data:
                try:
                    img_bytes = base64.b64decode(img_data)
                    img = Image.open(BytesIO(img_bytes)).convert("RGB")
                    images.append(img)
                except:
                    pass
        return images
    
    def _build_user_message(
        self, 
        question: str, 
        long_term_memory: List[Dict], 
        short_term_memory: List[Dict]
    ) -> List[Dict]:
        """构建用户消息内容（与训练时一致，返回 content list 格式）"""
        stm_frame_count = len(short_term_memory)
        
        if long_term_memory:
            ltm_info = f"We have {len(long_term_memory)} events in the long-term memory."
        else:
            ltm_info = "We have no events in the long-term memory, you shouldn't search the long-term memory."
        
        content = []
        
        if stm_frame_count > 0:
            # 先添加文本
            content.append({"type": "text", "text": f"{ltm_info}\nRecent Video ({stm_frame_count} frames) in the short-term memory:\n"})
            # 添加图片占位符
            for _ in range(stm_frame_count):
                content.append({"type": "image"})
            # 添加问题
            content.append({"type": "text", "text": f"\nQuestion: {question}\nPlease analyze the video and answer the question."})
        else:
            content.append({"type": "text", "text": f"{ltm_info}\nQuestion: {question}\nPlease analyze the video and answer the question."})
        
        return content
    
    def _apply_chat_template(
        self, 
        messages: List[Dict], 
        images: List[Image.Image],
        add_generation_prompt: bool = True
    ) -> Tuple[str, List[int]]:
        """应用 chat template（与训练时完全一致）"""
        raw_prompt = self.processor.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
        
        model_inputs = self.processor(
            text=[raw_prompt], 
            images=images if images else None, 
            return_tensors="pt"
        )
        prompt_ids = model_inputs["input_ids"].squeeze(0).tolist()
        
        return raw_prompt, prompt_ids
    
    def _apply_chat_template_for_tool_response(
        self, 
        tool_messages: List[Dict], 
        new_images: List[Image.Image]
    ) -> List[int]:
        """为工具响应应用 chat template（与训练时一致）"""
        raw_tool_response = self.processor.apply_chat_template(
            tool_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        
        current_images = new_images if new_images else None
        model_inputs = self.processor(
            text=[raw_tool_response], 
            images=current_images, 
            return_tensors="pt"
        )
        response_ids = model_inputs["input_ids"].squeeze(0).tolist()
        
        return response_ids

    def _generate(
        self, 
        prompt_ids: List[int], 
        image_data: List[Image.Image]
    ) -> Tuple[str, List[int]]:
        """使用 vLLM 生成（与训练时完全一致）"""
        from vllm import TokensPrompt
        
        sampling_params = self.SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        
        # 与训练时完全一致：先对 image tokens 去重
        prompt_ids_deduped = _qwen2_5_vl_dedup_image_tokens(prompt_ids, self.processor)
        
        prompt = TokensPrompt(
            prompt_token_ids=prompt_ids_deduped,
            multi_modal_data={"image": image_data} if image_data else None
        )
        
        outputs = self.llm.generate(
            prompts=prompt,
            sampling_params=sampling_params,
        )
        
        response_text = outputs[0].outputs[0].text
        response_ids = list(outputs[0].outputs[0].token_ids)
        
        return response_text.strip(), response_ids
    
    def _extract_tool_calls(self, response_text: str) -> List[Dict]:
        """从响应中提取工具调用（hermes 格式）"""
        tool_calls = []
        pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                tool_call = json.loads(match)
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue
        
        return tool_calls
    
    def _extract_final_answer(self, response_text: str) -> Optional[str]:
        """从响应中提取最终答案"""
        pattern = r'\\boxed\{([^}]*)\}'
        match = re.search(pattern, response_text)
        if match:
            return match.group(1)
        return None

    def _run_inference_loop(
        self,
        messages: List[Dict],
        image_data: List[Image.Image],
        long_term_memory: List[Dict],
        short_term_memory_raw: List[Dict],
    ) -> InferenceResult:
        """运行推理循环（与训练时 tool_agent_loop.py 完全一致）"""
        import asyncio
        
        tool_calls_history = []
        assistant_turns = 0
        user_turns = 0
        total_output_tokens = 0  # 记录总输出 token 数
        
        # 训练时：short_term_memory = image_data
        short_term_memory = image_data
        
        # 初始化工具
        self.tool_manager.initialize(long_term_memory, short_term_memory_raw, short_term_memory)
        
        # === PENDING 状态 ===
        raw_prompt, prompt_ids = self._apply_chat_template(messages, image_data)
        initial_input_tokens = len(prompt_ids)  # 记录初始输入 token 数
        
        print("\n" + "=" * 60)
        print("🎬 Starting Inference Loop")
        print("=" * 60)
        
        while assistant_turns < self.max_assistant_turns:
            # === GENERATING 状态 ===
            response_text, response_ids = self._generate(prompt_ids, image_data)
            assistant_turns += 1
            total_output_tokens += len(response_ids)  # 累加输出 token
            
            prompt_ids = prompt_ids + response_ids
            
            # 美化输出
            print(f"\n{'─' * 60}")
            print(f"🤖 [Turn {assistant_turns}] Assistant Response:")
            print(f"{'─' * 60}")
            print(response_text)
            
            # 检查最终答案
            final_answer = self._extract_final_answer(response_text)
            if final_answer:
                print(f"\n{'=' * 60}")
                print(f"✅ Final Answer: {final_answer}")
                print(f"{'=' * 60}\n")
                messages.append({"role": "assistant", "content": response_text})
                return InferenceResult(
                    answer=final_answer,
                    messages=messages,
                    tool_calls=tool_calls_history,
                    num_turns=assistant_turns + user_turns,
                    input_tokens=initial_input_tokens,
                    output_tokens=total_output_tokens,
                )
            
            # 提取工具调用
            tool_calls = self._extract_tool_calls(response_text)
            
            if not tool_calls:
                print(f"\n{'=' * 60}")
                print(f"⚠️  No tool call or final answer found")
                print(f"{'=' * 60}\n")
                messages.append({"role": "assistant", "content": response_text})
                return InferenceResult(
                    answer=response_text,
                    messages=messages,
                    tool_calls=tool_calls_history,
                    num_turns=assistant_turns + user_turns,
                    input_tokens=initial_input_tokens,
                    output_tokens=total_output_tokens,
                )
            
            # === PROCESSING_TOOLS 状态 ===
            messages.append({"role": "assistant", "content": response_text})
            
            add_messages = []
            new_images_this_turn = []
            
            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("arguments", {})
                
                tool_calls_history.append(tool_call)
                
                print(f"\n{'─' * 60}")
                print(f"🔧 Tool Call: {tool_name}")
                print(f"   Arguments: {json.dumps(tool_args, ensure_ascii=False)}")
                print(f"{'─' * 60}")
                
                tool_response, new_images = asyncio.run(
                    self.tool_manager.execute_tool(tool_name, tool_args)
                )
                
                print(f"📋 Tool Response:")
                # 限制输出长度，避免太长
                if len(tool_response) > 500:
                    print(f"{tool_response[:500]}...")
                    print(f"   (... truncated, total {len(tool_response)} chars)")
                else:
                    print(tool_response)
                
                if new_images:
                    print(f"🖼️  Returned {len(new_images)} image(s)")
                
                # 构建工具响应消息（与训练时格式一致）
                if new_images:
                    content = []
                    for img in new_images:
                        content.append({"type": "image"})
                        new_images_this_turn.append(img)
                    content.append({"type": "text", "text": tool_response})
                    add_messages.append({"role": "tool", "content": content})
                else:
                    add_messages.append({"role": "tool", "content": tool_response})
            
            messages.extend(add_messages)
            
            # 为工具响应应用 chat template
            tool_response_ids = self._apply_chat_template_for_tool_response(
                add_messages, 
                new_images_this_turn
            )
            
            prompt_ids = prompt_ids + tool_response_ids
            
            # 更新图片列表
            if new_images_this_turn:
                image_data.extend(new_images_this_turn)
            
            user_turns += 1
        
        print(f"\n{'=' * 60}")
        print(f"⏱️  Max turns ({self.max_assistant_turns}) reached")
        print(f"{'=' * 60}\n")
        
        return InferenceResult(
            answer="Max turns reached",
            messages=messages,
            tool_calls=tool_calls_history,
            num_turns=assistant_turns + user_turns,
            input_tokens=initial_input_tokens,
            output_tokens=total_output_tokens,
        )
    
    def inference_offline(
        self, 
        parquet_path: str, 
        sample_idx: int = 0
    ) -> InferenceResult:
        """离线推理：使用预处理好的 parquet 数据"""
        df = _load_parquet_with_nested_data(parquet_path)
        sample = df.iloc[sample_idx]
        
        # 处理可能是 numpy 数组的字段
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
        
        image_data = self._load_images_from_short_term_memory(short_term_memory_raw)
        
        system_prompt = self._create_system_prompt()
        user_content = self._build_user_message(question, long_term_memory, short_term_memory_raw)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        
        return self._run_inference_loop(messages, image_data, long_term_memory, short_term_memory_raw)
    
    def inference_online(
        self, 
        video_path: str, 
        question: str
    ) -> InferenceResult:
        """在线推理：直接输入视频文件
        
        使用独立的冻结 4B caption 模型：
        1. 先处理视频，构建长期记忆和短期记忆
        2. 然后进行推理
        """
        # Build memory independently from the trained policy.
        memory_builder = self._get_memory_builder()
        long_term_memory, short_term_memory_raw = memory_builder.process_video(video_path)
        
        # 加载图片
        image_data = self._load_images_from_short_term_memory(short_term_memory_raw)
        
        # 构建消息
        system_prompt = self._create_system_prompt()
        user_content = self._build_user_message(question, long_term_memory, short_term_memory_raw)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        
        # 运行推理
        return self._run_inference_loop(messages, image_data, long_term_memory, short_term_memory_raw)
