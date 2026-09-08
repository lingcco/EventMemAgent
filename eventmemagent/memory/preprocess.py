#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频记忆预处理脚本

将视频预处理为离线的长期记忆和短期记忆，用于训练。
输出格式为parquet文件，包含：
- messages: 对话历史
- long_term_memory: 长期记忆列表（事件列表）
- short_term_memory: 短期记忆帧列表
- ground_truth: 标准答案
- question_type: 问题类型

使用方法：
    # 1. 首先启动 vLLM API 服务器
    bash scripts/services/memory.sh
    
    # 2. 然后运行预处理脚本
    python -m eventmemagent.memory.preprocess \
        --input_json /path/to/dataset.json \
        --video_dir /path/to/videos \
        --output_dir /path/to/output \
        --api_base http://localhost:8000/v1 \
        --sample_fps 1.0
"""

import argparse
import json
import os
import sys
import base64
import time
import requests
import gc
import glob
import random
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from PIL import Image

def encode_pil_image(pil_image, format="JPEG"):

    buffered = BytesIO()
    pil_image.save(buffered, format=format, quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return img_str

@dataclass
class FrameInfo:
    """单帧信息"""
    image_base64: str  # base64编码的图像
    timestamp: float
    frame_idx: int


@dataclass
class EventMemory:
    """长期记忆中的单个事件"""
    event_id: int
    start_time: float
    end_time: float
    first_frame_base64: str  # base64编码的首帧图像
    caption: str
    embedding: Optional[List[float]] = None
    frame_count: int = 0
    # 变化日志（参考 NumPro 的时间理解增强）
    change_from_previous: Optional[str] = None  # 从上一个事件到当前事件的变化
    change_to_next: Optional[str] = None        # 从当前事件到下一个事件的变化


class VideoMemoryPreprocessor:
    """视频记忆预处理器（使用 vLLM API）"""
    
    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model_name: str = "Qwen3-VL-4B-Instruct",
        embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        sample_fps: float = 1.0,
        short_term_max_frames: int = 32,
        short_term_min_frames: int = 8,
        caption_sample_frames: int = 32,
        change_threshold: float = 0.2,
        detection_method: str = "histogram",
        device: str = "cuda",
        frames_dir: str = None,  # 帧保存目录
        max_frame_resolution: Tuple[int, int] = (420, 360),  # 最大分辨率 (width, height)
        max_retries: int = 3,
        timeout: int = 120,
    ):
        # API 配置
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout = timeout
        
        # 其他配置
        self.embedding_model_name = embedding_model_name
        self.sample_fps = sample_fps
        self.short_term_max_frames = short_term_max_frames
        self.short_term_min_frames = short_term_min_frames
        self.caption_sample_frames = caption_sample_frames
        self.change_threshold = change_threshold
        self.detection_method = detection_method
        self.device = device
        self.frames_dir = frames_dir
        self.max_frame_resolution = max_frame_resolution
        
        # 延迟加载 embedding 模型
        self.embedding_model = None
        self._embedding_initialized = False
    
    def _init_models(self):
        """初始化 embedding 模型（VLM 通过 API 调用）"""
        if self._embedding_initialized:
            return
        
        self._embedding_initialized = True
        
        # 检查 API 服务是否可用
        try:
            response = requests.get(f"{self.api_base}/models", timeout=10)
            if response.status_code == 200:
                models = response.json().get("data", [])
                model_ids = [m.get("id", "") for m in models]
                print(f"[Preprocessor] API server available. Models: {model_ids}")
            else:
                print(f"[Warning] API server returned status {response.status_code}")
        except Exception as e:
            print(f"[Warning] Failed to connect to API server at {self.api_base}: {e}")
            print(f"[Warning] Make sure to start the vLLM server first: bash start_vllm_server.sh")
        
        # 加载 embedding 模型
        print(f"[Preprocessor] Loading embedding model: {self.embedding_model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                model_kwargs={
                    "device_map": "auto",
                    "torch_dtype": torch.float16
                },
                tokenizer_kwargs={"padding_side": "left"}
            )
        except Exception as e:
            print(f"[Warning] Failed to load embedding model: {e}")
            self.embedding_model = None
    
    def _call_vlm_api(self, messages: List[Dict], max_tokens: int = 256, extra_body: Dict = None) -> Optional[str]:
        """调用 vLLM API 生成文本"""
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        
        # 添加 extra_body 参数（用于传递 mm_processor_kwargs 等）
        if extra_body:
            payload.update(extra_body)
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url, 
                    headers=headers, 
                    json=payload, 
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    return content.strip()
                else:
                    print(f"[Warning] API returned status {response.status_code}: {response.text[:200]}")
                    
            except requests.exceptions.Timeout:
                print(f"[Warning] API timeout (attempt {attempt + 1}/{self.max_retries})")
            except Exception as e:
                print(f"[Warning] API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
        
        return None
    
    
    def _compute_histogram_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """计算两张图像的直方图相似度"""
        # 转换为灰度图
        gray1 = img1.convert("L")
        gray2 = img2.convert("L")
        
        # 计算直方图
        hist1 = np.array(gray1.histogram(), dtype=np.float32)
        hist2 = np.array(gray2.histogram(), dtype=np.float32)
        
        # 归一化
        hist1 /= hist1.sum() + 1e-8
        hist2 /= hist2.sum() + 1e-8
        
        # 计算相关性
        correlation = np.corrcoef(hist1, hist2)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def _is_event_boundary(self, avg_frame: Image.Image, new_frame: Image.Image) -> bool:
        """检测是否为事件边界"""
        similarity = self._compute_histogram_similarity(avg_frame, new_frame)
        return similarity < self.change_threshold
    
    def _generate_caption(self, frames: List[Image.Image], start_time: float, end_time: float) -> str:
        """为一组帧生成 caption（通过 vLLM API）"""
        if not frames:
            return f"Event from {start_time:.2f}s to {end_time:.2f}s"
        
        # 采样帧
        sample_count = min(self.caption_sample_frames, len(frames))
        if len(frames) <= sample_count:
            sampled_frames = frames
        else:
            indices = np.linspace(0, len(frames) - 1, sample_count, dtype=int)
            sampled_frames = [frames[i] for i in indices]
        
        try:
            # 使用多图片格式（方案三：与训练时一致）
            # 先计算统一缩放比例，避免高分辨率图片导致上下文过长
            target_w, target_h = self.max_frame_resolution
            min_scale = 1.0
            for frame in sampled_frames:
                orig_w, orig_h = frame.size
                scale = min(target_w / orig_w, target_h / orig_h)
                if scale < min_scale:
                    min_scale = scale
            
            # 每个帧作为单独的 image_url 内容项
            content = []
            for frame in sampled_frames:
                # 缩放图片以控制分辨率
                if min_scale < 1.0:
                    new_w = int(frame.size[0] * min_scale)
                    new_h = int(frame.size[1] * min_scale)
                    frame = frame.resize((new_w, new_h), Image.LANCZOS)
                img_base64 = encode_pil_image(frame)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                })
            
            # 添加文本提示
            content.append({
                    "type": "text",
                    "text": (
                        f"You are a video memory builder. Describe what happens in this event sequence.\n"
                        f"TimeRange: [{start_time:.2f}s - {end_time:.2f}s].\n"
                        "Output EXACTLY ONE concise paragraph (single line) describing the event.\n"
                        "Be specific about: subjects, actions, objects, tools, locations, manner (how), and outcomes.\n"
                        "Do NOT include timestamps or bullet lists. Avoid hallucination.\n"
                        "Example: 'A woman in an apron grasps a chef knife and slices a potato on a wooden cutting board into thin pieces.'"
                    )
            })
            
            messages = [{"role": "user", "content": content}]
            
            # 调用 API
            caption = self._call_vlm_api(messages, max_tokens=256)
            
            if caption:
                return caption.replace("\n", " ")
            else:
                return f"Event from {start_time:.2f}s to {end_time:.2f}s"
            
        except Exception as e:
            print(f"[Warning] Failed to generate caption: {e}")
            return f"Event from {start_time:.2f}s to {end_time:.2f}s"
    
    def _extract_embedding(self, text: str) -> Optional[List[float]]:
        """提取文本embedding"""
        if self.embedding_model is None or not text:
            return None
        
        try:
            embedding = self.embedding_model.encode(
                text,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return embedding.tolist()
        except Exception as e:
            print(f"[Warning] Failed to extract embedding: {e}")
            return None
    
    def _add_frame_number_to_image(self, image: Image.Image, frame_number: int, timestamp: float = None) -> Image.Image:
        """在图片左上角添加帧序号和时间戳标记（参考 NumPro）
        
        帮助模型理解时间维度，通过视觉标记区分不同时刻的帧
        
        Args:
            image: PIL Image对象
            frame_number: 帧序号（从0开始）
            timestamp: 时间戳（秒），可选
            
        Returns:
            添加了标记的图像
        """
        from PIL import ImageDraw, ImageFont
        
        # 创建副本避免修改原图
        img_with_label = image.copy()
        draw = ImageDraw.Draw(img_with_label)
        
        # 设置文字（包含帧序号和时间戳）
        if timestamp is not None:
            text = f"Frame {frame_number} | {timestamp:.1f}s"
        else:
            text = f"Frame {frame_number}"
        
        # 尝试加载字体，失败则使用默认字体
        try:
            # macOS 系统字体
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except:
            try:
                # Linux 系统字体
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                # 默认字体
                font = ImageFont.load_default()
        
        # 计算文字位置（左上角，留边距）
        padding = 10
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            # 兼容旧版本 PIL
            text_width, text_height = draw.textsize(text, font=font)
        
        # 绘制半透明黑色背景
        background_box = [
            padding, 
            padding, 
            padding + text_width + 10, 
            padding + text_height + 10
        ]
        draw.rectangle(background_box, fill=(0, 0, 0, 180))
        
        # 绘制白色文字
        draw.text((padding + 5, padding + 5), text, fill=(255, 255, 255), font=font)
        
        return img_with_label
    
    def _generate_change_description(
        self, 
        prev_image: Image.Image, 
        curr_image: Image.Image,
        prev_time_range: Tuple[float, float],
        curr_time_range: Tuple[float, float]
    ) -> str:
        """生成两个事件之间的变化描述
        
        通过对比前后两个事件的首帧图像，让模型描述发生了什么变化。
        这有助于模型理解事件之间的时间关系和因果关系。
        
        Args:
            prev_image: 上一个事件的图片（原始尺寸，不resize）
            curr_image: 当前事件的图片（原始尺寸，不resize）
            prev_time_range: 上一个事件的时间范围 (start, end)
            curr_time_range: 当前事件的时间范围 (start, end)
        
        Returns:
            变化描述文本
        """
        try:
            # 构建多图片消息（使用原始图片，不resize）
            content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_pil_image(prev_image)}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_pil_image(curr_image)}"
                    }
                },
                {
                    "type": "text",
                    "text": (
                        f"You are analyzing changes between two video events.\n"
                        f"Image 1: Event from {prev_time_range[0]:.2f}s to {prev_time_range[1]:.2f}s\n"
                        f"Image 2: Event from {curr_time_range[0]:.2f}s to {curr_time_range[1]:.2f}s\n\n"
                        "Describe what changed between these two events in ONE concise sentence.\n"
                        "Focus on: new objects/people appearing, actions starting/stopping, "
                        "location changes, or significant state changes.\n"
                        "Be specific and factual. Avoid speculation.\n"
                        "Example: 'The person moved from the kitchen to the living room and picked up a book.'"
                    )
                }
            ]
            
            messages = [{"role": "user", "content": content}]
            
            # 调用 API（使用较小的 max_tokens 以加快速度）
            change_desc = self._call_vlm_api(messages, max_tokens=128)
            
            if change_desc:
                return change_desc.replace("\n", " ").strip()
            else:
                return f"Transition from event at {prev_time_range[0]:.1f}s to event at {curr_time_range[0]:.1f}s"
        
        except Exception as e:
            print(f"[Warning] Failed to generate change description: {e}")
            return f"Transition from event at {prev_time_range[0]:.1f}s to event at {curr_time_range[0]:.1f}s"
    
    def _compute_uniform_scale_ratio(self, frames: List[Dict]) -> float:
        """计算所有帧的统一缩放比例，确保同一视频所有帧使用相同缩放比例
        
        Args:
            frames: 帧列表，每个帧包含 "image" 字段
            
        Returns:
            统一的缩放比例（取最小值以确保所有帧都能fit）
        """
        if not frames or self.max_frame_resolution is None:
            return 1.0
        
        target_w, target_h = self.max_frame_resolution
        min_scale = 1.0
        
        for f in frames:
            img = f["image"]
            orig_w, orig_h = img.size
            # 计算该帧需要的缩放比例
            scale = min(target_w / orig_w, target_h / orig_h)
            if scale < min_scale:
                min_scale = scale
        
        return min_scale
    
    def _resize_and_pad_image(self, image: Image.Image, scale_ratio: float) -> Image.Image:
        """缩放图像并添加padding，保持宽高比不变形
        
        Args:
            image: PIL Image对象
            scale_ratio: 缩放比例
            
        Returns:
            处理后的图像（尺寸为 max_frame_resolution）
        """
        target_w, target_h = self.max_frame_resolution
        orig_w, orig_h = image.size
        
        # 只有在需要缩小时才缩放
        if scale_ratio < 1.0:
            new_w = int(orig_w * scale_ratio)
            new_h = int(orig_h * scale_ratio)
            resized = image.resize((new_w, new_h), Image.LANCZOS)
        else:
            resized = image
            new_w, new_h = orig_w, orig_h
        
        # 如果图片本身就小于目标尺寸，直接使用原尺寸
        if new_w <= target_w and new_h <= target_h:
            # 创建目标尺寸的黑色背景图
            result = Image.new("RGB", (target_w, target_h), (0, 0, 0))
            # 将缩放后的图片居中放置
            paste_x = (target_w - new_w) // 2
            paste_y = (target_h - new_h) // 2
            result.paste(resized, (paste_x, paste_y))
            return result
        else:
            return resized
    
    def process_video(self, video_path: str, sample_id: str = None) -> Tuple[List[Dict], List[Dict]]:
        """
        处理单个视频，返回长期记忆和短期记忆列表
        
        新逻辑：短期记忆可以包含多个事件
        - 检测到事件分割点时，如果短期记忆未满，不转移到长期记忆
        - 只有短期记忆满了需要蓄水池采样时，才把第一个事件转移到长期记忆
        - 事件分割点的计算基于当前事件的平均帧（不是整个短期记忆）
        
        Args:
            video_path: 视频文件路径
            sample_id: 样本ID（保留参数以兼容旧代码，但不再使用）
        
        Returns:
            (long_term_memory, short_term_memory)
            - long_term_memory: 事件列表，每个事件包含 base64 图片
            - short_term_memory: 帧列表，每个帧包含 base64 图片和时间戳
        """
        self._init_models()
        
        from decord import VideoReader, cpu
        
        vr = None
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
        except Exception as e:
            print(f"[Warning] Failed to load video {video_path}: {e}")
            return [], []
        
        total_frames = len(vr)
        video_fps = float(vr.get_avg_fps()) if vr.get_avg_fps() > 0 else 25.0
        video_duration = total_frames / video_fps
        
        print(f"[Preprocessor] Video: {total_frames} frames, {video_fps:.2f} fps, {video_duration:.2f}s")
        
        # 计算采样间隔
        sample_interval = int(video_fps / self.sample_fps)
        if sample_interval < 1:
            sample_interval = 1
        
        # 生成采样帧索引
        sampled_indices = list(range(0, total_frames, sample_interval))
        
        # 长期记忆
        long_term_events = []
        
        # 短期记忆：包含多个事件，每个事件是一个字典
        short_term_events = []  # List[Dict]: 每个元素是一个事件
        
        # 当前正在构建的事件
        current_event = {
            "frames": [],
            "accumulated_sum": None,
            "total_frames_seen": 0,  # 用于蓄水池采样
        }
        
        next_event_id = 0
        
        def get_total_stm_frames():
            """获取短期记忆中的总帧数"""
            total = sum(len(e["frames"]) for e in short_term_events)
            total += len(current_event["frames"])
            return total
        
        def move_first_event_to_ltm():
            """将短期记忆中的第一个事件转移到长期记忆"""
            nonlocal next_event_id
            if not short_term_events:
                return
            
            first_event_data = short_term_events.pop(0)
            event = self._create_event(first_event_data["frames"], next_event_id)
            
            if event:
                # 生成变化描述
                if long_term_events:
                    prev_event = long_term_events[-1]
                    try:
                        prev_image = Image.open(BytesIO(base64.b64decode(prev_event.first_frame_base64)))
                    except Exception as e:
                        print(f"[Warning] Failed to decode previous event image: {e}")
                        prev_image = None
                    
                    if prev_image is not None:
                        curr_image = first_event_data["frames"][0]["image"]
                        change_desc = self._generate_change_description(
                            prev_image=prev_image,
                            curr_image=curr_image,
                            prev_time_range=(prev_event.start_time, prev_event.end_time),
                            curr_time_range=(event.start_time, event.end_time)
                        )
                        prev_event.change_to_next = change_desc
                        event.change_from_previous = change_desc
                
                long_term_events.append(event)
                next_event_id += 1
        
        total_sampled_frames = len(sampled_indices)
        for idx, frame_idx in enumerate(sampled_indices):
            is_last_frame = (idx == total_sampled_frames - 1)
            
            try:
                frame_array = vr[frame_idx].asnumpy()
            except Exception as e:
                print(f"[Warning] Failed to read frame {frame_idx}: {e}")
                continue
            
            frame_image = Image.fromarray(frame_array)
            timestamp = frame_idx / video_fps
            frame_array_float = frame_array.astype(np.float64)
            
            frame_info = {
                "image": frame_image,
                "timestamp": timestamp,
                "frame_idx": frame_idx,
                "array_float": frame_array_float
            }
            
            current_event_len = len(current_event["frames"])
            
            # 检测事件边界（只有当前事件达到最小帧数时才检测）
            # 注意：事件边界检测只基于当前事件的平均帧，与原逻辑一致
            is_boundary = False
            if current_event_len >= self.short_term_min_frames:
                avg_array = (current_event["accumulated_sum"] / current_event_len).astype(np.uint8)
                avg_frame = Image.fromarray(avg_array)
                is_boundary = self._is_event_boundary(avg_frame, frame_image)
            
            if is_boundary and current_event_len >= self.short_term_min_frames and not is_last_frame:
                # 情况 A：检测到事件边界
                # 将当前事件保存到短期记忆事件列表
                short_term_events.append(current_event)
                
                # 开始新事件（包含当前帧）
                current_event = {
                    "frames": [frame_info],
                    "accumulated_sum": frame_array_float.copy(),
                    "total_frames_seen": 1,
                }
                
                # 检查短期记忆是否超限，如果超限则转移第一个事件到长期记忆
                while get_total_stm_frames() > self.short_term_max_frames and len(short_term_events) > 0:
                    move_first_event_to_ltm()
                
            else:
                # 情况 B/C：需要添加帧到当前事件
                # 先检查是否需要腾出空间
                while get_total_stm_frames() >= self.short_term_max_frames and len(short_term_events) > 0:
                    # STM 满了且有多个事件，转移第一个事件腾出空间
                    move_first_event_to_ltm()
                
                # 重新计算当前事件是否满了
                current_event_len = len(current_event["frames"])
                current_event_full = current_event_len >= self.short_term_max_frames
                
                if current_event_full:
                    # 情况 B：当前事件满了，需要蓄水池采样
                    current_event["total_frames_seen"] += 1
                    accept_prob = self.short_term_max_frames / current_event["total_frames_seen"]
                    if random.random() < accept_prob:
                        remove_idx = random.randint(0, len(current_event["frames"]) - 1)
                        removed_frame = current_event["frames"].pop(remove_idx)
                        current_event["accumulated_sum"] -= removed_frame["array_float"]
                        current_event["accumulated_sum"] += frame_array_float
                        current_event["frames"].append(frame_info)
                    # 如果不接受，不添加新帧
                else:
                    # 情况 C：正常添加到当前事件
                    current_event["frames"].append(frame_info)
                    if current_event["accumulated_sum"] is None:
                        current_event["accumulated_sum"] = frame_array_float.copy()
                    else:
                        current_event["accumulated_sum"] += frame_array_float
                    current_event["total_frames_seen"] += 1
        
        # 收集所有短期记忆帧
        all_stm_frames = []
        for event_data in short_term_events:
            all_stm_frames.extend(event_data["frames"])
        all_stm_frames.extend(current_event["frames"])
        
        # 最多保留 max_frames 个帧
        frames_to_save = all_stm_frames[-self.short_term_max_frames:]
        
        # 计算同一视频所有帧的统一缩放比例
        uniform_scale_ratio = self._compute_uniform_scale_ratio(frames_to_save)
        
        # 将短期记忆帧转换为 base64 格式（兼容 qwen_vl_utils）
        short_term_memory_list = []
        for i, f in enumerate(frames_to_save):
            # 缩放并添加padding，保持宽高比
            processed_image = self._resize_and_pad_image(f["image"], uniform_scale_ratio)
            # 添加帧序号和时间戳标记
            labeled_image = self._add_frame_number_to_image(processed_image, i, f["timestamp"])
            
            # 转换为 base64 并使用 data URI 格式
            image_base64 = encode_pil_image(labeled_image)
            short_term_memory_list.append({
                "image": f"data:image/jpeg;base64,{image_base64}",
                "frame_index": i,
                "timestamp": f["timestamp"],
            })
        
        if long_term_events:
            long_term_memory = [asdict(event) for event in long_term_events]
        else:
            long_term_memory = []
        
        # 释放视频资源
        if vr is not None:
            del vr
        # 清理临时变量
        del short_term_events
        del current_event
        gc.collect()
        
        return long_term_memory, short_term_memory_list
    
    def _create_event(self, frames: List[Dict], event_id: int) -> Optional[EventMemory]:
        """从帧列表创建事件"""
        if not frames:
            return None
        
        start_time = frames[0]["timestamp"]
        end_time = frames[-1]["timestamp"]
        first_frame = frames[0]["image"]
        
        # 生成caption
        images = [f["image"] for f in frames]
        caption = self._generate_caption(images, start_time, end_time)
        
        # 提取embedding
        embedding = self._extract_embedding(caption)
        
        return EventMemory(
            event_id=event_id,
            start_time=start_time,
            end_time=end_time,
            first_frame_base64=encode_pil_image(first_frame),
            caption=caption,
            embedding=embedding,
            frame_count=len(frames)
        )


def create_system_prompt() -> str:
    """创建系统prompt
    
    注意：工具的 schema 会由 verl 通过 apply_chat_template(tools=...) 自动添加，
    这里只需要定义任务相关的说明。
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
2) Action: a short description of what action you're taking (search memory, perform OCR, detect objects, or provide final answer). If using a tool, make a tool call. If answering, output \\boxed{your answer}.

Rules:
- Be brief: one sentence for Thought, one for Action.
- Use OCR only when text is too small or blurry to read directly.
- Use detect_objects when objects are ambiguous or partially occluded.
- Pay attention to frame numbers in short-term memory to understand temporal order.
- If the question doesn't mention a specific time, then only search long-term memory if short-term memory is unrelated to the question; otherwise, only refer to short-term memory for the answer.
- **CRITICAL: You have a maximum of 10 turns to complete the task (This includes the number of turns in the response). Plan efficiently and avoid unnecessary tool calls. After 10 turns, the task will be forcibly terminated.**
- **NEVER loop through frames one by one.** If you need to check multiple frames, select only 1-2 most relevant frames.
- Output your final answer using \\boxed{} format. For example: \\boxed{A} or \\boxed{The cat is sleeping}.
"""


def process_single_sample(args):
    """处理单个样本（用于多线程，共享 preprocessor）"""
    idx, sample, video_dir, preprocessor = args
    
    system_prompt = create_system_prompt()
    
    video_file = sample.get("video", sample.get("path", ""))
    
    if not video_file:
        print(f"[Warning] Sample {idx} has no video field")
        return None
    
    video_path = os.path.join(video_dir, video_file)
    if not os.path.exists(video_path):
        print(f"[Warning] Video not found: {video_path}")
        return None
    
    # 处理视频
    sample_id = sample.get("id", str(idx))
    try:
        long_term_memory, short_term_memory = preprocessor.process_video(video_path, sample_id=sample_id)
    except Exception as e:
        print(f"[Error] Failed to process video {video_path}: {e}")
        return None
    finally:
        # 处理完成后清理资源
        gc.collect()
    
    # 跳过没有短期记忆帧的样本
    if not short_term_memory:
        print(f"[Warning] Sample {idx} has no short-term memory frames, skipping")
        return None
    
    # 构建消息
    question = sample.get("problem", sample.get("prompt", ""))
    ground_truth = sample.get("solution", sample.get("ground_truth", ""))
    if "<answer>" in ground_truth:
        ground_truth = ground_truth.split('>')[1].split('<')[0]
    question_type = sample.get("question_type", "choice")
    problem_type = sample.get("problem_type", "unknown")  # 保留QA类型信息
    
    stm_frame_count = len(short_term_memory)
    
    if long_term_memory:
        ltm_info = f"We have {len(long_term_memory)} events in the long-term memory."
    else:
        ltm_info = "We have no events in the long-term memory, you shouldn't search the long-term memory."
    
    if stm_frame_count > 0:
        image_placeholders = "<image>" * stm_frame_count
        user_content = f"{ltm_info}\nRecent Video ({stm_frame_count} frames) in the short-term memory:\n{image_placeholders}\nQuestion: {question}\nPlease analyze the video and answer the question."
    else:
        user_content = f"{ltm_info}\nQuestion: {question}\nPlease analyze the video and answer the question."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    tools_kwargs = {
        "search_memory": {
            "create_kwargs": {"use_sample_memory": True},
            "calc_reward_kwargs": None,
            "release_kwargs": None,
        },
        "ocr": {
            "create_kwargs": {"use_sample_memory": True},
            "calc_reward_kwargs": None,
            "release_kwargs": None,
        },
        "detect_objects": {
            "create_kwargs": {"use_sample_memory": True},
            "calc_reward_kwargs": None,
            "release_kwargs": None,
        },
    }
    
    processed_sample = {
        "sample_id": sample_id,
        "video_path": video_file,
        "prompt": messages,
        "long_term_memory": long_term_memory,
        "short_term_memory": short_term_memory,
        "ground_truth": ground_truth,
        "question_type": question_type,
        "problem_type": problem_type,  # 保留QA类型信息
        "question": question,
        "data_source": "online_video_agent",
        "agent_name": "tool_agent",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "index": idx,
            "answer": ground_truth,
            "question": question,
            "tools_kwargs": tools_kwargs,
            "need_tools_kwargs": True,
        },
    }
    
    return processed_sample


def _save_batch(samples: List[Dict], output_dir: str, filename: str, batch_idx: int):
    """保存一批样本到临时文件"""
    if not samples:
        return
    
    df = pd.DataFrame(samples)
    temp_path = os.path.join(output_dir, f"{filename}.batch_{batch_idx}.parquet")
    df.to_parquet(temp_path, index=False)
    print(f"[Main] Saved batch {batch_idx}: {len(samples)} samples to {temp_path}")


def _load_existing_batches(output_dir: str, filename: str) -> Tuple[List[int], int]:
    """加载已存在的批次文件，返回已处理的样本索引集合和下一个批次编号"""
    pattern = os.path.join(output_dir, f"{filename}.batch_*.parquet")
    batch_files = glob.glob(pattern)
    
    if not batch_files:
        return set(), 0
    
    processed_indices = set()
    max_batch_idx = -1
    
    for batch_file in batch_files:
        try:
            # 从文件名提取批次编号
            batch_idx = int(batch_file.split('.batch_')[1].split('.parquet')[0])
            max_batch_idx = max(max_batch_idx, batch_idx)
            
            # 读取批次文件，获取已处理的样本索引
            df = pd.read_parquet(batch_file)
            if 'extra_info' in df.columns:
                for idx, row in df.iterrows():
                    extra_info = row['extra_info']
                    if isinstance(extra_info, dict) and 'index' in extra_info:
                        processed_indices.add(extra_info['index'])
        except Exception as e:
            print(f"[Warning] Failed to load batch file {batch_file}: {e}")
    
    next_batch_idx = max_batch_idx + 1
    print(f"[Main] Found {len(batch_files)} existing batch files, {len(processed_indices)} samples already processed")
    print(f"[Main] Next batch index: {next_batch_idx}")
    
    return processed_indices, next_batch_idx


def _merge_batches(output_dir: str, filename: str, num_batches: int = None):
    """合并所有批次文件为最终文件"""
    # 自动查找所有批次文件
    pattern = os.path.join(output_dir, f"{filename}.batch_*.parquet")
    batch_files = sorted(glob.glob(pattern), key=lambda x: int(x.split('.batch_')[1].split('.parquet')[0]))
    
    if not batch_files:
        print("[Main] No batch files found to merge")
        return
    
    # 如果指定了 num_batches，只合并前 num_batches 个
    if num_batches is not None:
        batch_files = [f for f in batch_files if int(f.split('.batch_')[1].split('.parquet')[0]) < num_batches]
    
    if len(batch_files) == 1:
        # 只有一个批次，直接重命名
        batch_path = batch_files[0]
        final_path = os.path.join(output_dir, filename)
        os.rename(batch_path, final_path)
        print(f"[Main] Renamed single batch to {final_path}")
        return
    
    # 合并多个批次
    print(f"[Main] Merging {len(batch_files)} batches...")
    dfs = []
    for batch_path in sorted(batch_files):
        try:
            df = pd.read_parquet(batch_path)
            dfs.append(df)
        except Exception as e:
            print(f"[Warning] Failed to read batch {batch_path}: {e}")
    
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        final_path = os.path.join(output_dir, filename)
        final_df.to_parquet(final_path, index=False)
        print(f"[Main] Merged {len(batch_files)} batches into {final_path} ({len(final_df)} total samples)")
        
        # 删除临时批次文件
        for batch_path in batch_files:
            try:
                os.remove(batch_path)
            except Exception as e:
                print(f"[Warning] Failed to remove batch file {batch_path}: {e}")


def process_dataset(
    input_json: str,
    video_dir: str,
    output_dir: str,
    api_base: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    model_name: str = "Qwen3-VL-4B-Instruct",
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    sample_fps: float = 1.0,
    max_samples: int = -1,
    num_workers: int = 1,
    change_threshold: float = 0.2,
    output_filename: str = "data.parquet",
    batch_size: int = 50
):
    """处理整个数据集（支持批量保存和断点续传）"""
    
    # 读取数据集
    print(f"[Main] Loading dataset from: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    if max_samples > 0:
        dataset = dataset[:max_samples]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查已存在的批次文件（断点续传）
    processed_indices, next_batch_idx = _load_existing_batches(output_dir, output_filename)
    
    # 检查是否所有样本都已处理
    if processed_indices and len(processed_indices) >= len(dataset):
        print("[Main] All samples already processed. Merging batches...")
        _merge_batches(output_dir, output_filename)
        return
    
    # 准备未处理的样本列表（保持原始索引）
    args_list = []
    remaining_count = 0
    for idx, sample in enumerate(dataset):
        if idx not in processed_indices:
            args_list.append((idx, sample, video_dir, None))  # preprocessor 稍后设置
            remaining_count += 1
    
    if remaining_count == 0:
        print("[Main] All samples already processed. Merging batches...")
        _merge_batches(output_dir, output_filename)
        return
    
    if processed_indices:
        print(f"[Main] Resuming: {len(processed_indices)} samples already processed, {remaining_count} samples remaining")
    else:
        print(f"[Main] Starting fresh processing of {len(dataset)} samples")
    
    print(f"[Main] Processing {remaining_count} samples with {num_workers} workers")
    print(f"[Main] Batch size: {batch_size}")
    print(f"[Main] Using API: {api_base}")
    
    # 初始化共享的预处理器（只加载一次 embedding 模型）
    print(f"[Main] Initializing shared preprocessor...")
    print(f"[Main] Change threshold: {change_threshold}")
    preprocessor = VideoMemoryPreprocessor(
        api_base=api_base,
        api_key=api_key,
        model_name=model_name,
        embedding_model_name=embedding_model,
        sample_fps=sample_fps,
        frames_dir=None,
        change_threshold=change_threshold,
    )
    # 提前初始化模型，避免多线程同时初始化
    preprocessor._init_models()
    
    # 设置 preprocessor 到参数列表
    args_list = [(idx, sample, video_dir, preprocessor) for idx, sample, video_dir, _ in args_list]
    
    # 多线程处理（共享 embedding 模型）
    processed_samples = []
    batch_count = next_batch_idx
    
    if num_workers > 1:
        print(f"[Main] Using {num_workers} threads (shared embedding model)")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_single_sample, args) for args in args_list]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
                result = future.result()
                if result is not None:
                    processed_samples.append(result)
                    
                    # 批量保存
                    if len(processed_samples) >= batch_size:
                        _save_batch(processed_samples, output_dir, output_filename, batch_count)
                        processed_samples = []  # 清空列表释放内存
                        batch_count += 1
                        gc.collect()  # 强制垃圾回收
    else:
        # 单线程模式
        print(f"[Main] Using single thread mode")
        for args in tqdm(args_list, desc="Processing videos"):
            result = process_single_sample(args)
            if result is not None:
                processed_samples.append(result)
                
                # 批量保存
                if len(processed_samples) >= batch_size:
                    _save_batch(processed_samples, output_dir, output_filename, batch_count)
                    processed_samples = []
                    batch_count += 1
                    gc.collect()  # 强制垃圾回收
    
    # 保存剩余的样本
    if processed_samples:
        _save_batch(processed_samples, output_dir, output_filename, batch_count)
        batch_count += 1
    
    # 合并所有批次（包括之前已存在的批次）
    print(f"[Main] Merging all batches...")
    _merge_batches(output_dir, output_filename)
    
    print(f"[Main] Processing completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess videos for online video agent training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
    # 1. 首先启动 vLLM API 服务器
    bash start_vllm_server.sh 0 8000
    
    # 2. 然后运行预处理脚本
    python preprocess_video_memory.py \\
        --input_json /path/to/dataset.json \\
        --video_dir /path/to/videos \\
        --output_dir /path/to/output \\
        --api_base http://localhost:8000/v1 \\
        --sample_fps 1.0
        """
    )
    parser.add_argument("--input_json", type=str, required=True, help="Path to input dataset JSON")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for parquet files")
    
    # API 配置
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1",
                        help="vLLM API base URL (default: http://localhost:8000/v1)")
    parser.add_argument("--api_key", type=str, default="EMPTY",
                        help="API key (default: EMPTY)")
    parser.add_argument("--model_name", type=str, default="Qwen3-VL-4B-Instruct",
                        help="Model name for API calls")
    
    # 其他配置
    parser.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                        help="Embedding model name or path")
    parser.add_argument("--sample_fps", type=float, default=1.0,
                        help="Video sampling FPS (default: 1.0)")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max samples to process (-1 for all)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--change_threshold", type=float, default=0.2,
                        help="Change threshold for event boundary detection (default: 0.2)")
    parser.add_argument("--output_filename", type=str, default="data.parquet",
                        help="Output parquet filename (default: data.parquet)")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Batch size for saving intermediate results (default: 50)")
    
    args = parser.parse_args()
    
    process_dataset(
        input_json=args.input_json,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        api_base=args.api_base,
        api_key=args.api_key,
        model_name=args.model_name,
        embedding_model=args.embedding_model,
        sample_fps=args.sample_fps,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        change_threshold=args.change_threshold,
        output_filename=args.output_filename,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
