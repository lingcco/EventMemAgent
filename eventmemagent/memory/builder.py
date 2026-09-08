#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在线记忆构建器

从视频实时构建长期记忆和短期记忆，复用预处理代码的逻辑
支持两种模式：
1. 使用 vLLM Python API（与推理共享模型）
2. 使用 HTTP API（独立服务）
"""

import base64
import gc
import random
from io import BytesIO
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def encode_pil_image(pil_image: Image.Image, format: str = "JPEG") -> str:
    """将 PIL Image 编码为 base64"""
    buffered = BytesIO()
    pil_image.save(buffered, format=format, quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _qwen_vl_dedup_image_tokens(prompt_ids: List[int], processor) -> List[int]:
    """
    对 Qwen2.5-VL / Qwen3-VL 的 image tokens 进行去重
    
    vLLM 会根据 image_data 自动扩展 <|image_pad|> token，
    所以需要先去重，只保留一个 <|image_pad|>
    """
    if processor is None:
        return prompt_ids
    
    image_processor_name = getattr(processor, 'image_processor', None)
    if image_processor_name:
        image_processor_name = image_processor_name.__class__.__name__
    else:
        return prompt_ids
    
    is_qwen_vl = "Qwen2VL" in image_processor_name or "Qwen3VL" in image_processor_name
    
    if is_qwen_vl and hasattr(processor, 'image_token_id'):
        prompt_ids_arr = np.array(prompt_ids)
        mask = np.ones(len(prompt_ids_arr), dtype=bool)
        is_value = prompt_ids_arr == processor.image_token_id
        mask[1:] &= ~(is_value[1:] & is_value[:-1])
        return prompt_ids_arr[mask].tolist()
    
    return prompt_ids


class OnlineMemoryBuilder:
    """在线记忆构建器"""
    
    def __init__(
        self,
        # vLLM Python API 模式（与推理共享模型）
        llm: Any = None,
        processor: Any = None,
        sampling_params_class: Any = None,
        # HTTP API 模式（独立服务）
        api_base: str = None,
        api_key: str = "EMPTY",
        model_name: str = "Qwen3-VL-4B-Instruct",
        # 通用参数
        embedding_model_name: str = None,
        sample_fps: float = 1.0,
        short_term_max_frames: int = 32,
        short_term_min_frames: int = 8,
        caption_sample_frames: int = 32,
        change_threshold: float = 0.2,
        max_frame_resolution: Tuple[int, int] = (420, 360),
        max_retries: int = 3,
        timeout: int = 120,
    ):
        # vLLM Python API 模式
        self.llm = llm
        self.processor = processor
        self.sampling_params_class = sampling_params_class
        
        # HTTP API 模式
        self.api_base = api_base.rstrip('/') if api_base else None
        self.api_key = api_key
        self.model_name = model_name
        
        # 通用参数
        self.embedding_model_name = embedding_model_name
        self.sample_fps = sample_fps
        self.short_term_max_frames = short_term_max_frames
        self.short_term_min_frames = short_term_min_frames
        self.caption_sample_frames = caption_sample_frames
        self.change_threshold = change_threshold
        self.max_frame_resolution = max_frame_resolution
        self.max_retries = max_retries
        self.timeout = timeout
        
        self.embedding_model = None
        self._initialized = False
        
        # 确定使用哪种模式
        self.use_python_api = (llm is not None and processor is not None)
    
    def _init_models(self):
        """初始化模型"""
        if self._initialized:
            return
        self._initialized = True
        
        if self.embedding_model_name:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(
                    self.embedding_model_name,
                    model_kwargs={"device_map": "auto", "torch_dtype": torch.float16},
                    tokenizer_kwargs={"padding_side": "left"}
                )
            except Exception as e:
                print(f"[Warning] Failed to load embedding model: {e}")
    
    def _call_vlm_python_api(self, images: List[Image.Image], prompt: str, max_tokens: int = 256) -> Optional[str]:
        """使用 vLLM Python API 生成文本"""
        if not self.llm or not self.processor:
            return None
        
        try:
            from vllm import TokensPrompt
            
            # 构建消息
            content = []
            for _ in images:
                content.append({"type": "image"})
            content.append({"type": "text", "text": prompt})
            
            messages = [{"role": "user", "content": content}]
            
            # 使用 processor 处理
            raw_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            
            model_inputs = self.processor(
                text=[raw_prompt],
                images=images,
                return_tensors="pt"
            )
            prompt_ids = model_inputs["input_ids"].squeeze(0).tolist()
            
            # 对 image tokens 去重（与训练时一致）
            prompt_ids = _qwen_vl_dedup_image_tokens(prompt_ids, self.processor)
            
            # 生成
            sampling_params = self.sampling_params_class(
                temperature=0.7,
                max_tokens=max_tokens,
            )
            
            prompt_obj = TokensPrompt(
                prompt_token_ids=prompt_ids,
                multi_modal_data={"image": images}
            )
            
            outputs = self.llm.generate(
                prompts=prompt_obj,
                sampling_params=sampling_params,
            )
            
            return outputs[0].outputs[0].text.strip()
        except Exception as e:
            print(f"[Warning] vLLM Python API call failed: {e}")
            return None
    
    def _call_vlm_http_api(self, images: List[Image.Image], prompt: str, max_tokens: int = 256) -> Optional[str]:
        """使用 HTTP API 生成文本"""
        import requests
        import time
        
        if not self.api_base:
            return None
        
        url = f"{self.api_base}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        
        content = []
        for img in images:
            img_base64 = encode_pil_image(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            })
        content.append({"type": "text", "text": prompt})
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"[Warning] HTTP API call failed (attempt {attempt + 1}): {e}")
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)
        raise RuntimeError("Caption service failed after retries; refusing to build placeholder memory")
    
    def _call_vlm(self, images: List[Image.Image], prompt: str, max_tokens: int = 256) -> Optional[str]:
        """调用 VLM 生成文本（自动选择模式）"""
        if self.use_python_api:
            return self._call_vlm_python_api(images, prompt, max_tokens)
        else:
            return self._call_vlm_http_api(images, prompt, max_tokens)
    
    def _compute_histogram_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """计算直方图相似度"""
        gray1 = img1.convert("L")
        gray2 = img2.convert("L")
        hist1 = np.array(gray1.histogram(), dtype=np.float32)
        hist2 = np.array(gray2.histogram(), dtype=np.float32)
        hist1 /= hist1.sum() + 1e-8
        hist2 /= hist2.sum() + 1e-8
        correlation = np.corrcoef(hist1, hist2)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def _is_event_boundary(self, avg_frame: Image.Image, new_frame: Image.Image) -> bool:
        """检测事件边界"""
        similarity = self._compute_histogram_similarity(avg_frame, new_frame)
        return similarity < self.change_threshold
    
    def _generate_caption(self, frames: List[Image.Image], start_time: float, end_time: float) -> str:
        """生成事件 caption"""
        if not frames:
            return f"Event from {start_time:.2f}s to {end_time:.2f}s"
        
        sample_count = min(self.caption_sample_frames, len(frames))
        if len(frames) <= sample_count:
            sampled_frames = frames
        else:
            indices = np.linspace(0, len(frames) - 1, sample_count, dtype=int)
            sampled_frames = [frames[i] for i in indices]
        
        prompt = (
            f"You are a video memory builder. Describe what happens in this event sequence.\n"
            f"TimeRange: [{start_time:.2f}s - {end_time:.2f}s].\n"
            "Output EXACTLY ONE concise paragraph (single line) describing the event."
        )
        
        caption = self._call_vlm(sampled_frames, prompt, max_tokens=256)
        if caption:
            return caption.replace("\n", " ")
        return f"Event from {start_time:.2f}s to {end_time:.2f}s"
    
    def _extract_embedding(self, text: str) -> Optional[List[float]]:
        """提取文本 embedding"""
        if not self.embedding_model or not text:
            return None
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False, normalize_embeddings=True)
            return embedding.tolist()
        except:
            return None
    
    def _add_frame_label(self, image: Image.Image, frame_number: int, timestamp: float = None) -> Image.Image:
        """添加帧标签"""
        img_with_label = image.copy()
        draw = ImageDraw.Draw(img_with_label)
        
        if timestamp is not None:
            text = f"Frame {frame_number} | {timestamp:.1f}s"
        else:
            text = f"Frame {frame_number}"
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                font = ImageFont.load_default()
        
        padding = 10
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except:
            text_width, text_height = draw.textsize(text, font=font)
        
        draw.rectangle([padding, padding, padding + text_width + 10, padding + text_height + 10], fill=(0, 0, 0, 180))
        draw.text((padding + 5, padding + 5), text, fill=(255, 255, 255), font=font)
        
        return img_with_label

    def _generate_change_description(
        self, 
        prev_image: Image.Image, 
        curr_image: Image.Image,
        prev_time_range: Tuple[float, float],
        curr_time_range: Tuple[float, float]
    ) -> str:
        """生成变化描述"""
        prompt = (
            f"Describe what changed between these two events in ONE concise sentence.\n"
            f"Image 1: {prev_time_range[0]:.2f}s - {prev_time_range[1]:.2f}s\n"
            f"Image 2: {curr_time_range[0]:.2f}s - {curr_time_range[1]:.2f}s"
        )
        
        change_desc = self._call_vlm([prev_image, curr_image], prompt, max_tokens=128)
        if change_desc:
            return change_desc.replace("\n", " ").strip()
        return f"Transition from {prev_time_range[0]:.1f}s to {curr_time_range[0]:.1f}s"
    
    def _resize_and_pad_image(self, image: Image.Image, scale_ratio: float) -> Image.Image:
        """缩放并填充图片"""
        target_w, target_h = self.max_frame_resolution
        
        if scale_ratio < 1.0:
            new_w = int(image.size[0] * scale_ratio)
            new_h = int(image.size[1] * scale_ratio)
            resized = image.resize((new_w, new_h), Image.LANCZOS)
        else:
            resized = image
            new_w, new_h = image.size
        
        if new_w <= target_w and new_h <= target_h:
            result = Image.new("RGB", (target_w, target_h), (0, 0, 0))
            paste_x = (target_w - new_w) // 2
            paste_y = (target_h - new_h) // 2
            result.paste(resized, (paste_x, paste_y))
            return result
        return resized
    
    def _compute_uniform_scale_ratio(self, frames: List[Dict]) -> float:
        """计算统一缩放比例"""
        if not frames:
            return 1.0
        target_w, target_h = self.max_frame_resolution
        min_scale = 1.0
        for f in frames:
            img = f["image"]
            scale = min(target_w / img.size[0], target_h / img.size[1])
            if scale < min_scale:
                min_scale = scale
        return min_scale
    
    def process_video(self, video_path: str) -> Tuple[List[Dict], List[Dict]]:
        """处理视频，返回长期记忆和短期记忆
        
        新逻辑：短期记忆可以包含多个事件
        - 检测到事件分割点时，如果短期记忆未满，不转移到长期记忆
        - 只有短期记忆满了需要蓄水池采样时，才把第一个事件转移到长期记忆
        - 事件分割点的计算基于当前事件的平均帧（不是整个短期记忆）
        """
        self._init_models()
        
        from decord import VideoReader, cpu
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
        except Exception as e:
            print(f"[Error] Failed to load video: {e}")
            return [], []
        
        total_frames = len(vr)
        video_fps = float(vr.get_avg_fps()) if vr.get_avg_fps() > 0 else 25.0
        
        sample_interval = max(1, int(video_fps / self.sample_fps))
        sampled_indices = list(range(0, total_frames, sample_interval))
        
        long_term_events = []
        next_event_id = 0
        
        # 短期记忆：包含多个事件
        short_term_events = []  # List[Dict]: 每个元素是一个事件
        
        # 当前正在构建的事件
        current_event = {
            "frames": [],
            "accumulated_sum": None,
            "total_frames_seen": 0,
        }
        
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
                print(f"[MemoryBuilder] Moving event {next_event_id} to LTM: {event['start_time']:.1f}s - {event['end_time']:.1f}s")
                
                # 生成变化描述
                if long_term_events:
                    prev_event = long_term_events[-1]
                    try:
                        prev_image = Image.open(BytesIO(base64.b64decode(prev_event["first_frame_base64"])))
                    except:
                        prev_image = None
                    
                    if prev_image:
                        change_desc = self._generate_change_description(
                            prev_image, first_event_data["frames"][0]["image"],
                            (prev_event["start_time"], prev_event["end_time"]),
                            (event["start_time"], event["end_time"])
                        )
                        prev_event["change_to_next"] = change_desc
                        event["change_from_previous"] = change_desc
                
                long_term_events.append(event)
                next_event_id += 1
        
        print(f"[MemoryBuilder] Processing video: {video_path}")
        print(f"[MemoryBuilder] Total frames: {total_frames}, FPS: {video_fps:.2f}, Sampled: {len(sampled_indices)}")
        
        for idx, frame_idx in enumerate(sampled_indices):
            is_last_frame = (idx == len(sampled_indices) - 1)
            
            try:
                frame_array = vr[frame_idx].asnumpy()
            except:
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
                        removed = current_event["frames"].pop(remove_idx)
                        current_event["accumulated_sum"] -= removed["array_float"]
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
        
        # 构建短期记忆
        frames_to_save = all_stm_frames[-self.short_term_max_frames:]
        uniform_scale = self._compute_uniform_scale_ratio(frames_to_save)
        
        short_term_memory = []
        for i, f in enumerate(frames_to_save):
            processed = self._resize_and_pad_image(f["image"], uniform_scale)
            labeled = self._add_frame_label(processed, i, f["timestamp"])
            img_base64 = encode_pil_image(labeled)
            short_term_memory.append({
                "image": f"data:image/jpeg;base64,{img_base64}",
                "frame_index": i,
                "timestamp": f["timestamp"],
            })
        
        print(f"[MemoryBuilder] Done. Long-term events: {len(long_term_events)}, Short-term frames: {len(short_term_memory)}, STM events: {len(short_term_events) + 1}")
        
        del vr
        gc.collect()
        
        return long_term_events, short_term_memory
    
    def _create_event(self, frames: List[Dict], event_id: int) -> Optional[Dict]:
        """创建事件"""
        if not frames:
            return None
        
        start_time = frames[0]["timestamp"]
        end_time = frames[-1]["timestamp"]
        first_frame = frames[0]["image"]
        
        images = [f["image"] for f in frames]
        caption = self._generate_caption(images, start_time, end_time)
        embedding = self._extract_embedding(caption)
        
        return {
            "event_id": event_id,
            "start_time": start_time,
            "end_time": end_time,
            "first_frame_base64": encode_pil_image(first_frame),
            "caption": caption,
            "embedding": embedding,
            "frame_count": len(frames),
            "change_from_previous": None,
            "change_to_next": None,
        }
