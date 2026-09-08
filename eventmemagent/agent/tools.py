#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具管理器

管理推理时使用的工具：search_memory, ocr, detect_objects

与训练保持一致：
- short_term_memory 是 List[Image.Image]（PIL 图片列表），与 image_data 相同
- 工具返回 ToolResponse 对象，与训练时格式一致
"""

import json
import base64
import asyncio
import aiohttp
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import torch
import numpy as np
from PIL import Image


@dataclass
class ToolResponse:
    """工具响应（与训练时 verl.tools.schemas.ToolResponse 一致）"""
    text: Optional[str] = None
    image: Optional[Any] = None  # PIL Image or List[PIL Image]
    video: Optional[Any] = None
    content: Optional[List[Dict]] = None  # 混合内容格式


class ToolManager:
    """工具管理器"""
    
    def __init__(
        self,
        tool_config_path: str,
        embedding_model_path: str = None,
        ocr_api_url: str = "http://localhost:8089/ocr",
        detection_api_url: str = "http://localhost:8090/detect",
    ):
        self.tool_config_path = tool_config_path
        self.embedding_model_path = embedding_model_path
        self.ocr_api_url = ocr_api_url
        self.detection_api_url = detection_api_url
        self.top_k = 3
        self.similarity_threshold = 0.3
        
        # 加载工具配置
        self.tool_schemas = self._load_tool_schemas()
        
        # 延迟加载 embedding 模型
        self.embedding_model = None
        
        # 当前实例的记忆数据
        # long_term_memory: List[Dict] - 事件列表（包含 caption, embedding, first_frame_base64 等）
        # short_term_memory: List[Image.Image] - PIL 图片列表（与训练时 image_data 一致）
        # short_term_memory_raw: List[Dict] - 原始字典列表（保留元数据）
        self.long_term_memory = []
        self.short_term_memory = []  # 这是 PIL Image 列表！
        self.short_term_memory_raw = []  # 原始字典列表
        self.search_history = []
    
    def _load_tool_schemas(self) -> List[Dict]:
        """加载工具 schema"""
        from omegaconf import OmegaConf

        config = OmegaConf.to_container(OmegaConf.load(self.tool_config_path), resolve=True)
        
        schemas = []
        for tool in config.get("tools", []):
            schema = tool.get("tool_schema", {})
            schemas.append(schema)
            function_name = schema.get("function", {}).get("name")
            tool_options = tool.get("config", {})
            if function_name == "search_memory":
                self.top_k = int(tool_options.get("top_k", self.top_k))
                self.similarity_threshold = float(
                    tool_options.get("similarity_threshold", self.similarity_threshold)
                )
            elif function_name == "ocr":
                self.ocr_api_url = tool_options.get("api_url", self.ocr_api_url)
            elif function_name == "detect_objects":
                self.detection_api_url = tool_options.get("api_url", self.detection_api_url)
        return schemas
    
    def get_tool_schemas(self) -> List[Dict]:
        """获取工具 schema 列表"""
        return self.tool_schemas
    
    def initialize(
        self, 
        long_term_memory: List[Dict], 
        short_term_memory_raw: List[Dict],
        short_term_memory_images: List[Image.Image] = None
    ):
        """初始化工具实例
        
        Args:
            long_term_memory: 长期记忆事件列表
            short_term_memory_raw: 短期记忆原始数据（字典列表）
            short_term_memory_images: 短期记忆 PIL 图片列表（与训练时 image_data 一致）
        """
        self.long_term_memory = long_term_memory or []
        self.short_term_memory_raw = short_term_memory_raw or []  # 字典列表（保留元数据）
        self.short_term_memory = short_term_memory_images or []  # PIL Image 列表
        self.search_history = []
    
    def _ensure_embedding_model(self):
        """确保 embedding 模型已加载"""
        if self.embedding_model is not None:
            return
        
        if not self.embedding_model_path:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            # 使用 CPU 避免多 GPU 环境下的设备不一致问题
            self.embedding_model = SentenceTransformer(
                self.embedding_model_path,
                device="cpu",
                tokenizer_kwargs={"padding_side": "left"}
            )
        except Exception as e:
            print(f"[Warning] Failed to load embedding model: {e}")
    
    async def execute_tool(self, tool_name: str, arguments: Dict) -> Tuple[str, List[Image.Image]]:
        """执行工具
        
        Returns:
            (response_text, new_images) - 与训练时工具返回格式对应
        """
        if tool_name == "search_memory":
            return await self._search_memory(arguments)
        elif tool_name == "ocr":
            return await self._ocr(arguments)
        elif tool_name == "detect_objects":
            return await self._detect_objects(arguments)
        else:
            return f"Unknown tool: {tool_name}", []
    
    async def _search_memory(self, arguments: Dict) -> Tuple[str, List[Image.Image]]:
        """搜索长期记忆"""
        query = arguments.get("query")
        start_time = arguments.get("start_time")
        end_time = arguments.get("end_time")
        
        if not self.long_term_memory:
            return "No events in long-term memory.", []
        
        events = []
        
        # 时间范围搜索
        if start_time is not None and end_time is not None:
            search_key = f"time:{start_time:.1f}-{end_time:.1f}"
            if search_key in self.search_history:
                return f"You have already searched for time range [{start_time:.1f}s - {end_time:.1f}s].", []
            self.search_history.append(search_key)
            
            for event in self.long_term_memory:
                event_start = event.get("start_time", 0)
                event_end = event.get("end_time", 0)
                overlap_start = max(start_time, event_start)
                overlap_end = min(end_time, event_end)
                overlap = max(0, overlap_end - overlap_start)
                if overlap > 0:
                    events.append((event, overlap))
            events.sort(key=lambda x: x[1], reverse=True)
            events = [e[0] for e in events[: self.top_k]]
            search_desc = f"time range [{start_time:.1f}s - {end_time:.1f}s]"
        
        # 文本搜索
        elif query:
            search_key = f"query:{query}"
            if search_key in self.search_history:
                return f"You have already searched for '{query}'.", []
            self.search_history.append(search_key)
            
            self._ensure_embedding_model()
            
            if self.embedding_model:
                # 在 CPU 上进行 embedding 计算，避免多 GPU 设备不一致
                query_emb = self.embedding_model.encode(
                    query, convert_to_tensor=True, normalize_embeddings=True
                )
                scored = []
                for event in self.long_term_memory:
                    emb = event.get("embedding")
                    # 检查 embedding 是否存在（兼容 numpy 数组）
                    if emb is not None and (hasattr(emb, '__len__') and len(emb) > 0):
                        # 在 CPU 上计算相似度
                        event_emb = torch.tensor(
                            emb, dtype=torch.float32, device="cpu"
                        )
                        sim = torch.nn.functional.cosine_similarity(
                            query_emb.unsqueeze(0), event_emb.unsqueeze(0)
                        ).item()
                        if sim >= self.similarity_threshold:
                            scored.append((event, sim))
                scored.sort(key=lambda x: x[1], reverse=True)
                events = [e[0] for e in scored[: self.top_k]]
            else:
                # 简单关键词匹配
                query_words = query.lower().split()
                for event in self.long_term_memory:
                    caption = event.get("caption", "").lower()
                    if any(w in caption for w in query_words):
                        events.append(event)
                events = events[: self.top_k]
            search_desc = f"'{query}'"
        else:
            return "Error: either query or (start_time and end_time) is required", []
        
        if not events:
            return f"No events found for {search_desc}", []
        
        # 格式化结果
        new_images = []
        result_lines = [f"Found {len(events)} relevant event(s) for {search_desc}:"]
        
        for event in events:
            event_id = event.get("event_id", "?")
            start = event.get("start_time", 0)
            end = event.get("end_time", 0)
            caption = event.get("caption", "No caption")
            change_from = event.get("change_from_previous")
            change_to = event.get("change_to_next")
            
            # 解码图片
            if event.get("first_frame_base64"):
                try:
                    img_data = base64.b64decode(event["first_frame_base64"])
                    img = Image.open(BytesIO(img_data)).convert("RGB")
                    new_images.append(img)
                except:
                    pass
            
            result_lines.append(f"\nEvent {event_id} [{start:.1f}s - {end:.1f}s]:")
            result_lines.append(f"Caption: {caption}")
            if change_from:
                result_lines.append(f"Change from previous: {change_from}")
            if change_to:
                result_lines.append(f"Change to next: {change_to}")
        
        return "\n".join(result_lines), new_images


    def _get_image_from_event(self, event_id: int) -> Optional[Image.Image]:
        """从长期记忆获取图片"""
        for event in self.long_term_memory:
            if event.get("event_id") == event_id:
                if event.get("first_frame_base64"):
                    try:
                        img_data = base64.b64decode(event["first_frame_base64"])
                        return Image.open(BytesIO(img_data)).convert("RGB")
                    except:
                        pass
        # 尝试按索引
        if 0 <= event_id < len(self.long_term_memory):
            event = self.long_term_memory[event_id]
            if event.get("first_frame_base64"):
                try:
                    img_data = base64.b64decode(event["first_frame_base64"])
                    return Image.open(BytesIO(img_data)).convert("RGB")
                except:
                    pass
        return None
    
    def _get_image_from_frame(self, frame_index: int) -> Optional[Image.Image]:
        """从短期记忆获取图片
        
        short_term_memory 是 PIL Image 列表（与训练时一致）
        """
        if 0 <= frame_index < len(self.short_term_memory):
            img = self.short_term_memory[frame_index]
            if isinstance(img, Image.Image):
                return img
        return None
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """将图片转换为 base64"""
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    async def _ocr(self, arguments: Dict) -> Tuple[str, List[Image.Image]]:
        """执行 OCR"""
        event_id = arguments.get("event_id")
        frame_index = arguments.get("frame_index")
        
        if event_id is None and frame_index is None:
            return "Error: Must specify either event_id or frame_index", []
        
        # 获取图片
        if event_id is not None:
            image = self._get_image_from_event(event_id)
            source = f"long_term_memory (event {event_id})"
        else:
            image = self._get_image_from_frame(frame_index)
            source = f"short_term_memory (frame {frame_index})"
        
        if image is None:
            return f"Error: Failed to load image from {source}", []
        
        # 调用 OCR API
        try:
            payload = {
                "image_base64": self._image_to_base64(image),
                "base_size": 1024,
                "image_size": 640,
                "crop_mode": True
            }
            
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.ocr_api_url, json=payload) as response:
                    if response.status != 200:
                        return f"OCR API error: status {response.status}", []
                    result = await response.json()
                    if result.get("success"):
                        text = result.get("text", "")
                        if text:
                            return f"OCR on {source}:\n{text}", []
                        else:
                            return f"OCR on {source}: No text detected", []
                    else:
                        return f"OCR error: {result.get('error', 'Unknown')}", []
        except Exception as e:
            return f"OCR error: {str(e)}", []
    
    async def _detect_objects(self, arguments: Dict) -> Tuple[str, List[Image.Image]]:
        """执行目标检测"""
        labels = arguments.get("labels", [])
        event_id = arguments.get("event_id")
        frame_index = arguments.get("frame_index")
        
        if not labels:
            return "Error: Must specify labels to detect", []
        
        if isinstance(labels, str):
            labels = [labels]
        
        # 获取图片
        if event_id is not None:
            image = self._get_image_from_event(event_id)
            source = f"long_term_memory (event {event_id})"
        elif frame_index is not None:
            image = self._get_image_from_frame(frame_index)
            source = f"short_term_memory (frame {frame_index})"
        else:
            # 默认最后一帧
            if self.short_term_memory:
                frame_index = len(self.short_term_memory) - 1
                image = self._get_image_from_frame(frame_index)
                source = f"short_term_memory (frame {frame_index})"
            else:
                return "Error: No frames available", []
        
        if image is None:
            return f"Error: Failed to load image from {source}", []
        
        # 调用检测 API
        try:
            payload = {
                "image_base64": self._image_to_base64(image),
                "labels": labels,
                "box_threshold": 0.4,
                "text_threshold": 0.3
            }
            
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.detection_api_url, json=payload) as response:
                    if response.status != 200:
                        return f"Detection API error: status {response.status}", []
                    result = await response.json()
                    if result.get("success"):
                        detections = result.get("detections", [])
                        if detections:
                            lines = [f"Detection on {source} for labels {labels}:"]
                            for i, det in enumerate(detections):
                                label = det.get("label", "unknown")
                                conf = det.get("confidence", 0)
                                box = det.get("box", [0, 0, 0, 0])
                                lines.append(f"  {i+1}. {label}: confidence={conf:.3f}, box={box}")
                            return "\n".join(lines), []
                        else:
                            return f"Detection on {source}: No objects detected for {labels}", []
                    else:
                        return f"Detection error: {result.get('error', 'Unknown')}", []
        except Exception as e:
            return f"Detection error: {str(e)}", []
