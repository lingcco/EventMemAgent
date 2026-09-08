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
视频记忆搜索工具
用于在线视频理解Agent的训练，支持从预处理的离线memory中检索相关事件
"""

import logging
import os
from typing import Any, Optional
from uuid import uuid4
import base64
from io import BytesIO

import torch
from PIL import Image

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MemorySearchTool(BaseTool):
    """
    视频记忆搜索工具
    
    在训练时，从预处理的长期记忆中检索与query相关的事件。
    每个事件包含：事件ID、时间范围、首帧图像、caption和embedding。
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        
        self.top_k = config.get("top_k", 3)
        self.similarity_threshold = config.get("similarity_threshold", 0.3)
        self.use_image_in_response = config.get("use_image_in_response", True)
        
        self.embedding_model = None
        self.embedding_model_name = config.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B")
        
        logger.info(f"Initialized MemorySearchTool with config: {config}")

    def _ensure_embedding_model(self):
        """确保embedding模型已加载（懒加载）"""
        if self.embedding_model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                model_kwargs={
                    "device_map": "auto",
                    "torch_dtype": torch.float16
                },
                tokenizer_kwargs={"padding_side": "left"}
            )
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            self.embedding_model = None

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, 
        instance_id: Optional[str] = None,
        long_term_memory: Optional[list] = None,
        **kwargs
    ) -> tuple[str, ToolResponse]:
        """创建工具实例
        
        Args:
            instance_id: 实例ID
            long_term_memory: 长期记忆列表（事件列表）
            **kwargs: 额外参数，可能包含 create_kwargs（用于兼容不同调用方式）
        """
        if instance_id is None:
            instance_id = str(uuid4())
        
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            if create_kwargs.get("use_sample_memory", False):
                long_term_memory = create_kwargs.get("long_term_memory", long_term_memory)
            else:
                long_term_memory = kwargs.get("long_term_memory", long_term_memory)
        
        self._instance_dict[instance_id] = {
            "long_term_memory": long_term_memory or [],
            "search_history": [],
            "search_count": 0,
            "reward": 0.0,
        }
        
        return instance_id, ToolResponse()

    def _compute_similarity(self, query_emb: torch.Tensor, event_emb: list) -> float:
        """计算查询和事件embedding的余弦相似度"""
        try:
            # 统一转换为 float32，避免 Half 和 float 类型不匹配
            # Archived embeddings are CPU lists; the encoder may return CUDA tensors.
            # Compare on CPU instead of silently returning zero for device mismatch.
            event_tensor = torch.as_tensor(event_emb, dtype=torch.float32, device="cpu")
            query_emb = query_emb.detach().to(device="cpu", dtype=torch.float32)
            
            if len(query_emb.shape) == 1:
                query_emb = query_emb.unsqueeze(0)
            if len(event_tensor.shape) == 1:
                event_tensor = event_tensor.unsqueeze(0)
            
            query_norm = torch.nn.functional.normalize(query_emb, dim=-1)
            event_norm = torch.nn.functional.normalize(event_tensor, dim=-1)
            
            similarity = torch.mm(query_norm, event_norm.t()).item()
            return similarity
        except Exception as e:
            logger.warning(f"Failed to compute similarity: {e}")
            return 0.0

    def _search_memory_by_query(self, instance_id: str, query: str) -> list:
        """通过文本query在长期记忆中搜索相关事件"""
        instance = self._instance_dict[instance_id]
        long_term_memory = instance["long_term_memory"]
        
        if not long_term_memory:
            return []
        
        self._ensure_embedding_model()
        
        query_emb = None
        if self.embedding_model is not None:
            try:
                query_emb = self.embedding_model.encode(
                    query,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    prompt_name="query"
                )
            except Exception as e:
                logger.warning(f"Failed to encode query: {e}")
        
        scored_events = []
        for event in long_term_memory:
            embedding = event.get("embedding")
            if query_emb is not None and embedding is not None and len(embedding) > 0:
                similarity = self._compute_similarity(query_emb, event["embedding"])
            else:
                caption = event.get("caption", "").lower()
                query_words = query.lower().split()
                match_count = sum(1 for w in query_words if w in caption)
                similarity = match_count / max(len(query_words), 1)
            
            if similarity >= self.similarity_threshold:
                scored_events.append((event, similarity))
        
        scored_events.sort(key=lambda x: x[1], reverse=True)
        return [e[0] for e in scored_events[:self.top_k]]

    def _search_memory_by_time(self, instance_id: str, start_time: float, end_time: float) -> list:
        """通过时间范围在长期记忆中搜索事件，返回交集最大的事件"""
        instance = self._instance_dict[instance_id]
        long_term_memory = instance["long_term_memory"]
        
        if not long_term_memory:
            return []
        
        # 计算每个事件与查询时间范围的交集
        scored_events = []
        for event in long_term_memory:
            event_start = event.get("start_time", 0)
            event_end = event.get("end_time", 0)
            
            # 计算交集
            overlap_start = max(start_time, event_start)
            overlap_end = min(end_time, event_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > 0:
                scored_events.append((event, overlap))
        
        # 按交集大小排序
        scored_events.sort(key=lambda x: x[1], reverse=True)
        return [e[0] for e in scored_events[:self.top_k]]

    def _search_memory(self, instance_id: str, query: str = None, start_time: float = None, end_time: float = None) -> list:
        """在长期记忆中搜索相关事件（支持文本query或时间范围）"""
        if start_time is not None and end_time is not None:
            return self._search_memory_by_time(instance_id, start_time, end_time)
        elif query:
            return self._search_memory_by_query(instance_id, query)
        else:
            return []

    def _format_search_results(self, events: list, query: str) -> ToolResponse:
        """格式化搜索结果，使用混合内容格式：每个事件的图片和文本交替出现"""
        if not events:
            return ToolResponse(text=f"No events found for query: '{query}'")
        
        # 使用混合内容格式
        content = [{"type": "text", "text": f"Found {len(events)} relevant event(s) for '{query}':\n"}]
        
        for event in events:
            event_id = event.get("event_id", "?")
            start_time = event.get("start_time", 0)
            end_time = event.get("end_time", 0)
            caption = event.get("caption", "No caption")
            frame_count = event.get("frame_count", 0)
            change_from_previous = event.get("change_from_previous")
            change_to_next = event.get("change_to_next")
            
            # 先添加图片
            if self.use_image_in_response and event.get("first_frame_base64"):
                try:
                    img_data = base64.b64decode(event["first_frame_base64"])
                    img = Image.open(BytesIO(img_data))
                    content.append({"type": "image", "image": img})
                except Exception as e:
                    logger.warning(f"Failed to decode image: {e}")
            
            # 再添加事件描述（包含 change_log）
            event_text = (
                f"\nEvent {event_id} [{start_time:.1f}s - {end_time:.1f}s]:\n"
                f"Caption: {caption}\n"
                f"Duration: {end_time - start_time:.1f}s, Frames: {frame_count}"
            )
            
            # 添加 change_log 信息（如果存在）
            if change_from_previous:
                event_text += f"\nChange from previous event: {change_from_previous}"
            if change_to_next:
                event_text += f"\nChange to next event: {change_to_next}"
            
            content.append({"type": "text", "text": event_text})
        
        return ToolResponse(content=content)

    @rollout_trace_op
    async def execute(
        self, 
        instance_id: str, 
        parameters: dict[str, Any], 
        **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        query = parameters.get("query", "")
        start_time = parameters.get("start_time")
        end_time = parameters.get("end_time")
        
        # 判断搜索模式
        use_time_search = start_time is not None and end_time is not None
        
        if not query and not use_time_search:
            return ToolResponse(text="Error: either query or (start_time and end_time) is required"), 0, {"error": "empty_params"}
        
        instance = self._instance_dict[instance_id]
        instance["search_count"] += 1
        
        # 构建搜索key用于去重
        if use_time_search:
            search_key = f"time:{start_time:.1f}-{end_time:.1f}"
            search_desc = f"time range [{start_time:.1f}s - {end_time:.1f}s]"
        else:
            search_key = f"query:{query}"
            search_desc = f"'{query}'"
        
        if search_key in instance["search_history"]:
            return ToolResponse(
                text=f"You have already searched for {search_desc}. Try a different search."
            ), 0, {"duplicate_search": True}
        
        instance["search_history"].append(search_key)
        
        # 执行搜索
        if use_time_search:
            events = self._search_memory(instance_id, start_time=start_time, end_time=end_time)
        else:
            events = self._search_memory(instance_id, query=query)
        
        response = self._format_search_results(events, search_desc)
        
        tool_reward = 0.0
        
        metrics = {
            "search_type": "time" if use_time_search else "query",
            "query": query if query else f"{start_time}-{end_time}",
            "results_count": len(events),
            "search_count": instance["search_count"],
        }

        return response, tool_reward, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        instance = self._instance_dict.get(instance_id)
        if not instance:
            return 0.0
        
        reward = 0.0
        
        return reward

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
