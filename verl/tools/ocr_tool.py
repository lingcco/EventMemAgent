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
OCR 工具
用于对长期记忆中的事件图片或短期记忆中的帧进行 OCR 识别
通过 API 调用 DeepSeek-OCR 服务
"""

import logging
import os
import base64
from typing import Any, Optional
from uuid import uuid4
from io import BytesIO

import aiohttp
from PIL import Image

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class OCRTool(BaseTool):
    """
    OCR 工具
    
    支持对长期记忆中的事件图片或短期记忆中的帧进行 OCR 识别。
    通过 API 调用 DeepSeek-OCR 服务。
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        
        # OCR API 配置
        self.api_url = config.get("api_url", "http://localhost:8089/ocr")
        self.base_size = config.get("base_size", 1024)
        self.image_size = config.get("image_size", 640)
        self.crop_mode = config.get("crop_mode", True)
        self.timeout = config.get("timeout", 60)  # API 超时时间（秒）
        
        logger.info(f"Initialized OCRTool with API URL: {self.api_url}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, 
        instance_id: Optional[str] = None,
        long_term_memory: Optional[list] = None,
        short_term_memory: Optional[list] = None,
        **kwargs
    ) -> tuple[str, ToolResponse]:
        """创建工具实例
        
        Args:
            instance_id: 实例ID
            long_term_memory: 长期记忆列表（事件列表）
            short_term_memory: 短期记忆列表（帧列表）
            **kwargs: 额外参数
        """
        if instance_id is None:
            instance_id = str(uuid4())
        
        # 兼容两种调用方式：
        # 1. 新方式：use_sample_memory=True，从 kwargs 中获取记忆（由 agent_loop 传递）
        # 2. 旧方式：直接从 create_kwargs 中获取记忆（向后兼容）
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            if create_kwargs.get("use_sample_memory", False):
                # 新方式：从外部传入的 kwargs 中获取
                long_term_memory = create_kwargs.get("long_term_memory", long_term_memory)
                short_term_memory = create_kwargs.get("short_term_memory", short_term_memory)
            else:
                # 旧方式：从 create_kwargs 中获取
                long_term_memory = kwargs.get("long_term_memory", long_term_memory)
                short_term_memory = kwargs.get("short_term_memory", short_term_memory)
        
        self._instance_dict[instance_id] = {
            "long_term_memory": long_term_memory or [],
            "short_term_memory": short_term_memory or [],
            "ocr_history": [],
            "ocr_count": 0,
        }
        
        return instance_id, ToolResponse()

    def _get_image_from_event(self, event: dict) -> Optional[Image.Image]:
        """从事件中获取图片"""
        # 尝试从 base64 获取
        if event.get("first_frame_base64"):
            try:
                img_data = base64.b64decode(event["first_frame_base64"])
                return Image.open(BytesIO(img_data)).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to decode event image from base64: {e}")
        
        # 尝试从文件路径获取
        if event.get("first_frame_path"):
            path = event["first_frame_path"]
            if path.startswith("file://"):
                path = path[7:]
            try:
                return Image.open(path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load event image from path: {e}")
        
        return None

    def _image_to_base64(self, image: Image.Image) -> str:
        """将 PIL Image 转换为 base64 字符串"""
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    async def _perform_ocr(self, image: Image.Image) -> str:
        """通过 API 执行 OCR"""
        try:
            # 将图片转换为 base64
            image_base64 = self._image_to_base64(image)
            
            # 构建请求
            payload = {
                "image_base64": image_base64,
                "base_size": self.base_size,
                "image_size": self.image_size,
                "crop_mode": self.crop_mode
            }
            
            # 发送请求
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.api_url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OCR API error: {response.status} - {error_text}")
                        return f"OCR error: API returned status {response.status}"
                    
                    result = await response.json()
                    
                    if result.get("success"):
                        return result.get("text", "")
                    else:
                        # breakpoint()
                        error = result.get("error", "Unknown error")
                        logger.error(f"OCR failed: {error}")
                        return f"OCR error: {error}"
        
        except aiohttp.ClientError as e:
            logger.error(f"OCR API connection error: {e}")
            return f"OCR error: Connection failed - {str(e)}"
        except Exception as e:
            # breakpoint()
            logger.error(f"OCR failed: {e}")
            return f"OCR error: {str(e)}"

    @rollout_trace_op
    async def execute(
        self, 
        instance_id: str, 
        parameters: dict[str, Any], 
        **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """执行 OCR
        
        参数支持两种模式：
        1. event_id: 对长期记忆中指定事件的图片进行 OCR
        2. frame_index: 对短期记忆中指定帧进行 OCR
        """
        event_id = parameters.get("event_id")
        frame_index = parameters.get("frame_index")
        
        if event_id is None and frame_index is None:
            return ToolResponse(text="Error: Must specify either event_id or frame_index"), 0, {"error": "missing_parameter"}
        
        instance = self._instance_dict.get(instance_id)
        if not instance:
            return ToolResponse(text="Error: Instance not found"), 0, {"error": "instance_not_found"}
        
        instance["ocr_count"] += 1
        
        image = None
        source_type = None
        source_id = None
        
        # 从长期记忆获取图片
        if event_id is not None:
            long_term_memory = instance["long_term_memory"]
            # 支持 event_id 作为索引或实际 ID
            target_event = None
            for event in long_term_memory:
                # event 可能是 dict 或 tuple
                if isinstance(event, dict):
                    if event.get("event_id") == event_id:
                        target_event = event
                        break
                elif isinstance(event, (list, tuple)):
                    # tuple 格式转换为 dict
                    event = self._normalize_event(event)
                    if event.get("event_id") == event_id:
                        target_event = event
                        break
            
            # 如果按 ID 找不到，尝试按索引
            if target_event is None and isinstance(event_id, int) and 0 <= event_id < len(long_term_memory):
                event = long_term_memory[event_id]
                if isinstance(event, (list, tuple)):
                    event = self._normalize_event(event)
                target_event = event
            
            if target_event:
                image = self._get_image_from_event(target_event)
                source_type = "long_term_memory"
                source_id = event_id
            else:
                return ToolResponse(text=f"Error: Event {event_id} not found in long-term memory"), 0, {"error": "event_not_found"}
        
        # 从短期记忆获取图片
        elif frame_index is not None:
            short_term_memory = instance["short_term_memory"]
            if not isinstance(frame_index, int) or frame_index < 0 or frame_index >= len(short_term_memory):
                return ToolResponse(
                    text=f"Error: Invalid frame_index {frame_index}. Valid range: 0-{len(short_term_memory)-1}"
                ), 0, {"error": "invalid_frame_index"}
            
            image = short_term_memory[frame_index]
            source_type = "short_term_memory"
            source_id = frame_index
        
        if image is None:
            return ToolResponse(text="Error: Failed to load image"), 0, {"error": "image_load_failed"}
        
        # 检查是否重复 OCR
        ocr_key = f"{source_type}:{source_id}"
        if ocr_key in instance["ocr_history"]:
            return ToolResponse(
                text=f"You have already performed OCR on {source_type} {source_id}. The result was already provided."
            ), 0, {"duplicate_ocr": True}
        
        instance["ocr_history"].append(ocr_key)
        
        # 执行 OCR
        ocr_result = await self._perform_ocr(image)

        if not ocr_result or ocr_result.startswith("OCR error"):
            response_text = f"OCR on {source_type} (index {source_id}): No text detected or OCR failed."
            tool_reward = 0
        else:
            response_text = f"OCR on {source_type} (index {source_id}):\n{ocr_result}"
            tool_reward = 0.0
        
        metrics = {
            "source_type": source_type,
            "source_id": source_id,
            "ocr_count": instance["ocr_count"],
            "text_length": len(ocr_result) if ocr_result else 0,
        }

        return ToolResponse(text=response_text), tool_reward, metrics

    def _normalize_event(self, event) -> dict:
        """将 event 从 tuple 转换为 dict"""
        if isinstance(event, dict):
            return event
        if isinstance(event, (list, tuple)):
            # 假设 tuple 格式为 (event_id, start_time, end_time, caption, embedding, first_frame_base64, ...)
            keys = ["event_id", "start_time", "end_time", "caption", "embedding", 
                    "first_frame_base64", "first_frame_path", "frame_count"]
            return {k: v for k, v in zip(keys, event) if v is not None}
        return {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """计算工具使用的奖励"""
        instance = self._instance_dict.get(instance_id)
        if not instance:
            return 0.0
        
        reward = 0.0
        ocr_count = instance["ocr_count"]
        
        # 过多的 OCR 调用惩罚
        
        return reward

    async def release(self, instance_id: str, **kwargs) -> None:
        """释放工具实例"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
