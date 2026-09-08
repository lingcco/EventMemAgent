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
目标检测工具
用于对长期记忆中的事件图片或短期记忆中的帧进行目标检测
通过 API 调用 Grounding DINO 服务
"""

import logging
import os
import base64
from typing import Any, List, Optional
from uuid import uuid4
from io import BytesIO

import aiohttp
from PIL import Image

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DetectionTool(BaseTool):
    """
    目标检测工具
    
    支持对长期记忆中的事件图片或短期记忆中的帧进行目标检测。
    通过 API 调用 Grounding DINO 服务进行零样本目标检测。
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        
        # 目标检测 API 配置
        self.api_url = config.get("api_url", "http://localhost:8090/detect")
        self.box_threshold = config.get("box_threshold", 0.4)
        self.text_threshold = config.get("text_threshold", 0.3)
        self.timeout = config.get("timeout", 60)  # API 超时时间（秒）
        
        logger.info(f"Initialized DetectionTool with API URL: {self.api_url}")

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
            "detection_history": [],
            "detection_count": 0,
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

    async def _perform_detection(self, image: Image.Image, labels: List[str]) -> dict:
        """通过 API 执行目标检测
        
        返回: {"success": bool, "detections": [...], "error": str}
        """
        try:
            # 将图片转换为 base64
            image_base64 = self._image_to_base64(image)
            
            # 构建请求
            payload = {
                "image_base64": image_base64,
                "labels": labels,
                "box_threshold": self.box_threshold,
                "text_threshold": self.text_threshold
            }
            
            # 发送请求
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.api_url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Detection API error: {response.status} - {error_text}")
                        return {"success": False, "error": f"API returned status {response.status}", "detections": []}
                    
                    result = await response.json()
                    return result
        
        except aiohttp.ClientError as e:
            logger.error(f"Detection API connection error: {e}")
            return {"success": False, "error": f"Connection failed - {str(e)}", "detections": []}
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return {"success": False, "error": str(e), "detections": []}

    def _format_detection_results(self, detections: List[dict]) -> str:
        """格式化检测结果为文本"""
        if not detections:
            return "No objects detected."
        
        lines = []
        for i, det in enumerate(detections):
            label = det.get("label", "unknown")
            confidence = det.get("confidence", 0)
            box = det.get("box", [0, 0, 0, 0])
            lines.append(f"  {i+1}. {label}: confidence={confidence:.3f}, box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
        
        return "Detected objects:\n" + "\n".join(lines)

    @rollout_trace_op
    async def execute(
        self, 
        instance_id: str, 
        parameters: dict[str, Any], 
        **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """执行目标检测
        
        参数:
        - labels: 要检测的目标标签列表（如 ["dog", "cat", "person"]）
        - event_id: (可选) 对长期记忆中指定事件的图片进行检测
        - frame_index: (可选) 对短期记忆中指定帧进行检测
        
        如果不指定 event_id 和 frame_index，默认检测短期记忆的最后一帧
        """
        labels = parameters.get("labels", [])
        event_id = parameters.get("event_id")
        frame_index = parameters.get("frame_index")
        
        if not labels:
            return ToolResponse(text="Error: Must specify labels to detect (e.g., ['dog', 'cat'])"), 0, {"error": "missing_labels"}
        
        # 确保 labels 是列表
        if isinstance(labels, str):
            labels = [labels]
        
        instance = self._instance_dict.get(instance_id)
        if not instance:
            return ToolResponse(text="Error: Instance not found"), 0, {"error": "instance_not_found"}
        
        instance["detection_count"] += 1
        
        image = None
        source_type = None
        source_id = None
        
        # 从长期记忆获取图片
        if event_id is not None:
            long_term_memory = instance["long_term_memory"]
            target_event = None
            for event in long_term_memory:
                if isinstance(event, dict):
                    if event.get("event_id") == event_id:
                        target_event = event
                        break
                elif isinstance(event, (list, tuple)):
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
        
        # 默认检测最后一帧
        else:
            short_term_memory = instance["short_term_memory"]
            if short_term_memory:
                frame_index = len(short_term_memory) - 1
                image = short_term_memory[frame_index]
                source_type = "short_term_memory"
                source_id = frame_index
            else:
                return ToolResponse(text="Error: No frames available in short-term memory"), 0, {"error": "no_frames"}
        
        if image is None:
            return ToolResponse(text="Error: Failed to load image"), 0, {"error": "image_load_failed"}
        
        # 执行检测
        result = await self._perform_detection(image, labels)
        
        if not result.get("success"):
            error = result.get("error", "Unknown error")
            response_text = f"Detection on {source_type} (index {source_id}) failed: {error}"
            tool_reward = 0
        else:
            detections = result.get("detections", [])
            formatted_results = self._format_detection_results(detections)
            response_text = f"Detection on {source_type} (index {source_id}) for labels {labels}:\n{formatted_results}"
            tool_reward = 0.0 
        
        metrics = {
            "source_type": source_type,
            "source_id": source_id,
            "detection_count": instance["detection_count"],
            "num_detections": len(result.get("detections", [])),
            "labels_queried": labels,
        }

        return ToolResponse(text=response_text), tool_reward, metrics

    def _normalize_event(self, event) -> dict:
        """将 event 从 tuple 转换为 dict"""
        if isinstance(event, dict):
            return event
        if isinstance(event, (list, tuple)):
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
        
        return reward

    async def release(self, instance_id: str, **kwargs) -> None:
        """释放工具实例"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

