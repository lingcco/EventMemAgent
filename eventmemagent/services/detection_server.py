#!/usr/bin/env python3
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
目标检测 API 服务器
使用 Grounding DINO 模型提供零样本目标检测服务
"""

import argparse
import base64
import logging
import os
from pathlib import Path
from io import BytesIO
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Object Detection API Server", version="1.0.0")

# 全局模型变量
model = None
processor = None
config = {}


class DetectionRequest(BaseModel):
    """目标检测请求"""
    # 图片数据（base64 编码）
    image_base64: Optional[str] = None
    # 或者图片文件路径
    image_path: Optional[str] = None
    # 要检测的目标标签列表（如 ["a dog", "a cat"]）
    # 注意：标签应该小写，并以句号结尾效果更好
    labels: List[str]
    # 检测阈值
    box_threshold: float = 0.4
    text_threshold: float = 0.3


class DetectionResult(BaseModel):
    """单个检测结果"""
    label: str
    confidence: float
    box: List[float]  # [x1, y1, x2, y2]


class DetectionResponse(BaseModel):
    """目标检测响应"""
    success: bool
    detections: List[DetectionResult] = []
    error: Optional[str] = None


def load_model(model_path: str, device: str = "cuda"):
    """加载目标检测模型"""
    global model, processor, config
    
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    
    logger.info(f"Loading detection model from: {model_path}")
    
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(device)
    model = model.eval()
    
    config["model_path"] = model_path
    config["device"] = device
    
    logger.info("Detection model loaded successfully")


def perform_detection(
    image: Image.Image, 
    labels: List[str],
    box_threshold: float = 0.4,
    text_threshold: float = 0.3
) -> List[DetectionResult]:
    """执行目标检测"""
    global model, processor, config
    
    if model is None or processor is None:
        raise RuntimeError("Model not loaded")
    
    device = config.get("device", "cuda")
    
    # 确保标签格式正确（小写，以句号结尾）
    processed_labels = []
    for label in labels:
        label = label.lower().strip()
        processed_labels.append(label)
    
    # 处理输入
    # Grounding DINO 需要 text 是 List[List[str]] 格式
    inputs = processor(
        images=image, 
        text=processed_labels,  # 包装成 batch
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 后处理
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]]  # (height, width)
    )
    
    # 解析结果
    detections = []
    if results and len(results) > 0:
        result = results[0]
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        result_labels = result.get("labels", [])
        
        for box, score, label in zip(boxes, scores, result_labels):
            box_list = [round(x, 2) for x in box.tolist()]
            detections.append(DetectionResult(
                label=label,
                confidence=round(score.item(), 3),
                box=box_list
            ))
    print(detections)
    return detections


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "config": config
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect_endpoint(request: DetectionRequest):
    """目标检测接口"""
    try:
        # 获取图片
        image = None
        
        if request.image_base64:
            # 从 base64 解码
            try:
                img_data = base64.b64decode(request.image_base64)
                image = Image.open(BytesIO(img_data)).convert("RGB")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {e}")
        
        elif request.image_path:
            # 从文件路径加载
            path = request.image_path
            if path.startswith("file://"):
                path = path[7:]
            try:
                image = Image.open(path).convert("RGB")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to load image from path: {e}")
        
        else:
            raise HTTPException(status_code=400, detail="Must provide either image_base64 or image_path")
        
        if not request.labels:
            raise HTTPException(status_code=400, detail="Must provide at least one label to detect")
        
        # 执行检测
        detections = perform_detection(
            image,
            labels=request.labels,
            box_threshold=request.box_threshold,
            text_threshold=request.text_threshold
        )
        
        return DetectionResponse(success=True, detections=detections)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return DetectionResponse(success=False, error=str(e))


def main():
    parser = argparse.ArgumentParser(description="Object Detection API Server")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "checkpoints/tools/grounding-dino-base"),
        help="Path to the detection model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to load model on"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8090,
        help="Port to bind to"
    )
    
    args = parser.parse_args()
    
    # 加载模型
    load_model(args.model_path, args.device)
    
    # 启动服务器
    logger.info(f"Starting Detection API server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
