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
OCR API 服务器 (高并发稳健版)
特性：
1. 使用 Queue 队列模式，防止并发导致显存溢出
2. 定期清理显存，防止长期运行碎片化报错
3. 线程安全设计
"""

import argparse
import base64
import logging
import os
import sys
from pathlib import Path
import threading
import queue
import asyncio
import gc  # 垃圾回收
from io import BytesIO
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="OCR API Server", version="1.1.0")

# 全局变量
model = None
tokenizer = None
config = {}

# --- 核心组件：任务队列 ---
# 所有请求都会进入这个队列排队，确保 GPU 同一时间只处理一张图
inference_queue = queue.Queue()

class InferenceTask:
    """内部使用的任务包装类"""
    def __init__(self, image, base_size, image_size, crop_mode, future):
        self.image = image
        self.base_size = base_size
        self.image_size = image_size
        self.crop_mode = crop_mode
        self.future = future  # 用于将结果传回给等待的 HTTP 请求线程

class OCRRequest(BaseModel):
    """API 请求体"""
    image_base64: Optional[str] = None
    image_path: Optional[str] = None
    base_size: int = 1024
    image_size: int = 640
    crop_mode: bool = True

class OCRResponse(BaseModel):
    """API 响应体"""
    success: bool
    text: str = ""
    error: Optional[str] = None

def load_model(model_path: str, dtype: str = "bfloat16", device: str = "cuda"):
    """加载模型"""
    global model, tokenizer, config
    
    logger.info(f"Loading OCR model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    
    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    model = model.eval()
    
    config["model_path"] = model_path
    config["dtype"] = dtype
    config["device"] = device
    
    logger.info("OCR model loaded successfully")

def perform_ocr_internal(image: Image.Image, base_size: int, image_size: int, crop_mode: bool) -> str:
    """
    实际执行推理的函数
    注意：这个函数只会被后台 Worker 线程调用，是串行的
    """
    global model, tokenizer
    
    if model is None:
        raise RuntimeError("Model is not loaded")

    prompt = "<image>\nFree OCR. "
    
    # 这里的参数根据 DeepSeek-OCR 的实际 API 调整
    result = model.infer(
        tokenizer,
        prompt=prompt,
        pil_image=image,
        output_path='',
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        save_results=False,
        test_compress=True,
        eval_mode=True
    )
    print(result)
    return result if result else ""

def worker_loop():
    """
    后台消费者线程：
    持续从队列中取出任务 -> 执行 OCR -> 返回结果 -> 清理显存
    """
    logger.info("🔥 Background Worker Thread Started! Waiting for tasks...")
    
    processed_count = 0
    
    while True:
        # 1. 阻塞等待任务，直到队列中有东西
        task = inference_queue.get()
        
        # 打印队列深度，方便监控是否积压
        q_size = inference_queue.qsize()
        if q_size > 0:
            logger.info(f"🚦 Current Queue Depth: {q_size} requests waiting...")
        
        try:
            # 检查客户端是否已经断开（如果 Future 已经被取消，就没必要跑了）
            if task.future.done():
                logger.warning("Task cancelled by client before execution, skipping.")
                continue

            logger.info(f"Processing Request #{processed_count + 1}...")
            
            # 2. 执行推理 (耗时操作)
            result = perform_ocr_internal(
                task.image, 
                task.base_size, 
                task.image_size, 
                task.crop_mode
            )
            
            # 3. 将结果设置回 Future，唤醒等待的 HTTP 请求
            # 使用 call_soon_threadsafe 确保跨线程安全
            if not task.future.done():
                task.future.get_loop().call_soon_threadsafe(task.future.set_result, result)
            else:
                logger.warning(f"Request #{processed_count + 1} timeout/cancelled during inference.")

        except Exception as e:
            logger.error(f"Inference Logic Error: {e}")
            if not task.future.done():
                task.future.get_loop().call_soon_threadsafe(task.future.set_exception, e)
        
        finally:
            # 标记队列任务完成
            inference_queue.task_done()
            processed_count += 1
            
            # --- 关键修复：显存清理 ---
            # 每处理 5 个请求，强制清理一次显存
            # 这能有效解决“跑着跑着就报错 (CUDA OOM/Fragmentation)”的问题
            if processed_count % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                # logger.info("🧹 Periodic GC & CUDA Cache Cleared")

@app.on_event("startup")
async def startup_event():
    """应用启动时，启动后台 Worker 线程"""
    t = threading.Thread(target=worker_loop, daemon=True)
    t.start()

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "queue_depth": inference_queue.qsize(), # 返回当前排队数量
        "model_loaded": model is not None
    }

@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(request: OCRRequest):
    """
    OCR HTTP 接口
    改为异步非阻塞模式：只负责接单，实际处理由后台线程完成
    """
    try:
        # 1. 解码图片 (CPU 操作，可以在这里做)
        image = None

        if request.image_base64:
            try:
                img_data = base64.b64decode(request.image_base64)
                image = Image.open(BytesIO(img_data)).convert("RGB")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {e}")
        
        elif request.image_path:
            path = request.image_path
            if path.startswith("file://"):
                path = path[7:]
            try:
                image = Image.open(path).convert("RGB")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to load image from path: {e}")
        
        else:
            raise HTTPException(status_code=400, detail="Must provide either image_base64 or image_path")
        
        # 2. 创建一个 Future 对象，相当于“取餐号”
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        # 3. 将任务打包放入队列
        task = InferenceTask(
            image=image,
            base_size=request.base_size,
            image_size=request.image_size,
            crop_mode=request.crop_mode,
            future=future
        )
        
        # put 是瞬时的，不会阻塞
        inference_queue.put(task)
        
        # 4. 异步等待结果
        # 这里会挂起当前请求，直到后台 Worker 处理完
        result_text = await future
        
        return OCRResponse(success=True, text=result_text)
    
    except asyncio.CancelledError:
        logger.warning("Client disconnected waiting for response")
        # 即使客户端断开，future 也会在 worker 中被标记处理，不会死锁
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Endpoint Error: {e}")
        return OCRResponse(success=False, error=str(e))

def main():
    parser = argparse.ArgumentParser(description="OCR API Server")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "checkpoints/tools/DeepSeek-OCR-updated"),
        help="Path to the OCR model"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype"
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
        default=8089,
        help="Port to bind to"
    )
    
    args = parser.parse_args()
    
    # 加载模型
    load_model(args.model_path, args.dtype, args.device)
    
    # 启动服务器
    logger.info(f"Starting OCR API server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
