from fastapi import FastAPI, Query
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
from sse_starlette.sse import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
from threading import Thread
from fastapi.security import APIKeyHeader
from fastapi import Depends, HTTPException
import asyncio

api_key_header = APIKeyHeader(name="X-API-Key")

auth_keys = ["fill your keys here",
             "support long string"]

# 限流
semaphore = asyncio.Semaphore(5)

# 定义请求参数格式
class Request(BaseModel):
    prompt: str
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

# 加载模型和分词器
model_name = "./weights/DeepSeek-R1-Distill-Qwen-14B"  # 替换为实际路径
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # 使用 BF16 节省显存
    device_map="auto",           # 自动分配 GPU/CPU
    low_cpu_mem_usage=True
).eval()

# 创建 FastAPI 实例
app = FastAPI()

# 允许所有来源的跨域请求（生产环境应限制域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# 保留之前的模型加载和 /generate 接口代码

@app.post("/generate")
async def generate_text(request: Request, api_key: str = Depends(api_key_header)):
    async with semaphore:
        if api_key not in auth_keys:
            raise HTTPException(status_code=403, detail="Invalid API Key")
        
        # 编码输入
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        
        # 生成文本
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码结果
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": response}


@app.get("/", response_class=FileResponse)
async def read_root():
    # 假设 index.html 文件位于当前目录
    file_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(file_path)


@app.get("/stream-generate")
async def stream_generate(
    prompt: str = Query(..., description="输入提示语"),
    max_length: int = Query(512, gt=0, le=2048),
    temperature: float = Query(0.7, ge=0.0, le=2.0),
    api_key: str = Query(..., description="API Key")
):
    if api_key not in auth_keys:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt",
                            truncation=True, max_length=4096).to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, decode_kwargs={"errors": "ignore"})
    
    def event_generator():
        #yield tokenizer.decode(streamer, skip_special_tokens=True)
        thread = Thread(target=model.generate, kwargs={**inputs, "streamer": streamer, "max_new_tokens": max_length, "temperature": temperature, "pad_token_id":tokenizer.eos_token_id})
        thread.start()

        is_thinking = True
        for text in streamer:
            if '<think>' in text:
                is_thinking = True
                continue
            elif '</think>' in text:
                is_thinking = False
                yield {"data": '\n'}
                continue
            elif '<｜end▁of▁sentence｜>' in text:
                yield {"data": text.replace('<｜end▁of▁sentence｜>', ' ')}
                break
            
            if is_thinking:
                text = text.replace('\n', '\n> ')

            yield {"data": text}
        yield {"data": "[DONE]"}
    
    return EventSourceResponse(event_generator())