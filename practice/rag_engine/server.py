"""Day 24：把 RAGPipeline 包成 HTTP 服务。

启动（在 practice/ 目录下，PowerShell）：
    $env:DEEPSEEK_API_KEY="..."
    python -m uvicorn rag_engine.server:app --port 8000

调用：
    curl -s http://127.0.0.1:8000/healthz
    curl -s -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
         -d "{\"question\":\"哈希表里元素数量与桶数量之比叫什么\",\"k\":3}"
    curl -N -X POST http://127.0.0.1:8000/ask/stream -H "Content-Type: application/json" \
         -d "{\"question\":\"哈希表里元素数量与桶数量之比叫什么\",\"k\":3}"
"""
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datetime import datetime

from fastapi import FastAPI,HTTPException,Query,Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .errors import PermanentError, RAGError, RetryableError
from .pipeline import RAGPipeline
from . import db


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PDF = BASE_DIR / "data" / "hello-algo.pdf"
CACHE = BASE_DIR / "cache" / "embed"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载一次，之后所有请求共用。

    索引加载要几秒（模型 + 1854 个 chunk 的 BM25）。写在端点里 = 每个请求几秒。
    """
    db.init_pool()
    db.init_schema()

    logger.info("加载索引…")
    app.state.pipe = RAGPipeline(str(PDF), CACHE)
    logger.info("就绪：%d chunks", len(app.state.pipe.chunks))
    yield
    db.close_pool()
    logger.info("服务退出")


app = FastAPI(title="RAG 问答服务", lifespan=lifespan)
#lifespan=lifespan 这个参数就是"注册"：告诉 FastAPI，启动和关闭的时候调用它。不写这一句，lifespan 就是一个没人调用的普通函数，索引永远不会加载



class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    k: int = Field(default=3, ge=1, le=10)
    session_id:int |None = Field(default=None,ge=1)

#ge:>= ; le:<=


class AskResponse(BaseModel):
    answer: str
    sources: list[int]
    session_id:int
    message_id:int
    latency_ms:int


class MessageOut(BaseModel):
    id:int
    role:str
    content:str
    sources:list[int] | None
    latency_ms:int |None
    created_at:datetime


@app.get("/healthz")
def healthz():
    # 能返回 chunk 数，说明 lifespan 跑完了、索引真的在
    return {"status": "ok", "chunks": len(app.state.pipe.chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # def 而不是 async def：管道是同步阻塞的（模型 + API 调用）。
    # FastAPI 会把同步端点丢进线程池，不阻塞事件循环；写成 async def 反而会卡住整个服务。
    """/ask 端点：FastAPI 解析请求 → 调 pipe.ask 生成回答并计时 
    → 调 save_turn 存库（会话不存在则 404）
    → 把回答内容和解包结果 + 会话 id、回答消息 id、耗时一起返回，
      由 FastAPI 按 AskResponse 校验并序列化"""
    started = time.perf_counter()
    result = app.state.pipe.ask(req.question,req.k)
    latency_ms = int((time.perf_counter() - started) * 1000)

    try:
        saved = db.save_turn(req.session_id,req.question,
                             result["answer"],result["sources"],latency_ms,
                             )
        #answer和sources是在返回的result里面去取，不是AskRequest！！！！
    except db.SessionNotFound:
        raise HTTPException(status_code=404, detail=f"会话 {req.session_id} 不存在")
    
    return {
        **result,
        #**代表了字典解包：把 result 里的所有键值对"摊平"放进新字典，就像下面的三个一样
        "session_id" : saved["session_id"],
        "message_id" : saved["answer_message_id"],
        "latency_ms" : latency_ms
    }


def sse(payload: dict) -> str:
    """一条 SSE 消息 = 'data: ' + 内容 + 两个换行。格式错了浏览器就收不到。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    def events():
        try:
            for event, value in app.state.pipe.ask_stream(req.question, req.k):
                yield sse({"type": event, "value": value})
        except RAGError as exc:
            # 流已经开始了，响应头（200）早就发出去了，改不了状态码，
            # 只能作为一条普通事件告诉客户端"后面没有了"。
            yield sse({"type": "error", "value": str(exc)})
        except Exception:
            # 兜底：没归类的异常（IndexError 之类）也要收尾，
            # 否则客户端只看到连接断了，分不清「说完了」和「炸了」。
            logger.exception("流式生成失败（未归类异常）")
            yield sse({"type": "error", "value": "服务内部错误"})
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
    #StreamingResponse(内容源, media_type=类型)


@app.exception_handler(RetryableError)
def on_retryable(request: Request, exc: RetryableError):
    # 能走到这里，说明 retry_with_backoff 已经试满了 —— 上游现在确实不可用
    return JSONResponse(status_code=503, content={"error": str(exc)})


@app.exception_handler(PermanentError)
def on_permanent(request: Request, exc: PermanentError):
    # 有 status_code：上游（DeepSeek）明确拒绝 → 502 网关错误
    # 没有：本地配置/文件问题（缺 key、PDF 不存在）→ 500 服务端错误
    code = 502 if exc.status_code else 500
    return JSONResponse(status_code=code, content={"error": str(exc)})

@app.get("/history/{session_id}",response_model=list[MessageOut])
def history(session_id:int,limit:int = Query(default=50,ge=1,le=200)):
    # 会话不存在返回空列表 []，不报 404：
    # 「没有消息」和「没有这个会话」对调用方来说都是"没东西"，多一个分支没收益。
    return db.get_history(session_id,limit)