"""Day 22：pipeline 端到端验收。"""
import logging
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from .pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
PDF = BASE_DIR / "data" / "hello-algo.pdf"
CACHE = BASE_DIR / "cache" / "embed"

if __name__ == "__main__":
    pipe = RAGPipeline(str(PDF), CACHE)

    for q in [
        "哈希冲突时，按规则探测下一个空闲桶的方法叫什么",
        "今天天气怎么样",
    ]:
        print(f"\n问题：{q}")
        print(pipe.ask(q))
