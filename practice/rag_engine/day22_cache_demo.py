"""Day 22：embedding 缓存冷 / 热耗时对比。"""
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from sentence_transformers import SentenceTransformer

from .cache import EmbeddingCache

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "cache" / "embed"

if __name__ == "__main__":
    model = SentenceTransformer(MODEL_NAME)
    cache = EmbeddingCache(CACHE_DIR, MODEL_NAME)

    texts = ["哈希表", "二叉搜索树", "动态规划"] * 300   # 900 条，只有 3 条唯一

    t = time.time()
    v1 = cache.encode(texts, model)
    print(f"第一次（冷 / 半热）：{time.time() - t:.2f} 秒，shape={v1.shape}")

    t = time.time()
    v2 = cache.encode(texts, model)
    print(f"第二次（全命中）：{time.time() - t:.4f} 秒，shape={v2.shape}")

    print("两次结果一致：", bool((v1 == v2).all()))
