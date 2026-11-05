# Day 17：chunk_size × top_k 检索命中率对比
# 基于 RAG6 改写：换 PDF、换测试集、补 top_k 循环

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = str(BASE_DIR / "data" / "hello-algo.pdf")

# 中文模型：BAAI/bge-small-zh-v1.5
# 换回 "all-MiniLM-L6-v2" 可复现英文模型的结果（同为 chunk_size=200 时 top_k=3 只有 2/11）
MODEL_NAME = "BAAI/bge-small-zh-v1.5"

def load_pdf(path):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append((i + 1, text))
    return pages


def chunk_text(pages, chunk_size, overlap=50):
    chunks = []
    for _, text in pages:
        start = 0
        while start < len(text):
            chunks.append(text[start:start + chunk_size])
            start += chunk_size - overlap
    return chunks


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def normalize(s):
    return "".join(s.split())  # 去掉所有空格/换行，PDF 提取的换行不是语义


def top_k_index(query_emb, chunk_emb, k=3):
    scores = [cosine(query_emb, c) for c in chunk_emb]
    return np.argsort(scores)[::-1][:k]


TEST_SET = [
    ("哈希冲突时，把冲突元素串在同一个桶的链表里，这种方法叫什么", "链式地址"),
    ("哈希冲突时，按规则探测下一个空闲桶的方法叫什么", "开放寻址"),
    ("哈希表中元素数量与桶数量之比叫什么", "负载因子"),
    ("用数组实现队列时，为了避免元素搬移，通常把数组看作什么", "环形数组"),
    ("堆通常用来实现哪种抽象数据类型", "优先队列"),
    ("对二叉搜索树进行中序遍历，得到的序列是什么顺序", "升序"),
    ("动态规划中描述状态之间转移关系的方程叫什么", "状态转移方程"),
    ("在递归中加入缓存以避免重复计算，这种方法叫什么", "记忆化搜索"),
    ("AVL 树用哪个指标判断是否失衡", "平衡因子"),
    ("堆是一种满足什么性质的二叉树", "完全二叉树"),
    ("背包问题中，每种物品可以选无限多次，这种问题叫什么", "完全背包"),
]

CHUNK_SIZES = [200, 500, 1000]
TOP_KS = [3, 5, 10]


def main():
    print(f"加载模型：{MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    pages = load_pdf(PDF_PATH)
    full_text = normalize("".join(t for _, t in pages))
    print(f"PDF：{len(pages)} 页，{len(full_text)} 字")

    # 先确认每个答案关键词在全文里真的存在
    print("\n关键词检查：")
    for _, kw in TEST_SET:
        print(f"  {kw}: {full_text.count(normalize(kw))} 次")

    # BGE 中文模型建议给 query 加检索指令；英文模型没有这个 prompt，会自动走 except
    try:
        q_embs = [model.encode(q, prompt_name="query") for q, _ in TEST_SET]
    except Exception:
        q_embs = [model.encode(q) for q, _ in TEST_SET]

    print("\nchunk_size | top_k | 命中率")
    print("-----------+-------+-------")
    for cs in CHUNK_SIZES:
        chunks = chunk_text(pages, cs)
        chunk_embs = model.encode(chunks)
        for k in TOP_KS:
            hit = 0
            for (_, kw), q_emb in zip(TEST_SET, q_embs):
                idxs = top_k_index(q_emb, chunk_embs, k)
                if any(normalize(kw) in normalize(chunks[i]) for i in idxs):
                    hit += 1
            print(f"  {cs:>6}   |  {k:>3}  | {hit}/{len(TEST_SET)}")


if __name__ == "__main__":
    main()
