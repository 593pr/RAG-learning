"""LlamaIndex 最小 Quickstart：用自己写的 Obsidian 笔记建索引，然后问答。

W3 D4（2026-11-19）。只了解概念，不深入。

LlamaIndex 的一条默认流水线（每个环节都能替换）：
  Reader      SimpleDirectoryReader  → Document（带 file_name 等 metadata）
  NodeParser  SentenceSplitter       → Node（chunk + 溯源 metadata）——按 token 数切，不是按字数
  Index       VectorStoreIndex       → 向量化 + 存储
  Retriever   index.as_retriever()   → top-k
  QueryEngine retriever + synthesizer → 拼 prompt、调 LLM、带 sources 返回

对照点：手写 RAG 里没有「Node」这个中间层，chunk 只是 (页码, 文本) 元组；
LlamaIndex 把 chunk、原文、metadata、embedding 打包成一个对象，所以能直接回溯来源。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# 模型已在 HF 缓存里；不设 offline 会先去 huggingface.co 查 metadata，国内直接 WinError 10054
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from huggingface_hub import snapshot_download
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

VAULT = Path(__file__).resolve().parents[3]  # D:\AI learning
NOTES_DIR = VAULT / "AI学习" / "概念"

QUESTIONS = [
    "MCP client 的 stderr 管道不消费，为什么会让 server 卡死？",
    "评测的价值是什么？用一句话说清楚。",
    "为什么 rerank 的候选深度取 10 比 20 好？",
]


def main() -> None:
    # 用 OpenAILike 而不是 OpenAI：后者会查「模型名是不是 OpenAI 官方模型」，
    # deepseek-chat 不在列表里直接 ValueError。OpenAILike 不查名字，但要自己声明上下文窗口。
    Settings.llm = OpenAILike(
        model="deepseek-chat",
        api_base="https://api.deepseek.com/v1",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        context_window=64_000,
        is_chat_model=True,
        is_function_calling_model=False,
        temperature=0.3,
    )
    # 复用全局环境里已下载的 bge-small-zh-v1.5，和手写 RAG 用同一个模型，才有一点可比性。
    # 注意：直接传 "BAAI/bge-small-zh-v1.5" 会让它去 llama-index 自己的 cache_folder 找，
    # 那里没有模型 → 离线时拿到空配置 → Pooling 报「缺 embedding_dimension」。
    # 先解析成 HF 缓存里的本地快照路径，框架就只能用它，不会再去别处找。
    model_path = snapshot_download("BAAI/bge-small-zh-v1.5", local_files_only=True)
    Settings.embed_model = HuggingFaceEmbedding(model_name=model_path, embed_batch_size=16)
    # chunk_size 这里是 token 数，不是字数。手写版的 250 字 ≈ 250 个中文 token 左右
    Settings.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=50)

    print(f"语料目录：{NOTES_DIR}")
    documents = SimpleDirectoryReader(input_dir=str(NOTES_DIR), required_exts=[".md"]).load_data()
    print(f"1. Reader  → {len(documents)} 个 Document（每个 md 文件一个）")
    print(f"   例：{documents[0].metadata.get('file_name')}，{len(documents[0].text)} 字")

    nodes = Settings.node_parser.get_nodes_from_documents(documents)
    print(f"2. Parser  → {len(nodes)} 个 Node（按 token 切，带 metadata）")
    print(f"   例：{nodes[0].metadata.get('file_name')} | {nodes[0].text[:40]!r}...")

    started = time.time()
    index = VectorStoreIndex(nodes)
    print(f"3. Index   → 向量化并建索引，耗时 {time.time() - started:.1f}s")

    query_engine = index.as_query_engine(similarity_top_k=3)

    for q in QUESTIONS:
        started = time.time()
        response = query_engine.query(q)
        elapsed = time.time() - started
        print(f"\n{'=' * 60}\nQ: {q}\n耗时 {elapsed:.1f}s")
        print(f"A: {str(response).strip()}")
        print("来源：")
        for i, node in enumerate(response.source_nodes, 1):
            name = node.metadata.get("file_name", "?")
            print(f"  [{i}] {name}  score={node.score:.4f}")


if __name__ == "__main__":
    main()
