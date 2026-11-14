"""Day 23：跑固定测试集，输出汇总表 + 逐题明细 + 失败模式。

运行（在 practice/ 目录下）：
    python -m rag_engine.day23_eval_demo

需要 DEEPSEEK_API_KEY（答案正确率要真调 LLM）。
"""
import logging
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from .evaluate import answer_rows, detail_table, diagnose, recall_at_k, summary_table
from .pipeline import RAGPipeline

logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
PDF = BASE_DIR / "data" / "hello-algo.pdf"
CACHE = BASE_DIR / "cache" / "embed"

if __name__ == "__main__":
    pipe = RAGPipeline(str(PDF), CACHE)

    recall = {k: recall_at_k(pipe, k) for k in (1, 3, 5)}
    answers = {k: answer_rows(pipe, k=k) for k in (3, 5)}

    print("\n## 汇总\n")
    print(summary_table(recall, answers))

    print("\n## 逐题明细（k=3）\n")
    print(detail_table(recall[3], answers[3]))

    for k in (3, 5):
        print(f"\n## 失败模式（k={k}）\n")
        problems = diagnose(recall[k], answers[k])
        if not problems:
            print("- 无")
        for problem in problems:
            print("-", problem)

    print("\n## 回答原文（k=3）\n")
    for i, row in enumerate(answers[3], 1):
        print(f"{i}. {row['q']}")
        print(f"   → {row['answer']}\n")
