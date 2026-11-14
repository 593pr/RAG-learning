"""Day 22：阈值校准 —— 域内 vs 域外问题的最高相似度分布。"""
import logging
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from .pipeline import RAGPipeline

logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent

IN_DOMAIN = [
    "哈希冲突时，把冲突元素串在同一个桶的链表里，这种方法叫什么",
    "哈希冲突时，按规则探测下一个空闲桶的方法叫什么",
    "哈希表中元素数量与桶数量之比叫什么",
    "用数组实现队列时，为了避免元素搬移，通常把数组看作什么",
    "堆通常用来实现哪种抽象数据类型",
    "对二叉搜索树进行中序遍历，得到的序列是什么顺序",
    "动态规划中描述状态之间转移关系的方程叫什么",
    "在递归中加入缓存以避免重复计算，这种方法叫什么",
    "AVL 树用哪个指标判断是否失衡",
    "堆是一种满足什么性质的二叉树",
    "背包问题中，每种物品可以选无限多次，这种问题叫什么",
]

OUT_DOMAIN = [
    "今天天气怎么样",
    "红烧肉怎么做",
    "明天股市会涨吗",
    "怎么申请出国签证",
    "推荐一部好看的电影",
    "如何提高英语口语",
    "怎样在一个月内减肥十斤",
    "Python 怎么读取一个文件",
]


def top1(pipe, q):
    return float(pipe._semantic_scores(q).max())


if __name__ == "__main__":
    pipe = RAGPipeline(str(BASE_DIR / "data" / "hello-algo.pdf"), BASE_DIR / "cache" / "embed")

    print("\n=== 域内（应高于阈值）===")
    in_scores = [(top1(pipe, q), q) for q in IN_DOMAIN]
    for s, q in sorted(in_scores):
        print(f"{s:.3f}  {q}")

    print("\n=== 域外（应低于阈值）===")
    out_scores = [(top1(pipe, q), q) for q in OUT_DOMAIN]
    for s, q in sorted(out_scores, reverse=True):
        print(f"{s:.3f}  {q}")

    print("\n域内最低：", f"{min(s for s, _ in in_scores):.3f}")
    print("域外最高：", f"{max(s for s, _ in out_scores):.3f}")
