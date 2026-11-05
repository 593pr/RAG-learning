
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path

from sentence_transformers import CrossEncoder
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import string

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = str(BASE_DIR / "data" / "hello-algo.pdf")
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
CHUNK_SIZE = 200
OVERLAP = 50

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

def normalize(s):
    return "".join(s.split())  # 去掉所有空格/换行，PDF 提取的换行不是语义
#不带参数调用 split() 时，Python 会按任意空白字符（空格、换行 \n、制表符 \t、回车 \r 等）切分字符串，
#并且自动丢弃连续空白和首尾空白，结果是一个列表。
#join 是字符串的方法，把可迭代对象里的所有字符串用调用者作为分隔符拼接起来。
#这里调用者是空字符串 ""，所以就是无分隔符直接连起来

PUNCT = set("，。！？；：、“”‘’（）【】《》…—‑–,.!?;:\"'()[]{}") | set(string.punctuation)

def tokenize(text):
    words = jieba.lcut(text)
    #过滤空白和标点，返回词列表
    return [
        w for w in words
        if w.strip()
        and w not in PUNCT
    ]

def cosine(a,b):
    return float(np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def semantic_scores(query) -> np.ndarray:
    q = model.encode(query,prompt_name="query")
    return np.array([cosine(q,c) for c in chunk_embs])

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

pages =load_pdf(PDF_PATH)
chunks = chunk_text(pages,CHUNK_SIZE,OVERLAP)
model = SentenceTransformer(MODEL_NAME)
RERANKER_PATH = str(BASE_DIR / "models" / "bge-reranker-base")
reranker = CrossEncoder(RERANKER_PATH, max_length=512)
print("编码chunk中...")
chunk_embs = model.encode(chunks)
print("编码完成")

print("构建BM25索引...")
for w in ["哈希表", "哈希冲突", "哈希函数", "链式地址", "开放寻址",
          "负载因子", "环形数组", "优先队列", "状态转移方程",
          "记忆化搜索", "完全二叉树", "完全背包", "二叉搜索树"]:
    jieba.add_word(w)
tokenized_chunks = [tokenize(c) for c in chunks]
bm25 = BM25Okapi(tokenized_chunks)
print("索引完成")

def bm25_scores(query) -> np.ndarray:
    return bm25.get_scores(tokenize(query))

def semantic_topk(query,k):
    scores = semantic_scores(query)
    return np.argsort(scores)[::-1][:k]

def bm25_topk(query,k):
    scores = bm25_scores(query)
    return np.argsort(scores)[::-1][:k]

def rrf(rank_lists,k=60):
    total = {}
    for ranking in rank_lists:
        for rank,idx in enumerate(ranking,start=1):
            total[idx] = total.get(idx,0) + 1 / (k + rank)
    return sorted(total,key=total.get,reverse=True)

def hybrid_topk(query,k,depth=20):
    return rrf([semantic_topk(query,depth),bm25_topk(query,depth)]) [:k]

#评估函数
def evaluate(get_topk,k):
    hits = 0
    misses = []
    for question,keyword in TEST_SET:
        top = get_topk(question,k)
        if any(keyword in normalize(chunks[i]) for i in top):
            hits += 1
        else:
            misses.append(keyword)
    return hits,misses

def first_hit_rank(ranking, keyword):
    for rank, idx in enumerate(ranking, start=1):
        if keyword in normalize(chunks[idx]):
            return rank
    return None

def minmax(scores):
    lo = scores.min()
    hi = scores.max()
    return (scores - lo) / (hi - lo + 1e-9)

def hybrid_score_topk(query, k):
    sem = minmax(semantic_scores(query))
    bm = minmax(bm25_scores(query))
    combined = 0.5 * sem + 0.5 * bm
    return np.argsort(combined)[::-1][:k]

def rerank_topk(query,k,depth=20):
    candidates = hybrid_score_topk(query,depth)
    pairs = [(query,chunks[i]) for i in candidates]
    scores = reranker.predict(pairs,batch_size=32)
    order = np.argsort(scores)[::-1]
    return [candidates[j] for j in order[:k]]


if __name__ == "__main__":
    hits, _ = evaluate(hybrid_score_topk, 20)
    print(f"一阶段天花板 top-20: {hits}/11")

    for name, fn in [("混合-分数", hybrid_score_topk), ("rerank", rerank_topk)]:
        row = []
        for k in [1, 3, 5]:
            hits, _ = evaluate(fn, k)
            row.append(f"{hits}/11")
        print(f"{name:10} {row}")
    
    for depth in [10,20,50]:
        fn = lambda q,k,d=depth:rerank_topk(q,k,d)
        r1,_ = evaluate(fn,1)
        r3,_ = evaluate(fn,3)
        print(f"depth={depth:<3} @1={r1}/11 @3={r3}/11")
