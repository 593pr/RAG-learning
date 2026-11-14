"""RAG pipeline：load → chunk → 缓存编码 → 混合检索 → LLM。"""
import logging
import string
from pathlib import Path

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .cache import EmbeddingCache
from .errors import EmptyRetrievalError
from .llm import chat,chat_stream
from .loader import chunk_pages,load_pdf

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
MIN_SIMILARITY = 0.65

PUNCT = set("，。！？；：、“”‘’（）【】《》…—,.!?;:\"'()[]{}") | set(string.punctuation)

def tokenize(text:str)->list[str]:
    return [w for w in jieba.lcut(text) if w.strip() and w not in PUNCT]

def minmax(scores:np.ndarray)->np.ndarray:
    lo,hi =  scores.min(),scores.max()
    return(scores - lo) / (hi - lo + 1e-9)
#Min-Max 归一化，把分数线性映射到 [0, 1] 区间

class RAGPipeline:
    def __init__(self,pdf_path:str,cache_dir:Path,model_name:str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.cache = EmbeddingCache(Path(cache_dir),model_name)

        pairs = chunk_pages(load_pdf(pdf_path),200,50)
        self.chunks = [t for _,t in pairs]
        self.pages = [p for p,_  in pairs]

        self.chunk_vectors = self.cache.encode(self.chunks,self.model)

        self.chunk_vectors /= np.linalg.norm(self.chunk_vectors,axis=1,keepdims=True)
        #把每个向量缩成长度 1，避免了长文本“作弊”，这样后面用"点积"算相似度，就等于在用"余弦相似度"——又快又准。

        for w in ["哈希表", "链式地址", "开放寻址", "负载因子",
                        "状态转移方程", "记忆化搜索", "完全背包"]:
                    jieba.add_word(w)
        self.bm25 = BM25Okapi([tokenize(c) for c in self.chunks])

        logger.info("索引就绪：%d chunks", len(self.chunks))

    
    def _semantic_scores(self,query:str)->np.ndarray:
          q = self.model.encode(query,prompt_name="query")
          q = q / np.linalg.norm(q)
          return self.chunk_vectors  @ q
    #query和每个chunk之间的余弦相似度
    
    def _bm25_scores(self,query:str)->np.ndarray:
          return self.bm25.get_scores(tokenize(query))
    #计算query和chunk之间的关键词匹配分数
    
    def search(self,query:str,k:int = 3)->list[int]:
          raw_sem = self._semantic_scores(query)
          fused = 0.5 * minmax(raw_sem) + 0.5 * minmax(self._bm25_scores(query))
          indices = np.argsort(fused)[::-1][:k]

          logger.info("检索最高相似度：%.3f",raw_sem[indices[0]])

          if raw_sem[indices[0]] < MIN_SIMILARITY:
                raise EmptyRetrievalError(query)
          
          return list(indices)
    

    def context_messages(self,query:str,idx:list[int])->list[dict]:
          """检索结果 + 问题 → messages。流式和非流式共用，避免两套 prompt 漂移。"""
          context = "\n\n".join(
             f"[第{self.pages[i]}页] {self.chunks[i]}" for i in idx
            )
          return [
                {
                      "role":"system",
                      "content":(
                            "你是一个严谨的问答助手。\n"
                            "1. 只能依据用户提供的资料回答，不要使用资料之外的知识。\n"
                            "2. 资料不足以回答时，直接回复「资料中没有找到答案」，不要猜测。\n"
                            "3. 每个结论后面标注来源页码，格式为 [第X页]。"
                      ),
                },
                {
                      "role":"user",
                      "content":f"资料：\n{context}\n\n问题：{query}",
                },
          ]

    def ask(self,query:str,k:int = 3)->dict:
          try:
                idx = self.search(query,k)
          except EmptyRetrievalError:
                return {"answer": "资料中没有找到相关内容。", "sources": []}

          answer = chat(self.context_messages(query,idx))

          return {"answer": answer, "sources": sorted({self.pages[i] for i in idx})}

    def ask_stream(self,query:str,k:int = 3):
          """流式作答：先 yield 页码，再逐段 yield 文字。

          页码必须排在第一个 —— 检索在生成之前就完成了，
          如果边生成边检索，前端拿不到来源。
          """
          try:
                idx = self.search(query,k)
          except EmptyRetrievalError:
                yield ("sources",[])
                yield ("token","资料中没有找到相关内容。")
                return

          yield ("sources",sorted({self.pages[i] for i in idx}))
          for piece in chat_stream(self.context_messages(query,idx)):
                yield ("token",piece)

