
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

def load_pdf(path):
    reader = PdfReader(path)
    pages = []
    for i,page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append((i+1,text))
        
    return pages

def chunk_text(pages,chunk_size,overlap = 50):
    chunks = []
    for _,text in pages:
        start = 0
        while start < len(text):
            chunks.append(text[start:start + chunk_size])
            start += chunk_size - overlap
    return chunks

def cosine(a,b):
    return float(np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def normalize(s):
    return "".join(s.split())   # 去掉所有空格/换行/tab，PDF 提取的换行不是语义

def top_k_index(query_emb,chunk_emb,k = 3):
    scores = [cosine(query_emb,c) for c in chunk_emb]
    return np.argsort(scores)[::-1][:k]

#np.argsort默认从小到大排序，[::-1]将其翻转，从大到小

TEST_SET = [
    ("GAINAX 原本企划中使徒设定数量是多少", "17"),
    ("碇真嗣重复说的内心座右铭是什么", "绝对不能逃"),
    ("哪个秘密组织隐瞒了第二次冲击的真相", "SEELE"),
    ("为了开发 EVA 而设立的组织叫什么", "Gehirn"),
    ("剧中两位女主角分别是明日香和谁", "绫波丽"),
    ("来自 SEELE、取代明日香成为贰号机驾驶员的是谁", "渚薰"),
    ("使徒拥有的 A.T.力场又叫什么", "绝对领域"),
    ("绫波丽是谁的克隆体", "碇唯"),
    ("绫波丽的声优是谁", "林原惠"),
    ("庵野秀明担任监督时，副导演是谁", "摩砂雪"),
    ("碇真嗣驾驶的是哪台 EVA", "初号机"),
]

model = SentenceTransformer('all-MiniLM-L6-v2')
pages = load_pdf(r"C:\Users\21215\Downloads\新世纪福音战士 - 维基百科，自由的百科全书.pdf")

full_text = "".join(text for _, text in pages)
full_text_norm = normalize(full_text)
for kw in ["绝对不能逃", "SEELE", "Gehirn", "绫波丽", "渚薰", "绝对领域", "碇唯", "林原惠", "摩砂雪", "初号机"]:
    print(kw, "出现", full_text_norm.count(normalize(kw)), "次")

for chunk_size in [200,500,1000]:
    chunks = chunk_text(pages,chunk_size)
    chunk_emb = model.encode(chunks)

    hit = 0
    for question,keyword in TEST_SET:
        q_emb = model.encode(question)
        idxs = top_k_index(q_emb,chunk_emb,k=3)
        top_chunks =[chunks[i] for i in idxs]
        if any(normalize(keyword) in normalize(c) for c in top_chunks):
            hit += 1
        else:
            print(f"  未命中 | chunk={chunk_size} | 关键词「{keyword}」全文出现 {full_text_norm.count(normalize(keyword))} 次")

    print(f"chunk_size={chunk_size}: {hit}/{len(TEST_SET)} 命中")

    #计算命中率，如果这几个筛选出来的top_chunk有标准答案，那么就算命中
