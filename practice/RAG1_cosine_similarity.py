# embedding_demo.py
import os
#os用于处理接触操作系统
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像，不设这行会下载失败

from sentence_transformers import SentenceTransformer
import numpy as np
#numpy用于高效计算数组和矩阵

model = SentenceTransformer('all-MiniLM-L6-v2')  # 首次运行自动下载，约 90MB

sentences = [
    "苹果公司发布了新款 iPhone",
    "iPhone 是苹果公司最畅销的手机",
    "今天天气很好，适合去公园散步",
]

embeddings = model.encode(sentences)

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print("句1 vs 句2（相关）:", cosine_similarity(embeddings[0], embeddings[1]))
print("句1 vs 句3（无关）:", cosine_similarity(embeddings[0], embeddings[2]))
