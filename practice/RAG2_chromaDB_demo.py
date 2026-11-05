
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

client =chromadb.Client()
#数据库连接，现在是内存模式，运行结束之后数据就没了

collection = client.create_collection("demo")
#在库里面创建一张叫"demo"的表

docs = [
    "苹果发布了新款 iPhone 手机",
    "iPhone 的拍照功能很强大",
    "Python 是一门编程语言",
    "Java 和 Python 都是后端常用语言",
    "今天天气很好适合去公园散步",
] 

collection.add(
    ids = [f"id{i}" for i in range(len(docs))],
    documents = docs,
    embeddings = model.encode(docs).tolist(),
)
#tolist()把numpy数组转成python列表（chromaDB只收列表格式）

question = "哪家公司发布了iphone?"

results = collection.query(
    query_embeddings = [model.encode(question).tolist()],
    n_results = 3,
)
#query_embeddings和n_results是固定的参数名，不能更改
#排出最近的3条然后返回

print("问题:", question)
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"  distance={dist:.4f}  {doc}")