
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pypdf import PdfReader
from sentence_transformers import  SentenceTransformer
import chromadb
from openai import OpenAI

def load_pdf(path:str):
    reader = PdfReader(path)
    pages = []
    for i,page in enumerate(reader.pages):
        text = page.extract_text()
        if text :
            pages.append((i+1,text))
    return pages
# 现在返回 [(页码, 文本), ...] 列表，页码从 1 开始


def chunk_text(pages,chunk_size:int=250,overlap:int=50):
    chunks =[]
    for page,text in pages:
        start = 0
        while start <len(text):
            chunks.append((page,text[start:start + chunk_size]))
            start += chunk_size - overlap
    return chunks

class RAGPipeline:
    def __init__(self,api_key:str):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("pdf_docs")
        self.llm = OpenAI(api_key = api_key, base_url = "https://api.deepseek.com")


    def add_pdf(self,path:str):
        pages = load_pdf(path)
        chunks = chunk_text(pages)
        self.collection.add(
            ids=[f"chunk{i}" for i in range(len(chunks))],
            documents=[c[1] for c in chunks],
            metadatas=[{"page":c[0]} for c in chunks],
            embeddings = self.embedder.encode([c[1] for c in chunks]).tolist(),
        )
        return len(chunks)
    
    def query(self,question:str,top_k:int = 10)->str:
        result = self.collection.query(
            query_embeddings = [self.embedder.encode(question).tolist()],
            n_results = top_k,
        )
        docs = result["documents"][0]
        metas = result["metadatas"][0]

        context_parts = []
        for doc, meta in zip(docs, metas):
            context_parts.append(f"[第 {meta['page']} 页] {doc}")
        
        context = "\n\n".join(context_parts)

        prompt = f"""基于以下资料回答问题。
        规则：
        1. 如果资料中有答案，回答后必须注明来源页码，格式：[来源：第X页]
        2. 如果资料中没有答案，直接回答"我不知道"，不要编造
        资料: {context}
        问题: {question}
        答案: """

        response = self.llm.chat.completions.create(
            model = "deepseek-chat",
            messages = [{"role" : "user","content" : prompt}],
        )
        return response.choices[0].message.content
    

if __name__ == "__main__":
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("请先设置环境变量 DEEPSEEK_API_KEY")
    rag = RAGPipeline(api_key=api_key)
    n = rag.add_pdf(r"C:\Users\21215\Downloads\实验一指导-路由交换3-静态（默认）路由与DHCP、NAT.pdf")
    print(f"已入库 {n} 个chunk")

    while True:
        q = input("\n提问(输入 exit 退出):")
        if q.strip().lower() == "exit":
            break
        print("答:",rag.query(q))
#test：碇真嗣重复说的一句内心座右铭是什么？庵野秀明如何评价碇真嗣的性格？GAINAX 原本企划中使徒设定数量是多少，正式播出改成多少个？