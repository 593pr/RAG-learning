
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pypdf import PdfReader
from sentence_transformers import  SentenceTransformer
import chromadb
from openai import OpenAI

def load_pdf(path:str):
    reader = PdfReader(path)
    pages = [p.extract_text() for p in reader.pages if p.extract_text()]
    return "\n".join(pages)
#用 "\n"（换行符）作为分隔符，把 pages 列表里的所有字符串拼接成一个完整的字符串。


def chunk_text(text:str,chunk_size:int=250,overlap:int=50):
    chunks =[]
    start = 0
    while start <len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    
    return chunks

class RAGPipeline:
    def __init__(self,api_key:str):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("pdf_docs")
        self.llm = OpenAI(api_key = api_key, base_url = "https://api.deepseek.com")


    def add_pdf(self,path:str):
        text = load_pdf(path)
        chunks = chunk_text(text)
        self.collection.add(
            ids = [f"chunk{i}" for i in range(len(chunks))],
            documents = chunks,

        embeddings = self.embedder.encode(chunks).tolist(),
        )
        return len(chunks)
    
    def query(self,question:str,top_k:int = 10)->str:
        result = self.collection.query(
            query_embeddings = [self.embedder.encode(question).tolist()],
            n_results = top_k,
        )
        context = "\n\n".join(result["documents"][0])
        prompt = f"""基于以下资料回答问题。如果资料中没有答案，请直接回答"我不知道"
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
    
    t = load_pdf(r"C:\Users\21215\Downloads\实验一指导-路由交换3-静态（默认）路由与DHCP、NAT.pdf")
    for kw in ["DHCP", "NAT", "静态路由"]:
        print(kw, "出现", t.count(kw), "次")

    while True:
        q = input("\n提问(输入 exit 退出):")
        if q.strip().lower() == "exit":
            break
        print("答:",rag.query(q))
#test：碇真嗣重复说的一句内心座右铭是什么？庵野秀明如何评价碇真嗣的性格？GAINAX 原本企划中使徒设定数量是多少，正式播出改成多少个？