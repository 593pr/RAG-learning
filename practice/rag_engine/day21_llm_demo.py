"""Day 21：llm.py 的两条验收。"""
from .errors import RAGError
from .llm import chat

if __name__ == "__main__":
    try:
        answer = chat([{"role": "user", "content": "只回复两个字：收到"}])
        print("回答：", answer)
    except RAGError as e:
        print(f"失败：{type(e).__name__}, status_code={getattr(e, 'status_code', None)}")
        print(f"原因：{e}")
