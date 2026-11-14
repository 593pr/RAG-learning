"""Day 21：loader.py 的三种坏输入验收。"""
from .errors import RAGError
from .loader import load_pdf

if __name__ == "__main__":
    for path in ["data/hello-algo.pdf", "data/不存在.pdf", "data/corrupt.pdf"]:
        try:
            print(f"{path} -> {len(load_pdf(path))} 页")
        except RAGError as e:
            print(f"{path} -> {type(e).__name__} : {e}")
