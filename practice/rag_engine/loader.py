"""PDF 读取 + 切块（带错误处理）。"""
import logging

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .errors import PermanentError

logger = logging.getLogger(__name__)


def load_pdf(path: str) -> list[tuple[int, str]]:
    """读取 PDF，返回 [(页码, 文本)]。任何读不了的情况都抛 PermanentError。"""
    try:
        reader = PdfReader(path)
    except FileNotFoundError as e:
        # TODO 1：文件不存在 → PermanentError，消息里带上 path
        raise PermanentError(f"{path}文件不存在") from e
    except PdfReadError as e:
        # TODO 2：文件损坏 → PermanentError
        raise PermanentError(f"文件损坏:{path}") from e

    # TODO 3：加密检查。reader.is_encrypted 为 True 就直接抛 PermanentError
    if reader.is_encrypted:
        raise PermanentError(f"PDF已加密，无法读取:{path}")

    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append((i + 1, text))
        else:
            logger.warning("第 %d 页没有可提取文本，已跳过", i + 1)

    # TODO 4：一页都没读出来 → PermanentError（这是数据问题，不是"检索为空"）
    if not pages:
        raise PermanentError(f"PDF中没有可用文本：{path}")
    return pages

def chunk_pages(pages:list[tuple[int,str]],chunk_size:int=200,overlap:int = 50)->list[tuple[int,str]]:
    out = []
    for page_no,text in pages:
        start = 0
        while start < len(text):
            out.append((page_no,text[start:start + chunk_size]))
            start += chunk_size - overlap
    return out

def chunk_text(pages: list[tuple[int, str]], chunk_size: int = 200, overlap: int = 50) -> list[str]:
    chunks = []
    for _, text in pages:
        start = 0
        while start < len(text):
            chunks.append(text[start:start + chunk_size])
            start += chunk_size - overlap
    return chunks
