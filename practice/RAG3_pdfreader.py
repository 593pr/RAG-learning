
from pypdf import PdfReader

def load_pdf(path:str)->str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)



def chunk_text(text:str,chunk_size:int = 250,overlap:int = 50)->list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    text = load_pdf(r"C:\Users\21215\Downloads\实验一指导-路由交换3-静态（默认）路由与DHCP、NAT.pdf")
    print(f"PDF total{len(text)} zi")
    chunks = chunk_text(text)
    print(f"切成 {len(chunks)} 块，第一块：\n{chunks[0]}")