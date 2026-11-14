"""Day 24 验收客户端：调 /ask 和 /ask/stream，用时间戳证明流式是真的"边生成边到"。

先起服务（另一个终端）：
    python -m uvicorn rag_engine.server:app --port 8000

再跑（在 practice/ 目录下）：
    python -m rag_engine.day24_client
    python -m rag_engine.day24_client 8011      # 换端口

不用 requests：标准库 urllib 就够，少一个依赖。
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = f"http://127.0.0.1:{sys.argv[1] if len(sys.argv) > 1 else 8000}"

QUESTION = "哈希表里元素数量与桶数量之比叫什么"
LONG_QUESTION = "请详细说明开放寻址法的原理，包括线性探测、平方探测和多次哈希"


def post_json(path: str, payload: dict) -> tuple[int, dict]:
    """一次性请求。返回 (状态码, 响应体)；4xx/5xx 也当数据返回，不抛异常。"""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def stream(path: str, payload: dict) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    first = None
    text = []
    with urllib.request.urlopen(req) as resp:
        print(f"  响应头：{resp.status} {resp.headers.get('content-type')}")
        # 服务端每条消息以 "\n\n" 结尾，按行迭代就能一块块读到
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue  # 空行 / 注释行 / event: 字段
            body = line[len("data: "):]
            if body == "[DONE]":
                print(f"\n  首字 {first or 0:.2f}s · 共 {len(''.join(text))} 字 · 总耗时 {time.perf_counter() - start:.2f}s")
                return
            event = json.loads(body)
            if event["type"] == "token":
                if first is None:
                    first = time.perf_counter() - start  # 用户看到第一个字的时刻
                text.append(event["value"])
                print(event["value"], end="", flush=True)
            else:
                print(f"  {event['type']}: {event['value']}")


def main() -> None:
    print(f"目标：{BASE}\n")

    print("## 1. 健康检查")
    with urllib.request.urlopen(BASE + "/healthz") as resp:
        print(" ", resp.status, resp.read().decode("utf-8"))

    print("\n## 2. 空问题 → 422（Pydantic 校验）")
    status, body = post_json("/ask", {"question": "", "k": 3})
    print(f"  {status} {body}")

    print("\n## 3. 非流式 /ask（用户在这段时间里只能看白屏）")
    start = time.perf_counter()
    status, body = post_json("/ask", {"question": QUESTION, "k": 3})
    print(f"  [{time.perf_counter() - start:5.2f}s] {status} {body}")

    print("\n## 4. 流式 /ask/stream")
    stream("/ask/stream", {"question": QUESTION, "k": 3})

    print("\n## 5. 长答案对比：答案越长，流式优势越明显")
    start = time.perf_counter()
    status, body = post_json("/ask", {"question": LONG_QUESTION, "k": 5})
    print(f"  非流式 [{time.perf_counter() - start:5.2f}s] 共 {len(body['answer'])} 字")
    stream("/ask/stream", {"question": LONG_QUESTION, "k": 5})


if __name__ == "__main__":
    main()
