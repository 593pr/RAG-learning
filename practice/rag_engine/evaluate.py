"""Day 23：端到端评测 —— 检索 recall@k + 生成答案正确率。

评测只做两件事：
1. 用固定测试集算指标（不打分的主观判断）
2. 把失败模式列出来（检索漏了 / 检索中了但答错）
"""

from .errors import EmptyRetrievalError
from .testset import TEST_SET


def normalize(text: str) -> str:
    """去掉所有空白：PDF 提取的换行不是语义。"""
    return "".join(text.split())


def recall_at_k(pipe, k: int) -> list[bool]:
    """每题一个 True/False：top-k 里有没有包含答案关键词的 chunk。"""
    flags = []
    for item in TEST_SET:
        try:
            idx = pipe.search(item["q"], k)
        except EmptyRetrievalError:
            flags.append(False)
            continue

        # TODO 1：hit = item["keyword"] 是否出现在 idx 指向的任一 chunk 中
        # 提示：any(... for i in idx)，两边都过 normalize()
        hit = any(
            normalize(item["keyword"]) in normalize(pipe.chunks[i]) for i in idx
        )
        flags.append(hit)

    return flags


def answer_rows(pipe, k: int = 3) -> list[dict]:
    """逐题调用 ask()，关键词出现在回答里算答对。"""
    rows = []
    for item in TEST_SET:
        result = pipe.ask(item["q"], k=k)

        # TODO 2：ok = item["keyword"] 是否出现在 result["answer"] 中
        ok = normalize(item["keyword"]) in normalize(result["answer"])

        rows.append({
            "q": item["q"],
            "keyword": item["keyword"],
            "ok": ok,
            "answer": result["answer"],
            "sources": result["sources"],
        })

    return rows


def summary_table(recall: dict[int, list[bool]], answers: dict[int, list[dict]]) -> str:
    """汇总表：一眼看到全部指标。"""
    n = len(TEST_SET)
    lines = ["| 指标 | 结果 |", "|------|------|"]
    for k, flags in sorted(recall.items()):
        lines.append(f"| recall@{k} | {sum(flags)}/{n} |")
    for k, rows in sorted(answers.items()):
        lines.append(f"| 答案正确率（k={k}） | {sum(r['ok'] for r in rows)}/{n} |")
    return "\n".join(lines)


def detail_table(recall3: list[bool], rows: list[dict]) -> str:
    """逐题明细：同一行里对照「检索」和「生成」。"""
    lines = [
        "| # | 问题 | 检索命中@3 | 答案正确 | 来源页 |",
        "|---|------|-----------|---------|--------|",
    ]
    for i, (hit, row) in enumerate(zip(recall3, rows), 1):
        q = row["q"] if len(row["q"]) <= 24 else row["q"][:24] + "…"
        pages = ",".join(str(p) for p in row["sources"]) or "—"
        hit_mark = "✓" if hit else "✗"
        ok_mark = "✓" if row["ok"] else "✗"
        lines.append(f"| {i} | {q} | {hit_mark} | {ok_mark} | {pages} |")
    return "\n".join(lines)


def diagnose(recall3: list[bool], rows: list[dict]) -> list[str]:
    """列出两种失败模式，供人看，不参与打分。

    ① 检索漏召回：检索层就没把答案捞上来
    ② 检索中了但答错：召回没问题，问题在 prompt / 生成 / 答案抽取
    """
    # TODO 3：遍历 zip(recall3, rows)，收集三种情况
    #   ① not hit              → f"检索漏召回：{row['q']}"
    #   ② hit and not ok       → f"检索中了但答错：{row['q']}"
    #   ③ not hit and ok       → f"检索漏了但答对（疑似 LLM 自带知识）：{row['q']}"
    #   提示：① 和 ③ 都是 not hit，用 elif 的顺序区分
    problems =[]
    for hit,row in zip(recall3,rows):
        if not hit and row["ok"]:
            problems.append(f"检索漏了但答对（疑似 LLM 自带知识）：{row['q']}")
        elif not hit:
            problems.append(f"检索漏召回：{row['q']}")
        elif not row["ok"]:
            problems.append(f"检索中了但答错：{row['q']}")
    return problems
