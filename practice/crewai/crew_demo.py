"""CrewAI 最小 Quickstart：2 个 agent 接力，把 RAG 评测数字写成面试话术。

W3 D4（2026-11-19）。只了解概念，不深入。

CrewAI 的三个概念：
  Agent  = role / goal / backstory 拼成的 system prompt + 一个内置的 ReAct 循环
  Task   = description / expected_output 拼成的 user prompt，外加一个 agent 作为执行者
  Crew   = 调度器：按 process 决定任务顺序，把上一个 task 的输出通过 context 传下去

和 LangGraph 的区别：LangGraph 你要自己画节点和边；CrewAI 你只声明「谁负责什么」，
执行顺序由 process 决定（sequential 就是按 tasks 列表顺序跑）。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv

# 复用 practice/.env 里的 DEEPSEEK_API_KEY（scripts 在 practice/crewai/ 下一层）
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
#Path(__file__).resolve() 先拿到当前文件的绝对路径

from crewai import LLM, Agent, Crew, Process, Task

# 这里用 LiteLLM 的 provider 前缀 deepseek/，CrewAI 底下走的是 litellm
llm = LLM(
    model="deepseek/deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
    temperature=0.3,
)

METRICS = """\
固定 11 题测试集（bge-small-zh-v1.5 + BM25 混合检索 + bge-reranker 重排）：
- recall@1 = 8/11，recall@3 = 9/11，recall@5 = 11/11
- 答案正确率：k=3 时 9/11，k=5 时 11/11，失败的两题是 #7 #8，都是检索漏召回
- rerank 把 @1 从 8 提到 10，@3 从 9 提到 11
- 经历过一次回归：新 pipeline 的标点过滤漏掉 string.punctuation，recall@5 从 11 掉到 10
"""

analyst = Agent(
    role="RAG 评测分析师",
    goal="从评测数字里挑出唯一值得在面试里讲的结论，并用两个数字支撑它",
    backstory=(
        "你在准备 AI Agent 实习面试。面试官只给你 30 秒，"
        "所以你的结论必须是一个可验证的判断，而不是一句「效果不错」。"
    ),
    llm=llm,
    verbose=True,
)

writer = Agent(
    role="面试话术写手",
    goal="把分析结论写成面试里能直接说出口的话",
    backstory=(
        "你写过很多技术复盘，熟悉「问题 → 怎么定位 → 怎么解决 → 学到什么」这套结构。"
        "你讨厌空话，每句话都要带一个数字。"
    ),
    llm=llm,
    verbose=True,
)

analyze_task = Task(
    description=(
        "分析下面这组 RAG 评测数据，找出最重要的一个结论。\n\n"
        f"{METRICS}"
    ),
    expected_output="一段 100 字以内的分析，必须引用至少两个数字。",
    agent=analyst,
)

write_task = Task(
    description="把分析结论写成面试可用的三句话，第一人称，口语化，不堆术语。",
    expected_output="三句话，每句不超过 40 字，合计至少包含两个具体数字。",
    agent=writer,
    context=[analyze_task],  # 显式声明依赖：写手要看到分析师的结果
)

crew = Crew(
    agents=[analyst, writer],
    tasks=[analyze_task, write_task],
    process=Process.sequential,
    verbose=True,
)

if __name__ == "__main__":
    started = time.time()
    result = crew.kickoff()
    elapsed = time.time() - started

    print(f"\n=== 耗时 {elapsed:.1f}s ===")
    print("=== 最终输出 ===")
    print(result.raw)

    usage = getattr(result, "token_usage", None)
    if usage is not None:
        print("\n=== token 用量 ===")
        print(
            f"prompt={usage.prompt_tokens} completion={usage.completion_tokens} "
            f"total={usage.total_tokens} requests={usage.successful_requests}"
        )
