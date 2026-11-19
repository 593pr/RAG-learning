"""给 CrewAI 补上决策层的眼睛：订阅事件总线。

W3 D4。`verbose=True` 只打编排层（task / agent 边界）。这里订阅：

  调用层  LLMCallStartedEvent    → model、messages
          LLMCallCompletedEvent  → usage（真实 token）、finish_reason
  决策层  ToolUsageStartedEvent  → tool_name、tool_args   ← verbose 里没有的东西
          ToolUsageFinishedEvent → output、耗时

为了让决策层有东西可看，给 agent 一个必须调用的工具。
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from crewai import LLM, Agent, Crew, Process, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallStartedEvent
from crewai.events.types.tool_usage_events import (
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai.tools import tool

LOG_PATH = Path(__file__).with_name("events.jsonl")
_started_at: dict[str, float] = {}


def emit(event: str, **fields) -> None:
    """一行一个 JSON，和 D3 的 JsonFormatter 同一个形状。"""
    record = {"ts": time.time(), "event": event, **fields}
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@crewai_event_bus.on(LLMCallStartedEvent)
def on_llm_start(source, event: LLMCallStartedEvent) -> None:
    _started_at["llm"] = time.time()
    emit(
        "llm.call",
        model=event.model,
        msg_count=len(event.messages) if isinstance(event.messages, list) else 1,
    )


@crewai_event_bus.on(LLMCallCompletedEvent)
def on_llm_end(source, event: LLMCallCompletedEvent) -> None:
    ms = int((time.time() - _started_at.get("llm", time.time())) * 1000)
    emit(
        "llm.done",
        ms=ms,
        finish_reason=event.finish_reason,
        usage=event.usage,
    )


@crewai_event_bus.on(ToolUsageStartedEvent)
def on_tool_start(source, event: ToolUsageStartedEvent) -> None:
    emit("tool.call", tool=event.tool_name, args=event.tool_args)


@crewai_event_bus.on(ToolUsageFinishedEvent)
def on_tool_end(source, event: ToolUsageFinishedEvent) -> None:
    ms = int((event.finished_at - event.started_at).total_seconds() * 1000)
    emit("tool.done", tool=event.tool_name, ms=ms, output=str(event.output)[:120])


@tool("recall_rate")
def recall_rate(hits: int, total: int) -> str:
    """计算召回率，返回百分比。hits 是命中的题数，total 是题目总数。"""
    return f"{hits}/{total} = {hits / total * 100:.1f}%"


llm = LLM(
    model="deepseek/deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
    temperature=0,
)

analyst = Agent(
    role="RAG 评测分析师",
    goal="用工具算出三个召回率，然后给出一句结论",
    backstory="你必须用 recall_rate 工具算百分比，不许心算。",
    llm=llm,
    tools=[recall_rate],  # 有工具，才会产生 ToolUsage 事件
    verbose=False,  # 关掉 verbose，只看我们自己订阅到的东西
)

task = Task(
    description=(
        "分别用 recall_rate 工具计算三组召回率：8/11、9/11、11/11。"
        "算完后用一句话说明这三个数字对应评测里的哪个阶段。"
    ),
    expected_output="三个百分比 + 一句话结论。",
    agent=analyst,
)

crew = Crew(agents=[analyst], tasks=[task], process=Process.sequential)

if __name__ == "__main__":
    LOG_PATH.unlink(missing_ok=True)
    started = time.time()
    result = crew.kickoff()
    print(f"\n=== 总耗时 {time.time() - started:.1f}s ===")
    print(result.raw)
    print(f"\n=== 结构化日志（{LOG_PATH.name}）===")
    print(LOG_PATH.read_text(encoding="utf-8"))
