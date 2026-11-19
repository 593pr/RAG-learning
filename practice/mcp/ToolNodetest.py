import asyncio
import contextvars
import threading
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

_tid = contextvars.ContextVar("tid", default="-")


def where(tag):
    return f"{tag}: thread={threading.current_thread().name} tid={_tid.get()}"


@tool
def sync_tool(x: int) -> str:
    """同步工具"""
    return where("sync")


@tool
async def async_tool(x: int) -> str:
    """异步工具"""
    return where("async")


class State(TypedDict):
    messages: Annotated[list, add_messages]


async def main():
    _tid.set("abc123")

    b = StateGraph(State)
    b.add_node("tools", ToolNode([sync_tool, async_tool]))
    b.add_edge(START, "tools")
    graph = b.compile()

    msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "sync_tool",  "args": {"x": 1}, "id": "c1"},
            {"name": "async_tool", "args": {"x": 2}, "id": "c2"},
        ],
    )
    out = await graph.ainvoke({"messages": [msg]})
    for m in out["messages"][1:]:
        print(m.content)


asyncio.run(main())

