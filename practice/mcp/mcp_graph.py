import asyncio
import os
import sys
from typing import Annotated, TypedDict
from log import get_logger,new_trace,timed

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_tools import to_lc_tool

log = get_logger("mcp_graph")

# 元 / 百万 token。全程按空闲时段计价（高峰为两倍）——练习假设，不是事实
PRICE = {"hit":0.02,"miss":1.0,"out":4.0}
MODEL = "deepseek-flash" 


def cost_of(resp)->dict:
    u = resp.usage_metadata or {}
    raw = resp.response_metadata.get("token_usage",{}) or {}

    hit = raw.get("prompt_cache_hit_tokens", 0) or 0
    in_total = u.get("input_tokens", 0) or 0
    miss = raw.get("prompt_cache_miss_tokens", in_total - hit) or 0
    out = u.get("output_tokens", 0) or 0

    cny = round((hit * PRICE["hit"] + miss * PRICE["miss"] + out * PRICE["out"]) / 1e6, 6)
    return {"model": MODEL, "hit": hit, "miss": miss, "out": out, "cny": cny}

class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(tools):
    llm = ChatOpenAI(
        model="deepseek-flash",
        base_url="https://api.deepseek.com/v1",
        api_key=os.environ["DEEPSEEK_API_KEY"],
    )
    llm_with_tools = llm.bind_tools(tools)

    async def llm_node(state: State):
        with timed(log, "node.llm"):
            resp = await llm_with_tools.ainvoke(state["messages"])

        log.info("llm.usage", extra={"fields": cost_of(resp)})
        return {"messages": [resp]}


    def route(state: State):
        return "tools" if state["messages"][-1].tool_calls else END

    b = StateGraph(State)
    b.add_node("llm", llm_node)
    b.add_node("tools", ToolNode(tools))
    b.add_edge(START, "llm")
    b.add_conditional_edges("llm", route)
    b.add_edge("tools", "llm")
    return b.compile()


async def main():
    new_trace()
    params = StdioServerParameters(command=sys.executable, args=["mcp_server.py"])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            tools = [to_lc_tool(session, t) for t in listed.tools]
            print("动态加载:", [t.name for t in tools])

            graph = build_graph(tools)
            result = await graph.ainvoke(
                {"messages": [("user", "先算 37*48，再把结果乘以 2，顺便看看北京天气")]},
                config={"recursion_limit": 10},
            )

            print("\n--- 消息链 ---")
            for m in result["messages"]:
                if type(m).__name__ == "AIMessage":
                    print(f"AIMessage: tool_calls={[tc['name'] for tc in m.tool_calls]}, content={m.content!r}")
                else:
                    print(f"{type(m).__name__}: {m.content!r}")


asyncio.run(main())
