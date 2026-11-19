import os
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

from tool_lc import tools

class State(TypedDict):
    messages: Annotated[list,add_messages]

llm = ChatOpenAI(
    model = "deepseek-chat",
    base_url = "https://api.deepseek.com/v1",
    api_key = os.environ["DEEPSEEK_API_KEY"],
)

llm_with_tools = llm.bind_tools(tools)

def llm_node(state:State):
    response = llm_with_tools.invoke(state["messages"])
    print(f"[llm] tool_calls = {[tc['name'] for tc in response.tool_calls]}")
    return {"messages":[response]}

def route(state:State):
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return END
    
builder = StateGraph(State)

builder.add_node("llm",llm_node)
builder.add_node("tools",ToolNode(tools))

builder.add_edge(START,"llm")
builder.add_conditional_edges("llm",route)
builder.add_edge("tools","llm")

graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke(
        {"messages": [("user", "先算 37*48，再把结果乘以 2 告诉我")]},
        config={"recursion_limit": 10},
    )

    print("\n--- 消息链 ---")
    for m in result["messages"]:
        name = type(m).__name__
        if name == "AIMessage":
            print(f"{name}: tool_calls={[tc['name'] for tc in m.tool_calls]}, content={m.content!r}")
        else:
            print(f"{name}: content={m.content!r}")



