
from typing import TypedDict,Annotated
from langgraph.graph import StateGraph,START,END

import operator

class State(TypedDict):
    count:int
    trace:Annotated[list[str],operator.add]

def node_a(state : State):
    print(f"A 收到的 count = {state['count']}")
    return {"count": state["count"] + 1,
            "trace": ["a"]}

def node_b(state : State):
    print(f"B 收到的 count = {state['count']}")
    return {"count": state["count"] + 10,
            "trace":["b"]}
#只返回要更新的字段

builder = StateGraph(State)

builder.add_node("a",node_a)
builder.add_node("b",node_b)

builder.add_edge(START,"a")
builder.add_edge("a","b")
builder.add_edge("b",END)


#编译：检查图结构，返回一个可执行对象
graph = builder.compile()

result = graph.invoke({"count" : 0,"trace" : []})
print("最终结果:",result)
