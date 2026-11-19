
import os
from typing import TypedDict,Annotated
from langgraph.graph import StateGraph , START , END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

class State(TypedDict):
    messages : Annotated[list,add_messages]

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key= os.environ["DEEPSEEK_API_KEY"],
)

def chat(state:State):
    response = llm.invoke(state["messages"])
    return {"messages":[response]}


builder = StateGraph(State)

builder.add_node("chat",chat)

builder.add_edge(START,"chat")
builder.add_edge("chat",END)

graph = builder.compile()

result = graph.invoke({"messages" : [("user","你好")]})

print(result)



