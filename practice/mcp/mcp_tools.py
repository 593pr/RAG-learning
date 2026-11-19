from mcp import ClientSession
from pydantic import Field,create_model
from langchain_core.tools import StructuredTool

from log import get_logger,gettid,timed

log = get_logger("mcp_client")

TYPE_MAP = {"string":str,"integer":int,"number":float,"boolean":bool}

def to_lc_tool(session:ClientSession,t)->StructuredTool:
    schema = t.inputSchema
    required = set(schema.get("required",[]))

    fields = {}
    for name,prop in schema.get("properties",{}).items():
        py_type = TYPE_MAP.get(prop.get("type",str))
        #把json类型转换成python类型

        desc = prop.get("description","")
        if name in required:
            fields[name] = (py_type,Field(...,description=desc))
        else:
            fields[name] = (py_type | None,Field(default=None,description=desc))

    Args = create_model(f"{t.name}_args",**fields)


    async def _run(**kwargs):
        with timed(log,"tool.call",tool=t.name):
                res = await session.call_tool(t.name,kwargs,meta={"trace_id":gettid()})
        text = "\n".join(c.text for c in res.content if getattr(c, "type", None) == "text")
        return f"[工具报错] {text}" if res.isError else text
    #调用的时候就运行_run

    return StructuredTool.from_function(
        coroutine=_run,
        name=t.name,
        description=t.description or "",
        args_schema=Args,
    )
    #把工具包装成Langgraph能调用的样子







"""
① LLM 看到工具列表：name="weather", description="查天气", args_schema=Args
        │
        │  LLM 决定调用，生成参数 {"city": "北京", "days": 3}
        ▼
② LangChain 用 Args 校验参数 → 通过
        │
        │  调用 _run(city="北京", days=3)
        ▼
③ _run 里：kwargs = {"city": "北京", "days": 3}
        │
        │  await session.call_tool("weather", {"city": "北京", "days": 3})
        ▼
④ MCP server 执行，返回 res
        │
        ▼
⑤ 从 res.content 提取文本 → text = "北京今天晴，25度"
        │
        │  res.isError == False，直接返回 text
        ▼
⑥ LLM 拿到 "北京今天晴，25度"，继续生成回答
"""