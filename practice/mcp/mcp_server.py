
from mcp.server.fastmcp import FastMCP,Context

from log import get_logger,timed,bind_trace

mcp = FastMCP("my-first-mcp")
log = get_logger("mcp_server")

@mcp.tool()
def calculator(expression:str,ctx: Context)->str:
    """计算数学表达式，比如1+1=2"""
    bind_trace(getattr(ctx.request_context.meta, "trace_id", None) or "-")
    with timed(log,"tool.exec",tool="calculator"):
        return str(eval(expression))

@mcp.tool()
def weather(city:str,ctx: Context)->str:
    """查询指定城市的天气"""
    bind_trace(getattr(ctx.request_context.meta, "trace_id", None) or "-")
    with timed(log, "tool.exec", tool="weather"):
        return f"{city}今天是晴天，25℃"


if __name__ == "__main__":
    mcp.run()
    