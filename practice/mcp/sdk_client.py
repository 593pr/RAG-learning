
import sys
import asyncio

from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command=sys.executable,args=["mcp_server.py"])

    async with stdio_client(params) as (read,write):
        async with ClientSession(read,write) as session:
            info = await session.initialize()
            print("server:", info.serverInfo.name, info.serverInfo.version)
            print("protocol:", info.protocolVersion)

            listed = await session.list_tools()
            for t in listed.tools:
                print(f"- {t.name}: {t.description}")
                print("  schema:", t.inputSchema)

            res = await session.call_tool("calculator",{"expression":"37*48"})
            print("isError:",res.isError)
            print("content:", [c.text for c in res.content])
            print("structured:", res.structuredContent)

            res2 = await session.call_tool("weather", {"city": "北京"})
            print("weather:", [c.text for c in res2.content])

asyncio.run(main())
