import asyncio

CURRENT = {"tid":None}

async def handle(name,delay):
    CURRENT["tid"] = name
    await asyncio.sleep(delay)
    print(f"{name} 读到的 tid = {CURRENT['tid']}")

async def main():
    await asyncio.gather(handle("A",0.2),handle("B",0.1))

asyncio.run(main())