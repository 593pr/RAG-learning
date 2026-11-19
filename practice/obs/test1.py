import asyncio  
import contextvars

_tid = contextvars.ContextVar("tid",default=None)

async def handle(name,delay):
    print(f"{name} set 之前读到 = {_tid.get()}")
    await asyncio.sleep(delay)
    _tid.set(name)
    print(f"{name} 读到的 tid = {_tid.get()}")

async def main():
    await asyncio.gather(handle("A",0.2),handle("B",0.1))

asyncio.run(main())