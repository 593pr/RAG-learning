import asyncio
import contextvars

_tid = contextvars.ContextVar("tid",default=None)

def blocking_work():
    return _tid.get()

async def main():
    _tid.set("abc123")
    loop = asyncio.get_running_loop()

    print("run_in_executor:",await loop.run_in_executor(None,blocking_work))
    print("to_thread    :",await asyncio.to_thread(blocking_work))


asyncio.run(main())