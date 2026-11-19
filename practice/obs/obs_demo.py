import asyncio

from log import get_logger, new_trace, timed

log = get_logger("obs")


async def handle(q: str, delay: float):
    new_trace()
    with timed(log, "request", q=q):
        await asyncio.sleep(delay)


async def main():
    new_trace()                                   # 入口
    await asyncio.gather(handle("问题A", 0.2), handle("问题B", 0.1))

    with timed(log, "step.ok"):
        await asyncio.sleep(0.15)

    try:
        with timed(log, "step.boom"):
            await asyncio.sleep(0.05)
            raise ValueError("故意失败")
    except ValueError:
        pass


asyncio.run(main())

