"""Day 21：错误处理 + 指数退避重试。"""
import logging
import time

from .errors import EmptyRetrievalError, PermanentError, RetryableError
from .retry import retry_with_backoff

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def check_retryable():
    """连续 3 次 503 → 第 4 次成功。"""
    calls = {"n": 0}
    times = []

    @retry_with_backoff(max_attempts=4, base_delay=0.2, max_delay=5, jitter=0.1)
    def flaky():
        calls["n"] += 1
        times.append(time.time())
        if calls["n"] < 4:
            raise RetryableError("模拟服务不可用", status_code=503)
        return "成功"

    result = flaky()

    assert calls["n"] == 4, f"期望调用 4 次，实际 {calls['n']}"
    assert result == "成功"

    # TODO 1：算相邻两次调用的时间差，打印出来
    # 提示：zip(times, times[1:]) 配对，b - a 就是间隔，round(x, 2) 保留两位
    intervals = [round(b - a, 2) for a, b in zip(times, times[1:])]
#           ↑ 方括号 = 列表推导式，先算出每一项，再交给 round
 
    print(f"[PASS] 重试 {calls['n']} 次成功，实际间隔：{intervals}")


def check_not_retried(name, exc):
    """不可重试的错误：只调用 1 次就往上抛。"""
    calls = {"n": 0}

    @retry_with_backoff(max_attempts=4, base_delay=0.2)
    def always_fail():
        calls["n"] += 1
        raise exc

    try:
        always_fail()
    except Exception:          # 故意兜住，只为验证"抛出来了"
        pass

    assert calls["n"] == 1, f"{name}：期望只调用 1 次，实际 {calls['n']} 次"
    print(f"[PASS] {name}：只调用 1 次，未重试")


if __name__ == "__main__":
    check_retryable()

    # TODO 2：调用 check_not_retried 三次，分别传：
    #   ① PermanentError("key 无效", status_code=401)
    #   ② EmptyRetrievalError("什么是红黑树")
    #   ③ ValueError("代码 bug")   ← 没见过的异常也必须立刻抛
    check_not_retried("PermanentError", PermanentError("key 无效", status_code=401))
    check_not_retried("EmptyRetrievalError", EmptyRetrievalError("什么是红黑树"))
    check_not_retried("ValueError", ValueError("代码 bug"))
    print("\n全部通过")
