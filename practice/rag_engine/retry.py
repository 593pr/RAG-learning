
import logging
import random
import time
from functools import wraps

from .errors import RetryableError

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: float = 0.5,
):
    """瞬时故障自动重试。PermanentError 和其他异常直接抛出，不重试。"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except RetryableError as e:
                    if attempt == max_attempts:
                        # TODO 1：记一条 error 日志（说清是第几次、什么错），然后 raise
                        logger.error(f"这是第{attempt}次尝试最终失败：{e}(status_code={e.status_code})")
                        raise
                    # TODO 2：算 delay（两行：封顶 + jitter）
                    delay = min(base_delay * 2 **(attempt - 1),max_delay)
                    delay = delay + random.uniform(0,jitter)
                    # TODO 3：记一条 warning 日志，说明"第 X 次失败，Y 秒后重试"
                    logger.warning(f"这是第{attempt}次失败，{delay:.2f}秒后重试")
                    time.sleep(delay)
        return wrapper
    return decorator
