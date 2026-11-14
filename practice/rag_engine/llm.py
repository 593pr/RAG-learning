import os
import openai
from openai import OpenAI

from .errors import PermanentError,RAGError,RetryableError
from .retry import retry_with_backoff

BASE_URL = "https://api.deepseek.com"

def _get_client()->OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise PermanentError("未设置DEEPSEEK_API_KEY")
    return OpenAI(
        api_key = api_key,
        base_url = BASE_URL,
        timeout = 30.0,
        max_retries = 0,
    )


def _translate(e:Exception)->RAGError | None:
    """OpenAI SDK 异常 → 自有异常；认不出来的返回 None，由调用方原样抛出。

    顺序有讲究：APITimeoutError 是 APIConnectionError 的子类，
    各种 4xx 是 APIStatusError 的子类 —— 先判具体，最后判兜底。
    """
    if isinstance(e,openai.APITimeoutError):
        return RetryableError(f"请求超时：{e}")
    if isinstance(e,openai.APIConnectionError):
        return RetryableError(f"连接失败：{e}")
    if isinstance(e,openai.RateLimitError):
        return RetryableError(f"限流：{e}",status_code=429)
    if isinstance(e,openai.InternalServerError):
        return RetryableError(f"服务端错误：{e}",status_code=e.status_code)
    if isinstance(e,openai.AuthenticationError):
        return PermanentError(f"鉴权失败：{e}",status_code=401)
    if isinstance(e,openai.PermissionDeniedError):
        return PermanentError(f"无权限：{e}",status_code=403)
    if isinstance(e,openai.NotFoundError):
        return PermanentError(f"资源不存在：{e}",status_code=404)
    if isinstance(e,openai.BadRequestError):
        return PermanentError(f"请求参数错误：{e}",status_code=400)
    if isinstance(e,openai.APIStatusError):
        if e.status_code >= 500:
            return RetryableError(f"HTTP {e.status_code}：{e}",status_code=e.status_code)
        return PermanentError(f"HTTP {e.status_code}：{e}",status_code=e.status_code)
    return None


@retry_with_backoff(max_attempts=3,base_delay=1.0,max_delay=10.0)
def chat(messages:list[dict],model:str = "deepseek-chat",temperature:float = 0.0)->str:
    client  = _get_client()
    try:
        resp = client.chat.completions.create(
            model = model,
            messages = messages,
            temperature = temperature,
        )
        return resp.choices[0].message.content
    except Exception as e:
        err = _translate(e)
        if err is None:
            raise
        raise err from e


@retry_with_backoff(max_attempts=3,base_delay=1.0,max_delay=10.0)
def _open_stream(messages:list[dict],model:str,temperature:float):
    """只负责「建立流」：此刻还没有任何 token 发出去，重试是安全的。"""
    client = _get_client()
    try:
        return client.chat.completions.create(
            model = model,
            messages = messages,
            temperature = temperature,
            stream = True,
        )
    except Exception as e:
        err = _translate(e)
        if err is None:
            raise
        raise err from e


def chat_stream(messages:list[dict],model:str = "deepseek-chat",temperature:float = 0.0):
    """逐 token 产出答案。

    注意：这个函数没有加 @retry_with_backoff —— 装饰器只能重试「造生成器」，
    而生成器函数在调用时并不执行。重试放在 _open_stream（还没吐字），
    循环里的异常直接抛：客户端已经收到半句话，重连只会得到重复内容。
    """
    stream = _open_stream(messages,model,temperature)
    try:
        for chunk in stream:
            if not chunk.choices:
                continue  # 有些兼容实现会额外发一条只带 usage 的尾块，choices 是空的
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        err = _translate(e)
        if err is None:
            raise
        raise err from e
    