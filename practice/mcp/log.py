import contextvars
import json
import logging
import time
from contextlib import contextmanager
#@contextmanager 来自 contextlib，把下面这个生成器函数变成一个可以用 with 的对象。


_tid = contextvars.ContextVar("tid",default="-")

def new_trace():
    import uuid
    tid = uuid.uuid4().hex[:8]
    _tid.set(tid)
    return tid

def gettid():
    return _tid.get()

def bind_trace(value:str)->str:
    """让当前 context 采用外部传来的 trace id"""
    _tid.set(value)
    return value

class JsonFormatter(logging.Formatter):
    def format(self,record):
        entry = {
            "ts":time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
                + f".{int(record.msecs):03d}",
            "level":record.levelname,
            "tid":_tid.get(),
            "event":record.getMessage(),
        }
        entry.update(getattr(record,"fields",{}))
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info).replace("\n"," | ")
        return json.dumps(entry,ensure_ascii=False)
    

def get_logger(name:str,path:str = "run.jsonl")->logging.Logger:
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    lg.propagate = False
    if not lg.handlers:
        h = logging.FileHandler(path,encoding="utf-8")
        h.setFormatter(JsonFormatter())
        lg.addHandler(h)
    return lg

@contextmanager
def timed(log,event,**fields):
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        ms = int((time.perf_counter() - t0) * 1000)
        log.error(f"{event}.error", exc_info=True,
                  extra={"fields": {**fields, "ms": ms}})
        raise
    else:
        ms = int((time.perf_counter() - t0) * 1000)
        log.info(event, extra={"fields": {**fields, "ms": ms}})