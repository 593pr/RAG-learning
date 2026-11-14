class RAGError(Exception):
    
    """RAG模块所有的异常的基类，便于调用方用一条except兜住"""


class RetryableError(RAGError):
    """瞬时故障，稍后重试可能成功：超时，429限流，500/502/503/504"""
    def __init__(self, message:str,status_code:int | None = None):
        super().__init__(message)
        self.status_code = status_code

class PermanentError(RAGError):
    """永久故障，不可能成功：400/401/文件不存在"""
    def __init__(self, message:str,status_code:int | None = None):

        super().__init__(message)
        self.status_code = status_code
#加了status_code因为有状态码能更准确的定位错误，便于纠正



class EmptyRetrievalError(RAGError):
    """检索结果为空时抛出"""

    def __init__(self, query:str):
        super().__init__(f"检索不到与[{query}]相关的内容")
        self.query = query