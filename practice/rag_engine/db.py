"""W2 D6：Postgres 持久化对话记录。

两张表：
    sessions   一次会话
    messages   每条消息（user / assistant），外键指向会话

连接池在 lifespan 里建一次，请求只借还，不重复建连。
"""
import logging
import os

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

# 逐条执行：psycopg 带参数的查询走扩展协议，一次只能一条语句，拆成列表最不容易踩坑。
# 真项目里这部分交给 Alembic 管版本；今天的 IF NOT EXISTS 够用。
SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id         bigserial PRIMARY KEY,
        created_at timestamptz NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id         bigserial PRIMARY KEY,
        session_id bigint NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        role       text   NOT NULL CHECK (role IN ('user', 'assistant')),
        content    text   NOT NULL,
        sources    int[],
        latency_ms int,
        created_at timestamptz NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_messages_session
        ON messages (session_id, id)
    """,
]

_pool: ConnectionPool | None = None


class SessionNotFound(Exception):
    """客户端传了不存在的 session_id。端点是唯一把它翻成 404 的地方。"""


def init_pool() -> None:
    """启动时调用一次。open=False 是为了能显式 wait。"""
    global _pool
    # os.environ[] 而不是 get()：配置缺失就在启动时大声失败
    _pool = ConnectionPool(os.environ["DATABASE_URL"], min_size=1, max_size=5, open=False)
    _pool.open()
    _pool.wait(timeout=30)   # 等第一条连接真的建立；db 没就绪就在这里阻塞
    logger.info("数据库连接池就绪")


def close_pool() -> None:
    if _pool is not None:
        _pool.close()


def init_schema() -> None:
    with _pool.connection() as conn:
        for stmt in SCHEMA:
            conn.execute(stmt)
    logger.info("表初始化就绪")


def save_turn(session_id, question, answer, sources, latency_ms) -> dict:
    """写一轮问答，返回 {"session_id", "user_message_id", "answer_message_id"}。

    session_id 为 None 就新建会话；给了但不存在则抛 SessionNotFound。
    两条 INSERT 在同一个事务里：要么都有，要么都没有，不会留下「有提问没回答」的残缺记录。
    """
    with _pool.connection() as conn:        # 正常退出自动 commit，抛异常自动 rollback
        if session_id is None:
            session_id = conn.execute(
                "INSERT INTO sessions DEFAULT VALUES RETURNING id"
            ).fetchone()[0]
        else:
            exists = conn.execute(
                "SELECT 1 FROM sessions WHERE id = %s", (session_id,)
            ).fetchone()
            if exists is None:
                raise SessionNotFound(session_id)

        user_id = conn.execute(
            """INSERT INTO messages (session_id, role, content)
               VALUES (%s, 'user', %s) RETURNING id""",
            (session_id, question),
        ).fetchone()[0]

        answer_id = conn.execute(
            """INSERT INTO messages (session_id, role, content, sources, latency_ms)
               VALUES (%s, 'assistant', %s, %s, %s) RETURNING id""",
            (session_id, answer, sources, latency_ms),
        ).fetchone()[0]

    return {
        "session_id": session_id,
        "user_message_id": user_id,
        "answer_message_id": answer_id,
    }


def get_history(session_id, limit=50) -> list[dict]:
    """按写入顺序读一个会话的消息。索引 (session_id, id) 正好覆盖 where + order by。"""
    with _pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT id, role, content, sources, latency_ms, created_at
                   FROM messages
                   WHERE session_id = %s
                   ORDER BY id
                   LIMIT %s""",
                (session_id, limit),
            )
            return cur.fetchall()
