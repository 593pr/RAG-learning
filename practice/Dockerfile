FROM python:3.12-slim
#slim 是精简版 Debian，没有 gcc、curl 这些

# 不写 .pyc；日志不缓冲（否则 docker logs 要等缓冲满才看得到）
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/models/hf

WORKDIR /app

# torch 需要 OpenMP 运行库，slim 镜像没带。
# Debian 官方源在国内超时，换腾讯云（实测 3.3MB/s）
RUN sed -i 's|deb.debian.org|mirrors.cloud.tencent.com|g' /etc/apt/sources.list.d/debian.sources \
 && apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# 1) 先装 CPU 版 torch。Linux 上装默认版 torch 会拖进 2GB+ 的 nvidia-* 依赖，
#    而这里根本没有 GPU。官方源实测 35KB/s，换腾讯云 wheel 镜像（6.6MB/s）。
RUN pip install --no-cache-dir torch==2.6.0+cpu \
    --index-url https://mirrors.cloud.tencent.com/pytorch-wheels/whl/cpu/ \
    --extra-index-url https://mirrors.cloud.tencent.com/pypi/simple/

# 2) 依赖单独一层：只改代码时这一层直接命中缓存
COPY requirements.txt .
# 国内网络走腾讯云源；官方 PyPI 实测 20s 超时
RUN pip install --no-cache-dir -r requirements.txt \
    --index-url https://mirrors.cloud.tencent.com/pypi/simple/

# 3) 只复制服务代码，不要 COPY .（否则 models/ 1.2GB 全进镜像）
COPY rag_engine ./rag_engine

EXPOSE 8000

# 用 python 探活，省得为了 healthcheck 装 curl
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

# 必须 --host 0.0.0.0：默认 127.0.0.1 只在容器内部回环，端口映射进不来
CMD ["uvicorn", "rag_engine.server:app", "--host", "0.0.0.0", "--port", "8000"]
