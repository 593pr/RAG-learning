# 从零手写 RAG：召回 → 混合 → 重排

> 不用框架，实现完整的 RAG 检索链路：切块 → 语义 / BM25 混合召回 → cross-encoder 精排。
> 11 题测试集上，recall@3 从 2/11 提升到 11/11。

全程用 Python 实现，目标是理解 RAG 链路的结构和基础原理，而不是调用现成框架。核心是 `RAG1`–`RAG9` 九个递进脚本，每个环节都有可复现的对比实验。

## 核心结果

| 指标 | 结果 |
|------|------|
| 英文 embedding（all-MiniLM-L6-v2） | recall@3 = 2/11 |
| 中文 embedding（bge-small-zh-v1.5） | recall@3 = 9/11 |
| 混合检索（BM25 + 语义，分数融合） | recall@5 = 11/11 |
| 一阶段召回天花板（top-20） | recall@20 = 11/11 |
| rerank（bge-reranker-base，top-20 → 精排） | recall@1 8/11 → 10/11；recall@3 9/11 → 11/11 |

> 测试集 11 题，答案关键词均已在原文验证。语料：hello-algo 1.3.0（361 页 / 216,183 字 / 1854 chunks，chunk_size=200，overlap=50）。

## 检索链路

```text
离线（建库，只跑一次）
  PDF → 按页切块 → bge-small-zh 编码 → ChromaDB
                    ↘ jieba 分词 → BM25 索引

在线（每次查询）
  query ─┬→ 语义召回 top-20 ┐
         └→ BM25 召回 top-20 ┴→ min-max 分数融合 → 候选 top-20
                                              → cross-encoder 精排
                                              → top-k chunk
                                              → 拼 prompt（带页码）
                                              → DeepSeek 生成答案
```

**离线 vs 在线的区别不是"有没有网络"**，而是一次性预处理 vs 每次查询都要跑。模型都在本地加载，网络只用在两处：首次下载模型、最后调用 DeepSeek API。

**top-20 是二阶段的候选深度**：一阶段故意多召回，给精排留纠错空间。但候选不是越多越好——见「已知限制」。

> `RAG6`–`RAG9` 只做检索评估，不调用 LLM；拼 prompt 和生成答案见 `RAG4` / `RAG5`。

## 三个关键决策

### 1. 换中文 embedding（recall@3：2/11 → 9/11）

同一份语料、同一套题、同一种切分，只换模型。诊断方法是打印正确答案在全部 1854 个 chunk 里的名次——英文模型下是 53、76、167、972……，说明不是 top_k 太小，是整体排序错了。

### 2. 混合检索用分数融合，不用 RRF（recall@5：10/11 → 11/11）

BM25 补上语义漏的「负载因子」，语义补上 BM25 漏的「状态转移方程」。但 RRF 只数名次：全文有 18 个 chunk 含「记忆化搜索」，两路各投不同副本，票被分散。改成 min-max 归一化后按分数加权，保留分差，k=5 达到 11/11。

### 3. 加 cross-encoder 重排（recall@1：8/11 → 10/11，recall@3：9/11 → 11/11）

一阶段 top-20 的召回已经 11/11，说明瓶颈在排序而不是召回。用 bge-reranker-base 对候选逐对打分，@3 打到天花板。

但深度不是越大越好：depth=10 是 11/11，depth=20/50 反而 10/11。一段 DP 代码碎片靠 `dp` / `mem` 的字面重叠骗过 reranker（0.9007 vs 正确段落的 0.8247）。

## 如何运行

### 环境

```bash
pip install sentence-transformers pypdf chromadb rank-bm25 jieba openai numpy
```

Python 3.13 / Windows 测试通过。

### 数据

从 [krahets/hello-algo 1.3.0](https://github.com/krahets/hello-algo/releases/tag/1.3.0) Releases 下载 `hello-algo_1.3.0_zh_go.pdf`，重命名为 `hello-algo.pdf`，放到 `practice/data/`。

> 必须用 **Go 版**：不同语言版本的代码示例不同，切块数量（1854）和评测数字都会变。

`data/` 和 `models/` 都已在 `.gitignore` 中，不进仓库。

### 模型

embedding 模型首次运行时自动下载（脚本已设置 hf-mirror 镜像）。

reranker 约 1 GB，需要手动下载：

```bash
python -c "from modelscope.hub.snapshot_download import snapshot_download; snapshot_download('BAAI/bge-reranker-base', local_dir='practice/models/bge-reranker-base', ignore_patterns=['onnx/*', '*.bin'])"
```

### 运行

```bash
python practice/RAG9.rerank.py
```

脚本用 `Path(__file__)` 定位文件，不依赖固定路径，可以在任意目录执行。

`RAG4` / `RAG5` 需要调用 DeepSeek API：

```bash
# Windows
set DEEPSEEK_API_KEY=your_key_here

# Linux / macOS
export DEEPSEEK_API_KEY=your_key_here
```

## 文件导航

| 文件 | 内容 |
|------|------|
| `RAG1_cosine_similarity.py` | 手写余弦相似度，验证 embedding 的语义距离 |
| `RAG2_chromaDB_demo.py` | ChromaDB 存取 + 语义检索 |
| `RAG3_pdfreader.py` | pypdf 读 PDF + 按页切块 |
| `RAG4.rag_use.py` | 完整 RAG 流水线（load → chunk → embed → retrieve → LLM） |
| `RAG5.citation.py` | 来源标注：页码跟着 chunk 走完整条流水线 |
| `RAG6.test_chunksize.py` | chunk_size × top_k 评估脚本 |
| `RAG7.topk.py` | 11 题测试集 + 检索评估 |
| `RAG8_hybrid.py` | 语义 + BM25 混合召回（RRF / 分数融合） |
| `RAG9.rerank.py` | cross-encoder 二阶段重排 |
| `experiment_results.md` | 全部实验记录与结论 |

`practice/` 下另有早期 Python 练习脚本和 `weather-cli`，不是本项目的核心内容。

## 已知限制

- 测试集只有 11 题，1 题就是 9 个百分点，±1 不能当作提升。
- rerank 的候选深度没有在独立验证集上调过。
- 语料里有 18 个重复 chunk，没有做去重；RRF 分票的根因在这里。
- 只评估了检索层（recall@k），没有评估 LLM 答案正确率。
- PDF 提取会把部分汉字映射成康熙部首字符，评估前做了空白归一化。
- `RAG4` / `RAG5` 使用的是另一份实验 PDF，路径需要自行修改。

## 相关仓库

- [agent_loop](https://github.com/593pr/agent_loop)：手写 Function Calling Agent（工具注册、调度、记忆、循环）
