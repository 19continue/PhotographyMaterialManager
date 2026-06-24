# 本地处理 + 国内大模型智能搜索方案

最终分工：

- 语音转写：本地 `openai-whisper`，不上传音频。
- 语义召回：本地 `sentence-transformers` + `BAAI/bge-small-zh-v1.5`，不上传字幕向量化请求。
- 素材索引：当前 MVP 使用 SQLite/FTS/本地向量；大规模目标使用 Milvus。
- 智能搜索：国内 OpenAI-compatible 大模型 API，只接收用户问题和已召回的少量候选字幕片段。

## 配置

默认 `.env`：

```text
PMM_TRANSCRIPTION_BACKEND=local-whisper
PMM_EMBEDDING_BACKEND=local
PMM_LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
PMM_ENABLE_ASSISTANT=true
PMM_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
PMM_LLM_API_KEY=
PMM_LLM_MODEL=deepseek-v4-flash
```

`PMM_LLM_BASE_URL` 保持 DashScope OpenAI-compatible 地址不变。`PMM_LLM_MODEL` 当前按你的方案使用 `deepseek-v4-flash`。填入对应厂商 Key 后，“智能搜索”按钮会启用。

## 数据流

1. 本地 Whisper 从视频抽音频并转成字幕。
2. 本地 embedding 模型把字幕片段和前后文转成向量。
3. 用户输入自然语言描述。
4. 系统用本地字幕索引和本地向量索引召回候选。
5. 本地 reranker 对候选做二次排序。
6. 国内大模型只读取候选字幕、文件名和时间码，做扩写、重排、解释和总结。

## 隐私边界

不会上传：

- 原视频
- 抽取音频
- 本地向量
- 全量字幕库

会上传给国内大模型：

- 用户当前问题
- 本地召回的少量候选字幕
- 候选素材文件名和时间码

如果要完全离线运行，保持 `PMM_LLM_API_KEY` 为空即可；此时智能搜索会回退到本地检索。

## 模型选择

- 本地 embedding 默认 `BAAI/bge-small-zh-v1.5`：适合当前 CPU MVP。
- 可升级 `BAAI/bge-m3`：语义能力更强，但模型更大、CPU 更慢。
- 本地 reranker 默认 `BAAI/bge-reranker-base`：只重排前几十条候选，提升准确率。
- 国内大模型使用 `deepseek-v4-flash`：负责智能搜索理解、扩写、整理候选。

## 大规模检索

当前 SQLite 向量扫描只适合 MVP。素材量增大后，字幕向量迁移到 Milvus：

- Milvus 负责 topK 向量召回。
- PostgreSQL 保存项目、素材、转写任务和权限元数据。
- OpenSearch / Elasticsearch 保存字幕全文索引。
- 大模型仍只看最终候选，不直接访问全量素材库。
