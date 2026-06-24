# 大规模素材库架构建议

当前 MVP 使用 SQLite 保存字幕、全文索引和本地向量，适合低成本快速验证。当素材增长到几十 TB 到几百 TB 后，应把“原始素材存储”和“可检索索引”分层，向量库目标方案使用 Milvus。

## 目标架构

- 原始素材层：NAS / 对象存储 / MinIO / S3 只保存原视频，应用不复制 0.5-1T/天的素材。
- 元数据层：PostgreSQL 保存素材、项目、拍摄日期、人员、路径、时长、状态、权限和处理任务。
- 全文检索层：OpenSearch / Elasticsearch 保存字幕全文、分词、过滤字段和高亮。
- 语义检索层：Milvus 保存字幕片段向量，用 HNSW/IVF 等 ANN 索引提升大规模召回速度。
- 处理队列：Celery/RQ/Arq + Redis，把抽音频、转写、向量化、代理文件生成拆成异步任务。
- 预览层：生成低码率 proxy 视频和缩略图，网页播放 proxy，剪辑导出时再回到原素材路径。

## 检索范围

当前方案聚焦对白和音频转写检索：

- 支持：字幕精确检索、字幕全文检索、字幕语义检索、智能搜索重排和整理。
- 暂不纳入：字幕和音频转写以外的扩展索引链路。

原因是这些能力会引入新的处理链路和模型依赖，不适合当前验证阶段。后续如要扩展，再单独设计，不混进当前字幕检索方案。

## 检索性能原则

- 查询先按项目、日期、拍摄组、人员、素材类型过滤，再做全文或向量召回。
- 字幕不要按整条视频建索引，应按 5-30 秒上下文窗口建索引，返回时映射回原视频秒数。
- Milvus 负责大规模向量 topK 召回，本地 reranker 只重排前 50-100 条候选。
- 国内大模型只接收最终候选，不读取全量素材库，因此大模型响应速度不随素材总量线性增长。
- 热门项目保留本地高速 SSD 缓存；冷素材只保留元数据、字幕、向量和 proxy。
- 入库采用增量策略：文件路径、大小、mtime 或 hash 未变化时不重复转写。

## Milvus 集合建议

第一阶段只需要一个核心集合：

```text
collection: transcript_segments

vector:
  embedding: FloatVector

scalar fields:
  segment_id
  media_id
  project_id
  filename
  path
  start_seconds
  end_seconds
  text
  transcript_model
  embedding_model
  shoot_date
  created_at
```

推荐索引：

- 小规模验证：HNSW，调参简单，召回质量稳定。
- 更大规模：IVF_FLAT / IVF_PQ，按数据量、内存和召回要求调参。

查询流程：

1. 用户输入自然语言。
2. 本地 embedding 生成查询向量。
3. PostgreSQL / Milvus scalar filter 先限制项目、日期、素材范围。
4. Milvus 返回 topK 字幕候选。
5. 本地 reranker 重排候选。
6. 国内大模型只整理最终候选，返回文件名和时间段。

## 演进路径

1. 2 小时样本：当前 SQLite MVP 验证“描述式搜索能找到素材”。
2. 1 天素材：验证转写速度、语义召回质量和剪辑师工作流。
3. 1-5 TB：保留 SQLite 应用，但增加项目/日期过滤和 proxy 预览。
4. 10 TB 以上：迁移 PostgreSQL + Milvus + OpenSearch，后台任务队列化。
5. 100 TB 以上：NAS/对象存储 + proxy 预览 + 分布式转写/向量化，按项目分区和生命周期管理。
