# Photography Material Manager

## 演示视频

[查看 6月9日演示视频](https://github.com/19continue/PhotographyMaterialManager/releases/download/demo-2026-06-09/demo-2026-06-09.mp4)

本项目是视频素材对白检索助手的低成本 MVP。素材保留在本机或移动硬盘，系统只保存文件路径、压缩音频分块和转写索引。第一版支持按对白搜索，返回素材文件、时间范围，并在网页里跳秒预览。

## 功能

- 扫描本地视频目录，支持 `mp4/mov/mxf/mkv/avi/webm` 等常见格式。
- 使用 `ffmpeg` 抽取 16kHz 单声道低码率音频，并按默认 15 分钟分块。
- 默认使用 FunASR SenseVoiceSmall 离线转写，保存字幕段落、时间戳和 SQLite FTS 索引。
- 默认使用本地 embedding 生成语义向量，支持用自然语言描述找对白素材。
- 可接国内 OpenAI-compatible 大模型做智能搜索理解、重排和总结。
- 可切换回本地开源 Whisper 后端，离线完成音频转文字。
- 本地网页提供入库队列、运行状态、对白搜索、视频预览和 CSV 导出。

## 环境准备

1. 安装 Python 3.9 或更高版本。
2. 安装 ffmpeg，并确保 `ffmpeg` 和 `ffprobe` 在命令行可用。
3. 创建 `.env`：

```powershell
Copy-Item .env.example .env
notepad .env
```

语音转写默认走离线 FunASR，不需要 OpenAI Key。`.env` 中可保留：

```text
PMM_TRANSCRIPTION_BACKEND=funasr-sensevoice
PMM_FUNASR_MODEL=iic/SenseVoiceSmall
PMM_FUNASR_VAD_MODEL=fsmn-vad
PMM_FUNASR_PUNC_MODEL=ct-punc
PMM_FUNASR_DEVICE=cpu
PMM_FUNASR_LANGUAGE=zh
PMM_OUTPUT_SIMPLIFIED_CHINESE=true
```

如果要使用智能搜索，再填入国内大模型 Key，例如 `PMM_LLM_API_KEY`。语音和语义向量仍在本地处理。

如果 ffmpeg 没有加入 PATH，可在 `.env` 里指定：

```text
PMM_FFMPEG_BIN=C:\path\to\ffmpeg.exe
PMM_FFPROBE_BIN=C:\path\to\ffprobe.exe
```

## 智能检索

默认开启本地语义索引：

```text
PMM_ENABLE_EMBEDDINGS=true
PMM_EMBEDDING_BACKEND=local
PMM_LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

入库完成后，系统会用本地模型把每段字幕写入向量索引。搜索时不是只匹配关键词，而是混合三类结果：

- `字幕精确`：原文里直接包含这句话。
- `语义`：用户描述和字幕意思接近，即使关键词不同也可能命中。
- `关键词`：SQLite FTS 兜底召回。

如果已有字幕但没有语义索引，在网页点击“补齐语义索引”。

## 离线转写

默认离线方案使用 FunASR SenseVoiceSmall 在本机运行。切回 Whisper 的方式：

```text
PMM_TRANSCRIPTION_BACKEND=local-whisper
PMM_LOCAL_WHISPER_MODEL=small
PMM_LOCAL_WHISPER_DEVICE=cpu
```

并安装离线转写依赖：

```powershell
.\setup-offline.ps1
```

FunASR 和 Whisper 首次使用都需要模型权重；如果机器完全离线，需要提前下载并缓存模型。当前默认使用 CPU，后续可按机器环境再评估 GPU。

## 启动

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

打开：

```text
http://127.0.0.1:8000
```

## 首轮验证流程

1. 选择约 2 小时代表素材放在同一个目录中。
2. 在网页左侧输入素材目录，点击“扫描并转写”。
3. 等队列状态变为 `done` 后，在顶部搜索框输入对白。
4. 点击结果，确认视频能跳到对应素材时间。
5. 点击“导出 CSV”，给剪辑师核对素材名、起止秒和字幕文本。

## 成本控制

默认分块参数：

- `PMM_CHUNK_SECONDS=900`
- `PMM_CHUNK_OVERLAP_SECONDS=5`
- `PMM_AUDIO_BITRATE=32k`
