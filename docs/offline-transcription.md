# 离线语音转字幕方案

默认使用 FunASR `SenseVoiceSmall` 在本地转写，不上传音频。第一次运行某个模型时会下载模型权重到本地缓存。

## 配置

```text
PMM_TRANSCRIPTION_BACKEND=funasr-sensevoice
PMM_FUNASR_MODEL=iic/SenseVoiceSmall
PMM_FUNASR_VAD_MODEL=fsmn-vad
PMM_FUNASR_PUNC_MODEL=ct-punc
PMM_FUNASR_DEVICE=cpu
PMM_FUNASR_LANGUAGE=zh
PMM_FUNASR_BATCH_SIZE_S=60
PMM_FUNASR_MERGE_LENGTH_S=15
PMM_OUTPUT_SIMPLIFIED_CHINESE=true
```

`PMM_FUNASR_LANGUAGE=zh` 会强制按中文识别。`PMM_OUTPUT_SIMPLIFIED_CHINESE=true` 会把转写结果统一转成简体中文。`fsmn-vad` 用于长音频切分，`ct-punc` 用于标点和句子级时间段。

## Whisper 回退配置

如果 FunASR 在某些素材上效果不合适，可以切回 Whisper：

```text
PMM_TRANSCRIPTION_BACKEND=local-whisper
PMM_LOCAL_WHISPER_MODEL=small
PMM_LOCAL_WHISPER_DEVICE=cpu
PMM_LOCAL_WHISPER_LANGUAGE=zh
PMM_LOCAL_WHISPER_TASK=transcribe
PMM_OUTPUT_SIMPLIFIED_CHINESE=true
```

## 模型选择

当前推荐先验证 `SenseVoiceSmall`。它比原版 `openai-whisper` 更适合中文离线快速验证，也能配合 VAD 处理长音频。

Whisper 可用模型包括 `small`、`medium`、`large-v3`、`large-v3-turbo` 和 `turbo`。

- `small`：当前默认，速度快，适合 MVP 批量验证。
- `medium`：可以使用，准确率通常比 `small` 好，但 CPU 转写会明显更慢。
- `large-v3-turbo` / `turbo`：更新的高质量候选，通常比完整 `large-v3` 更适合速度和准确率折中，但仍建议先用 10-20 分钟素材实测。
- `large-v3`：质量强，但资源和耗时最高，不建议当前 CPU 配置直接批量跑。

你的机器有 GTX 1650 4GB，但当前 PyTorch 是 CPU 版，所以 `PMM_LOCAL_WHISPER_DEVICE=cuda` 暂时不会生效。要用 GPU，需要安装匹配 CUDA 的 PyTorch；4GB 显存对 `medium`、`large-v3-turbo` 也偏紧，可能仍要回退 CPU。

## 安装

```powershell
.\setup-offline.ps1
```

如果无法从国外下载模型，可先设置镜像或提前把 Whisper 模型缓存到本机。
