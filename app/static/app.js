const state = {
  lastQuery: "",
  activeSegmentId: null,
  ingestMode: "directory",
  selectedPaths: [],
  health: null,
  startup: null,
  smartStreamAnswer: "",
  smartStreamTerms: [],
};

const $ = (selector) => document.querySelector(selector);

const healthList = $("#healthList");
const progressStats = $("#progressStats");
const currentProgress = $("#currentProgress");
const taskWindow = $("#taskWindow");
const taskSummary = $("#taskSummary");
const taskToggleBtn = $("#taskToggleBtn");
const resultList = $("#resultList");
const resultCount = $("#resultCount");
const searchStatus = $("#searchStatus");
const mediaList = $("#mediaList");
const mediaCount = $("#mediaCount");
const video = $("#videoPreview");
const videoPlaceholder = $("#videoPlaceholder");
const selectedMeta = $("#selectedMeta");
const selectedTime = $("#selectedTime");
const exportLink = $("#exportLink");
const assistantAnswer = $("#assistantAnswer");
const assistantBtn = $("#assistantBtn");
const directoryInput = $("#directoryInput");
const limitInput = $("#limitInput");
const ingestStatus = $("#ingestStatus");
const selectedFiles = $("#selectedFiles");
const pickDirectoryBtn = $("#pickDirectoryBtn");
const pickFilesBtn = $("#pickFilesBtn");
const startupPanel = $("#startupPanel");
const startupToggleBtn = $("#startupToggleBtn");
const dependencySummary = $("#dependencySummary");
const dependencyStatusText = $("#dependencyStatusText");
const dependencyDetail = $("#dependencyDetail");
const dependencyProgress = $("#dependencyProgress");
const dependencyLog = $("#dependencyLog");
const dependencyPath = $("#dependencyPath");

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      if (Array.isArray(payload.detail)) {
        message = payload.detail.map((item) => item.msg || String(item)).join("；") || message;
      } else {
        message = payload.detail || message;
      }
    } catch {
      // Keep the HTTP message.
    }
    throw new Error(message);
  }
  return response.json();
}

function statusPill(ok) {
  return statusPillText(ok ? "可用" : "缺失", ok ? "" : "bad");
}

function statusPillText(label, tone = "") {
  return `<span class="status-pill ${tone}">${escapeHtml(label)}</span>`;
}

function aiDependencyPill(status) {
  const value = String(status || "unknown").toLowerCase();
  if (value === "ready") return statusPillText("已就绪");
  if (value === "installing") return statusPillText("安装中", "warn");
  if (value === "checking") return statusPillText("自检中", "warn");
  if (value === "skipped") return statusPillText("已跳过", "warn");
  if (value === "failed") return statusPillText("失败", "bad");
  if (value === "path-too-long") return statusPillText("路径过长", "bad");
  if (value === "missing-requirements") return statusPillText("缺清单", "bad");
  return statusPillText("待检查", "warn");
}

function dependencyStatusMeta(status) {
  const value = String(status || "unknown").toLowerCase();
  if (value === "ready") {
    return {
      label: "已就绪",
      tone: "",
      detail: "本地语音识别、语义检索和模型自检已完成。",
      visible: false,
    };
  }
  if (value === "checking") {
    return {
      label: "自检中",
      tone: "warn",
      detail: "正在检查本地语音识别、语义检索模型。完成前暂不能入库处理和搜索。",
      visible: true,
    };
  }
  if (value === "installing") {
    return {
      label: "安装中",
      tone: "warn",
      detail: "正在下载和安装本地 AI 依赖。完成前暂不能入库处理和搜索。",
      visible: true,
    };
  }
  if (value === "failed") {
    return {
      label: "失败",
      tone: "bad",
      detail: "首次环境准备失败。请检查网络后重新双击启动，程序会再次尝试；详细原因见本地日志文件。",
      visible: true,
    };
  }
  if (value === "path-too-long") {
    return {
      label: "路径过长",
      tone: "bad",
      detail: "当前解压路径过长，可能导致依赖或模型下载失败。请先关闭程序，把整个文件夹移动到 C:\\PMM 或 E:\\PMM 后重新启动。",
      visible: true,
    };
  }
  if (value === "skipped") {
    return {
      label: "已跳过",
      tone: "warn",
      detail: "当前跳过了 AI 依赖安装，转写和本地语义检索暂不可用。",
      visible: true,
    };
  }
  if (value === "missing-requirements") {
    return {
      label: "缺清单",
      tone: "bad",
      detail: "缺少 requirements-ai.txt，无法自动安装 AI 依赖。",
      visible: true,
    };
  }
  return {
    label: "待检查",
    tone: "warn",
    detail: "正在等待启动器写入首次环境准备状态。完成前暂不能入库处理和搜索。",
    visible: true,
  };
}

function runtimeStatus(payload = state.startup) {
  const progress = payload?.ai_dependency_progress || {};
  return String(progress.status || payload?.ai_dependency_status || "unknown").toLowerCase();
}

function runtimeReady(payload = state.startup) {
  return runtimeStatus(payload) === "ready";
}

function runtimeBlockMessage(payload = state.startup) {
  const status = runtimeStatus(payload);
  const meta = dependencyStatusMeta(status);
  return `首次环境准备未完成：${meta.detail}`;
}

function updateRuntimeGate(payload = state.startup) {
  const ready = runtimeReady(payload);
  const message = ready ? "" : runtimeBlockMessage(payload);
  document.querySelectorAll("[data-runtime-required]").forEach((control) => {
    if ("disabled" in control) {
      control.disabled = !ready;
    } else {
      control.classList.toggle("disabled", !ready);
      control.setAttribute("aria-disabled", ready ? "false" : "true");
      control.tabIndex = ready ? 0 : -1;
    }
    if (message) {
      control.title = message;
    } else {
      control.removeAttribute("title");
    }
  });
}

function ensureRuntimeReady(target = "search") {
  if (runtimeReady()) return true;
  const message = runtimeBlockMessage();
  if (startupPanel) startupPanel.classList.remove("hidden");
  if (target === "ingest") {
    setIngestStatus(message, "bad");
  } else {
    setSearchStatus(message, { error: true });
  }
  return false;
}

function friendlyProgressMessage(progress, meta) {
  const status = String(progress.status || "").toLowerCase();
  const stage = String(progress.stage || status || "").toLowerCase();
  const packageName = progress.package || "";
  if (status === "ready") return meta.detail;
  if (status === "failed") return meta.detail;
  if (status === "path-too-long") return meta.detail;
  if (status === "missing-requirements" || status === "skipped") return meta.detail;
  if (stage === "dependencies-ready") return "依赖已安装，正在加载本地模型做自检。";
  if (stage === "speech-model") return "正在加载离线语音识别模型，首次运行会下载模型文件。";
  if (stage === "embedding-model") return "正在加载本地语义检索模型，首次运行会下载模型文件。";
  if (stage === "reranker-model") return "正在检查本地重排模型。";
  if (stage === "installing" || stage === "downloading") {
    return packageName ? `正在安装依赖：${packageName}` : meta.detail;
  }
  if (stage === "prepare") return "正在准备安装本地 AI 依赖。";
  return meta.detail;
}

function renderDependencyProgress(progress, payload) {
  if (!dependencyProgress) return;
  const percent = Math.max(0, Math.min(100, Number(progress.percent || 0)));
  const total = Number(progress.total || 0);
  const current = Number(progress.current || 0);
  const packageName = progress.package || "--";
  const status = String(progress.status || payload.ai_dependency_status || "unknown").toLowerCase();
  const meta = dependencyStatusMeta(status);
  const running = status === "installing" || status === "checking";
  const failed = status === "failed" || status === "missing-requirements" || status === "path-too-long";
  const message = friendlyProgressMessage(progress, meta);
  const installDir = payload.dependency_install_dir || progress.dependency_install_dir || "--";
  const cacheDir = payload.model_cache_dir || progress.model_cache_dir || "--";
  dependencyProgress.innerHTML = `
    <div class="progress-card dependency-card ${running ? "running" : ""} ${failed ? "failed" : ""}">
      <div class="progress-stage-row">
        <span class="progress-stage-pill">${escapeHtml(stageLabel(progress.stage || status))}</span>
        <strong class="progress-percent">${formatPercent(percent)}</strong>
      </div>
      <div class="progress-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${percent.toFixed(0)}">
        <span class="progress-bar-fill" style="width: ${percent}%"></span>
      </div>
      <div class="progress-message">${escapeHtml(message)}</div>
      <div class="progress-detail-grid dependency-detail-grid">
        <div><span>当前步骤</span><strong>${escapeHtml(stageLabel(progress.stage || status))}</strong></div>
        <div><span>步骤进度</span><strong>${total ? `${current} / ${total}` : "--"}</strong></div>
        <div><span>依赖目录</span><strong title="${escapeHtml(installDir)}">${escapeHtml(installDir)}</strong></div>
        <div><span>模型缓存</span><strong title="${escapeHtml(cacheDir)}">${escapeHtml(cacheDir)}</strong></div>
      </div>
    </div>
  `;
}

function formatSeconds(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) return "--:--";
  const value = Math.max(0, Math.floor(Number(seconds)));
  const h = Math.floor(value / 3600);
  const m = Math.floor((value % 3600) / 60);
  const s = value % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function formatSize(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = Number(bytes);
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(value >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatPercent(value) {
  const percent = Math.max(0, Math.min(100, Number(value || 0)));
  return `${percent.toFixed(percent >= 10 || percent === 0 ? 0 : 1)}%`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setIngestStatus(message, tone = "") {
  if (!ingestStatus) return;
  ingestStatus.textContent = message;
  ingestStatus.className = `muted ${tone}`.trim();
}

function setSearchStatus(message, options = {}) {
  if (!searchStatus) return;
  const { busy = false, error = false } = options;
  if (!message) {
    searchStatus.className = "work-status hidden";
    searchStatus.innerHTML = "";
    return;
  }
  searchStatus.className = `work-status ${busy ? "busy" : ""} ${error ? "bad" : ""}`.trim();
  searchStatus.innerHTML = `${busy ? '<span class="status-dot"></span>' : ""}<span>${escapeHtml(message)}</span>`;
}

function renderLoading(message) {
  resultCount.textContent = "处理中";
  resultList.className = "result-list empty loading";
  resultList.innerHTML = `
    <div class="inline-loader"></div>
    <p>${escapeHtml(message)}</p>
  `;
}

function renderSelectedFiles(paths) {
  if (!selectedFiles) return;
  if (!paths.length) {
    selectedFiles.classList.add("hidden");
    selectedFiles.innerHTML = "";
    return;
  }
  selectedFiles.classList.remove("hidden");
  const visible = paths.slice(0, 4);
  const extra = paths.length - visible.length;
  selectedFiles.innerHTML = `
    ${visible.map((path) => `<div>${escapeHtml(path)}</div>`).join("")}
    ${extra > 0 ? `<div>另有 ${extra} 个文件</div>` : ""}
  `;
}

function showTaskWindow() {
  if (!taskWindow) return;
  taskWindow.classList.remove("minimized");
  localStorage.setItem("pmmTaskMinimized", "0");
  if (taskToggleBtn) taskToggleBtn.textContent = "缩小";
}

async function refreshStartupStatus() {
  if (!startupPanel) return null;
  const payload = await api("/api/startup-status");
  state.startup = payload;
  const progress = payload.ai_dependency_progress || {};
  const status = progress.status || payload.ai_dependency_status;
  const meta = dependencyStatusMeta(status);
  updateRuntimeGate(payload);
  startupPanel.classList.toggle("hidden", !meta.visible);
  startupPanel.classList.toggle("active", status === "installing" || status === "checking");
  if (meta.visible && localStorage.getItem("pmmStartupMinimized") === "1") {
    startupPanel.classList.add("minimized");
  }
  if (startupToggleBtn) startupToggleBtn.textContent = startupPanel.classList.contains("minimized") ? "展开" : "缩小";
  if (dependencySummary) {
    const current = Number(progress.current || 0);
    const total = Number(progress.total || 0);
    const suffix = total ? ` · ${current}/${total}` : "";
    dependencySummary.textContent = `${meta.label} ${formatPercent(progress.percent || 0)}${suffix}`;
  }
  if (dependencyStatusText) {
    dependencyStatusText.className = `status-pill ${meta.tone}`.trim();
    dependencyStatusText.textContent = meta.label;
  }
  if (dependencyDetail) dependencyDetail.textContent = friendlyProgressMessage(progress, meta);
  renderDependencyProgress(progress, payload);
  if (dependencyLog) {
    dependencyLog.innerHTML = "";
    dependencyLog.style.display = "none";
  }
  if (dependencyPath) {
    dependencyPath.innerHTML = [
      payload.log_path ? `启动日志：${escapeHtml(payload.log_path)}` : "",
      payload.model_log_path ? `模型日志：${escapeHtml(payload.model_log_path)}` : "",
      payload.dependency_install_dir ? `依赖：${escapeHtml(payload.dependency_install_dir)}` : "",
      payload.model_cache_dir ? `模型/缓存：${escapeHtml(payload.model_cache_dir)}` : "",
    ]
      .filter(Boolean)
      .join("<br />");
  }
  return payload;
}

async function refreshHealth() {
  const health = await api("/api/health");
  state.health = health;
  if (!state.startup) {
    updateRuntimeGate({ ai_dependency_status: health.ai_dependency_status });
  }
  if (!healthList) return health;
  const embeddingLabel = health.embedding_model
    ? `语义: ${health.embedding_backend} / ${health.embedding_model.split("/").pop()}`
    : `语义: ${health.embedding_backend || "未配置"}`;
  const assistantLabel = health.assistant_enabled
    ? `智能搜索: ${health.llm_model || "未配置"}`
    : "智能搜索: 关闭";
  const rerankerLabel = health.local_reranker_enabled
    ? `重排: ${String(health.local_reranker_model || "--").split("/").pop()}`
    : "重排: 关闭";
  const transcriptionModelLabel = String(health.transcription_backend || "").includes("funasr")
    ? `FunASR: ${String(health.funasr_model || "--").split("/").pop()}`
    : `Whisper: ${health.local_whisper_model || "--"}`;
  const transcriptionLanguage = String(health.transcription_backend || "").includes("funasr")
    ? health.funasr_language
    : health.local_whisper_language;
  healthList.innerHTML = [
    ["ffmpeg", health.ffmpeg_available],
    ["ffprobe", health.ffprobe_available],
    ["AI依赖", aiDependencyPill(health.ai_dependency_status), true],
    [`转写: ${health.transcription_backend}`, health.transcription_available],
    [transcriptionModelLabel, health.transcription_available],
    [`语言: ${transcriptionLanguage || "auto"} / ${health.output_simplified_chinese ? "简体" : "原文"}`, health.transcription_available],
    [embeddingLabel, health.embedding_available],
    [rerankerLabel, health.local_reranker_available],
    [assistantLabel, health.assistant_available],
  ]
    .map(([label, ok, rawPill]) => `<div class="status-row"><span>${escapeHtml(label)}</span>${rawPill ? ok : statusPill(ok)}</div>`)
    .join("");
  return health;
}

async function refreshProgress() {
  const progress = await api("/api/progress");
  const media = progress.media || {};
  const chunks = progress.chunks || {};
  const current = progress.current || {};
  const active = Number(current.active || 0) === 1;
  const totalMedia = Object.values(media).reduce((a, b) => a + Number(b || 0), 0);
  const totalChunks = Object.values(chunks).reduce((a, b) => a + Number(b || 0), 0);
  if (taskWindow) taskWindow.classList.toggle("active", active);
  if (taskSummary) {
    taskSummary.textContent = active
      ? `${stageLabel(current.stage)} ${formatPercent(current.percent)} · ${current.filename || "处理中"}`
      : `${current.message || "队列空闲"} · ${totalMedia} 个素材`;
  }
  progressStats.innerHTML = `
    <div class="stat-total"><dt>全部素材</dt><dd>${totalMedia}</dd></div>
    <div><dt>待处理素材</dt><dd>${media.pending || 0}</dd></div>
    <div><dt>处理中素材</dt><dd>${media.processing || 0}</dd></div>
    <div><dt>已完成素材</dt><dd>${media.done || 0}</dd></div>
    <div><dt>失败素材</dt><dd>${media.failed || 0}</dd></div>
    <div><dt>音频分块</dt><dd>${totalChunks}</dd></div>
    <div><dt>字幕段落</dt><dd>${progress.segments || 0}</dd></div>
    <div><dt>语义向量</dt><dd>${progress.embeddings || 0}</dd></div>
  `;
  renderCurrentProgress(current);
}

function renderCurrentProgress(current) {
  const percent = Math.max(0, Math.min(100, Number(current.percent || 0)));
  const totalChunks = Number(current.total_chunks || 0);
  const chunkLabel = totalChunks
    ? `第 ${Number(current.current_chunk || 0)} / ${totalChunks} 段`
    : "未开始";
  const timeLabel = Number(current.total_seconds || 0)
    ? `${formatSeconds(current.current_seconds)} / ${formatSeconds(current.total_seconds)}`
    : "--:--";
  const activeClass = Number(current.active || 0) === 1 ? "active" : "";
  const tone = progressTone(current);
  const message = current.message || (tone === "idle" ? "队列空闲" : "等待任务状态");
  currentProgress.innerHTML = `
    <div class="progress-card ${activeClass} ${tone}">
      <div class="progress-stage-row">
        <span class="progress-stage-pill">${escapeHtml(stageLabel(current.stage || "idle"))}</span>
        <strong class="progress-percent">${escapeHtml(formatPercent(percent))}</strong>
      </div>
      <div class="progress-bar" aria-label="处理进度" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${percent.toFixed(0)}">
        <span class="progress-bar-fill" style="width: ${percent}%"></span>
      </div>
      <div class="progress-message">${escapeHtml(message)}</div>
      <div class="progress-detail-grid">
        <div>
          <span>素材</span>
          <strong title="${escapeHtml(current.filename || "")}">${escapeHtml(current.filename || "--")}</strong>
        </div>
        <div>
          <span>分块</span>
          <strong>${escapeHtml(chunkLabel)}</strong>
        </div>
        <div>
          <span>时间</span>
          <strong>${escapeHtml(timeLabel)}</strong>
        </div>
      </div>
    </div>
  `;
}

function progressTone(current) {
  const active = Number(current.active || 0) === 1;
  const stage = String(current.stage || "idle");
  if (stage === "failed") return "failed";
  if (active) return "running";
  if (stage === "done" || Number(current.percent || 0) >= 100) return "done";
  return "idle";
}

function stageLabel(stage) {
  const labels = {
    checking: "检查环境",
    probing: "读取素材",
    extracting: "抽取音频",
    transcribing: "离线转写",
    indexing: "生成索引",
    prepare: "准备安装",
    installing: "安装依赖",
    downloading: "下载依赖",
    installed: "依赖完成",
    "dependencies-ready": "依赖完成",
    "speech-model": "语音模型",
    "embedding-model": "语义模型",
    "reranker-model": "重排模型",
    "path-too-long": "路径过长",
    blocked: "等待移动",
    unknown: "等待检查",
    skipped: "已跳过",
    done: "已完成",
    failed: "失败",
    idle: "空闲",
  };
  return labels[stage] || stage || "空闲";
}

async function refreshMedia() {
  const rows = await api("/api/media?limit=80");
  mediaCount.textContent = `${rows.length} 个文件`;
  if (!rows.length) {
    mediaList.innerHTML = `<p class="muted">还没有入库素材。</p>`;
    return;
  }
  mediaList.innerHTML = rows
    .map(
      (row) => `
        <div class="media-row" title="${escapeHtml(row.error_message || row.path)}">
          <div class="media-copy">
            <div class="media-name">${escapeHtml(row.filename)}</div>
            <div class="media-path">${escapeHtml(formatSize(row.size_bytes))} · ${escapeHtml(formatSeconds(row.duration_seconds))}</div>
          </div>
          <span class="media-status ${escapeHtml(row.status)}">${escapeHtml(row.status)}</span>
        </div>
      `,
    )
    .join("");
}

function setAssistantAnswer(payload) {
  const terms = payload?.expanded_terms || [];
  if (!payload || (!payload.answer && !terms.length)) {
    assistantAnswer.classList.add("hidden");
    assistantAnswer.innerHTML = "";
    return;
  }
  assistantAnswer.classList.remove("hidden");
  assistantAnswer.innerHTML = `
    ${payload.answer ? `<p>${escapeHtml(payload.answer)}</p>` : ""}
    ${
      terms.length
        ? `<div class="term-list">${terms.map((term) => `<span>${escapeHtml(term)}</span>`).join("")}</div>`
        : ""
    }
  `;
}

async function readSseEvents(response, onEvent) {
  if (!response.body) {
    throw new Error("当前浏览器不支持流式读取。");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      const rawEvent = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = rawEvent
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (data) onEvent(JSON.parse(data));
      boundary = buffer.indexOf("\n\n");
    }
  }
}

async function search(query) {
  state.lastQuery = query.trim();
  exportLink.href = `/api/export.csv?q=${encodeURIComponent(state.lastQuery)}`;
  setAssistantAnswer(null);
  if (!state.lastQuery) {
    setSearchStatus("");
    renderEmpty("输入一句描述或对白后，结果会显示素材文件和时间范围。");
    return;
  }
  if (!ensureRuntimeReady("search")) return;
  setSearchStatus("正在检索字幕、语义向量和本地重排结果", { busy: true });
  renderLoading("正在整理匹配素材");
  try {
    const payload = await api(`/api/search?q=${encodeURIComponent(state.lastQuery)}&limit=80`);
    renderResults(payload.results || [], `${payload.count || 0} 条`);
    setSearchStatus(payload.count ? `已找到 ${payload.count} 条候选素材` : "没有找到匹配素材");
  } catch (error) {
    renderEmpty("搜索失败。");
    setSearchStatus(error.message, { error: true });
  }
}

async function assistantSearch(query) {
  state.lastQuery = query.trim();
  exportLink.href = `/api/export.csv?q=${encodeURIComponent(state.lastQuery)}`;
  if (!state.lastQuery) {
    setSearchStatus("");
    renderEmpty("输入一句描述后，智能搜索会理解需求并整理候选素材。");
    return;
  }
  if (!ensureRuntimeReady("search")) return;
  assistantBtn.disabled = true;
  assistantBtn.textContent = "整理中";
  state.smartStreamAnswer = "";
  state.smartStreamTerms = [];
  setAssistantAnswer(null);
  setSearchStatus("智能搜索正在连接大模型", { busy: true });
  renderLoading("智能搜索正在整理候选素材");
  let finalPayload = null;
  try {
    const response = await fetch("/api/smart-search/stream", {
      headers: { "Content-Type": "application/json" },
      method: "POST",
      body: JSON.stringify({ query: state.lastQuery, limit: 12 }),
    });
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    await readSseEvents(response, (event) => {
      if (event.type === "status") {
        setSearchStatus(event.message || "智能搜索处理中", { busy: true });
        return;
      }
      if (event.type === "terms") {
        state.smartStreamTerms = event.terms || [];
        setSearchStatus("已理解搜索意图，正在召回本地候选", { busy: true });
        setAssistantAnswer({
          answer: state.smartStreamAnswer || event.intent || "",
          expanded_terms: state.smartStreamTerms,
        });
        return;
      }
      if (event.type === "candidates") {
        setSearchStatus(`已召回 ${event.count || 0} 条候选，正在筛选`, { busy: true });
        return;
      }
      if (event.type === "answer_delta") {
        state.smartStreamAnswer += event.text || "";
        setAssistantAnswer({
          answer: state.smartStreamAnswer,
          expanded_terms: state.smartStreamTerms,
        });
        return;
      }
      if (event.type === "final") {
        finalPayload = event.payload || {};
        setAssistantAnswer(finalPayload);
        renderResults(finalPayload.results || [], `${(finalPayload.results || []).length} 条`);
        setSearchStatus(
          (finalPayload.results || []).length ? "智能搜索已整理候选素材" : "智能搜索没有找到合适候选",
        );
      }
    });
    if (!finalPayload) {
      throw new Error("智能搜索没有返回完整结果。");
    }
  } catch (error) {
    renderEmpty("智能搜索处理失败。");
    setSearchStatus(error.message, { error: true });
  } finally {
    assistantBtn.disabled = false;
    assistantBtn.textContent = "智能搜索";
    updateRuntimeGate();
  }
}

function renderEmpty(message) {
  resultCount.textContent = "0 条";
  resultList.className = "result-list empty";
  resultList.innerHTML = `<p>${escapeHtml(message)}</p>`;
}

function renderResults(results, label) {
  resultCount.textContent = label;
  if (!results.length) {
    resultList.className = "result-list empty";
    resultList.innerHTML = `<p>没有匹配结果。</p>`;
    return;
  }
  resultList.className = "result-list";
  resultList.innerHTML = results.map(renderResult).join("");
  resultList.querySelectorAll(".result-item").forEach((button) => {
    button.addEventListener("click", () => {
      const result = JSON.parse(button.dataset.result);
      selectResult(result);
    });
  });
}

function renderResult(result) {
  const safe = escapeHtml(JSON.stringify(result));
  const score = Number.isFinite(Number(result.score)) ? Number(result.score).toFixed(3) : "--";
  const isSemantic = String(result.match_type || "").includes("语义");
  return `
    <button class="result-item" data-result="${safe}" data-segment="${result.segment_id}" type="button">
      <div class="result-main">
        <span class="result-file">${escapeHtml(result.filename)}</span>
        <span class="timecode">${formatSeconds(result.start_seconds)} - ${formatSeconds(result.end_seconds)}</span>
      </div>
      <div class="result-badges">
        <span class="match-type ${isSemantic ? "semantic" : ""}">${escapeHtml(result.match_type || "匹配")}</span>
        <span class="result-score">${escapeHtml(score)}</span>
      </div>
      <div class="result-text">${escapeHtml(result.preview_text || result.text)}</div>
      ${result.reason ? `<div class="result-reason">${escapeHtml(result.reason)}</div>` : ""}
      <div class="result-path">${escapeHtml(result.path)}</div>
    </button>
  `;
}

function selectResult(result) {
  state.activeSegmentId = result.segment_id;
  resultList.querySelectorAll(".result-item").forEach((item) => {
    item.classList.toggle("active", Number(item.dataset.segment) === Number(result.segment_id));
  });

  videoPlaceholder.style.display = "none";
  const target = Math.max(0, Number(result.start_seconds || 0));
  video.src = `/api/media/${result.media_id}/file#t=${target}`;
  video.addEventListener(
    "loadedmetadata",
    () => {
      video.currentTime = target;
    },
    { once: true },
  );
  selectedTime.textContent = `${formatSeconds(result.start_seconds)} - ${formatSeconds(result.end_seconds)}`;
  selectedMeta.innerHTML = `
    <div><strong>${escapeHtml(result.filename)}</strong></div>
    <div>${escapeHtml(result.text)}</div>
    ${result.reason ? `<div>${escapeHtml(result.reason)}</div>` : ""}
    <div>匹配: ${escapeHtml(result.match_type || "匹配")} / 分数 ${escapeHtml(Number(result.score || 0).toFixed(3))}</div>
    <div>${escapeHtml(result.path)}</div>
  `;
}

async function refreshAll() {
  await Promise.all([refreshHealth(), refreshProgress(), refreshMedia(), refreshStartupStatus()]);
}

async function pickDirectory() {
  pickDirectoryBtn.disabled = true;
  setIngestStatus("正在打开文件夹窗口");
  try {
    const payload = await api("/api/pick-directory");
    if (!payload.path) {
      setIngestStatus("未选择文件夹");
      return;
    }
    state.ingestMode = "directory";
    state.selectedPaths = [];
    directoryInput.value = payload.path;
    renderSelectedFiles([]);
    setIngestStatus("已选择文件夹");
  } catch (error) {
    setIngestStatus(error.message, "bad");
  } finally {
    pickDirectoryBtn.disabled = false;
  }
}

async function pickFiles() {
  pickFilesBtn.disabled = true;
  setIngestStatus("正在打开文件窗口");
  try {
    const payload = await api("/api/pick-files");
    const paths = payload.paths || [];
    if (!paths.length) {
      setIngestStatus("未选择文件");
      return;
    }
    state.ingestMode = "files";
    state.selectedPaths = paths;
    directoryInput.value = paths.length === 1 ? paths[0] : `已选择 ${paths.length} 个文件`;
    renderSelectedFiles(paths);
    setIngestStatus(`已选择 ${paths.length} 个文件`);
  } catch (error) {
    setIngestStatus(error.message, "bad");
  } finally {
    pickFilesBtn.disabled = false;
  }
}

$("#ingestForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!ensureRuntimeReady("ingest")) return;
  const directory = directoryInput.value.trim();
  const limitRaw = limitInput.value.trim();
  const limit = limitRaw ? Number(limitRaw) : null;
  if (state.ingestMode !== "files" && !directory) {
    setIngestStatus("请先打开文件夹、打开文件，或粘贴本机路径", "bad");
    return;
  }
  showTaskWindow();
  setIngestStatus("正在加入入库队列");
  try {
    const result =
      state.ingestMode === "files" && state.selectedPaths.length
        ? await api("/api/ingest-files", {
            method: "POST",
            body: JSON.stringify({ paths: state.selectedPaths, limit, start_processing: true }),
          })
        : await api("/api/ingest", {
            method: "POST",
            body: JSON.stringify({ directory, limit, start_processing: true }),
          });
    setIngestStatus(
      `已加入 ${result.added || 0} 个，已存在 ${result.existing || 0} 个，跳过 ${result.skipped || 0} 个`,
    );
    await refreshAll();
  } catch (error) {
    setIngestStatus(error.message, "bad");
  }
});

$("#processBtn").addEventListener("click", async () => {
  if (!ensureRuntimeReady("ingest")) return;
  showTaskWindow();
  setIngestStatus("正在继续处理队列");
  try {
    await api("/api/process", { method: "POST", body: "{}" });
    await refreshAll();
  } catch (error) {
    setIngestStatus(error.message, "bad");
  }
});

$("#retryBtn").addEventListener("click", async () => {
  if (!ensureRuntimeReady("ingest")) return;
  showTaskWindow();
  setIngestStatus("正在重试失败任务");
  try {
    await api("/api/retry-failed", { method: "POST", body: "{}" });
    await refreshAll();
  } catch (error) {
    setIngestStatus(error.message, "bad");
  }
});

$("#semanticBtn").addEventListener("click", async () => {
  if (!ensureRuntimeReady("ingest")) return;
  showTaskWindow();
  setIngestStatus("正在补齐语义索引");
  try {
    await api("/api/semantic-index", { method: "POST", body: "{}" });
    await refreshAll();
  } catch (error) {
    setIngestStatus(error.message, "bad");
  }
});

$("#refreshBtn").addEventListener("click", refreshAll);

pickDirectoryBtn.addEventListener("click", pickDirectory);
pickFilesBtn.addEventListener("click", pickFiles);
directoryInput.addEventListener("input", () => {
  if (!directoryInput.value.trim()) return;
  state.ingestMode = "directory";
  state.selectedPaths = [];
  renderSelectedFiles([]);
  setIngestStatus("使用手动输入路径");
});

taskToggleBtn.addEventListener("click", () => {
  const minimized = !taskWindow.classList.contains("minimized");
  taskWindow.classList.toggle("minimized", minimized);
  localStorage.setItem("pmmTaskMinimized", minimized ? "1" : "0");
  taskToggleBtn.textContent = minimized ? "展开" : "缩小";
});

if (startupToggleBtn) {
  startupToggleBtn.addEventListener("click", () => {
    const minimized = !startupPanel.classList.contains("minimized");
    startupPanel.classList.toggle("minimized", minimized);
    localStorage.setItem("pmmStartupMinimized", minimized ? "1" : "0");
    startupToggleBtn.textContent = minimized ? "展开" : "缩小";
  });
}

$("#searchForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  await search($("#searchInput").value);
});

assistantBtn.addEventListener("click", async () => {
  await assistantSearch($("#searchInput").value);
});

exportLink.addEventListener("click", (event) => {
  if (runtimeReady()) return;
  event.preventDefault();
  ensureRuntimeReady("search");
});

updateRuntimeGate();

refreshAll().catch((error) => {
  setIngestStatus(`服务连接失败：${error.message}`, "bad");
});

if (localStorage.getItem("pmmTaskMinimized") === "1") {
  taskWindow.classList.add("minimized");
  taskToggleBtn.textContent = "展开";
}

setInterval(() => {
  refreshStartupStatus().catch(() => {});
  refreshProgress().catch(() => {});
  refreshMedia().catch(() => {});
}, 3000);
