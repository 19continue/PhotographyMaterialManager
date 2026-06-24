from __future__ import annotations

import csv
import io
import math
import sqlite3
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from typing import Iterator

from .config import Settings
from .database import Database, row_to_dict
from .embeddings import (
    build_embedding_client,
    embedding_model_key,
    is_embedding_available,
    blob_to_vector,
    cosine_similarity,
    vector_to_blob,
)
from .llm_assistant import LLMClient, LLMError, is_llm_available
from .media import (
    SUPPORTED_VIDEO_EXTENSIONS,
    check_media_tools,
    extract_audio_chunk,
    iter_video_files,
    probe_duration,
)
from .reranker import build_reranker, is_local_reranker_available
from .text_utils import normalize_text, snippet
from .transcription import Transcriber, build_transcriber, transcription_backend_status


@dataclass(frozen=True)
class SearchResult:
    segment_id: int
    media_id: int
    filename: str
    path: str
    start_seconds: float
    end_seconds: float
    text: str
    preview_text: str
    score: float
    match_type: str
    reason: str = ""
    confidence: float | None = None


def _ordered_unique(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = value.strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        output.append(clean)
    return output


class MaterialService:
    def __init__(self, settings: Settings, db: Database) -> None:
        self.settings = settings
        self.db = db
        self.transcriber: Transcriber | None = None
        self.embedding_client = build_embedding_client(settings)
        self.reranker = build_reranker(settings)
        self.llm_client = LLMClient(settings)
        self._lock = threading.Lock()
        self._semantic_autofill_lock = threading.Lock()
        self._semantic_autofill_attempted = False

    def scan_directory(self, directory: Path, limit: int | None = None) -> dict[str, int]:
        try:
            directory = directory.expanduser().resolve()
        except OSError as exc:
            raise ValueError(f"Directory path is invalid: {directory}") from exc
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Directory does not exist: {directory}")

        added = 0
        existing = 0
        skipped = 0
        with self.db.connect() as conn:
            for index, media_path in enumerate(iter_video_files(directory)):
                if limit is not None and index >= limit:
                    break
                result = self._insert_media(conn, media_path)
                if result == "added":
                    added += 1
                elif result == "existing":
                    existing += 1
                else:
                    skipped += 1
            self._event(
                conn,
                "info",
                f"Scanned {directory}: {added} new, {existing} existing, {skipped} skipped",
            )
        return {"added": added, "existing": existing, "skipped": skipped}

    def scan_files(self, paths: list[Path], limit: int | None = None) -> dict[str, int]:
        if not paths:
            raise ValueError("No media files selected.")

        added = 0
        existing = 0
        skipped = 0
        with self.db.connect() as conn:
            for index, media_path in enumerate(paths):
                if limit is not None and index >= limit:
                    break
                result = self._insert_media(conn, media_path)
                if result == "added":
                    added += 1
                elif result == "existing":
                    existing += 1
                else:
                    skipped += 1
            self._event(
                conn,
                "info",
                f"Selected files: {added} new, {existing} existing, {skipped} skipped",
            )
        return {"added": added, "existing": existing, "skipped": skipped}

    def _insert_media(self, conn: sqlite3.Connection, media_path: Path) -> str:
        try:
            media_path = media_path.expanduser()
            if (
                not media_path.exists()
                or not media_path.is_file()
                or media_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS
            ):
                return "skipped"
            media_path = media_path.resolve()
            stat = media_path.stat()
        except OSError:
            return "skipped"
        try:
            conn.execute(
                """
                INSERT INTO media(path, filename, extension, size_bytes, status)
                VALUES (?, ?, ?, ?, 'pending')
                """,
                (
                    str(media_path),
                    media_path.name,
                    media_path.suffix.lower(),
                    stat.st_size,
                ),
            )
        except sqlite3.IntegrityError:
            return "existing"
        return "added"

    def process_pending(self) -> None:
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._set_progress(
                active=True,
                stage="checking",
                message="检查 ffmpeg、转写后端和待处理队列",
                percent=0.0,
            )
            tools = check_media_tools(self.settings)
            if not tools.ffmpeg_available or not tools.ffprobe_available:
                message = "ffmpeg/ffprobe is not available. Install ffmpeg or set PMM_FFMPEG_BIN and PMM_FFPROBE_BIN."
                with self.db.connect() as conn:
                    conn.execute(
                        "UPDATE media SET status='failed', error_message=?, updated_at=CURRENT_TIMESTAMP WHERE status='pending'",
                        (message,),
                    )
                    self._event(conn, "error", message)
                self._set_progress(active=False, stage="failed", message=message, percent=0.0)
                return
            backend_status = transcription_backend_status(self.settings)
            if not backend_status["available"]:
                message = f"Transcription backend is not available: {backend_status['detail']}"
                with self.db.connect() as conn:
                    conn.execute(
                        "UPDATE media SET status='failed', error_message=?, updated_at=CURRENT_TIMESTAMP WHERE status='pending'",
                        (message,),
                    )
                    self._event(conn, "error", message)
                self._set_progress(active=False, stage="failed", message=message, percent=0.0)
                return

            while True:
                media = self._next_pending_media()
                if media is None:
                    self._set_progress(active=False, stage="idle", message="队列已处理完成", percent=100.0)
                    break
                self._process_media(media)
        finally:
            self._lock.release()

    def _next_pending_media(self) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM media
                WHERE status = 'pending'
                ORDER BY id
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE media SET status='processing', error_message=NULL, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (row["id"],),
            )
            return row_to_dict(row)

    def retry_failed(self) -> int:
        with self.db.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE media
                SET status='pending', error_message=NULL, updated_at=CURRENT_TIMESTAMP
                WHERE status='failed'
                """
            )
            count = cursor.rowcount
            self._event(conn, "info", f"Reset {count} failed media files to pending")
        return int(count)

    def _process_media(self, media: dict[str, Any]) -> None:
        media_id = int(media["id"])
        media_path = Path(str(media["path"]))
        try:
            self._set_progress(
                active=True,
                stage="probing",
                media_id=media_id,
                filename=media_path.name,
                message=f"读取素材信息：{media_path.name}",
                percent=1.0,
            )
            if not media_path.exists():
                raise FileNotFoundError(f"Media file no longer exists: {media_path}")

            duration = probe_duration(media_path, self.settings)
            if duration is None or duration <= 0:
                raise RuntimeError("Could not read a positive media duration.")

            with self.db.connect() as conn:
                conn.execute(
                    """
                    UPDATE media
                    SET duration_seconds=?, updated_at=CURRENT_TIMESTAMP
                    WHERE id=?
                    """,
                    (duration, media_id),
                )
                conn.execute("DELETE FROM transcript_word WHERE media_id=?", (media_id,))
                conn.execute("DELETE FROM transcript_segment WHERE media_id=?", (media_id,))
                conn.execute("DELETE FROM audio_chunk WHERE media_id=?", (media_id,))

            chunks = self._create_chunks(media_id, media_path, duration)
            total_chunks = len(chunks)
            for index, chunk in enumerate(chunks):
                self._set_progress(
                    active=True,
                    stage="transcribing",
                    media_id=media_id,
                    filename=media_path.name,
                    current_chunk=index + 1,
                    total_chunks=total_chunks,
                    current_seconds=float(chunk["end_seconds"]),
                    total_seconds=duration,
                    percent=30.0 + (index / max(total_chunks, 1)) * 50.0,
                    message=f"转写第 {index + 1}/{total_chunks} 段：{media_path.name}",
                )
                with self.db.connect() as conn:
                    conn.execute(
                        "UPDATE audio_chunk SET status='transcribing', updated_at=CURRENT_TIMESTAMP WHERE id=?",
                        (chunk["id"],),
                    )
                last_inner_percent = -1.0

                def chunk_progress(current: int, total: int) -> None:
                    nonlocal last_inner_percent
                    if total <= 0:
                        return
                    fraction = max(0.0, min(1.0, float(current) / float(total)))
                    percent = 30.0 + ((index + fraction) / max(total_chunks, 1)) * 50.0
                    if percent - last_inner_percent < 1.0 and current < total:
                        return
                    last_inner_percent = percent
                    self._set_progress(
                        active=True,
                        stage="transcribing",
                        media_id=media_id,
                        filename=media_path.name,
                        current_chunk=index + 1,
                        total_chunks=total_chunks,
                        current_seconds=float(chunk["start_seconds"])
                        + (float(chunk["end_seconds"]) - float(chunk["start_seconds"])) * fraction,
                        total_seconds=duration,
                        percent=percent,
                        message=f"转写第 {index + 1}/{total_chunks} 段：{media_path.name}",
                    )

                payload = self._get_transcriber().transcribe(
                    Path(chunk["audio_path"]),
                    progress_callback=chunk_progress,
                )
                self._store_transcript(media_id, int(chunk["id"]), float(chunk["start_seconds"]), payload)
                with self.db.connect() as conn:
                    conn.execute(
                        "UPDATE audio_chunk SET status='done', updated_at=CURRENT_TIMESTAMP WHERE id=?",
                        (chunk["id"],),
                    )
                self._set_progress(
                    active=True,
                    stage="transcribing",
                    media_id=media_id,
                    filename=media_path.name,
                    current_chunk=index + 1,
                    total_chunks=total_chunks,
                    current_seconds=float(chunk["end_seconds"]),
                    total_seconds=duration,
                    percent=30.0 + ((index + 1) / max(total_chunks, 1)) * 50.0,
                    message=f"已完成第 {index + 1}/{total_chunks} 段转写：{media_path.name}",
                )

            self._set_progress(
                active=True,
                stage="indexing",
                media_id=media_id,
                filename=media_path.name,
                current_chunk=total_chunks,
                total_chunks=total_chunks,
                current_seconds=duration,
                total_seconds=duration,
                percent=85.0,
                message=f"生成本地语义索引：{media_path.name}",
            )
            try:
                self.embed_missing_segments(media_id=media_id)
            except Exception as exc:
                with self.db.connect() as conn:
                    self._event(
                        conn,
                        "warning",
                        f"Semantic index skipped for {media_path.name}: {exc}",
                    )

            with self.db.connect() as conn:
                conn.execute(
                    "UPDATE media SET status='done', error_message=NULL, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (media_id,),
                )
                self._event(conn, "info", f"Processed {media_path.name}")
            self._set_progress(
                active=True,
                stage="done",
                media_id=media_id,
                filename=media_path.name,
                current_chunk=total_chunks,
                total_chunks=total_chunks,
                current_seconds=duration,
                total_seconds=duration,
                percent=100.0,
                message=f"完成：{media_path.name}",
            )
        except Exception as exc:
            with self.db.connect() as conn:
                conn.execute(
                    """
                    UPDATE media
                    SET status='failed', error_message=?, updated_at=CURRENT_TIMESTAMP
                    WHERE id=?
                    """,
                    (str(exc), media_id),
                )
                self._event(conn, "error", f"{media_path.name}: {exc}")
            self._set_progress(
                active=False,
                stage="failed",
                media_id=media_id,
                filename=media_path.name,
                message=f"处理失败：{media_path.name} - {exc}",
                percent=0.0,
            )

    def _get_transcriber(self) -> Transcriber:
        if self.transcriber is None:
            self.transcriber = build_transcriber(self.settings)
        return self.transcriber

    def warm_runtime(self) -> None:
        try:
            self._get_transcriber()
        except Exception as exc:
            with self.db.connect() as conn:
                self._event(conn, "warning", f"Silent speech model warmup skipped: {exc}")
        try:
            if self.settings.enable_embeddings and is_embedding_available(self.settings):
                self.embedding_client.embed_texts(["环境预热"])
        except Exception as exc:
            with self.db.connect() as conn:
                self._event(conn, "warning", f"Silent embedding model warmup skipped: {exc}")
        try:
            if self._ensure_reranker_available():
                self.reranker.score_pairs("环境预热", ["环境预热"])
        except Exception as exc:
            with self.db.connect() as conn:
                self._event(conn, "warning", f"Silent reranker warmup skipped: {exc}")

    def _create_chunks(self, media_id: int, media_path: Path, duration: float) -> list[dict[str, Any]]:
        chunk_seconds = max(60, self.settings.chunk_seconds)
        overlap = max(0, min(self.settings.chunk_overlap_seconds, chunk_seconds - 1))
        count = max(1, math.ceil(duration / chunk_seconds))
        created: list[dict[str, Any]] = []

        for index in range(count):
            start = max(0.0, index * chunk_seconds - (overlap if index > 0 else 0))
            end = min(duration, (index + 1) * chunk_seconds + overlap)
            chunk_duration = max(0.1, end - start)
            audio_path = self.settings.audio_dir / f"media_{media_id}" / f"chunk_{index:04d}.mp3"
            self._set_progress(
                active=True,
                stage="extracting",
                media_id=media_id,
                filename=media_path.name,
                current_chunk=index + 1,
                total_chunks=count,
                current_seconds=end,
                total_seconds=duration,
                percent=5.0 + (index / max(count, 1)) * 20.0,
                message=f"抽取音频第 {index + 1}/{count} 段：{media_path.name}",
            )
            extract_audio_chunk(media_path, audio_path, start, chunk_duration, self.settings)
            with self.db.connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO audio_chunk(media_id, chunk_index, start_seconds, end_seconds, audio_path, status)
                    VALUES (?, ?, ?, ?, ?, 'pending')
                    """,
                    (media_id, index, start, end, str(audio_path)),
                )
                row = conn.execute("SELECT * FROM audio_chunk WHERE id=?", (cursor.lastrowid,)).fetchone()
                created.append(row_to_dict(row) or {})
            self._set_progress(
                active=True,
                stage="extracting",
                media_id=media_id,
                filename=media_path.name,
                current_chunk=index + 1,
                total_chunks=count,
                current_seconds=end,
                total_seconds=duration,
                percent=5.0 + ((index + 1) / max(count, 1)) * 20.0,
                message=f"已抽取音频第 {index + 1}/{count} 段：{media_path.name}",
            )
        return created

    def _store_transcript(
        self,
        media_id: int,
        chunk_id: int,
        chunk_start: float,
        payload: dict[str, Any],
    ) -> None:
        segments = payload.get("segments") or []
        words = payload.get("words") or []

        with self.db.connect() as conn:
            if segments:
                for segment in segments:
                    text = str(segment.get("text") or "").strip()
                    if not text:
                        continue
                    start = chunk_start + float(segment.get("start") or 0)
                    end = chunk_start + float(segment.get("end") or start)
                    segment_id = conn.execute(
                        """
                        INSERT INTO transcript_segment(media_id, chunk_id, start_seconds, end_seconds, text, normalized_text)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (media_id, chunk_id, start, end, text, normalize_text(text)),
                    ).lastrowid
                    self._store_words_for_segment(conn, int(segment_id), media_id, segment, chunk_start)
            elif payload.get("text"):
                text = str(payload.get("text") or "").strip()
                conn.execute(
                    """
                    INSERT INTO transcript_segment(media_id, chunk_id, start_seconds, end_seconds, text, normalized_text)
                    SELECT ?, ?, start_seconds, end_seconds, ?, ? FROM audio_chunk WHERE id=?
                    """,
                    (media_id, chunk_id, text, normalize_text(text), chunk_id),
                )

            if words:
                self._store_orphan_words(conn, media_id, chunk_id, chunk_start, words)

    def _store_words_for_segment(
        self,
        conn: sqlite3.Connection,
        segment_id: int,
        media_id: int,
        segment: dict[str, Any],
        chunk_start: float,
    ) -> None:
        for word in segment.get("words") or []:
            text = str(word.get("word") or "").strip()
            if not text:
                continue
            start = chunk_start + float(word.get("start") or 0)
            end = chunk_start + float(word.get("end") or start)
            conn.execute(
                """
                INSERT INTO transcript_word(segment_id, media_id, word, normalized_word, start_seconds, end_seconds)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (segment_id, media_id, text, normalize_text(text), start, end),
            )

    def _store_orphan_words(
        self,
        conn: sqlite3.Connection,
        media_id: int,
        chunk_id: int,
        chunk_start: float,
        words: list[dict[str, Any]],
    ) -> None:
        existing = conn.execute(
            """
            SELECT COUNT(*) FROM transcript_word
            WHERE media_id=? AND segment_id IN (
                SELECT id FROM transcript_segment WHERE chunk_id=?
            )
            """,
            (media_id, chunk_id),
        ).fetchone()[0]
        if existing:
            return

        segment = conn.execute(
            """
            SELECT id FROM transcript_segment
            WHERE media_id=? AND chunk_id=?
            ORDER BY start_seconds
            LIMIT 1
            """,
            (media_id, chunk_id),
        ).fetchone()
        if segment is None:
            return
        for word in words:
            text = str(word.get("word") or "").strip()
            if not text:
                continue
            start = chunk_start + float(word.get("start") or 0)
            end = chunk_start + float(word.get("end") or start)
            conn.execute(
                """
                INSERT INTO transcript_word(segment_id, media_id, word, normalized_word, start_seconds, end_seconds)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (segment["id"], media_id, text, normalize_text(text), start, end),
            )

    def search(self, query: str, limit: int = 50) -> list[SearchResult]:
        clean_query = (query or "").strip()
        normalized = normalize_text(clean_query)
        if not clean_query:
            return []

        candidate_limit = max(
            limit,
            min(self.settings.semantic_candidates, limit * max(1, self.settings.search_candidate_multiplier)),
        )
        merged: dict[int, SearchResult] = {}

        if normalized:
            rows = self._search_normalized_phrase(normalized, candidate_limit)
            for row in rows:
                result = self._row_to_search_result(
                    row,
                    clean_query,
                    score=1.2,
                    match_type="字幕精确",
                    reason="字幕文本包含完整查询",
                )
                self._merge_search_result(merged, result)

        rows = self._search_fts(clean_query, candidate_limit, set())
        for row in rows:
            raw_rank = abs(float(row["rank"]))
            score = 0.65 if raw_rank == 0 else max(0.35, 0.65 - min(raw_rank, 0.3))
            result = self._row_to_search_result(
                row,
                clean_query,
                score=score,
                match_type="关键词",
                reason="全文索引匹配查询词",
            )
            self._merge_search_result(merged, result)

        self._ensure_semantic_index_for_search()
        for result in self._search_semantic(clean_query, candidate_limit, set()):
            self._merge_search_result(merged, result)

        results = [
            self._with_lexical_boost(result, clean_query)
            for result in merged.values()
        ]
        ranked = sorted(
            results,
            key=lambda result: (result.score, -result.start_seconds),
            reverse=True,
        )
        return self._rerank_locally(clean_query, ranked, limit)

    def assistant_search(self, query: str, limit: int = 12) -> dict[str, Any]:
        clean_query = (query or "").strip()
        if not clean_query:
            return {
                "query": clean_query,
                "answer": "",
                "expanded_terms": [],
                "results": [],
                "assistant_available": is_llm_available(self.settings),
                "used_llm": False,
            }

        if not is_llm_available(self.settings):
            fallback = self.search(clean_query, limit=limit)
            return {
                "query": clean_query,
                "answer": "大模型未配置，已使用本地检索返回候选结果。",
                "expanded_terms": [clean_query],
                "results": [result.__dict__ for result in fallback],
                "assistant_available": False,
                "used_llm": False,
            }

        try:
            expanded = self.llm_client.expand_query(
                clean_query,
                max_terms=self.settings.assistant_query_terms,
            )
            terms = [clean_query] + [
                term for term in expanded.search_terms if term and term != clean_query
            ]
            candidates = self._assistant_candidates(terms)
            if not candidates:
                return {
                    "query": clean_query,
                    "answer": "我理解了需求，但当前字幕索引里没有召回可用候选素材。",
                    "intent": expanded.intent,
                    "expanded_terms": terms,
                    "results": [],
                    "assistant_available": True,
                    "used_llm": True,
                }

            reranked = self.llm_client.rerank(
                clean_query,
                [candidate.__dict__ for candidate in candidates],
                limit=limit,
            )
            output = self._assistant_results_from_rerank(candidates, reranked, limit)
            return {
                "query": clean_query,
                "answer": str(reranked.get("answer") or expanded.intent or "已整理候选素材。"),
                "intent": expanded.intent,
                "expanded_terms": terms,
                "results": output,
                "assistant_available": True,
                "used_llm": True,
            }
        except (LLMError, Exception) as exc:
            fallback = self.search(clean_query, limit=limit)
            with self.db.connect() as conn:
                self._event(conn, "warning", f"Assistant search fallback: {exc}")
            return {
                "query": clean_query,
                "answer": f"智能搜索暂不可用，已回退到本地检索：{exc}",
                "expanded_terms": [clean_query],
                "results": [result.__dict__ for result in fallback],
                "assistant_available": True,
                "used_llm": False,
            }

    def assistant_search_stream(self, query: str, limit: int = 12) -> Iterator[dict[str, Any]]:
        clean_query = (query or "").strip()
        if not clean_query:
            yield {
                "type": "final",
                "payload": {
                    "query": clean_query,
                    "answer": "",
                    "expanded_terms": [],
                    "results": [],
                    "assistant_available": is_llm_available(self.settings),
                    "used_llm": False,
                },
            }
            return

        if not is_llm_available(self.settings):
            yield {"type": "status", "message": "大模型未配置，正在使用本地检索"}
            fallback = self.search(clean_query, limit=limit)
            yield {
                "type": "final",
                "payload": {
                    "query": clean_query,
                    "answer": "大模型未配置，已使用本地检索返回候选结果。",
                    "expanded_terms": [clean_query],
                    "results": [result.__dict__ for result in fallback],
                    "assistant_available": False,
                    "used_llm": False,
                },
            }
            return

        try:
            yield {"type": "status", "message": "正在理解搜索意图"}
            expanded = self.llm_client.expand_query_stream(
                clean_query,
                max_terms=self.settings.assistant_query_terms,
            )
            terms = [clean_query] + [
                term for term in expanded.search_terms if term and term != clean_query
            ]
            yield {
                "type": "terms",
                "intent": expanded.intent,
                "terms": terms,
            }

            yield {"type": "status", "message": "正在本地召回候选素材"}
            candidates = self._assistant_candidates(terms)
            yield {"type": "candidates", "count": len(candidates)}
            if not candidates:
                yield {
                    "type": "final",
                    "payload": {
                        "query": clean_query,
                        "answer": "我理解了需求，但当前字幕索引里没有召回可用候选素材。",
                        "intent": expanded.intent,
                        "expanded_terms": terms,
                        "results": [],
                        "assistant_available": True,
                        "used_llm": True,
                    },
                }
                return

            yield {"type": "status", "message": "大模型正在筛选候选素材"}
            reranked: dict[str, Any] | None = None
            for event in self.llm_client.rerank_events(
                clean_query,
                [candidate.__dict__ for candidate in candidates],
                limit=limit,
            ):
                if event.get("type") == "answer_delta":
                    yield event
                elif event.get("type") == "final":
                    payload = event.get("payload")
                    if isinstance(payload, dict):
                        reranked = payload

            if reranked is None:
                raise LLMError("LLM did not return rerank payload.")

            output = self._assistant_results_from_rerank(candidates, reranked, limit)
            yield {
                "type": "final",
                "payload": {
                    "query": clean_query,
                    "answer": str(reranked.get("answer") or expanded.intent or "已整理候选素材。"),
                    "intent": expanded.intent,
                    "expanded_terms": terms,
                    "results": output,
                    "assistant_available": True,
                    "used_llm": True,
                },
            }
        except (LLMError, Exception) as exc:
            fallback = self.search(clean_query, limit=limit)
            with self.db.connect() as conn:
                self._event(conn, "warning", f"Smart search fallback: {exc}")
            yield {"type": "status", "message": "大模型暂不可用，已回退到本地检索"}
            yield {
                "type": "final",
                "payload": {
                    "query": clean_query,
                    "answer": f"智能搜索暂不可用，已回退到本地检索：{exc}",
                    "expanded_terms": [clean_query],
                    "results": [result.__dict__ for result in fallback],
                    "assistant_available": True,
                    "used_llm": False,
                },
            }

    def _assistant_results_from_rerank(
        self,
        candidates: list[SearchResult],
        reranked: dict[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        by_id = {candidate.segment_id: candidate for candidate in candidates}
        output: list[dict[str, Any]] = []
        for item in reranked.get("items", []):
            try:
                segment_id = int(item.get("segment_id"))
            except (TypeError, ValueError):
                continue
            candidate = by_id.get(segment_id)
            if candidate is None:
                continue
            data = candidate.__dict__.copy()
            data["reason"] = str(item.get("reason") or "")
            try:
                data["confidence"] = float(item.get("confidence"))
            except (TypeError, ValueError):
                data["confidence"] = None
            output.append(data)
            if len(output) >= limit:
                break
        if not output:
            output = [candidate.__dict__ for candidate in candidates[:limit]]
        return output

    def _merge_search_result(
        self,
        merged: dict[int, SearchResult],
        result: SearchResult,
    ) -> None:
        existing = merged.get(result.segment_id)
        if existing is None:
            merged[result.segment_id] = result
            return

        labels = _ordered_unique(existing.match_type.split("/") + result.match_type.split("/"))
        reasons = _ordered_unique(
            [part for part in [existing.reason, result.reason] if part]
        )
        base = existing if existing.score >= result.score else result
        combined_score = min(1.35, max(existing.score, result.score) + 0.06)
        merged[result.segment_id] = replace(
            base,
            score=combined_score,
            match_type="/".join(labels),
            reason="；".join(reasons),
        )

    def _with_lexical_boost(self, result: SearchResult, query: str) -> SearchResult:
        lexical = self._lexical_similarity(query, result.text)
        if lexical <= 0:
            return result
        boosted = min(1.35, result.score + min(0.18, lexical * 0.18))
        if boosted <= result.score:
            return result
        reason = result.reason
        lexical_reason = f"字幕与查询字面重合 {lexical:.2f}"
        if lexical_reason not in reason:
            reason = "；".join(part for part in [reason, lexical_reason] if part)
        return replace(result, score=boosted, reason=reason)

    def _rerank_locally(
        self,
        query: str,
        results: list[SearchResult],
        limit: int,
    ) -> list[SearchResult]:
        if not results:
            return []
        if not self._ensure_reranker_available():
            return results[:limit]

        candidate_count = min(len(results), max(limit, self.settings.reranker_candidates))
        head = results[:candidate_count]
        tail = results[candidate_count:]
        try:
            raw_scores = self.reranker.score_pairs(query, [result.text for result in head])
        except Exception as exc:
            with self.db.connect() as conn:
                self._event(conn, "warning", f"Local reranker skipped: {exc}")
            return results[:limit]

        if len(raw_scores) != len(head):
            with self.db.connect() as conn:
                self._event(
                    conn,
                    "warning",
                    f"Local reranker returned {len(raw_scores)} scores for {len(head)} candidates; keeping original ranking",
                )
            return results[:limit]

        reranked: list[SearchResult] = []
        for result, raw_score in zip(head, raw_scores):
            exact_bonus = 0.18 if "字幕精确" in result.match_type else 0.0
            rerank_score = max(0.0, min(1.0, raw_score))
            combined_score = min(
                1.35,
                (result.score * 0.25) + (rerank_score * 0.75) + exact_bonus,
            )
            if "字幕精确" in result.match_type:
                combined_score = max(result.score, combined_score)
            reason = "；".join(
                part
                for part in [
                    result.reason,
                    f"本地重排分 {raw_score:.3f}",
                ]
                if part
            )
            reranked.append(replace(result, score=combined_score, reason=reason))

        return sorted(
            reranked + tail,
            key=lambda result: (result.score, -result.start_seconds),
            reverse=True,
        )[:limit]

    def _ensure_reranker_available(self) -> bool:
        if not is_local_reranker_available(self.settings):
            return False
        if self.reranker.__class__.__name__ == "NoopReranker":
            self.reranker = build_reranker(self.settings)
        return True

    def _ensure_semantic_index_for_search(self) -> None:
        if self._semantic_autofill_attempted:
            return
        if not self.settings.enable_embeddings or not is_embedding_available(self.settings):
            return
        with self._semantic_autofill_lock:
            if self._semantic_autofill_attempted:
                return
            segment_count, embedding_count = self._semantic_index_counts()
            if segment_count <= 0 or embedding_count > 0:
                return
            self._semantic_autofill_attempted = True
            try:
                self.embed_missing_segments(batch_size=32)
            except Exception as exc:
                with self.db.connect() as conn:
                    self._event(conn, "warning", f"Semantic index autofill skipped: {exc}")

    def _semantic_index_counts(self) -> tuple[int, int]:
        with self.db.connect() as conn:
            segment_count = conn.execute("SELECT COUNT(*) FROM transcript_segment").fetchone()[0]
            embedding_count = conn.execute(
                "SELECT COUNT(*) FROM transcript_embedding WHERE model=?",
                (embedding_model_key(self.settings),),
            ).fetchone()[0]
        return int(segment_count), int(embedding_count)

    def _lexical_similarity(self, query: str, text: str) -> float:
        query_norm = normalize_text(query)
        text_norm = normalize_text(text)
        if not query_norm or not text_norm:
            return 0.0
        if query_norm in text_norm:
            return 1.0
        grams = self._char_ngrams(query_norm)
        if not grams:
            return 0.0
        text_grams = self._char_ngrams(text_norm)
        if not text_grams:
            return 0.0
        return len(grams & text_grams) / len(grams)

    def _char_ngrams(self, value: str) -> set[str]:
        if not value:
            return set()
        if len(value) <= 2:
            return {value}
        return {value[index : index + 2] for index in range(len(value) - 1)}

    def _assistant_candidates(self, terms: list[str]) -> list[SearchResult]:
        merged: dict[int, SearchResult] = {}
        per_term_limit = max(10, self.settings.assistant_candidates)
        for term in terms:
            for result in self.search(term, limit=per_term_limit):
                self._merge_search_result(merged, result)
        return sorted(
            merged.values(),
            key=lambda item: (item.score, -item.start_seconds),
            reverse=True,
        )[: self.settings.assistant_candidates]

    def embed_missing_segments(self, media_id: int | None = None, batch_size: int = 64) -> int:
        if not self.settings.enable_embeddings:
            return 0
        if not is_embedding_available(self.settings):
            with self.db.connect() as conn:
                self._event(conn, "warning", "Semantic index skipped because embedding backend is not available")
            return 0

        model_key = embedding_model_key(self.settings)
        embedded = 0
        while True:
            rows = self._load_segments_missing_embeddings(media_id, batch_size)
            if not rows:
                break
            texts = [self._embedding_text(row) for row in rows]
            vectors = self.embedding_client.embed_texts(texts)
            with self.db.connect() as conn:
                for row, values in zip(rows, vectors):
                    blob, dimensions, norm = vector_to_blob(values)
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO transcript_embedding(
                            segment_id, model, dimensions, vector_norm, vector
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            row["id"],
                            model_key,
                            dimensions,
                            norm,
                            sqlite3.Binary(blob),
                        ),
                    )
                    embedded += 1
        if embedded:
            with self.db.connect() as conn:
                self._event(conn, "info", f"Embedded {embedded} transcript segments")
        return embedded

    def _load_segments_missing_embeddings(
        self,
        media_id: int | None,
        limit: int,
    ) -> list[sqlite3.Row]:
        params: list[Any] = [embedding_model_key(self.settings)]
        media_filter = ""
        if media_id is not None:
            media_filter = "AND s.media_id=?"
            params.append(media_id)
        params.append(limit)
        with self.db.connect() as conn:
            return conn.execute(
                f"""
                SELECT
                    s.id,
                    s.text,
                    s.start_seconds,
                    s.end_seconds,
                    m.filename,
                    (
                        SELECT group_concat(ordered.text, ' ')
                        FROM (
                            SELECT w.text
                            FROM transcript_segment w
                            WHERE w.media_id = s.media_id
                              AND w.start_seconds >= MAX(0, s.start_seconds - 15)
                              AND w.start_seconds <= s.end_seconds + 15
                            ORDER BY w.start_seconds
                        ) ordered
                    ) AS context_text
                FROM transcript_segment s
                JOIN media m ON m.id = s.media_id
                LEFT JOIN transcript_embedding e
                    ON e.segment_id = s.id AND e.model=?
                WHERE e.segment_id IS NULL
                {media_filter}
                ORDER BY s.id
                LIMIT ?
                """,
                params,
            ).fetchall()

    def _embedding_text(self, row: sqlite3.Row) -> str:
        context = str(row["context_text"] or row["text"])
        return (
            f"当前对白: {row['text']}\n"
            f"上下文对白: {context}"
        )

    def _search_normalized_phrase(self, normalized: str, limit: int) -> list[sqlite3.Row]:
        with self.db.connect() as conn:
            return conn.execute(
                """
                SELECT s.*, m.filename, m.path
                FROM transcript_segment s
                JOIN media m ON m.id = s.media_id
                WHERE s.normalized_text LIKE ?
                ORDER BY s.start_seconds
                LIMIT ?
                """,
                (f"%{normalized}%", limit),
            ).fetchall()

    def _search_fts(self, query: str, limit: int, seen: set[int]) -> list[sqlite3.Row]:
        terms = [term for term in query.replace('"', " ").split() if term]
        if not terms:
            terms = [query]
        fts_query = " OR ".join(f'"{term}"' for term in terms)
        params: list[Any] = [fts_query]
        exclusion = ""
        if seen:
            placeholders = ",".join("?" for _ in seen)
            exclusion = f"AND s.id NOT IN ({placeholders})"
            params.extend(sorted(seen))
        params.append(limit)

        with self.db.connect() as conn:
            try:
                return conn.execute(
                    f"""
                    SELECT s.*, m.filename, m.path, bm25(transcript_fts) AS rank
                    FROM transcript_fts
                    JOIN transcript_segment s ON s.id = transcript_fts.rowid
                    JOIN media m ON m.id = s.media_id
                    WHERE transcript_fts MATCH ?
                    {exclusion}
                    ORDER BY rank
                    LIMIT ?
                    """,
                    params,
                ).fetchall()
            except sqlite3.OperationalError:
                return []

    def _search_semantic(self, query: str, limit: int, seen: set[int]) -> list[SearchResult]:
        if not self.settings.enable_embeddings or not is_embedding_available(self.settings):
            return []
        try:
            query_vector = self.embedding_client.embed_texts([self._semantic_query_text(query)])[0]
        except Exception as exc:
            with self.db.connect() as conn:
                self._event(conn, "warning", f"Semantic search skipped: {exc}")
            return []

        query_norm = math.sqrt(sum(value * value for value in query_vector))
        candidates: list[SearchResult] = []
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT s.*, m.filename, m.path, e.vector, e.vector_norm
                FROM transcript_embedding e
                JOIN transcript_segment s ON s.id = e.segment_id
                JOIN media m ON m.id = s.media_id
                WHERE e.model=?
                """,
                (embedding_model_key(self.settings),),
            ).fetchall()

        for row in rows:
            if int(row["id"]) in seen:
                continue
            stored = blob_to_vector(row["vector"])
            score = cosine_similarity(query_vector, query_norm, stored.values, float(row["vector_norm"]))
            if score < self.settings.semantic_min_score:
                continue
            candidates.append(
                self._row_to_search_result(
                    row,
                    query,
                    score=score,
                    match_type="语义",
                    reason=f"本地向量相似度 {score:.3f}",
                )
            )

        candidates.sort(key=lambda result: result.score, reverse=True)
        return candidates[: min(limit, self.settings.semantic_candidates)]

    def _semantic_query_text(self, query: str) -> str:
        instruction = self.settings.embedding_query_instruction
        if not instruction and self.settings.embedding_backend == "local":
            model_name = self.settings.local_embedding_model.lower()
            if "bge" in model_name:
                instruction = "为这个句子生成表示以用于检索相关视频字幕："
        if not instruction:
            return query
        return f"{instruction}{query}"

    def _row_to_search_result(
        self,
        row: sqlite3.Row,
        query: str,
        score: float,
        match_type: str,
        reason: str = "",
    ) -> SearchResult:
        start = max(0.0, float(row["start_seconds"]) - 5.0)
        end = float(row["end_seconds"]) + 5.0
        return SearchResult(
            segment_id=int(row["id"]),
            media_id=int(row["media_id"]),
            filename=str(row["filename"]),
            path=str(row["path"]),
            start_seconds=start,
            end_seconds=end,
            text=str(row["text"]),
            preview_text=snippet(str(row["text"]), query),
            score=score,
            match_type=match_type,
            reason=reason,
        )

    def progress(self) -> dict[str, Any]:
        with self.db.connect() as conn:
            media_counts = {
                row["status"]: row["count"]
                for row in conn.execute(
                    "SELECT status, COUNT(*) AS count FROM media GROUP BY status"
                ).fetchall()
            }
            chunk_counts = {
                row["status"]: row["count"]
                for row in conn.execute(
                    "SELECT status, COUNT(*) AS count FROM audio_chunk GROUP BY status"
                ).fetchall()
            }
            total_segments = conn.execute("SELECT COUNT(*) FROM transcript_segment").fetchone()[0]
            total_words = conn.execute("SELECT COUNT(*) FROM transcript_word").fetchone()[0]
            total_embeddings = conn.execute(
                "SELECT COUNT(*) FROM transcript_embedding WHERE model=?",
                (embedding_model_key(self.settings),),
            ).fetchone()[0]
            progress_row = conn.execute(
                "SELECT * FROM processing_progress WHERE id=1"
            ).fetchone()
            events = [
                row_to_dict(row)
                for row in conn.execute(
                    "SELECT * FROM app_event ORDER BY id DESC LIMIT 20"
                ).fetchall()
            ]
        return {
            "media": media_counts,
            "chunks": chunk_counts,
            "segments": total_segments,
            "words": total_words,
            "embeddings": total_embeddings,
            "current": row_to_dict(progress_row)
            or {"active": 0, "stage": "idle", "percent": 0, "message": ""},
            "events": events,
        }

    def list_media(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, path, filename, size_bytes, duration_seconds, status, error_message, updated_at
                FROM media
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [row_to_dict(row) or {} for row in rows]

    def media_by_id(self, media_id: int) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM media WHERE id=?", (media_id,)).fetchone()
        return row_to_dict(row)

    def export_csv(self, query: str) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["media_id", "filename", "path", "start_seconds", "end_seconds", "text"])
        for result in self.search(query, limit=1000):
            writer.writerow(
                [
                    result.media_id,
                    result.filename,
                    result.path,
                    f"{result.start_seconds:.3f}",
                    f"{result.end_seconds:.3f}",
                    result.text,
                ]
            )
        return output.getvalue()

    def _event(self, conn: sqlite3.Connection, level: str, message: str) -> None:
        conn.execute("INSERT INTO app_event(level, message) VALUES (?, ?)", (level, message))

    def _set_progress(
        self,
        *,
        active: bool,
        stage: str,
        message: str,
        percent: float,
        media_id: int | None = None,
        filename: str | None = None,
        current_chunk: int = 0,
        total_chunks: int = 0,
        current_seconds: float = 0.0,
        total_seconds: float = 0.0,
    ) -> None:
        safe_percent = max(0.0, min(100.0, float(percent)))
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO processing_progress(
                    id, active, stage, media_id, filename, current_chunk, total_chunks,
                    current_seconds, total_seconds, percent, message, updated_at
                )
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    active=excluded.active,
                    stage=excluded.stage,
                    media_id=excluded.media_id,
                    filename=excluded.filename,
                    current_chunk=excluded.current_chunk,
                    total_chunks=excluded.total_chunks,
                    current_seconds=excluded.current_seconds,
                    total_seconds=excluded.total_seconds,
                    percent=excluded.percent,
                    message=excluded.message,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    1 if active else 0,
                    stage,
                    media_id,
                    filename or "",
                    int(current_chunk),
                    int(total_chunks),
                    float(current_seconds),
                    float(total_seconds),
                    safe_percent,
                    message,
                ),
            )
