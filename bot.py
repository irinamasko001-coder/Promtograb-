from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import math
import os
import re
import secrets
import socket
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiosqlite
import gdown
import httpx
import yt_dlp
from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandObject
from aiogram.types import (
    BotCommand,
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LabeledPrice,
    Message,
    PreCheckoutQuery,
)
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from openai import AsyncOpenAI


def env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    openai_key: str = os.getenv("OPENAI_API_KEY", "")
    kie_key: str = os.getenv("KIE_API_KEY", "")
    owner_id: int = int(os.getenv("OWNER_ID", "780477475"))
    data_dir: Path = Path(os.getenv("DATA_DIR", "/data"))
    max_video_seconds: int = int(os.getenv("MAX_VIDEO_SECONDS", "600"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "50"))
    starting_tokens: int = int(os.getenv("STARTING_TOKENS", "30"))
    tokens_per_second: int = int(os.getenv("TOKENS_PER_SECOND", "1"))
    public_base_url: str = os.getenv("PUBLIC_BASE_URL", "")
    payments_enabled: bool = env_bool("PAYMENTS_ENABLED", False)
    support_username: str = os.getenv("SUPPORT_USERNAME", "@support")
    transcription_model: str = os.getenv("TRANSCRIPTION_MODEL", "gpt-transcribe")
    kie_visual_model: str = os.getenv("KIE_VISUAL_MODEL", "gemini-3-8-flash")
    kie_final_model: str = os.getenv("KIE_FINAL_MODEL", "gpt-5-6-terra")
    moderation_model: str = os.getenv("MODERATION_MODEL", "omni-moderation-latest")
    moderation_enabled: bool = env_bool("MODERATION_ENABLED", True)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "bot.sqlite3"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def base_url(self) -> str:
        if self.public_base_url:
            return self.public_base_url.rstrip("/")
        domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "").strip()
        return f"https://{domain}" if domain else ""

    @property
    def packages(self) -> dict[int, int]:
        return {
            600: int(os.getenv("PACKAGE_600_STARS", "300")),
            1800: int(os.getenv("PACKAGE_1800_STARS", "700")),
            6000: int(os.getenv("PACKAGE_6000_STARS", "1900")),
        }

    def validate(self) -> None:
        missing = [
            name
            for name, value in (
                ("TELEGRAM_BOT_TOKEN", self.telegram_token),
                ("OPENAI_API_KEY", self.openai_key),
                ("KIE_API_KEY", self.kie_key),
            )
            if not value
        ]
        if missing:
            raise RuntimeError("Missing environment variables: " + ", ".join(missing))


S = Settings()
URL_RE = re.compile(r"https?://[^\s<>]+", re.IGNORECASE)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
MEDIA_FILES: dict[str, Path] = {}
WORK_SEMAPHORE = asyncio.Semaphore(2)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def split_text(text: str, limit: int = 3800) -> list[str]:
    result: list[str] = []
    rest = text.strip()
    while rest:
        if len(rest) <= limit:
            result.append(rest)
            break
        cut = rest.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = rest.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        result.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    return result


def chunks_with_intro(intro: str, text: str, limit: int = 3800) -> list[str]:
    """split_text, but fold a short intro line into the first chunk instead of
    sending it as its own message — keeps the reply count to a minimum."""
    chunks = split_text(text, limit)
    if not chunks:
        return [intro]
    combined = intro + "\n\n" + chunks[0]
    if len(combined) <= limit:
        chunks[0] = combined
    else:
        chunks.insert(0, intro)
    return chunks


def safe_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return clean[:120] or "video.mp4"


def fmt_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    return f"{total // 60}:{total % 60:02d}"


class Database:
    def __init__(self, path: Path):
        self.path = path

    async def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as db:
            await db.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    display_name TEXT NOT NULL,
                    balance INTEGER NOT NULL DEFAULT 0,
                    consented INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    source TEXT NOT NULL,
                    duration REAL NOT NULL DEFAULT 0,
                    token_cost INTEGER NOT NULL DEFAULT 0,
                    scenario_path TEXT,
                    error TEXT,
                    revisions INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    delta INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    reference TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS payments (
                    charge_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    stars INTEGER NOT NULL,
                    tokens INTEGER NOT NULL,
                    refunded INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_id, created_at);
                """
            )
            await db.commit()

    async def ensure_user(self, user_id: int, username: str | None, name: str) -> bool:
        async with aiosqlite.connect(self.path) as db:
            await db.execute("BEGIN IMMEDIATE")
            row = await (await db.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,))).fetchone()
            created = row is None
            if created:
                await db.execute(
                    "INSERT INTO users VALUES(?,?,?,?,0,?,?)",
                    (user_id, username, name, S.starting_tokens, now_iso(), now_iso()),
                )
                if S.starting_tokens:
                    await db.execute(
                        "INSERT INTO ledger(user_id,delta,reason,reference,created_at) VALUES(?,?,?,?,?)",
                        (user_id, S.starting_tokens, "welcome_bonus", None, now_iso()),
                    )
            else:
                await db.execute(
                    "UPDATE users SET username=?, display_name=?, updated_at=? WHERE user_id=?",
                    (username, name, now_iso(), user_id),
                )
            await db.commit()
            return created

    async def consent(self, user_id: int) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute("UPDATE users SET consented=1, updated_at=? WHERE user_id=?", (now_iso(), user_id))
            await db.commit()

    async def has_consent(self, user_id: int) -> bool:
        async with aiosqlite.connect(self.path) as db:
            row = await (await db.execute("SELECT consented FROM users WHERE user_id=?", (user_id,))).fetchone()
            return bool(row and row[0])

    async def balance(self, user_id: int) -> int:
        async with aiosqlite.connect(self.path) as db:
            row = await (await db.execute("SELECT balance FROM users WHERE user_id=?", (user_id,))).fetchone()
            return int(row[0]) if row else 0

    async def create_job(self, job_id: str, user_id: int, chat_id: int, source: str) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT INTO jobs(job_id,user_id,chat_id,status,source,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                (job_id, user_id, chat_id, "received", source, now_iso(), now_iso()),
            )
            await db.commit()

    async def reserve(self, job_id: str, user_id: int, duration: float, amount: int) -> bool:
        async with aiosqlite.connect(self.path) as db:
            await db.execute("BEGIN IMMEDIATE")
            if user_id != S.owner_id:
                row = await (await db.execute("SELECT balance FROM users WHERE user_id=?", (user_id,))).fetchone()
                if not row or int(row[0]) < amount:
                    await db.execute(
                        "UPDATE jobs SET status='insufficient',duration=?,token_cost=?,updated_at=? WHERE job_id=?",
                        (duration, amount, now_iso(), job_id),
                    )
                    await db.commit()
                    return False
                await db.execute("UPDATE users SET balance=balance-?,updated_at=? WHERE user_id=?", (amount, now_iso(), user_id))
                await db.execute(
                    "INSERT INTO ledger(user_id,delta,reason,reference,created_at) VALUES(?,?,?,?,?)",
                    (user_id, -amount, "video_analysis", job_id, now_iso()),
                )
            await db.execute(
                "UPDATE jobs SET status='processing',duration=?,token_cost=?,updated_at=? WHERE job_id=?",
                (duration, amount, now_iso(), job_id),
            )
            await db.commit()
            return True

    async def complete(self, job_id: str, scenario_path: Path) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "UPDATE jobs SET status='completed',scenario_path=?,updated_at=? WHERE job_id=?",
                (str(scenario_path), now_iso(), job_id),
            )
            await db.commit()

    async def refund_job(self, job_id: str, error: str) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute("BEGIN IMMEDIATE")
            row = await (
                await db.execute("SELECT user_id,status,token_cost FROM jobs WHERE job_id=?", (job_id,))
            ).fetchone()
            if not row:
                await db.rollback()
                return
            user_id, status, amount = int(row[0]), str(row[1]), int(row[2])
            if status == "processing" and amount > 0 and user_id != S.owner_id:
                await db.execute("UPDATE users SET balance=balance+?,updated_at=? WHERE user_id=?", (amount, now_iso(), user_id))
                await db.execute(
                    "INSERT INTO ledger(user_id,delta,reason,reference,created_at) VALUES(?,?,?,?,?)",
                    (user_id, amount, "automatic_refund", job_id, now_iso()),
                )
            await db.execute(
                "UPDATE jobs SET status='refunded',error=?,updated_at=? WHERE job_id=?",
                (error[:1000], now_iso(), job_id),
            )
            await db.commit()

    async def latest_job(self, user_id: int) -> dict[str, Any] | None:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            row = await (
                await db.execute(
                    "SELECT * FROM jobs WHERE user_id=? AND status='completed' ORDER BY created_at DESC LIMIT 1",
                    (user_id,),
                )
            ).fetchone()
            return dict(row) if row else None

    async def get_job(self, job_id: str, user_id: int) -> dict[str, Any] | None:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            row = await (
                await db.execute("SELECT * FROM jobs WHERE job_id=? AND user_id=?", (job_id, user_id))
            ).fetchone()
            return dict(row) if row else None

    async def save_revision(self, job_id: str) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute("UPDATE jobs SET revisions=revisions+1,updated_at=? WHERE job_id=?", (now_iso(), job_id))
            await db.commit()

    async def grant(self, user_id: int, amount: int, reference: str = "owner_grant") -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute("BEGIN IMMEDIATE")
            await db.execute("UPDATE users SET balance=balance+?,updated_at=? WHERE user_id=?", (amount, now_iso(), user_id))
            await db.execute(
                "INSERT INTO ledger(user_id,delta,reason,reference,created_at) VALUES(?,?,?,?,?)",
                (user_id, amount, reference, None, now_iso()),
            )
            await db.commit()

    async def add_payment(self, user_id: int, charge_id: str, payload: str, stars: int, tokens: int) -> bool:
        async with aiosqlite.connect(self.path) as db:
            await db.execute("BEGIN IMMEDIATE")
            seen = await (await db.execute("SELECT charge_id FROM payments WHERE charge_id=?", (charge_id,))).fetchone()
            if seen:
                await db.rollback()
                return False
            await db.execute(
                "INSERT INTO payments VALUES(?,?,?,?,?,0,?)",
                (charge_id, user_id, payload, stars, tokens, now_iso()),
            )
            await db.execute("UPDATE users SET balance=balance+?,updated_at=? WHERE user_id=?", (tokens, now_iso(), user_id))
            await db.execute(
                "INSERT INTO ledger(user_id,delta,reason,reference,created_at) VALUES(?,?,?,?,?)",
                (user_id, tokens, "stars_payment", charge_id, now_iso()),
            )
            await db.commit()
            return True

    async def payment_for_refund(self, charge_id: str) -> tuple[int, int] | None:
        async with aiosqlite.connect(self.path) as db:
            row = await (
                await db.execute("SELECT user_id,tokens FROM payments WHERE charge_id=? AND refunded=0", (charge_id,))
            ).fetchone()
            return (int(row[0]), int(row[1])) if row else None

    async def mark_refunded(self, charge_id: str, user_id: int, tokens: int) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute("BEGIN IMMEDIATE")
            await db.execute("UPDATE payments SET refunded=1 WHERE charge_id=?", (charge_id,))
            await db.execute("UPDATE users SET balance=MAX(0,balance-?),updated_at=? WHERE user_id=?", (tokens, now_iso(), user_id))
            await db.execute(
                "INSERT INTO ledger(user_id,delta,reason,reference,created_at) VALUES(?,?,?,?,?)",
                (user_id, -tokens, "stars_refund", charge_id, now_iso()),
            )
            await db.commit()

    async def stats(self) -> tuple[int, int, int]:
        async with aiosqlite.connect(self.path) as db:
            users = (await (await db.execute("SELECT COUNT(*) FROM users")).fetchone())[0]
            jobs = (await (await db.execute("SELECT COUNT(*) FROM jobs WHERE status='completed'")).fetchone())[0]
            sold = (await (await db.execute("SELECT COALESCE(SUM(tokens),0) FROM payments WHERE refunded=0")).fetchone())[0]
            return int(users), int(jobs), int(sold)


DB = Database(S.db_path)


async def run_command(*args: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode:
        raise RuntimeError(stderr.decode("utf-8", "replace")[-2000:])
    return stdout.decode("utf-8", "replace")


async def video_info(path: Path) -> dict[str, Any]:
    raw = await run_command(
        "ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height,r_frame_rate",
        "-select_streams", "v:0", "-of", "json", str(path)
    )
    data = json.loads(raw)
    duration = float(data.get("format", {}).get("duration") or 0)
    stream = (data.get("streams") or [{}])[0]
    return {"duration": duration, "width": stream.get("width"), "height": stream.get("height")}


async def extract_audio(video: Path, audio: Path) -> Path:
    await run_command(
        "ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", str(audio)
    )
    return audio


async def extract_analysis_frames(video: Path, target_dir: Path, duration: float) -> list[tuple[float, Path]]:
    """Extract timestamped JPEGs for a fallback when direct video analysis fails."""
    target_dir.mkdir(parents=True, exist_ok=True)
    if duration <= 15:
        fps = 3.0
    elif duration <= 60:
        fps = 2.0
    elif duration <= 180:
        fps = 1.0
    else:
        fps = 0.5
    frame_count = min(180, max(2, math.ceil(duration * fps) + 1))
    end = max(0.0, duration - 0.04)
    timestamps = [i * end / (frame_count - 1) for i in range(frame_count)]
    frames: list[tuple[float, Path]] = []
    for index, timestamp in enumerate(timestamps):
        output = target_dir / f"frame-{index:04d}.jpg"
        await run_command(
            "ffmpeg", "-y", "-ss", f"{timestamp:.3f}", "-i", str(video),
            "-frames:v", "1", "-vf", "scale='min(768,iw)':-2",
            "-pix_fmt", "yuvj420p", "-threads", "1", "-q:v", "3", str(output)
        )
        if output.exists() and output.stat().st_size:
            frames.append((timestamp, output))
    if not frames:
        raise RuntimeError("Не удалось извлечь кадры из видео")
    return frames


async def detect_cuts(video: Path, duration: float) -> list[float]:
    """Return likely hard-cut timestamps. Gemini still verifies them against the video."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-i",
        str(video),
        "-vf",
        "select=gt(scene\\,0.30),showinfo",
        "-an",
        "-f",
        "null",
        "-",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    text = stderr.decode("utf-8", "replace")
    found = [float(x) for x in re.findall(r"pts_time:([0-9.]+)", text)]
    points = [0.0]
    for value in sorted(found):
        if 0.08 < value < duration - 0.08 and value - points[-1] > 0.12:
            points.append(value)
    if duration - points[-1] > 0.08:
        points.append(duration)
    return points


def validate_public_url_sync(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Нужна публичная ссылка http или https")
    host = parsed.hostname.lower()
    if host in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Локальные ссылки не поддерживаются")
    try:
        addresses = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
        for item in addresses:
            ip = ipaddress.ip_address(item[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                raise ValueError("Локальные и закрытые адреса не поддерживаются")
    except socket.gaierror as exc:
        raise ValueError("Не удалось открыть адрес ссылки") from exc
    return url


def download_url_sync(url: str, target_dir: Path) -> Path:
    validate_public_url_sync(url)
    target_dir.mkdir(parents=True, exist_ok=True)
    if "drive.google.com" in url:
        output = target_dir / "source-video"
        result = gdown.download(url=url, output=str(output), quiet=True, fuzzy=True)
        if not result:
            raise RuntimeError("Google Drive не отдал файл. Включите доступ «всем, у кого есть ссылка».")
        return Path(result)
    options = {
        "format": "bv*+ba/b",
        "outtmpl": str(target_dir / "source.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "retries": 3,
        "max_filesize": S.max_upload_bytes,
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            requested = info.get("requested_downloads") or []
            candidates = [Path(x["filepath"]) for x in requested if x.get("filepath")]
            candidates.append(Path(ydl.prepare_filename(info)))
    except Exception as exc:
        raise RuntimeError(
            "Не удалось получить видео по ссылке. Пришлите сам файл или публичную ссылку Google Drive. "
            f"Техническая причина: {str(exc)[-500:]}"
        ) from exc
    candidates.extend(p for p in target_dir.iterdir() if p.is_file() and not p.name.endswith(".part"))
    for path in candidates:
        if path.exists() and path.stat().st_size:
            return path
    raise RuntimeError("Видео по ссылке не найдено")


async def download_url(url: str, target_dir: Path) -> Path:
    return await asyncio.to_thread(download_url_sync, url, target_dir)


def media_url(path: Path) -> tuple[str, str]:
    if not S.base_url:
        raise RuntimeError("Для видеоанализа ещё не создан публичный домен Railway")
    token = secrets.token_urlsafe(32)
    MEDIA_FILES[token] = path
    return f"{S.base_url}/media/{token}", token


def extract_kie_text(data: Any) -> str:
    if not isinstance(data, dict):
        return str(data)
    choices = data.get("choices") or []
    if choices:
        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(x.get("text", "") for x in content if isinstance(x, dict))
    candidates = data.get("candidates") or []
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "\n".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text"))
        if text:
            return text
    output = data.get("output") or []
    texts: list[str] = []
    for item in output:
        for part in item.get("content", []) if isinstance(item, dict) else []:
            if isinstance(part, dict) and part.get("text"):
                texts.append(part["text"])
    if texts:
        return "\n".join(texts)
    return data.get("output_text") or data.get("text") or json.dumps(data, ensure_ascii=False)


async def kie_request(
    url: str,
    payload: dict[str, Any],
    *,
    attempts: int = 3,
    timeout_seconds: float = 600.0,
) -> str:
    headers = {"Authorization": f"Bearer {S.kie_key}", "Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_seconds, connect=30.0)
            ) as client:
                response = await client.post(url, headers=headers, json=payload)
                if response.is_error:
                    detail = response.text.strip()[:1000]
                    raise RuntimeError(
                        f"Kie API HTTP {response.status_code}: {detail or response.reason_phrase}"
                    )
                ctype = response.headers.get("content-type", "")
                if "text/event-stream" in ctype:
                    final: dict[str, Any] | None = None
                    deltas: list[str] = []
                    for line in response.text.splitlines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        with suppress(json.JSONDecodeError):
                            event = json.loads(raw)
                            final = event.get("response") if isinstance(event, dict) and event.get("response") else final
                            delta = event.get("delta") if isinstance(event, dict) else None
                            if isinstance(delta, str):
                                deltas.append(delta)
                    return extract_kie_text(final) if final else "".join(deltas)
                data = response.json()
                if isinstance(data, dict):
                    code = data.get("code")
                    if isinstance(code, int) and code >= 400:
                        raise RuntimeError(f"Kie API code {code}: {data.get('msg') or data}")
                    if data.get("error"):
                        raise RuntimeError(f"Kie API error response: {data['error']}")
                result = extract_kie_text(data).strip()
                if not result:
                    raise RuntimeError("Kie API вернул пустой ответ")
                # Some upstream failures arrive inside a nominally successful
                # response as JSON text in the model's content field.
                with suppress(json.JSONDecodeError):
                    nested = json.loads(result)
                    if isinstance(nested, dict):
                        nested_code = nested.get("code")
                        if isinstance(nested_code, int) and nested_code >= 400:
                            raise RuntimeError(
                                f"Kie API code {nested_code}: {nested.get('msg') or nested}"
                            )
                        if nested.get("error"):
                            raise RuntimeError(f"Kie API error response: {nested['error']}")
                return result
        except (httpx.HTTPError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            if attempt < attempts - 1:
                await asyncio.sleep(2 ** attempt)
    raise RuntimeError(f"Kie API error: {last_error}")


OPENAI = AsyncOpenAI(api_key=S.openai_key) if S.openai_key else None


async def transcribe(audio: Path) -> str:
    if not OPENAI:
        raise RuntimeError("OPENAI_API_KEY не настроен")
    with audio.open("rb") as file:
        result = await OPENAI.audio.transcriptions.create(
            model=S.transcription_model,
            file=file,
            prompt=(
                "Точная дословная расшифровка исходной речи. Сохраняй язык, имена, междометия, "
                "паузы и незавершённые фразы. Не переписывай реплики более литературно."
            ),
        )
    return result.text.strip()


async def moderation_frames(video: Path, duration: float, target_dir: Path) -> list[Path]:
    """A handful of evenly spaced low-res frames used only for content moderation
    (not for the full visual analysis)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    count = min(6, max(2, math.ceil(duration / 5)))
    end = max(0.0, duration - 0.04)
    timestamps = [i * end / max(1, count - 1) for i in range(count)]
    frames: list[Path] = []
    for index, timestamp in enumerate(timestamps):
        output = target_dir / f"mod-{index:02d}.jpg"
        await run_command(
            "ffmpeg", "-y", "-ss", f"{timestamp:.3f}", "-i", str(video),
            "-frames:v", "1", "-vf", "scale='min(512,iw)':-2",
            "-pix_fmt", "yuvj420p", "-threads", "1", "-q:v", "5", str(output),
        )
        if output.exists() and output.stat().st_size:
            frames.append(output)
    return frames


async def check_moderation(frames: list[Path]) -> bool:
    """Return True if any sampled frame is flagged as sexual/explicit content.
    Moderation-service failures never block processing — only an explicit flag does."""
    if not S.moderation_enabled or not OPENAI or not frames:
        return False
    for frame in frames:
        data_url = "data:image/jpeg;base64," + base64.b64encode(frame.read_bytes()).decode("ascii")
        try:
            result = await OPENAI.moderations.create(
                model=S.moderation_model,
                input=[{"type": "image_url", "image_url": {"url": data_url}}],
            )
        except Exception:
            continue
        for item in result.results:
            categories = item.categories
            if getattr(categories, "sexual", False) or getattr(categories, "sexual_minors", False):
                return True
    return False


VISUAL_PROMPT = """Проанализируй ПРИКРЕПЛЁННОЕ ВИДЕО целиком как режиссёр монтажа. Ничего не додумывай.
Нужно вернуть подробные фактические наблюдения на русском языке:
1. Точная длительность и ориентация.
2. Каждая монтажная склейка с максимально точным таймкодом.
3. Для каждого плана: крупность, ракурс, положение и движение камеры, кто уже находится в первом кадре.
4. Положение каждого героя, движение корпуса, головы, плеч, рук, кистей и пальцев; направление взгляда.
5. Видимые эмоции и их изменение без психологических догадок.
6. Реплики с примерными таймкодами и говорящими; отдельно речь за кадром.
7. Свет, фон, предметы, музыка, шумы и синхронные звуки.
8. Каждый VFX как отдельное событие: момент возникновения, форма, движение и исчезновение.

Критически важно: слово «появляется» используй только если герой или объект реально входит в кадр.
Если герой уже стоит или сидит в первом кадре, так и напиши. Не добавляй паузы, замирания,
жесты или эффекты, которых нет. Особо внимательно отслеживай последовательность жестов обеими руками.
Верни хронологический анализ с таймкодами, пригодный для восстановления ролика план в план."""


def visual_payload(content: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "include_thoughts": False,
        "reasoning_effort": "medium",
    }


async def analyze_frame_batches(
    frames: list[tuple[float, Path]], cut_points: list[float]
) -> tuple[str, list[str]]:
    """Analyze timestamped frames in ordered batches."""
    tokens: list[str] = []
    reports: list[str] = []
    try:
        for batch_number, start in enumerate(range(0, len(frames), 12), 1):
            batch = frames[start:start + 12]
            content: list[dict[str, Any]] = [{
                "type": "text",
                "text": (
                    VISUAL_PROMPT
                    + "\n\nНиже идут последовательные кадры исходного ролика. Перед каждым "
                    "указан точный таймкод. Анализируй только этот диапазон, сохраняя порядок. "
                    f"Пакет {batch_number}; границы склеек детектора: "
                    + ", ".join(f"{value:.3f}" for value in cut_points)
                    + "."
                ),
            }]
            for timestamp, path in batch:
                url, token = media_url(path)
                tokens.append(token)
                content.append({"type": "text", "text": f"Кадр, таймкод {timestamp:.3f} с:"})
                content.append({"type": "image_url", "image_url": {"url": url}})
            report = await kie_request(
                "https://api.kie.ai/gemini-3-8-flash-openai/v1/chat/completions",
                visual_payload(content), attempts=3, timeout_seconds=300.0,
            )
            if is_refusal(report):
                raise RuntimeError(
                    "Модель отказалась анализировать видео из-за политики безопасности "
                    "(возможно, в кадре что-то похожее на оружие, насилие или другой "
                    "чувствительный элемент)."
                )
            reports.append(f"ПАКЕТ КАДРОВ {batch_number}:\n{report}")
        return "\n\n".join(reports), tokens
    except Exception:
        for token in tokens:
            MEDIA_FILES.pop(token, None)
        raise


POLICY_REFUSAL_MARKERS = (
    "prohibited use policy", "sensitive words", "could not be submitted",
    "generative ai prohibited", "content policy", "safety policy",
    "cannot generate", "cannot create", "i cannot assist", "i can't help with that",
    "against our usage policies", "policy violation", "violates google",
    "i'm not able to", "i am not able to", "i won't be able to",
    "нарушает политику", "против политики", "не могу сгенерировать",
    "не могу создать", "запрещённый контент", "не соответствует политике",
    "не могу помочь с этим запросом", "против правил использования",
)


def is_refusal(text: str) -> bool:
    """True only when the response as a whole looks like a refusal — short,
    with a refusal phrase in it. A genuine analysis/scenario/prompt is always
    long (that's what the system prompts demand), so gating on length first
    prevents false positives from a stray hedge phrase inside real, valid,
    long output (that was the bug: matching the marker anywhere in the text,
    with no length check, flagged perfectly good long responses as refusals)."""
    stripped = text.strip()
    if len(stripped) > 600:
        return False
    lowered = stripped.lower()
    return any(marker in lowered for marker in POLICY_REFUSAL_MARKERS)


async def analyze_video(
    url: str, cut_points: list[float], video: Path, target_dir: Path, duration: float
) -> tuple[str, list[str]]:
    cut_hint = ", ".join(f"{value:.3f}" for value in cut_points)
    content = [
        {
            "type": "text",
            "text": VISUAL_PROMPT
            + "\n\nЛокальный детектор кадров предложил следующие границы (секунды): "
            + cut_hint
            + ". Проверь каждую по самому видео, исправь ложные и добавь пропущенные.",
        },
        {"type": "image_url", "image_url": {"url": url}},
    ]
    try:
        result = await kie_request(
            "https://api.kie.ai/gemini-3-8-flash-openai/v1/chat/completions",
            visual_payload(content), attempts=3, timeout_seconds=300.0,
        )
        lowered = result.lower()
        error_markers = (
            '"code": 524', '"code":524', "http 400", "interal error",
        )
        # Gemini sometimes answers with a normal HTTP 200 but plain text saying
        # it couldn't actually see/reach the video (wrong content-type, slow
        # Railway response, oversized file, etc). That text doesn't match the
        # technical error markers above, so it used to be accepted as a valid
        # (but empty) analysis. Catch those phrasings explicitly.
        no_video_markers = (
            "не вижу", "не видно", "не удалось получить доступ", "не могу получить доступ",
            "не могу просмотреть", "не могу открыть", "не могу воспроизвести",
            "не найдено видео", "видео не найдено", "нет прикреплённого видео",
            "нет прикрепленного видео", "не был предоставлен", "не была предоставлена",
            "отсутствует видео", "нет видео", "не содержит видео", "недоступн",
            "cannot access", "can't access", "unable to access", "unable to view",
            "unable to load", "i don't see", "i do not see", "i cannot see",
            "no video", "no image or video", "video could not be", "failed to load",
            "unable to retrieve", "unable to fetch", "i don't have access",
            "i do not have access",
        )
        looks_empty = any(marker in lowered for marker in error_markers)
        looks_blind = any(marker in lowered for marker in no_video_markers)
        looks_refused = is_refusal(result)
        # A genuine per-shot breakdown for a real video is long. A refusal or
        # "I can't see it" reply is almost always short — use that as a second
        # signal so phrasings not covered by the lists above are still caught.
        looks_too_short = len(result.strip()) < 400
        if looks_empty or looks_blind or looks_refused or looks_too_short:
            raise RuntimeError(
                "Прямой анализ видео по ссылке не удался (пустой/короткий/отрицающий/отказной ответ модели)"
            )
        return result, []
    except RuntimeError:
        frames = await extract_analysis_frames(video, target_dir / "analysis-frames", duration)
        return await analyze_frame_batches(frames, cut_points)


SCENARIO_SYSTEM = """Ты создаёшь точную реконструкцию фактически проанализированного видео.
Опирайся только на переданные метаданные, дословную транскрипцию и визуальные наблюдения.
Не добавляй ради красоты действий, жестов, пауз, эмоций, дыма, света, предметов или склеек,
которых нет в данных. Если конкретная деталь не подтверждается, помести её в раздел проверки.

Реплики сохраняй дословно. Не исправляй странные фразы на более логичные. Для речи за кадром
явно пиши «за кадром». Если герой уже находится в первом кадре, пиши «уже стоит/сидит»;
не пиши, что он появляется. Описывай движения физически последовательно: исходная поза,
направление, движение, смысловой акцент, конечное положение. Не используй расплывчатое
«жестикулирует» без расшифровки движений ладоней и кистей.

ФОРМА ОТВЕТА:
# Название
- точная продолжительность, формат, разрешение, стиль, локация, количество планов

## Персонажи
Только фактически видимые постоянные признаки.

## Монтажная схема
| План | Таймкод | Длительность | Крупность | Персонажи | Переход |

## Подробный сценарий по планам
Для каждого плана:
### План N — название
**Таймкод:** 00:00.000–00:00.000
**Продолжительность:**
**Крупность:**
**Ракурс:**
**Камера:**
**Переход:**

Затем разбей действие внутри плана на точные временные отрезки. Для каждого отрезка укажи:
**Действие**, **Реплика**, **Эмоция**, **Жесты**, **Взгляд**, **Свет**, **Звук**, **VFX**, **Монтаж**.
Не заполняй поле выдумкой: если элемента нет, кратко укажи «нет».

## Полный диалог
Все реплики по порядку с именами говорящих.

## Места, требующие проверки
Добавляй только при реальной неопределённости с конкретным таймкодом и причиной.

Пиши по-русски, подробно и профессионально. Верни полный сценарий целиком."""


async def terra_text(instruction: str, content: str, effort: str = "medium") -> str:
    payload = {
        "model": S.kie_final_model,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": instruction + "\n\n" + content}],
            }
        ],
        "reasoning": {"effort": effort},
        "max_output_tokens": 22000,
        "stream": False,
    }
    try:
        # The Codex-compatible Kie endpoint can occasionally keep an SSE
        # connection open indefinitely. Cap the whole request and then use the
        # already configured Gemini endpoint as a text-only fallback.
        result = await asyncio.wait_for(
            kie_request(
                "https://api.kie.ai/codex/v1/responses",
                payload,
                attempts=1,
                timeout_seconds=240.0,
            ),
            timeout=300.0,
        )
        if is_refusal(result):
            raise RuntimeError("Codex endpoint refused the request")
        return result
    except (asyncio.TimeoutError, RuntimeError, httpx.HTTPError):
        fallback_payload = {
            "messages": [
                {
                    "role": "user",
                    "content": instruction + "\n\n" + content,
                }
            ],
            "stream": False,
            "include_thoughts": False,
            "reasoning_effort": effort,
        }
        result = await kie_request(
            "https://api.kie.ai/gemini-3-8-flash-openai/v1/chat/completions",
            fallback_payload,
            attempts=2,
            timeout_seconds=300.0,
        )
        if is_refusal(result):
            raise RuntimeError(
                "Модель отказалась выполнить запрос из-за политики безопасности "
                "(вероятно, в видео есть элемент, который она сочла чувствительным: "
                "оружие, насилие и т.п.)."
            )
        return result


async def create_scenario(meta: dict[str, Any], transcript: str, visual: str) -> str:
    content = (
        "МЕТАДАННЫЕ:\n"
        f"Длительность: {meta['duration']:.3f} секунды\n"
        f"Размер: {meta.get('width')}x{meta.get('height')}\n\n"
        "ДОСЛОВНАЯ ТРАНСКРИПЦИЯ OPENAI:\n"
        f"{transcript}\n\n"
        "АНАЛИЗ САМОГО ВИДЕО GEMINI:\n"
        f"{visual}"
    )
    return (await terra_text(SCENARIO_SYSTEM, content, "high")).strip()


SEEDANCE_SYSTEM = """Преобразуй точный реконструированный сценарий в готовые промпты Seedance 2.0.
Пиши только на русском языке — английской версии не давай вообще, даже частично.

ГЛАВНОЕ ПРАВИЛО ЯЗЫКА: пиши максимально просто и прямо, как инструкцию для нейросети-генератора,
а не как киноведческий разбор. Короткие прямые предложения. Никакого профессионального
киножаргона («мизансцена», «полиэкранная композиция», «внутрикадровый монтаж» и подобное) —
если нужно описать план из нескольких элементов, просто перечисли, что где находится, обычными
словами. Не усложняй описание движений: вместо длинных цепочек «сгибает, разгибает, ритмично
двигает в такт» пиши коротко и естественно, например «бежит с реалистичной физикой движений» —
детали физики нужны только там, где без них план непонятен. Не пиши двусмысленные фразы,
которые нейросеть может понять наоборот (например «лицо скрыто дистанцией» может быть прочитано
как указание что-то скрыть) — пиши прямо, что́ видно, а не что не видно.

Не добавляй ничего, чего нет в сценарии. Сохраняй точные реплики, эмоции, ключевые жесты,
направления взглядов и отдельные появления VFX — но формулируй компактно и без повторов.

ССЫЛКИ НА РЕФЕРЕНСЫ (@image1, @image2…)
Референсом обозначай только то, что реально можно сфотографировать и прикрепить отдельным
файлом: внешность каждого персонажа (один референс на персонажа) и отдельный характерный
предмет/реквизит крупным планом (оружие, бутылка, конкретный аксессуар). НЕ создавай отдельный
@image для локации, фона или общей атмосферы сцены — место действия и окружение просто описывай
текстом внутри плана, без номера референса. В самом начале одной строкой перечисли все
использованные референсы: «@image1 — кто/что», «@image2 — кто/что» и т.д., коротко.

ВРЕМЯ
Округляй все таймкоды до целых секунд (0–2s, 2–5s, 5–9s и т.д.), не пиши доли секунды и
миллисекунды.

КРИТИЧЕСКИЕ ОГРАНИЧЕНИЯ — ТОЛЬКО КОГДА РЕАЛЬНО НУЖНЫ
Не добавляй этот блок «на всякий случай». Пиши его только если в кадре одновременно несколько
персонажей (тогда одной строкой: «В кадре ровно N человек: [список]. Без дублей») или если есть
специфический технический риск (например, резкая смена стиля кадра посреди плана). Если такого
риска нет — просто не создавай этот блок вообще.

СТРУКТУРА КАЖДОЙ ЧАСТИ (максимум 15 секунд; длинный ролик дели по склейкам/смене говорящего).
Не пиши сплошным текстом — каждый параметр с новой строки, с жирной подписью, коротко:

**Референсы:** @image1 — …, @image2 — …
**Стиль:** фотореалистичное кинематографическое качество, не мультфильм, не пластик.
**Формат:** вертикальный 9:16, N шотов.

**Shot 1 (0–Ns)**
**Крупность:** (простыми словами: крупный план лица / по пояс / в полный рост)
**Камера:** где стоит и как снимает, простыми словами
**Положение:** кто где — LEFT / RIGHT / center
**Фон:** одной фразой, что видно позади (обязательно, если герой один в кадре)
**Действие:** что происходит, по порядку, простыми предложениями
**Эмоция:** что видно на лице — глаза, брови, губы (без одного слова вроде «грустная»)
**Реплика:** тон голоса коротко + сама реплика в кавычках
**VFX:** только если реально есть эффект

Если персонаж обращается к кому-то за кадром — одной строкой укажи, где этот кто-то за кадром
(LEFT/RIGHT) и куда смотрит говорящий; если адресат — зритель, пиши, что взгляд направлен в
камеру. Фон не переописывай заново, если он не поменялся с предыдущего Shot — пиши «тот же фон».
Внешность персонажа заново не пересказывай — она уже дана в референсах, в Shot ссылайся на
@imageN.

**Audio:** тип голоса + реплики по порядку + окружающие звуки; «без музыки», если музыки нет.
**Свет:** одна-две фразы.

Финальная строка: X seconds. Vertical 9:16. 720p. 24fps.

СЛОВ-ТРИГГЕРОВ ИЗБЕГАЙ (могут заблокировать генерацию):
— слова «юная/подросток/несовершеннолетн*» в описании внешности;
— «текстура кожи», «поры»;
— анатомические термины вроде «грудь» — опиши силуэт или одежду вместо этого;
— «светящиеся/сверкающие глаза» — вместо этого «яркие глаза», «широко раскрытые глаза»;
— КАПСЛОК в описании эмоций — обычный регистр;
— подряд несколько слов про злость/издёвку/травлю — разбавляй нейтральной лексикой;
— студийные имена и узнаваемые визуальные атрибуты персонажей известных франшиз — называй
  персонажа по роли («бог подземного царства», «девушка в сиреневом платье»), а не собственным
  именем конкретной франшизы.

Не пиши вступления и заключения — начинай сразу с первой части и заканчивай последней строкой
последнего шота. Каждую самостоятельную часть оформляй отдельным блоком в тройных обратных
кавычках."""


async def create_seedance(scenario: str) -> str:
    return (await terra_text(SEEDANCE_SYSTEM, "РЕКОНСТРУИРОВАННЫЙ СЦЕНАРИЙ:\n" + scenario, "high")).strip()


async def revise_scenario(scenario: str, correction: str) -> str:
    instruction = SCENARIO_SYSTEM + (
        "\n\nПравка пользователя имеет высший приоритет. Верни полный исправленный сценарий целиком, "
        "а не отдельный фрагмент. Сохрани все остальные утверждённые детали."
    )
    content = f"ПРАВКА ПОЛЬЗОВАТЕЛЯ:\n{correction}\n\nТЕКУЩИЙ СЦЕНАРИЙ:\n{scenario}"
    return (await terra_text(instruction, content, "high")).strip()


async def revise_seedance(prompt: str, correction: str) -> str:
    instruction = SEEDANCE_SYSTEM + (
        "\n\nПравка пользователя имеет высший приоритет. Верни полный исправленный промпт целиком, "
        "а не отдельный фрагмент. Сохрани все остальные утверждённые детали, которые правка не касается."
    )
    content = f"ПРАВКА ПОЛЬЗОВАТЕЛЯ:\n{correction}\n\nТЕКУЩИЙ ПРОМПТ:\n{prompt}"
    return (await terra_text(instruction, content, "high")).strip()


BOT = Bot(S.telegram_token) if S.telegram_token else None
DP = Dispatcher()
ROUTER = Router()
DP.include_router(ROUTER)


def consent_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Разрешаю обработку видео", callback_data="consent")]
        ]
    )


def buy_keyboard() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text=f"{tokens} токенов — {stars} ⭐", callback_data=f"buy:{tokens}")]
        for tokens, stars in S.packages.items()
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def scenario_keyboard(job_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Создать промпты Seedance", callback_data=f"seedance:{job_id}")]
        ]
    )


def seedance_keyboard(job_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✏️ Исправить промпт", callback_data=f"revise_prompt:{job_id}")]
        ]
    )


# user_id -> job_id: set when someone taps "✏️ Исправить промпт" and cleared once
# their next plain-text message is consumed as the correction (see `fallback`).
PENDING_PROMPT_REVISIONS: dict[int, str] = {}


async def ensure_known(message: Message) -> bool:
    if not message.from_user:
        return False
    await DB.ensure_user(message.from_user.id, message.from_user.username, message.from_user.full_name)
    if not await DB.has_consent(message.from_user.id):
        await message.answer(
            "Для анализа видео файл временно передаётся внешним автоматизированным сервисам обработки. "
            "Подтвердите, что у вас есть право использовать видео и вы разрешаете автоматическую обработку.",
            reply_markup=consent_keyboard(),
        )
        return False
    return True


@ROUTER.message(Command("start"))
async def command_start(message: Message) -> None:
    if not message.from_user:
        return
    created = await DB.ensure_user(message.from_user.id, message.from_user.username, message.from_user.full_name)
    if not await DB.has_consent(message.from_user.id):
        bonus = f" Вам начислено {S.starting_tokens} пробных токенов." if created and S.starting_tokens else ""
        await message.answer(
            "Бот принимает видеофайлы и публичные ссылки, распознаёт речь, анализирует планы, "
            "жесты и эмоции и выдаёт готовые промпты для Seedance 2.0." + bonus +
            "\n\nДля анализа файл временно передаётся внешним автоматизированным сервисам обработки. "
            "Подтвердите, что у вас есть право использовать видео и вы разрешаете обработку.",
            reply_markup=consent_keyboard(),
        )
        return
    await message.answer("Пришлите видеофайл или публичную ссылку на видео. Проверить баланс: /balance")


@ROUTER.callback_query(F.data == "consent")
async def callback_consent(callback: CallbackQuery) -> None:
    await DB.ensure_user(callback.from_user.id, callback.from_user.username, callback.from_user.full_name)
    await DB.consent(callback.from_user.id)
    await callback.answer("Разрешение сохранено")
    if callback.message:
        await callback.message.edit_text("Готово. Теперь пришлите видеофайл или публичную ссылку.")


@ROUTER.message(Command("balance"))
async def command_balance(message: Message) -> None:
    if not message.from_user:
        return
    await DB.ensure_user(message.from_user.id, message.from_user.username, message.from_user.full_name)
    if message.from_user.id == S.owner_id:
        await message.answer("Ваш баланс: без ограничений — вы владелец бота.")
        return
    balance = await DB.balance(message.from_user.id)
    await message.answer(
        f"Ваш баланс: {balance} видеотокенов.\n"
        f"Примерно {balance // 60} мин {balance % 60} сек видео.",
        reply_markup=buy_keyboard() if S.payments_enabled else None,
    )


@ROUTER.message(Command("buy"))
async def command_buy(message: Message) -> None:
    if not S.payments_enabled:
        await message.answer(
            "Автоматическое пополнение пока не включено. Обратитесь в поддержку: " + S.support_username
        )
        return
    await message.answer("Выберите пакет видеотокенов:", reply_markup=buy_keyboard())


@ROUTER.callback_query(F.data.startswith("buy:"))
async def callback_buy(callback: CallbackQuery) -> None:
    if not callback.data or not callback.message:
        return
    if not S.payments_enabled:
        await callback.answer("Пополнение пока не включено", show_alert=True)
        return
    tokens = int(callback.data.split(":", 1)[1])
    stars = S.packages.get(tokens)
    if not stars:
        await callback.answer("Пакет не найден", show_alert=True)
        return
    payload = f"tokens:{tokens}:stars:{stars}"
    await callback.message.answer_invoice(
        title=f"{tokens} видеотокенов",
        description=f"Баланс для анализа {tokens} секунд исходного видео",
        payload=payload,
        currency="XTR",
        prices=[LabeledPrice(label=f"{tokens} видеотокенов", amount=stars)],
        provider_token="",
    )
    await callback.answer()


@ROUTER.pre_checkout_query()
async def pre_checkout(query: PreCheckoutQuery) -> None:
    match = re.fullmatch(r"tokens:(\d+):stars:(\d+)", query.invoice_payload)
    if not match:
        await query.answer(ok=False, error_message="Некорректный платёж")
        return
    tokens, stars = map(int, match.groups())
    if S.packages.get(tokens) != stars or query.total_amount != stars or query.currency != "XTR":
        await query.answer(ok=False, error_message="Цена пакета изменилась. Откройте /buy снова.")
        return
    await query.answer(ok=True)


@ROUTER.message(F.successful_payment)
async def successful_payment(message: Message) -> None:
    if not message.from_user or not message.successful_payment:
        return
    payment = message.successful_payment
    match = re.fullmatch(r"tokens:(\d+):stars:(\d+)", payment.invoice_payload)
    if not match:
        return
    tokens, stars = map(int, match.groups())
    added = await DB.add_payment(
        message.from_user.id,
        payment.telegram_payment_charge_id,
        payment.invoice_payload,
        stars,
        tokens,
    )
    if added:
        await message.answer(f"Оплата получена. Начислено {tokens} видеотокенов.")


@ROUTER.message(Command("privacy"))
async def command_privacy(message: Message) -> None:
    await message.answer(
        "Видео временно хранится на сервере и передаётся внешним автоматизированным сервисам для "
        "распознавания речи, анализа изображения и подготовки результата. Исходные файлы удаляются "
        "после завершения обработки или ошибки. "
        "Не присылайте видео, на использование которых у вас нет права."
    )


@ROUTER.message(Command("terms"))
async def command_terms(message: Message) -> None:
    await message.answer(
        "1 видеотокен оплачивает анализ 1 секунды исходного видео. Продолжительность округляется вверх, "
        "минимальное списание — 15 токенов. При технической ошибке токены возвращаются автоматически. "
        "Результат создаётся нейросетями и может потребовать проверки пользователем."
    )


@ROUTER.message(Command("support"))
@ROUTER.message(Command("paysupport"))
async def command_support(message: Message) -> None:
    await message.answer("Поддержка и вопросы по платежам: " + S.support_username)


@ROUTER.message(Command("myid"))
async def command_myid(message: Message) -> None:
    if message.from_user:
        await message.answer(f"Ваш Telegram ID: {message.from_user.id}")


@ROUTER.message(Command("grant"))
async def command_grant(message: Message) -> None:
    if not message.from_user or message.from_user.id != S.owner_id:
        return
    parts = (message.text or "").split()
    if len(parts) != 3:
        await message.answer("Формат: /grant TELEGRAM_ID КОЛИЧЕСТВО")
        return
    try:
        user_id, amount = int(parts[1]), int(parts[2])
    except ValueError:
        await message.answer("ID и количество должны быть числами")
        return
    if amount <= 0:
        await message.answer("Количество должно быть больше нуля")
        return
    await DB.grant(user_id, amount)
    await message.answer(f"Пользователю {user_id} начислено {amount} токенов.")


@ROUTER.message(Command("stats"))
async def command_stats(message: Message) -> None:
    if not message.from_user or message.from_user.id != S.owner_id:
        return
    users, jobs, sold = await DB.stats()
    await message.answer(f"Пользователи: {users}\nГотовые анализы: {jobs}\nПродано токенов: {sold}")


@ROUTER.message(Command("refund"))
async def command_refund(message: Message) -> None:
    if not message.from_user or message.from_user.id != S.owner_id or not BOT:
        return
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) != 2:
        await message.answer("Формат: /refund TELEGRAM_PAYMENT_CHARGE_ID")
        return
    charge_id = parts[1].strip()
    payment = await DB.payment_for_refund(charge_id)
    if not payment:
        await message.answer("Платёж не найден или уже возвращён.")
        return
    user_id, tokens = payment
    await BOT.refund_star_payment(user_id=user_id, telegram_payment_charge_id=charge_id)
    await DB.mark_refunded(charge_id, user_id, tokens)
    await message.answer("Возврат выполнен.")


@ROUTER.message(Command("revise"))
async def command_revise(message: Message, command: CommandObject) -> None:
    if not await ensure_known(message) or not message.from_user:
        return
    correction = (command.args or "").strip()
    if not correction:
        await message.answer("Напишите правку после команды: /revise Аид уже стоит в первом кадре…")
        return
    job = await DB.latest_job(message.from_user.id)
    if not job or not job.get("scenario_path"):
        await message.answer("Сначала обработайте видео.")
        return
    path = Path(job["scenario_path"])
    if not path.exists():
        await message.answer("Файл сценария не найден.")
        return
    status = await message.answer("Переписываю полный сценарий с учётом правки…")
    try:
        updated = await revise_scenario(path.read_text(encoding="utf-8"), correction)
        path.write_text(updated, encoding="utf-8")
        await DB.save_revision(job["job_id"])
        await status.edit_text("Готово.")
        for chunk in chunks_with_intro("Исправленный сценарий (файл приложен):", updated):
            await message.answer(chunk)
        await message.answer_document(FSInputFile(path, filename=f"scenario-{job['job_id'][:8]}.md"))
    except Exception as exc:
        await status.edit_text("Не удалось применить правку. Попробуйте ещё раз.\n" + str(exc)[-500:])


@ROUTER.callback_query(F.data.startswith("seedance:"))
async def callback_seedance(callback: CallbackQuery) -> None:
    if not callback.data or not callback.message:
        return
    job_id = callback.data.split(":", 1)[1]
    job = await DB.get_job(job_id, callback.from_user.id)
    if not job or not job.get("scenario_path"):
        await callback.answer("Сценарий не найден", show_alert=True)
        return
    await callback.answer("Создаю полный комплект промптов")
    status = await callback.message.answer("Создаю промпты Seedance 2.0…")
    try:
        scenario = Path(job["scenario_path"]).read_text(encoding="utf-8")
        result = await create_seedance(scenario)
        path = Path(job["scenario_path"]).with_name("seedance-prompts.md")
        path.write_text(result, encoding="utf-8")
        await status.edit_text("Готово.")
        for chunk in chunks_with_intro("Промпт Seedance 2.0 (файл приложен):", result):
            await callback.message.answer(chunk)
        await callback.message.answer_document(
            FSInputFile(path, filename=f"seedance-{job_id[:8]}.md"),
            reply_markup=seedance_keyboard(job_id),
        )
    except Exception as exc:
        await status.edit_text("Не удалось создать промпт. Попробуйте позже.\n" + str(exc)[-500:])


async def process_job(job_id: str, user_id: int, chat_id: int, source: str, local_path: Path | None, status_id: int) -> None:
    if not BOT:
        return
    job_dir = S.jobs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    media_tokens: list[str] = []
    charged = False

    async def status(text: str) -> None:
        with suppress(Exception):
            await BOT.edit_message_text(text, chat_id=chat_id, message_id=status_id)

    async def keep_typing() -> None:
        # Telegram's native "typing…" indicator already animates with dots and
        # fades after ~5s, so we just need to keep refreshing it — this reads
        # as continuous activity without editing the status text every second.
        with suppress(Exception):
            while True:
                await BOT.send_chat_action(chat_id, ChatAction.TYPING)
                await asyncio.sleep(4.0)

    typing_task = asyncio.create_task(keep_typing())

    async with WORK_SEMAPHORE:
        try:
            video = local_path
            if video is None:
                await status("1/5 — Загружаю видео по ссылке…")
                video = await download_url(source, job_dir)
            if not video.exists() or not video.stat().st_size:
                raise RuntimeError("Видео не загрузилось")
            if video.stat().st_size > S.max_upload_bytes:
                raise RuntimeError(f"Файл больше лимита {S.max_upload_mb} МБ")

            await status("2/5 — Проверяю длительность и баланс…")
            meta = await video_info(video)
            duration = float(meta["duration"])
            if duration <= 0:
                raise RuntimeError("Не удалось определить длительность видео")
            if duration > S.max_video_seconds:
                raise RuntimeError(f"Максимальная длительность сейчас {S.max_video_seconds // 60} минут")

            if S.moderation_enabled:
                await status("2/5 — Проверяю содержание видео…")
                try:
                    mod_frames = await moderation_frames(video, duration, job_dir / "moderation-frames")
                except Exception:
                    mod_frames = []
                if await check_moderation(mod_frames):
                    await status(
                        "Это видео нельзя обработать: обнаружен откровенный сексуальный контент. "
                        "Пришлите другое видео."
                    )
                    return

            cost = max(15, math.ceil(duration * S.tokens_per_second))
            charged = await DB.reserve(job_id, user_id, duration, cost)
            if not charged:
                balance = await DB.balance(user_id)
                await status(
                    f"Недостаточно токенов. Видео {fmt_duration(duration)} стоит {cost} токенов, "
                    f"на балансе {balance}."
                )
                if S.payments_enabled:
                    await BOT.send_message(chat_id, "Выберите пополнение:", reply_markup=buy_keyboard())
                return

            await status(f"3/5 — Распознаю речь и анализирую видео… Списано {cost} токенов.")
            audio = await extract_audio(video, job_dir / "audio.mp3")
            cut_points = await detect_cuts(video, duration)
            public_url, media_token = media_url(video)
            media_tokens.append(media_token)
            transcript_task = asyncio.create_task(transcribe(audio))
            visual_task = asyncio.create_task(
                analyze_video(public_url, cut_points, video, job_dir, duration)
            )
            transcript, visual_result = await asyncio.gather(transcript_task, visual_task)
            visual, fallback_tokens = visual_result
            media_tokens.extend(fallback_tokens)

            await status("4/5 — Собираю планы, реплики, жесты и эмоции по секундам…")
            scenario = await create_scenario(meta, transcript, visual)
            scenario_path = job_dir / "scenario.md"
            scenario_path.write_text(scenario, encoding="utf-8")
            await status("5/5 — Создаю готовые промпты для Seedance 2.0…")
            seedance = await create_seedance(scenario)
            seedance_path = job_dir / "seedance-prompts.md"
            seedance_path.write_text(seedance, encoding="utf-8")
            await DB.complete(job_id, scenario_path)
            await status("5/5 — Готово.")

            for chunk in chunks_with_intro("Промпт Seedance 2.0 (файл приложен):", seedance):
                await BOT.send_message(chat_id, chunk)
            await BOT.send_document(
                chat_id,
                FSInputFile(seedance_path, filename=f"seedance-{job_id[:8]}.md"),
                caption="Промпт Seedance 2.0.",
                reply_markup=seedance_keyboard(job_id),
            )
        except Exception as exc:
            if charged:
                await DB.refund_job(job_id, str(exc))
            else:
                await DB.refund_job(job_id, str(exc))
            await status("Обработка не завершилась. Списанные токены возвращены автоматически.")
            await BOT.send_message(chat_id, "Причина: " + str(exc)[-1200:])
        finally:
            typing_task.cancel()
            with suppress(Exception):
                await typing_task
            for media_token in media_tokens:
                MEDIA_FILES.pop(media_token, None)
            if job_dir.exists():
                for path in job_dir.iterdir():
                    if path.name in {"scenario.md", "seedance-prompts.md"}:
                        continue
                    if path.is_dir():
                        for sub in path.iterdir():
                            sub.unlink(missing_ok=True)
                        with suppress(OSError):
                            path.rmdir()
                    elif path.is_file():
                        path.unlink(missing_ok=True)


@ROUTER.message(F.video | F.document)
async def receive_file(message: Message) -> None:
    if not await ensure_known(message) or not message.from_user or not BOT:
        return
    media = message.video or message.document
    if not media:
        return
    mime = getattr(media, "mime_type", "") or ""
    suffix = Path(getattr(media, "file_name", "") or "").suffix.lower()
    if message.document and not (mime.startswith("video/") or suffix in VIDEO_EXTENSIONS):
        await message.answer("Это не похоже на видеофайл.")
        return
    if media.file_size and media.file_size > S.max_upload_bytes:
        await message.answer(f"Файл больше {S.max_upload_mb} МБ. Пришлите публичную ссылку.")
        return
    job_id = uuid.uuid4().hex
    job_dir = S.jobs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    filename = safe_name(getattr(media, "file_name", "") or f"telegram-{media.file_unique_id}.mp4")
    path = job_dir / filename
    status = await message.answer("Загружаю видео из Telegram…")
    try:
        await BOT.download(media.file_id, destination=path)
    except Exception as exc:
        await status.edit_text("Telegram не отдал файл. Попробуйте отправить его как видео или дать ссылку.\n" + str(exc)[-500:])
        return
    await DB.create_job(job_id, message.from_user.id, message.chat.id, "telegram-file")
    await status.edit_text("Видео принято. Начинаю анализ…")
    asyncio.create_task(process_job(job_id, message.from_user.id, message.chat.id, "telegram-file", path, status.message_id))


@ROUTER.message(F.text.regexp(URL_RE))
async def receive_url(message: Message) -> None:
    if not await ensure_known(message) or not message.from_user:
        return
    match = URL_RE.search(message.text or "")
    if not match:
        return
    url = match.group(0).rstrip(".,);]")
    job_id = uuid.uuid4().hex
    status = await message.answer("Ссылка принята. Начинаю загрузку…")
    await DB.create_job(job_id, message.from_user.id, message.chat.id, url)
    asyncio.create_task(process_job(job_id, message.from_user.id, message.chat.id, url, None, status.message_id))


@ROUTER.callback_query(F.data.startswith("revise_prompt:"))
async def callback_revise_prompt(callback: CallbackQuery) -> None:
    if not callback.data or not callback.message or not callback.from_user:
        return
    job_id = callback.data.split(":", 1)[1]
    job = await DB.get_job(job_id, callback.from_user.id)
    if not job or not job.get("scenario_path"):
        await callback.answer("Промпт не найден", show_alert=True)
        return
    path = Path(job["scenario_path"]).with_name("seedance-prompts.md")
    if not path.exists():
        await callback.answer("Файл промпта не найден", show_alert=True)
        return
    PENDING_PROMPT_REVISIONS[callback.from_user.id] = job_id
    await callback.answer()
    await callback.message.answer("Напишите одним сообщением, что нужно исправить в промпте.")


@ROUTER.message()
async def fallback(message: Message) -> None:
    if not await ensure_known(message) or not message.from_user:
        return
    job_id = PENDING_PROMPT_REVISIONS.pop(message.from_user.id, None)
    if job_id:
        correction = (message.text or "").strip()
        if not correction:
            PENDING_PROMPT_REVISIONS[message.from_user.id] = job_id
            await message.answer("Пришлите правку текстом одним сообщением.")
            return
        job = await DB.get_job(job_id, message.from_user.id)
        if not job or not job.get("scenario_path"):
            await message.answer("Промпт не найден.")
            return
        path = Path(job["scenario_path"]).with_name("seedance-prompts.md")
        if not path.exists():
            await message.answer("Файл промпта не найден.")
            return
        status = await message.answer("Переписываю промпт с учётом правки…")
        try:
            updated = await revise_seedance(path.read_text(encoding="utf-8"), correction)
            path.write_text(updated, encoding="utf-8")
            await status.edit_text("Готово.")
            for chunk in chunks_with_intro("Исправленный промпт (файл приложен):", updated):
                await message.answer(chunk)
            await message.answer_document(
                FSInputFile(path, filename=f"seedance-{job_id[:8]}.md"),
                reply_markup=seedance_keyboard(job_id),
            )
        except Exception as exc:
            await status.edit_text("Не удалось применить правку. Попробуйте ещё раз.\n" + str(exc)[-500:])
        return
    await message.answer("Пришлите видеофайл или одну публичную ссылку на видео.")


POLLING_TASK: asyncio.Task[Any] | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global POLLING_TASK
    S.validate()
    S.data_dir.mkdir(parents=True, exist_ok=True)
    S.jobs_dir.mkdir(parents=True, exist_ok=True)
    await DB.init()
    if not BOT:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не настроен")
    await BOT.set_my_commands(
        [
            BotCommand(command="start", description="🚀 Запустить бота"),
            BotCommand(command="balance", description="💰 Баланс видеотокенов"),
            BotCommand(command="buy", description="⭐ Пополнить баланс"),
            BotCommand(command="privacy", description="Обработка и хранение видео"),
            BotCommand(command="terms", description="Условия использования"),
            BotCommand(command="support", description="Поддержка"),
            BotCommand(command="paysupport", description="Вопросы по платежам"),
        ]
    )
    POLLING_TASK = asyncio.create_task(DP.start_polling(BOT, allowed_updates=DP.resolve_used_update_types()))
    yield
    if POLLING_TASK:
        POLLING_TASK.cancel()
        with suppress(asyncio.CancelledError):
            await POLLING_TASK
    await BOT.session.close()


app = FastAPI(title="Telegram Video Script Bot", lifespan=lifespan)


@app.get("/")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "telegram-video-script-bot"}


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def guess_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image/jpeg" if suffix in {".jpg", ".jpeg"} else f"image/{suffix.lstrip('.')}"
    if suffix in VIDEO_EXTENSIONS:
        return "video/mp4" if suffix == ".mp4" else f"video/{suffix.lstrip('.')}"
    # Files downloaded via gdown (Google Drive) can end up without any
    # extension at all (see download_url_sync). Default to video/mp4 there —
    # everything this bot serves without a recognized suffix is source video,
    # never an arbitrary unknown type. Without an explicit content-type the
    # remote analysis service can fail to recognize the payload as a video.
    return "video/mp4"


@app.get("/media/{token}")
async def serve_media(token: str) -> FileResponse:
    path = MEDIA_FILES.get(token)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Media expired")
    return FileResponse(path, media_type=guess_media_type(path))
