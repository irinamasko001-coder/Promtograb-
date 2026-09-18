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
    tokens_per_photo: int = int(os.getenv("TOKENS_PER_PHOTO", "5"))
    public_base_url: str = os.getenv("PUBLIC_BASE_URL", "")
    payments_enabled: bool = env_bool("PAYMENTS_ENABLED", False)
    support_username: str = os.getenv("SUPPORT_USERNAME", "@support")
    transcription_model: str = os.getenv("TRANSCRIPTION_MODEL", "gpt-transcribe")
    kie_visual_model: str = os.getenv("KIE_VISUAL_MODEL", "gemini-3-8-flash")
    kie_final_model: str = os.getenv("KIE_FINAL_MODEL", "gpt-5-6-terra")
    openai_text_model: str = os.getenv("OPENAI_TEXT_MODEL", "gpt-5.6-terra")
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
            300: int(os.getenv("PACKAGE_300_STARS", "150")),
            600: int(os.getenv("PACKAGE_600_STARS", "250")),
            1200: int(os.getenv("PACKAGE_1200_STARS", "500")),
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
            # Remembered engine choice, so the user picks Seedance/Grok once and
            # every later link uses it. Added via migration for existing tables.
            with suppress(Exception):
                await db.execute("ALTER TABLE users ADD COLUMN engine TEXT")
            await db.commit()

    async def set_engine(self, user_id: int, engine: str) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "UPDATE users SET engine=?, updated_at=? WHERE user_id=?",
                (engine, now_iso(), user_id),
            )
            await db.commit()

    async def get_engine(self, user_id: int) -> str | None:
        async with aiosqlite.connect(self.path) as db:
            row = await (
                await db.execute("SELECT engine FROM users WHERE user_id=?", (user_id,))
            ).fetchone()
            return row[0] if row and row[0] else None

    async def ensure_user(self, user_id: int, username: str | None, name: str) -> bool:
        async with aiosqlite.connect(self.path) as db:
            await db.execute("BEGIN IMMEDIATE")
            row = await (await db.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,))).fetchone()
            created = row is None
            if created:
                # Columns are named explicitly: a positional INSERT breaks the
                # moment the table gains a column (that is exactly what the
                # `engine` migration did — every user lookup started failing,
                # so the bot stopped responding to anything).
                await db.execute(
                    "INSERT INTO users(user_id,username,display_name,balance,consented,"
                    "created_at,updated_at) VALUES(?,?,?,?,0,?,?)",
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

    async def charge(self, user_id: int, amount: int, reason: str) -> bool:
        """Deduct a flat fee (photo/idea prompts). Owner is never charged."""
        if user_id == S.owner_id:
            return True
        async with aiosqlite.connect(self.path) as db:
            await db.execute("BEGIN IMMEDIATE")
            row = await (await db.execute("SELECT balance FROM users WHERE user_id=?", (user_id,))).fetchone()
            if not row or int(row[0]) < amount:
                await db.commit()
                return False
            await db.execute(
                "UPDATE users SET balance=balance-?,updated_at=? WHERE user_id=?",
                (amount, now_iso(), user_id),
            )
            await db.execute(
                "INSERT INTO ledger(user_id,delta,reason,reference,created_at) VALUES(?,?,?,?,?)",
                (user_id, -amount, reason, None, now_iso()),
            )
            await db.commit()
            return True

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
                "INSERT INTO payments(charge_id,user_id,payload,stars,tokens,refunded,created_at) VALUES(?,?,?,?,?,0,?)",
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
        text = stderr.decode("utf-8", "replace")
        # ffmpeg prints a long version/metadata banner before the real error.
        # Keep only the meaningful lines so the user sees the cause, not the banner.
        noise = (
            "ffmpeg version", "built with", "configuration:", "libav", "libsw", "libpost",
            "Metadata:", "major_brand", "minor_version", "compatible_brands", "creation_time",
            "handler_name", "vendor_id", "encoder", "Duration:", "Stream #", "Input #",
            "Output #", "Stream mapping:", "Press [q]", "  ",
        )
        lines = [
            line for line in text.splitlines()
            if line.strip() and not any(line.startswith(n) or line.lstrip().startswith(n) for n in noise)
        ]
        raise RuntimeError(("\n".join(lines) or text)[-800:])
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


async def has_audio_stream(video: Path) -> bool:
    """Many downloaded reels are video-only (muted clip, or the audio track was
    never merged). Extracting audio from those makes ffmpeg fail outright, so
    check first instead of crashing the whole job."""
    try:
        raw = await run_command(
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_type", "-of", "json", str(video),
        )
        return bool(json.loads(raw).get("streams"))
    except Exception:
        return False


async def extract_audio(video: Path, audio: Path) -> Path | None:
    if not await has_audio_stream(video):
        return None
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
    # Extract every frame in ONE ffmpeg pass. The previous version spawned a
    # separate ffmpeg process per frame (31+ processes for a 10s clip, each
    # re-opening and seeking the file) which dominated total runtime.
    await run_command(
        "ffmpeg", "-y", "-i", str(video),
        "-vf", f"fps={fps},scale='min(768,iw)':-2",
        "-pix_fmt", "yuvj420p", "-threads", "1", "-q:v", "3",
        str(target_dir / "frame-%04d.jpg"),
    )
    extracted = sorted(target_dir.glob("frame-*.jpg"))
    frames: list[tuple[float, Path]] = []
    for index, path in enumerate(extracted[:frame_count]):
        if path.exists() and path.stat().st_size:
            frames.append((index / fps, path))
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
    # Single ffmpeg pass (was one process per frame).
    rate = max(count / duration, 0.01) if duration > 0 else 1.0
    await run_command(
        "ffmpeg", "-y", "-i", str(video),
        "-vf", f"fps={rate:.4f},scale='min(512,iw)':-2",
        "-pix_fmt", "yuvj420p", "-threads", "1", "-q:v", "5",
        str(target_dir / "mod-%02d.jpg"),
    )
    return [p for p in sorted(target_dir.glob("mod-*.jpg"))[:count] if p.stat().st_size]


async def check_moderation(frames: list[Path]) -> bool:
    """Return True if any sampled frame is flagged as sexual/explicit content.
    Moderation-service failures never block processing — only an explicit flag does."""
    if not S.moderation_enabled or not OPENAI or not frames:
        return False

    async def flagged(frame: Path) -> bool:
        data_url = "data:image/jpeg;base64," + base64.b64encode(frame.read_bytes()).decode("ascii")
        try:
            result = await OPENAI.moderations.create(
                model=S.moderation_model,
                input=[{"type": "image_url", "image_url": {"url": data_url}}],
            )
        except Exception:
            return False
        for item in result.results:
            categories = item.categories
            if getattr(categories, "sexual", False) or getattr(categories, "sexual_minors", False):
                return True
        return False

    # Check every frame at once instead of one-by-one.
    results = await asyncio.gather(*(flagged(f) for f in frames), return_exceptions=True)
    return any(r is True for r in results)


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
    batches = [frames[start:start + 12] for start in range(0, len(frames), 12)]

    async def analyze_batch(batch_number: int, batch: list[tuple[float, Path]]) -> str:
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": (
                VISUAL_PROMPT
                + "\n\nНиже идут последовательные кадры исходного ролика. Перед каждым "
                "указан точный таймкод. Анализируй только этот диапазон, сохраняя порядок. "
                f"Пакет {batch_number}; границы склеек детектора: "
                + ", ".join(f"{value:.3f}" for value in cut_points)
                + ".\n\nКРИТИЧЕСКИ ВАЖНО: это НЕ набор отдельных фотографий, а "
                "последовательность моментов одного непрерывного движения. Твоя главная "
                "задача — описать, что ПРОИСХОДИТ МЕЖДУ кадрами: какое действие началось, "
                "как оно развивается и чем заканчивается. Сравнивай каждый кадр с предыдущим "
                "и описывай изменение как непрерывное действие с глаголами движения "
                "(«поднимает», «срезает», «падает», «подбегает»), а не как статичное "
                "состояние («держит», «находится», «стоит»). Если между двумя кадрами "
                "предмет изменил положение — значит произошло действие, назови его. Не пиши "
                "покадровый список — пиши связное описание происходящего по таймкодам."
            ),
        }]
        for timestamp, path in batch:
            data_url = "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({"type": "text", "text": f"Кадр, таймкод {timestamp:.3f} с:"})
            content.append({"type": "image_url", "image_url": {"url": data_url}})
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
        return f"ПАКЕТ КАДРОВ {batch_number}:\n{report}"

    try:
        # Analyze all batches concurrently instead of one after another.
        reports = await asyncio.gather(
            *(analyze_batch(n, b) for n, b in enumerate(batches, 1))
        )
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

ПЛАН = МОНТАЖНАЯ СКЛЕЙКА. Новый план начинается ТОЛЬКО на границе склейки, указанной во
входных данных. Минимальная длина плана — 3 секунды; на 15-секундное видео обычно 3–5 планов,
а при съёмке одним куском — один план на всё видео. Смена крупности, движение камеры, новое
действие, смена эмоции или появление предмета — это НЕ новый план, всё это описывается внутри
текущего плана по секундам. Никогда не делай план на каждую секунду.

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


async def openai_text(instruction: str, content: str) -> str:
    """Text-only generation via OpenAI. Used as a fallback when the Kie
    endpoints refuse or fail — OpenAI's policy filters are considerably less
    trigger-happy on ordinary creative/scene description than Gemini's."""
    if not OPENAI:
        raise RuntimeError("OPENAI_API_KEY не настроен")
    response = await OPENAI.chat.completions.create(
        model=S.openai_text_model,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": content},
        ],
        max_completion_tokens=16000,
    )
    text = (response.choices[0].message.content or "").strip()
    # Mark truncation so callers can request a continuation instead of
    # silently handing the user a prompt that stops mid-sentence.
    if response.choices[0].finish_reason == "length":
        text += TRUNCATION_MARK
    return text


TRUNCATION_MARK = "\u0000TRUNCATED"


def looks_unfinished(text: str) -> bool:
    """Did generation stop mid-thought rather than complete?

    Only two signals count: the provider explicitly reported truncation, or the
    last line is a long sentence cut off mid-way. Short field-style endings like
    "План: средний план" are perfectly valid completions — treating those as
    truncated (the earlier bug) triggered pointless extra generation rounds that
    made every photo prompt several times slower and appended junk."""
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.endswith(TRUNCATION_MARK):
        return True
    tail = stripped[-200:].lower()
    if "fps" in tail:
        return False
    last_line = stripped.splitlines()[-1].strip()
    # A cut-off sentence is long and lacks closing punctuation. A finished field
    # value is short, so it never qualifies.
    return len(last_line) > 60 and last_line[-1] not in ".!?»\"')"


async def generate_complete(
    instruction: str, content: str, effort: str = "medium", max_rounds: int = 2
) -> str:
    """Run terra_text and, if the answer came back cut off, ask for the rest and
    stitch it on. Without this the user receives prompts that stop mid-sentence."""
    result = await terra_text(instruction, content, effort)
    result = result.replace(TRUNCATION_MARK, "").strip()

    rounds = 0
    while looks_unfinished(result) and rounds < max_rounds:
        rounds += 1
        tail = result[-1500:]
        continuation = await terra_text(
            instruction
            + "\n\nТЫ ПРОДОЛЖАЕШЬ РАНЕЕ НАЧАТЫЙ ОТВЕТ, КОТОРЫЙ ОБОРВАЛСЯ. "
            "Продолжи ровно с того места, где текст прервался. Не повторяй уже "
            "написанное, не начинай заново, не пиши вступлений — сразу дальше по тексту "
            "и доведи промпт до конца, включая финальную строку с форматом и fps.",
            f"ИСХОДНЫЕ ДАННЫЕ:\n{content}\n\nКОНЕЦ УЖЕ НАПИСАННОГО ТЕКСТА:\n...{tail}",
            effort,
        )
        continuation = continuation.replace(TRUNCATION_MARK, "").strip()
        if not continuation:
            break
        separator = "" if result.endswith(("\n", " ")) else " "
        result = (result + separator + continuation).strip()
    return result


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
            # Without an explicit cap this endpoint used the provider default,
            # which is far smaller than the primary path's budget — that was
            # why prompts arrived cut off mid-sentence.
            "max_tokens": 16000,
        }
        try:
            result = await kie_request(
                "https://api.kie.ai/gemini-3-8-flash-openai/v1/chat/completions",
                fallback_payload,
                attempts=2,
                timeout_seconds=300.0,
            )
        except Exception:
            # Both Kie endpoints are down/erroring (524, 500, timeouts). These
            # steps are text-only, so fall back to OpenAI instead of failing.
            return await openai_text(instruction, content)
        if is_refusal(result):
            # A short refusal-looking reply is sometimes a one-off fluke rather
            # than a real, persistent policy block — retry once before giving
            # up, so a transient hiccup doesn't abort the whole job.
            await asyncio.sleep(2.0)
            result = await kie_request(
                "https://api.kie.ai/gemini-3-8-flash-openai/v1/chat/completions",
                fallback_payload,
                attempts=2,
                timeout_seconds=300.0,
            )
        if is_refusal(result):
            # Both Kie endpoints refused. These are text-only steps (scenario,
            # prompt), so switch providers entirely rather than failing the job:
            # OpenAI rarely blocks ordinary scene description that Gemini's
            # stricter filters flag.
            try:
                openai_result = await openai_text(instruction, content)
            except Exception as exc:
                raise RuntimeError(
                    "Оба сервиса отказались выполнить запрос. "
                    f"Ошибка резервной модели: {str(exc)[-300:]}"
                ) from exc
            if is_refusal(openai_result) or not openai_result:
                raise RuntimeError(
                    "Запрос отклонён всеми доступными моделями. Попробуйте видео без "
                    "спорных элементов или переформулируйте правку."
                )
            return openai_result
        return result


async def create_scenario(
    meta: dict[str, Any], transcript: str, visual: str, cut_points: list[float] | None = None
) -> str:
    # Real cut boundaries are what the "one shot per cut" rule keys off — without
    # them the model invents a new shot for every micro-action.
    if cut_points:
        cuts_line = ", ".join(f"{value:.1f}" for value in cut_points)
        cuts = (
            f"ГРАНИЦЫ МОНТАЖНЫХ СКЛЕЕК (секунды): {cuts_line}\n"
            "Ровно по этим границам проходят смены кадра. Новый план начинается только здесь.\n\n"
        )
    else:
        cuts = (
            "ГРАНИЦЫ МОНТАЖНЫХ СКЛЕЕК: склеек не обнаружено — видео снято одним непрерывным "
            "планом, значит это ОДИН план на всё видео.\n\n"
        )
    content = (
        "МЕТАДАННЫЕ:\n"
        f"Длительность: {meta['duration']:.3f} секунды\n"
        f"Размер: {meta.get('width')}x{meta.get('height')}\n\n"
        + cuts +
        "ДОСЛОВНАЯ ТРАНСКРИПЦИЯ OPENAI:\n"
        f"{transcript}\n\n"
        "АНАЛИЗ САМОГО ВИДЕО GEMINI:\n"
        f"{visual}"
    )
    return (await generate_complete(SCENARIO_SYSTEM, content, "high")).strip()


SEEDANCE_SYSTEM = """Преобразуй реконструированный сценарий в готовый промпт Seedance 2.0.
Пиши только на русском языке.

САМОЕ ГЛАВНОЕ — НИКАКИХ ВСТУПЛЕНИЙ. Первая строка ответа — уже сам промпт (строка
"Референсы:"). Запрещено писать «Преобразую сценарий…», «Вот промпт…», «Сохраню точные
действия…» и любые объяснения того, что ты делаешь. Также никаких заключений в конце.

ЯЗЫК — ПРОСТОЙ И ЕСТЕСТВЕННЫЙ
Пиши так, как человек описывает происходящее, глядя на экран. Нейросеть понимает обычные
слова — не надо разжёвывать механику движений. Правильно: «правой рукой берёт свечу и убирает
её из кадра». Неправильно: «отгибает большой и указательный пальцы, подносит кисть под углом
90 градусов, захватывает свечу». Никакого киножаргона («мизансцена», «полиэкранная
композиция», «внутрикадровый монтаж») — только обычные слова.

ЭМОЦИИ И РЕПЛИКИ — ВНУТРИ ДЕЙСТВИЯ, А НЕ ОТДЕЛЬНЫМ СПИСКОМ
Это критично. Нельзя собирать эмоции и реплики в отдельные поля или перечислять их в конце —
непонятно, к какой секунде они относятся. Эмоция пишется прямо там, где она происходит, внутри
описания действия того же отрезка времени, и репликa тоже.
Правильно: «0–3 с: девушка удивлённо смотрит на свечи, брови приподнимаются, правой рукой
убирает свечу и говорит с лёгкой растерянностью: „Это что такое?“»
Неправильно: отдельные строки «Эмоция: удивление» и «Реплика: „Это что такое?“» в конце шота.

ДВИЖЕНИЕ, А НЕ ПОЗА
У действия должно быть начало, развитие и результат: «заносит ножницы, смыкает лезвия,
срезанные перья падают вниз», а не «держит ножницы у крыла». Глаголы динамические: срезает,
поднимает, падает, подбегает, разворачивается, уходит. Избегай статичных «держит»,
«находится», «расположен», «стоит» — если персонаж почти неподвижен, опиши микродвижение
(дыхание, поворот головы, движение ткани). Действие внутри шота разбивай по секундам и
показывай, как оно развивается.

ССЫЛКИ НА РЕФЕРЕНСЫ (@image1, @image2…)
Референс — только внешность персонажей, один @image на персонажа. НЕ создавай референс для
предметов, реквизита, локации или фона — их просто описывай текстом. Первая строка ответа:
«Референсы: @image1 — кто, @image2 — кто».

ВРЕМЯ
Только целые секунды (0–2s, 2–5s, 5–9s). Никаких долей и миллисекунд.

ФОРМАТИРОВАНИЕ — БЕЗ MARKDOWN
Текст идёт прямо в поле промпта Seedance: никаких звёздочек, решёток и обратных кавычек.
Подписи полей — через двоеточие, каждый параметр с новой строки.

ГЛАВНОЕ ПРАВИЛО НАРЕЗКИ: ШОТ = МОНТАЖНАЯ СКЛЕЙКА, А НЕ ОТДЕЛЬНОЕ ДЕЙСТВИЕ
Новый Shot начинается ТОЛЬКО там, где в исходном видео реальная монтажная склейка — смена
кадра. Ориентируйся на границы склеек, указанные в данных анализа.

Минимальная длина шота — 3 секунды. Шотов должно быть мало: на 15-секундное видео обычно
3–5 шотов, а если видео снято одним непрерывным планом — вообще ОДИН шот на всё видео.
Никогда не делай шот на каждую секунду и не создавай 10+ шотов — это дробит промпт и путает
нейросеть.

НЕ являются причиной начинать новый Shot (всё это описывается ВНУТРИ текущего шота, в поле
«Действие», по секундам): смена крупности, движение или наезд камеры, новое действие
персонажа, смена эмоции, появление предмета или эффекта, поворот головы.

Пример правильного шота:
Shot 1 (0–4s)
Действие: 0–1 с: крупный план ног на облаках, камера идёт вверх. 1–4 с: мальчик натягивает
тетиву, сосредоточенно смотрит вперёд, плечи напряжены.

Пример НЕПРАВИЛЬНОЙ нарезки (так делать нельзя): Shot 1 (0–1s), Shot 2 (1–2s), Shot 3 (2–3s).

СТРУКТУРА (максимум 15 секунд на часть; если видео длиннее, части отделяй строкой «ЧАСТЬ 2»,
«ЧАСТЬ 3»):

Референсы: @image1 — …, @image2 — …
Стиль: фотореалистичное кинематографическое качество, не мультфильм, не пластик.
Формат: вертикальный 9:16, N шотов.

Shot 1 (0–Ns)
Крупность: простыми словами — крупный план лица / по пояс / в полный рост. Если внутри шота
крупность меняется — напиши это одной фразой: «от общего плана к крупному».
Камера: где стоит и как снимает
Положение: кто где — LEFT / RIGHT / center
Фон: одной фразой, что видно позади (обязательно, если герой один в кадре; если фон не
менялся — «тот же фон»)
Действие: развитие действия по секундам внутри этого шота, с эмоциями и репликами внутри
текста (см. правило выше)
VFX: только если эффект реально есть

Критические ограничения добавляй ТОЛЬКО когда они нужны: если в кадре несколько персонажей —
одной строкой «В кадре ровно N человек: [список]. Без дублей». Если риска нет — блока нет.

Если персонаж обращается к кому-то за кадром, укажи, где тот находится (LEFT/RIGHT) и куда
смотрит говорящий; если адресат — зритель, взгляд направлен в камеру.

Audio: тип голоса, реплики по порядку, окружающие звуки; «без музыки», если музыки нет.
Свет: одна-две фразы.

Финальная строка: X seconds. Vertical 9:16. 720p. 24fps.

СЛОВА-ТРИГГЕРЫ, КОТОРЫХ ИЗБЕГАЙ (блокируют генерацию):
— «юная», «подросток», «несовершеннолетн*» в описании внешности;
— «текстура кожи», «поры»;
— анатомические термины вроде «грудь» — опиши силуэт или одежду;
— «светящиеся/сверкающие глаза» — пиши «яркие глаза», «широко раскрытые глаза»;
— КАПСЛОК в эмоциях;
— несколько слов про злость/издёвку подряд — разбавляй нейтральной лексикой;
— имена персонажей известных франшиз — называй по роли («девушка в сиреневом платье»)."""


GROK_SYSTEM = """Преобразуй реконструированный сценарий в промпт для Grok Imagine Video.
Пиши только на русском языке.

НИКАКИХ ВСТУПЛЕНИЙ И ЗАКЛЮЧЕНИЙ. Первая строка — уже сам промпт.

Grok устроен иначе, чем Seedance: ему НЕ нужна посекундная раскадровка по шотам, поля с
подписями и таблицы. Нужен компактный связный текст, максимум 2–3 абзаца.

Формула Grok: субъект → действие и движение → движение камеры → визуальный стиль → звук.
Самое важное ставь в первые 20–30 слов: Grok сильнее всего опирается на начало промпта.

Пиши так, будто смотришь на человека и описываешь, что он делает и что при этом чувствует —
естественными словами, коротко. Эмоция идёт прямо внутри действия: «девушка удивлённо смотрит
на свечи и подаётся вперёд», а не отдельным полем «Эмоция: удивление». Реплики — там же по
ходу текста, а не списком в конце.

Обязательно укажи: что именно движется в кадре, как движется камера, визуальный стиль одним
определением (не смешивай несколько эстетик), звук, примерную длительность («~10 секунд») и
вертикальный формат 9:16.

Не пиши механику движений по суставам и градусам — нейросеть понимает обычный язык.
Динамические глаголы обязательны: подбегает, срезает, разворачивается, падает.

Избегай слов-триггеров: «юная», «подросток», «текстура кожи», «поры», анатомических терминов,
«светящиеся глаза», КАПСЛОКА в эмоциях, имён персонажей известных франшиз."""


PHOTO_SKIN_BLOCK = (
    "Ultra-realistic expensive-looking skin with: visible pores, soft peach fuzz, realistic "
    "skin texture, subtle tonal variation, subsurface scattering on skin, authentic facial "
    "asymmetry. Skin must look healthy and professionally cared for, like after high-end "
    "cosmetology and luxury skincare treatments. Warm golden undertones with realistic depth "
    "and glow. Natural glossy highlights from lighting. No over-smoothing, no plastic texture, "
    "no CGI perfection, no fake beauty-filter effect."
)


PHOTO_SYSTEM = f"""Создай промпт для генерации фотографии. Пиши на русском языке.

НИКАКИХ ВСТУПЛЕНИЙ И ЗАКЛЮЧЕНИЙ. Первая строка — уже сам промпт.

СТРОГАЯ СТРУКТУРА (соблюдай порядок и подписи):

Используйте лицо со справочного фото: сохраните точные черты лица (форму лица, глаза, брови,
нос, губы, скулы), выражение и общее сходство. НЕ меняй идентичность лица. Цвет волос и длина
строго как на референсе.

• {PHOTO_SKIN_BLOCK}

• ОПИСАНИЕ ОБЪЕКТА:
Волосы:
Макияж:
Одежда:
Украшения:

• ПОЗА И ДЕЙСТВИЕ:

• ОКРУЖЕНИЕ:

• ОСВЕЩЕНИЕ:

• ТЕХНИКА:

• Ракурс:

• План:

ВАЖНЫЕ ПРАВИЛА:

1. Блок про кожу вставляй дословно как есть, на английском, ничего в нём не меняя.

2. НИКОГДА не указывай цвет волос и цвет глаз — они берутся с референса. Описывай только
длину, текстуру и укладку: «волнистые волосы до плеч», «прямые собранные в хвост», «короткая
стрижка». Слова «блондинка», «русые», «карие глаза» запрещены.

3. Если в кадре несколько людей — блок «ОПИСАНИЕ ОБЪЕКТА» повторяется для каждого: «ОПИСАНИЕ
ОБЪЕКТА 1», «ОПИСАНИЕ ОБЪЕКТА 2», у каждого свои волосы, макияж, одежда, украшения.

4. КАМЕРУ подбирай сам под сюжет, строго из трёх вариантов, копируя блоки дословно:

Canon G7X (контрастный, вечеринки, ночь, эффект плёнки, 2000-е):
ОСВЕЩЕНИЕ: контрастное драматичное освещение со встроенной вспышкой, снятое на Canon G7X.
ТЕХНИКА: Canon G7X, flash, 2000s aesthetic, Kodak Portra 400 film effect, film grain, halation,
vignette, analog mood, candid aesthetic, Instagram photo, Pinterest aesthetic, RAW photo.
Не размывать фон!

iPhone (повседневное, лайфстайл, дневной свет):
ОСВЕЩЕНИЕ: мягкое рассеянное естественное освещение.
ТЕХНИКА: iPhone shot, casual lifestyle aesthetic, Instagram photo, Pinterest aesthetic, RAW photo.
Не размывать фон!

Sony A7 III (только студийная съёмка):
ОСВЕЩЕНИЕ: освещение с мягким рассеянным светом, зернистость, студийный свет.
ТЕХНИКА: Sony A7 III, 85mm f/1.8, студийный свет, мягкие тени, чёткие детали, кинематографичный,
лёгкое зерно, RAW photo, высокая детализация.

5. РАКУРС выбирай из: анфас (эмоции, портрет), профиль (романтично, задумчиво), три четверти
(самый живой), снизу (сила, значимость), сверху (нежность, хрупкость), со спины (загадочность),
через плечо (эффект присутствия).

6. ПЛАН выбирай из: общий (в полный рост, видно окружение), средний (от пояса, видна одежда и
поза), крупный (лицо и плечи, акцент на эмоции), детальный макро (глаза, губы), американский
(от колен, для динамики).

7. Сочетай осмысленно: крупный + анфас = портрет и эмоция; средний + профиль = задумчивость;
общий + снизу = эпичность; крупный + сверху = нежность.

8. Пиши простыми словами, без киножаргона и разжёвывания механики движений."""


IDEA_SYSTEM = PHOTO_SYSTEM + """

ОСОБЕННОСТЬ ЭТОГО РЕЖИМА: фотографии нет, есть только текстовая идея пользователя. Разверни её
в полноценный промпт по структуре выше, додумывая недостающие детали сцены (окружение, одежду,
позу, свет) так, чтобы они логично подходили к идее. Строку про сохранение лица с референса
оставляй в промпте всегда — пользователь приложит своё фото при генерации."""


async def create_grok(scenario: str) -> str:
    return (await generate_complete(GROK_SYSTEM, "РЕКОНСТРУИРОВАННЫЙ СЦЕНАРИЙ:\n" + scenario, "medium")).strip()


async def create_photo_prompt(description: str) -> str:
    return (await generate_complete(PHOTO_SYSTEM, "ОПИСАНИЕ ФОТО:\n" + description, "medium")).strip()


async def create_idea_prompt(idea: str) -> str:
    return (await generate_complete(IDEA_SYSTEM, "ИДЕЯ ПОЛЬЗОВАТЕЛЯ:\n" + idea, "medium")).strip()


async def translate_prompt(prompt: str) -> str:
    instruction = (
        "Переведи промпт на английский язык. Никаких вступлений и комментариев — только сам "
        "перевод. Сохрани структуру, подписи полей и порядок строк. Технические блоки, которые "
        "уже на английском, оставь без изменений. Собственные термины генерации (RAW photo, "
        "film grain, Vertical 9:16, 24fps и подобные) не переводи."
    )
    return (await generate_complete(instruction, prompt, "medium")).strip()


async def create_seedance(scenario: str) -> str:
    # "medium" rather than "high": this step reformats an already-written
    # scenario, so extra reasoning depth adds little but costs real time.
    return (await generate_complete(SEEDANCE_SYSTEM, "РЕКОНСТРУИРОВАННЫЙ СЦЕНАРИЙ:\n" + scenario, "medium")).strip()


async def revise_scenario(scenario: str, correction: str) -> str:
    instruction = SCENARIO_SYSTEM + (
        "\n\nПравка пользователя имеет высший приоритет. Верни полный исправленный сценарий целиком, "
        "а не отдельный фрагмент. Сохрани все остальные утверждённые детали."
    )
    content = f"ПРАВКА ПОЛЬЗОВАТЕЛЯ:\n{correction}\n\nТЕКУЩИЙ СЦЕНАРИЙ:\n{scenario}"
    return (await generate_complete(instruction, content, "high")).strip()


async def revise_seedance(prompt: str, correction: str) -> str:
    instruction = SEEDANCE_SYSTEM + (
        "\n\nПравка пользователя имеет высший приоритет. Верни полный исправленный промпт целиком, "
        "а не отдельный фрагмент. Сохрани все остальные утверждённые детали, которые правка не касается."
    )
    content = f"ПРАВКА ПОЛЬЗОВАТЕЛЯ:\n{correction}\n\nТЕКУЩИЙ ПРОМПТ:\n{prompt}"
    return (await generate_complete(instruction, content, "high")).strip()


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
            [InlineKeyboardButton(text="✏️ Исправить промпт", callback_data=f"revise_prompt:{job_id}")],
            [InlineKeyboardButton(text="🇬🇧 Перевести на английский", callback_data=f"translate:{job_id}")],
        ]
    )


WELCOME_TEXT = (
    "🎬 VideoPrompt\n"
    "Превращаю видео и фото в готовые промпты для нейросетей.\n\n"
    "Что умею:\n"
    "🎬 Промпт для видео — пришлите видео или ссылку, получите промпт "
    "для Seedance 2.0 / 2.5 или Grok с раскадровкой по секундам\n"
    "🖼 Промпт по фото — пришлите фото, получите промпт для генерации изображения\n"
    "💡 Промпт по описанию — опишите идею словами, соберу промпт с нуля\n\n"
    "Каждый промпт можно исправить или перевести на английский прямо в чате.\n\n"
    "Баланс и пополнение — /balance\n\n"
    "Выберите режим ниже 👇"
)


def start_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🖼 Промпт по фото", callback_data="mode:photo")],
            [InlineKeyboardButton(text="💡 Промпт по описанию", callback_data="mode:idea")],
            [InlineKeyboardButton(text="🎬 Промпт для видео", callback_data="mode:video")],
        ]
    )


def video_engine_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🎬 Seedance 2.0 / 2.5", callback_data="engine:seedance")],
            [InlineKeyboardButton(text="🤖 Grok", callback_data="engine:grok")],
        ]
    )


def html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


async def send_prompt_block(
    chat_id: int, text: str, keyboard: InlineKeyboardMarkup | None = None
) -> None:
    """Send a prompt as a tap-to-copy code block instead of a file."""
    chunks = split_text(text, 3500)
    for index, chunk in enumerate(chunks):
        is_last = index == len(chunks) - 1
        await BOT.send_message(
            chat_id,
            f"<pre>{html_escape(chunk)}</pre>",
            parse_mode="HTML",
            reply_markup=keyboard if is_last else None,
        )


# user_id -> job_id: set when someone taps "✏️ Исправить промпт" and cleared once
# their next plain-text message is consumed as the correction (see `fallback`).
PENDING_PROMPT_REVISIONS: dict[int, str] = {}

# user_id -> "photo" | "idea": which standalone mode the user selected from the
# start menu, consumed by the next photo/text message they send.
PENDING_MODE: dict[int, str] = {}

# user_id -> "seedance" | "grok": which video engine to build the prompt for.
VIDEO_ENGINE: dict[int, str] = {}

# user_id -> latest prompt text, so the translate/revise buttons work for the
# photo and idea modes too (those have no job record in the database).
LAST_PROMPT: dict[int, str] = {}


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
        bonus = f"\n\nВам начислено {S.starting_tokens} пробных токенов." if created and S.starting_tokens else ""
        # Show the real welcome text right away; consent is asked underneath it
        # rather than replacing it.
        await message.answer(
            WELCOME_TEXT + bonus +
            "\n\nДля анализа файл временно передаётся внешним автоматизированным сервисам "
            "обработки. Подтвердите, что у вас есть право использовать видео и вы разрешаете "
            "обработку.",
            reply_markup=consent_keyboard(),
        )
        return
    await message.answer(WELCOME_TEXT, reply_markup=start_keyboard())


@ROUTER.message(Command("model"))
async def command_model(message: Message) -> None:
    if not await ensure_known(message) or not message.from_user:
        return
    current = VIDEO_ENGINE.get(message.from_user.id) or await DB.get_engine(message.from_user.id)
    name = {"seedance": "Seedance 2.0 / 2.5", "grok": "Grok"}.get(current or "", "не выбрана")
    await message.answer(
        f"Текущая модель для видео: {name}.\nВыберите, какую использовать дальше:",
        reply_markup=video_engine_keyboard(),
    )


@ROUTER.callback_query(F.data == "consent")
async def callback_consent(callback: CallbackQuery) -> None:
    await DB.ensure_user(callback.from_user.id, callback.from_user.username, callback.from_user.full_name)
    await DB.consent(callback.from_user.id)
    await callback.answer("Разрешение сохранено")
    if callback.message:
        await callback.message.edit_text(WELCOME_TEXT)
        await callback.message.answer("Выберите режим:", reply_markup=start_keyboard())


@ROUTER.callback_query(F.data.startswith("mode:"))
async def callback_mode(callback: CallbackQuery) -> None:
    if not callback.data or not callback.message or not callback.from_user:
        return
    mode = callback.data.split(":", 1)[1]
    await callback.answer()
    if mode == "video":
        await callback.message.answer(
            "Для какой нейросети сделать промпт?", reply_markup=video_engine_keyboard()
        )
        return
    PENDING_MODE[callback.from_user.id] = mode
    if mode == "photo":
        await callback.message.answer(
            f"Пришлите фото — соберу промпт для генерации изображения.\n"
            f"Стоимость: {S.tokens_per_photo} токенов."
        )
    else:
        await callback.message.answer(
            f"Опишите идею словами — соберу промпт с нуля.\n"
            f"Стоимость: {S.tokens_per_photo} токенов."
        )


@ROUTER.callback_query(F.data.startswith("engine:"))
async def callback_engine(callback: CallbackQuery) -> None:
    if not callback.data or not callback.message or not callback.from_user:
        return
    engine = callback.data.split(":", 1)[1]
    VIDEO_ENGINE[callback.from_user.id] = engine
    await DB.set_engine(callback.from_user.id, engine)
    PENDING_MODE.pop(callback.from_user.id, None)
    await callback.answer()
    name = "Seedance 2.0 / 2.5" if engine == "seedance" else "Grok"
    await callback.message.answer(
        f"Режим: {name}.\nПришлите видеофайл или публичную ссылку на видео."
    )


@ROUTER.callback_query(F.data.startswith("translate:"))
async def callback_translate(callback: CallbackQuery) -> None:
    if not callback.data or not callback.message or not callback.from_user:
        return
    key = callback.data.split(":", 1)[1]
    prompt = LAST_PROMPT.get(callback.from_user.id)
    if not prompt and key:
        job = await DB.get_job(key, callback.from_user.id)
        if job and job.get("scenario_path"):
            path = Path(job["scenario_path"]).with_name("seedance-prompts.md")
            if path.exists():
                prompt = path.read_text(encoding="utf-8")
    if not prompt:
        await callback.answer("Промпт не найден", show_alert=True)
        return
    await callback.answer()
    status = await callback.message.answer("Перевожу…")
    try:
        translated = await translate_prompt(prompt)
        with suppress(Exception):
            await status.delete()
        # No buttons under the English version: it is the final copy-and-use step.
        await send_prompt_block(callback.message.chat.id, translated)
    except Exception as exc:
        await status.edit_text("Не удалось перевести. Попробуйте ещё раз.\n" + str(exc)[-300:])


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
        LAST_PROMPT[callback.from_user.id] = result
        with suppress(Exception):
            await status.delete()
        await send_prompt_block(callback.message.chat.id, result, seedance_keyboard(job_id))
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

    # Track the last step we announced, so a failure or a hang reports where it
    # actually got stuck instead of a bare error with no context.
    current_step = {"name": "старт"}

    async def step(name: str, text: str) -> None:
        current_step["name"] = name
        await status(text)

    async with WORK_SEMAPHORE:
        try:
            video = local_path
            if video is None:
                await step("загрузка по ссылке", "1/5 — Загружаю видео по ссылке…")
                video = await download_url(source, job_dir)
            if not video.exists() or not video.stat().st_size:
                raise RuntimeError("Видео не загрузилось")
            if video.stat().st_size > S.max_upload_bytes:
                raise RuntimeError(f"Файл больше лимита {S.max_upload_mb} МБ")

            await step("проверка длительности", "2/5 — Проверяю длительность и баланс…")
            meta = await video_info(video)
            duration = float(meta["duration"])
            if duration <= 0:
                raise RuntimeError("Не удалось определить длительность видео")
            if duration > S.max_video_seconds:
                raise RuntimeError(f"Максимальная длительность сейчас {S.max_video_seconds // 60} минут")

            if S.moderation_enabled:
                await step("модерация", "2/5 — Проверяю содержание видео…")
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

            remaining = await DB.balance(user_id)
            await step(
                "расшифровка речи и анализ видео",
                f"3/5 — Распознаю речь и анализирую видео… "
                f"Списано {cost} токенов, осталось {remaining}.",
            )
            audio = await extract_audio(video, job_dir / "audio.mp3")
            cut_points = await detect_cuts(video, duration)
            public_url, media_token = media_url(video)
            media_tokens.append(media_token)

            async def get_transcript() -> str:
                if audio is None:
                    return "(в видео нет звуковой дорожки — речи нет)"
                return await transcribe(audio)

            transcript_task = asyncio.create_task(get_transcript())
            visual_task = asyncio.create_task(
                analyze_video(public_url, cut_points, video, job_dir, duration)
            )
            transcript, visual_result = await asyncio.wait_for(
                asyncio.gather(transcript_task, visual_task), timeout=900.0
            )
            visual, fallback_tokens = visual_result
            media_tokens.extend(fallback_tokens)

            await step("сценарий", "4/5 — Собираю планы, реплики, жесты и эмоции по секундам…")
            scenario = await asyncio.wait_for(
                create_scenario(meta, transcript, visual, cut_points), timeout=600.0
            )
            scenario_path = job_dir / "scenario.md"
            scenario_path.write_text(scenario, encoding="utf-8")
            engine = VIDEO_ENGINE.get(user_id) or await DB.get_engine(user_id) or "seedance"
            engine_name = "Grok" if engine == "grok" else "Seedance 2.0 / 2.5"
            await step("промпт видео", f"5/5 — Создаю промпт для {engine_name}…")
            builder = create_grok if engine == "grok" else create_seedance
            seedance = await asyncio.wait_for(builder(scenario), timeout=600.0)
            seedance_path = job_dir / "seedance-prompts.md"
            seedance_path.write_text(seedance, encoding="utf-8")
            await DB.complete(job_id, scenario_path)
            LAST_PROMPT[user_id] = seedance
            with suppress(Exception):
                await BOT.delete_message(chat_id, status_id)
            # Prompt goes out as a tap-to-copy code block; no file is sent.
            await send_prompt_block(chat_id, seedance, seedance_keyboard(job_id))
        except asyncio.TimeoutError:
            await DB.refund_job(job_id, "timeout")
            await status("Обработка не завершилась. Списанные токены возвращены автоматически.")
            await BOT.send_message(
                chat_id,
                f"Причина: превышено время ожидания на шаге «{current_step['name']}». "
                "Обычно это перегрузка сервиса анализа — попробуйте ещё раз через пару минут.",
            )
        except Exception as exc:
            await DB.refund_job(job_id, str(exc))
            await status("Обработка не завершилась. Списанные токены возвращены автоматически.")
            await BOT.send_message(
                chat_id,
                f"Причина (шаг «{current_step['name']}»): " + str(exc)[-1200:],
            )
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


async def charge_small(user_id: int) -> bool:
    """Charge the flat photo/idea prompt fee. Owner is never charged."""
    if user_id == S.owner_id:
        return True
    return await DB.charge(user_id, S.tokens_per_photo, "photo_prompt")


@ROUTER.message(F.photo)
async def handle_photo(message: Message) -> None:
    if not await ensure_known(message) or not message.from_user:
        return
    if PENDING_MODE.get(message.from_user.id) != "photo":
        await message.answer(
            "Чтобы сделать промпт по фото, сначала выберите режим 🖼 Промпт по фото в /start."
        )
        return
    if not await charge_small(message.from_user.id):
        balance = await DB.balance(message.from_user.id)
        await message.answer(
            f"Недостаточно токенов. Промпт по фото стоит {S.tokens_per_photo}, "
            f"на балансе {balance}."
        )
        if S.payments_enabled:
            await message.answer("Выберите пополнение:", reply_markup=buy_keyboard())
        return
    PENDING_MODE.pop(message.from_user.id, None)
    status = await message.answer("Собираю промпт по фото…")
    try:
        photo = message.photo[-1]
        file = await BOT.get_file(photo.file_id)
        buffer = await BOT.download_file(file.file_path)
        data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.read()).decode("ascii")
        described = await kie_request(
            "https://api.kie.ai/gemini-3-8-flash-openai/v1/chat/completions",
            visual_payload([
                {"type": "text", "text": (
                    "Подробно опиши это фото для последующего написания промпта: внешность и "
                    "количество людей, длина и укладка волос (БЕЗ цвета), макияж, одежда, "
                    "украшения, поза и действие, окружение, характер освещения, ракурс и "
                    "крупность плана. Не называй цвет волос и цвет глаз."
                )},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]),
            attempts=2, timeout_seconds=120.0,
        )
        prompt = await asyncio.wait_for(create_photo_prompt(described), timeout=300.0)
        LAST_PROMPT[message.from_user.id] = prompt
        with suppress(Exception):
            await status.delete()
        await send_prompt_block(message.chat.id, prompt, seedance_keyboard("photo"))
    except asyncio.TimeoutError:
        if message.from_user.id != S.owner_id:
            await DB.grant(message.from_user.id, S.tokens_per_photo, "photo_refund")
        await status.edit_text(
            "Сервис не ответил вовремя, токены возвращены. Попробуйте ещё раз."
        )
    except Exception as exc:
        if message.from_user.id != S.owner_id:
            await DB.grant(message.from_user.id, S.tokens_per_photo, "photo_refund")
        await status.edit_text(
            "Не удалось собрать промпт, токены возвращены.\n" + str(exc)[-400:]
        )


@ROUTER.message()
async def fallback(message: Message) -> None:
    if not await ensure_known(message) or not message.from_user:
        return
    user_id = message.from_user.id

    job_id = PENDING_PROMPT_REVISIONS.pop(user_id, None)
    if job_id:
        correction = (message.text or "").strip()
        if not correction:
            PENDING_PROMPT_REVISIONS[user_id] = job_id
            await message.answer("Пришлите правку текстом одним сообщением.")
            return
        prompt = LAST_PROMPT.get(user_id)
        path = None
        if job_id not in {"photo", "idea"}:
            job = await DB.get_job(job_id, user_id)
            if job and job.get("scenario_path"):
                candidate = Path(job["scenario_path"]).with_name("seedance-prompts.md")
                if candidate.exists():
                    path = candidate
                    prompt = candidate.read_text(encoding="utf-8")
        if not prompt:
            await message.answer("Промпт не найден.")
            return
        status = await message.answer("Переписываю промпт с учётом правки…")
        try:
            updated = await revise_seedance(prompt, correction)
            LAST_PROMPT[user_id] = updated
            if path:
                path.write_text(updated, encoding="utf-8")
            with suppress(Exception):
                await status.delete()
            await send_prompt_block(message.chat.id, updated, seedance_keyboard(job_id))
        except Exception as exc:
            await status.edit_text(
                "Не удалось применить правку. Попробуйте ещё раз.\n" + str(exc)[-500:]
            )
        return

    if PENDING_MODE.get(user_id) == "idea":
        idea = (message.text or "").strip()
        if not idea:
            await message.answer("Опишите идею текстом одним сообщением.")
            return
        if not await charge_small(user_id):
            balance = await DB.balance(user_id)
            await message.answer(
                f"Недостаточно токенов. Промпт по описанию стоит {S.tokens_per_photo}, "
                f"на балансе {balance}."
            )
            if S.payments_enabled:
                await message.answer("Выберите пополнение:", reply_markup=buy_keyboard())
            return
        PENDING_MODE.pop(user_id, None)
        status = await message.answer("Собираю промпт по описанию…")
        try:
            prompt = await asyncio.wait_for(create_idea_prompt(idea), timeout=300.0)
            LAST_PROMPT[user_id] = prompt
            with suppress(Exception):
                await status.delete()
            await send_prompt_block(message.chat.id, prompt, seedance_keyboard("idea"))
        except asyncio.TimeoutError:
            if user_id != S.owner_id:
                await DB.grant(user_id, S.tokens_per_photo, "idea_refund")
            await status.edit_text(
                "Сервис не ответил вовремя, токены возвращены. Попробуйте ещё раз."
            )
        except Exception as exc:
            if user_id != S.owner_id:
                await DB.grant(user_id, S.tokens_per_photo, "idea_refund")
            await status.edit_text(
                "Не удалось собрать промпт, токены возвращены.\n" + str(exc)[-400:]
            )
        return

    await message.answer("Выберите режим в /start или пришлите видео либо ссылку.")


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
            BotCommand(command="start", description="ℹ️ Что умеет бот"),
            BotCommand(command="balance", description="👤 Мой профиль"),
            BotCommand(command="buy", description="⭐ Пополнить баланс"),
            BotCommand(command="model", description="🎛 Сменить модель"),
            BotCommand(command="privacy", description="🔒 Обработка видео"),
            BotCommand(command="terms", description="📄 Условия использования"),
            BotCommand(command="support", description="💬 Поддержка"),
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
