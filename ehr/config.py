"""Central configuration. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _path(env: str, default: Path) -> Path:
    return Path(os.environ.get(env, default)).resolve()


@dataclass(frozen=True)
class Settings:
    raw_dir: Path = field(default_factory=lambda: _path("EHR_RAW_DIR", ROOT / "data" / "raw"))
    out_dir: Path = field(default_factory=lambda: _path("EHR_OUT_DIR", ROOT / "data" / "output"))
    db_path: Path = field(default_factory=lambda: _path("EHR_DB_PATH", ROOT / "data" / "store" / "ehr.sqlite3"))
    chroma_dir: Path = field(default_factory=lambda: _path("EHR_CHROMA_DIR", ROOT / "data" / "store" / "chroma"))

    # Summarization
    claude_model: str = field(default_factory=lambda: os.environ.get("EHR_CLAUDE_MODEL", "claude-opus-5-5"))
    # "auto" uses Claude when credentials exist and falls back to the offline summarizer otherwise.
    summarizer: str = field(default_factory=lambda: os.environ.get("EHR_SUMMARIZER", "auto"))

    # Search
    embedding_model: str = field(default_factory=lambda: os.environ.get("EHR_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    # "sentence-transformers" (default) or "hash" (dependency-free, for tests/offline demos).
    embedder: str = field(default_factory=lambda: os.environ.get("EHR_EMBEDDER", "sentence-transformers"))


def get_settings() -> Settings:
    return Settings()
