"""Configuration resolution for the pgvector memory provider.

Precedence, highest first:
  1. Environment variables (PGVECTOR_MEMORY_*)  -- secrets and overrides
  2. config.yaml under plugins.pgvector-memory  -- the canonical surface
  3. Built-in defaults

This mirrors Hermes' own rule: .env carries credentials, config.yaml carries
behaviour. A DSN can embed a password, so it is accepted from the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

DEFAULT_DSN = "postgresql:///hermes_memory?host=/run/postgresql"
DEFAULT_TABLE = "hermes_memories"


@dataclass
class Config:
    dsn: str = DEFAULT_DSN
    table: str = DEFAULT_TABLE

    ollama_host: str = "http://127.0.0.1:11434"
    embed_model: str = "nomic-embed-text"

    # Recall injected into the prompt before each turn.
    auto_recall: bool = True
    recall_limit: int = 5
    # Cosine similarity floor. Below ~0.5 nomic-embed-text results are mostly
    # topical noise; injecting them wastes context and dilutes attention.
    min_similarity: float = 0.55

    # Automatic capture of conversation turns. Off by default: it is lossy,
    # noisy, and grows the table fast. Explicit pgvector_remember calls and
    # mirrored MEMORY.md writes are the curated path.
    auto_capture_turns: bool = False
    # Skip turns shorter than this -- "ok", "thanks", "sim" carry no recall value.
    min_turn_chars: int = 80

    # Mirror built-in MEMORY.md / USER.md writes into Postgres.
    mirror_memory_tool: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "dsn": self.dsn,
            "table": self.table,
            "ollama_host": self.ollama_host,
            "embed_model": self.embed_model,
            "auto_recall": self.auto_recall,
            "recall_limit": self.recall_limit,
            "min_similarity": self.min_similarity,
            "auto_capture_turns": self.auto_capture_turns,
            "min_turn_chars": self.min_turn_chars,
            "mirror_memory_tool": self.mirror_memory_tool,
        }


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_config(cfg_get=None) -> Config:
    """Resolve configuration.

    ``cfg_get`` is Hermes' ``hermes_cli.config.cfg_get``, injected so this
    module stays importable (and testable) outside the agent process.
    """

    def _cfg(key: str, default: Any = None) -> Any:
        """Read one key, treating a missing value as absent.

        Measured behaviour, not the documented one: Hermes' ``cfg_get``
        returns ``None`` for an unset key even when a default is passed, so
        the default must be applied here. An empty string is also treated as
        absent — a blank DSN is never a deliberate choice.
        """
        if cfg_get is None:
            return default
        try:
            value = cfg_get(f"plugins.pgvector-memory.{key}")
        except Exception:
            return default
        if value is None or (isinstance(value, str) and not value.strip()):
            return default
        return value

    env = os.environ.get
    return Config(
        dsn=env("PGVECTOR_MEMORY_DSN") or _cfg("dsn", DEFAULT_DSN),
        table=env("PGVECTOR_MEMORY_TABLE") or _cfg("table", DEFAULT_TABLE),
        ollama_host=env("PGVECTOR_MEMORY_OLLAMA_HOST")
        or _cfg("ollama_host", "http://127.0.0.1:11434"),
        embed_model=env("PGVECTOR_MEMORY_EMBED_MODEL") or _cfg("embed_model", "nomic-embed-text"),
        auto_recall=_as_bool(_cfg("auto_recall"), True),
        recall_limit=_as_int(_cfg("recall_limit"), 5),
        min_similarity=_as_float(_cfg("min_similarity"), 0.55),
        auto_capture_turns=_as_bool(_cfg("auto_capture_turns"), False),
        min_turn_chars=_as_int(_cfg("min_turn_chars"), 80),
        mirror_memory_tool=_as_bool(_cfg("mirror_memory_tool"), True),
    )
