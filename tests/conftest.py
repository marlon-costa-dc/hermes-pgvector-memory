"""Test bootstrap.

The plugin imports ``agent.memory_provider`` from its host (Hermes). That
package does not exist on PyPI, so tests must resolve it one of two ways:

1. A real Hermes checkout, when one is present (HERMES_AGENT_PATH or the
   default install location). Tests then run against the true ABC.
2. A minimal stub, for CI and contributors without Hermes installed.

The stub is a fallback, not a substitute: ``test_contract.py`` asserts the
provider satisfies the REAL ABC and skips when only the stub is available, so
a host-side signature change cannot pass unnoticed on a developer machine.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

HERMES_AVAILABLE = False


def _try_real_hermes() -> bool:
    candidates = []
    if env_path := os.environ.get("HERMES_AGENT_PATH"):
        candidates.append(Path(env_path))
    candidates += [
        Path.home() / ".hermes" / "hermes-agent",
        Path.home() / "hermes-agent",
    ]
    for path in candidates:
        if (path / "agent" / "memory_provider.py").exists():
            sys.path.insert(0, str(path))
            try:
                import agent.memory_provider  # noqa: F401

                return True
            except Exception:
                sys.path.remove(str(path))
    return False


HERMES_AVAILABLE = _try_real_hermes()

if not HERMES_AVAILABLE:
    import types
    from abc import ABC, abstractmethod

    agent_pkg = types.ModuleType("agent")
    mp = types.ModuleType("agent.memory_provider")

    @dataclass
    class RecallStatus:  # type: ignore[no-redef]
        provider_label: str
        count: int
        glyph: str = "\N{EYE}"

    def is_trivial_prompt(text: str | None) -> bool:
        if not text:
            return True
        return text.strip().lower().rstrip("!?.,:; ") in {
            "yes",
            "no",
            "ok",
            "okay",
            "sure",
            "thanks",
            "hi",
            "hey",
            "hello",
            "continue",
            "go ahead",
            "do it",
            "done",
            "k",
        }

    class MemoryProvider(ABC):  # type: ignore[no-redef]
        pre_compress_checkpoint_api_version = 1

        @property
        @abstractmethod
        def name(self) -> str: ...

        @abstractmethod
        def is_available(self) -> bool: ...

        @abstractmethod
        def initialize(self, session_id: str, **kwargs) -> None: ...

        @abstractmethod
        def get_tool_schemas(self) -> list[dict[str, Any]]: ...

        def unavailable_reason(self) -> str:
            return ""

        def system_prompt_block(self) -> str:
            return ""

        def prefetch(self, query: str, *, session_id: str = "") -> str:
            return ""

        def queue_prefetch(self, query: str, *, session_id: str = "") -> None: ...
        def recall_status(self):
            return None

        def sync_turn(self, user_content, assistant_content, **kw) -> None: ...
        def handle_tool_call(self, tool_name, args, **kw) -> str:
            return ""

        def shutdown(self) -> None: ...
        def on_session_end(self, messages) -> None: ...
        def on_memory_write(self, action, target, content, metadata=None) -> None: ...
        def get_config_schema(self) -> list[dict[str, Any]]:
            return []

        def save_config(self, values, hermes_home) -> None: ...
        def backup_paths(self) -> list[str]:
            return []

    mp.MemoryProvider = MemoryProvider
    mp.RecallStatus = RecallStatus
    mp.is_trivial_prompt = is_trivial_prompt
    agent_pkg.memory_provider = mp
    sys.modules["agent"] = agent_pkg
    sys.modules["agent.memory_provider"] = mp
