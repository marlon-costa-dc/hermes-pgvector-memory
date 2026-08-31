"""pgvector-memory — Hermes memory provider backed by PostgreSQL.

Stores agent memories as embeddings in PostgreSQL, retrieves them with
pgvectorscale's DiskANN index fused with lexical ranking, and embeds locally
through Ollama. No data leaves the machine.

Config in $HERMES_HOME/config.yaml:

    memory:
      provider: pgvector-memory
    plugins:
      pgvector-memory:
        dsn: "postgresql:///hermes_memory?host=/run/postgresql"
        embed_model: nomic-embed-text
        auto_recall: true
        recall_limit: 5
        min_similarity: 0.55
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from agent.memory_provider import MemoryProvider, RecallStatus, is_trivial_prompt

from .config import Config, load_config
from .embeddings import EmbeddingError, OllamaEmbedder
from .store import MemoryStore, StoreError

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

REMEMBER_SCHEMA = {
    "name": "pgvector_remember",
    "description": (
        "Store a durable memory in PostgreSQL with a semantic embedding. "
        "Use for facts worth recalling in a LATER session: user preferences, "
        "project conventions, environment quirks, decisions and their reasons.\n\n"
        "Write one self-contained statement per call — it will be retrieved "
        "without surrounding conversation. Prefer 'The operator runs CachyOS "
        "with systemd-boot' over 'he uses that distro'.\n\n"
        "Do NOT store: transient task state, secrets, or anything trivially "
        "re-derivable from the filesystem."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The memory, as one self-contained declarative statement.",
            },
            "kind": {
                "type": "string",
                "enum": ["fact", "preference", "observation"],
                "description": (
                    "fact = objective and durable; preference = how the user wants "
                    "things done; observation = something noticed, weaker claim."
                ),
            },
            "tags": {
                "type": "string",
                "description": "Optional comma-separated tags (e.g. 'postgres,ollama').",
            },
        },
        "required": ["content"],
    },
}

RECALL_SCHEMA = {
    "name": "pgvector_recall",
    "description": (
        "Search stored memories by meaning. Combines vector similarity with "
        "keyword matching, so it finds paraphrases AND exact tokens like error "
        "codes or file paths.\n\n"
        "Call this before answering questions about the user's environment, "
        "preferences, or past decisions — recall injected automatically each "
        "turn is capped and may not cover the question at hand."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "kind": {
                "type": "string",
                "enum": ["fact", "preference", "observation", "turn"],
                "description": "Optional: restrict to one kind of memory.",
            },
            "limit": {"type": "integer", "description": "Max results (default 10)."},
        },
        "required": ["query"],
    },
}

FORGET_SCHEMA = {
    "name": "pgvector_forget",
    "description": (
        "Delete a memory by id, when it is wrong or obsolete. Get the id from "
        "pgvector_recall. Deletion is permanent."
    ),
    "parameters": {
        "type": "object",
        "properties": {"memory_id": {"type": "integer", "description": "Id to delete."}},
        "required": ["memory_id"],
    },
}


class PgVectorMemoryProvider(MemoryProvider):
    """MemoryProvider implementation over PostgreSQL + pgvector + pgvectorscale."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config
        self._store: MemoryStore | None = None
        self._embedder: OllamaEmbedder | None = None
        self._session_id = ""
        self._agent_identity = ""
        self._unavailable_reason = ""
        self._last_recall_count = 0
        # Prefetch runs on a background thread; the result is consumed on the
        # next turn. Guarded because the turn thread reads what it writes.
        self._prefetch_lock = threading.Lock()
        self._prefetched = ""

    # -- identity -----------------------------------------------------------

    @property
    def name(self) -> str:
        return "pgvector-memory"

    # -- lifecycle ----------------------------------------------------------

    @property
    def config(self) -> Config:
        if self._config is None:
            try:
                from hermes_cli.config import cfg_get
            except ImportError:
                cfg_get = None
            self._config = load_config(cfg_get)
        return self._config

    def is_available(self) -> bool:
        """Cheap readiness check: driver importable and Ollama serving the model.

        The host calls this before initialize() and documents it as
        no-network; the Ollama probe is a localhost GET with a 5s timeout,
        which is the only way to know the embedder exists at all.
        """
        try:
            import psycopg  # noqa: F401
        except ImportError:
            self._unavailable_reason = (
                "psycopg is not installed — run: pip install 'psycopg[binary]'"
            )
            return False

        embedder = OllamaEmbedder(self.config.ollama_host, self.config.embed_model)
        if not embedder.is_available():
            self._unavailable_reason = (
                f"Ollama at {self.config.ollama_host} is not serving "
                f"{self.config.embed_model!r} — run: "
                f"ollama pull {self.config.embed_model}"
            )
            return False

        self._embedder = embedder
        return True

    def unavailable_reason(self) -> str:
        return self._unavailable_reason

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._agent_identity = kwargs.get("agent_identity", "") or ""

        if self._embedder is None:
            self._embedder = OllamaEmbedder(self.config.ollama_host, self.config.embed_model)

        dims = self._embedder.dims
        if dims is None:
            # Unknown model: learn the width from one real call rather than
            # guessing, so the table is created with the correct dimension.
            dims = len(self._embedder.embed_one("dimension probe"))

        self._store = MemoryStore(self.config.dsn, dims, self.config.table)
        self._store.connect()
        self._store.ensure_schema()
        logger.info(
            "pgvector-memory ready: %s (%d dims, model %s)",
            self.config.table,
            dims,
            self.config.embed_model,
        )

    def shutdown(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None

    # -- prompt integration -------------------------------------------------

    def system_prompt_block(self) -> str:
        if self._store is None:
            return ""
        try:
            stats = self._store.stats()
        except Exception as exc:
            logger.debug("stats failed: %s", exc)
            return ""
        if not stats["total"]:
            return ""
        return (
            f"Persistent memory: {stats['total']} entries in PostgreSQL "
            f"({stats['facts']} facts, {stats['preferences']} preferences, "
            f"{stats['observations']} observations). Use pgvector_recall to "
            f"search them and pgvector_remember to add durable ones."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return recall context for this turn.

        Consumes whatever the background thread prepared after the previous
        turn. Returns quickly and never embeds on the turn thread.
        """
        if not self.config.auto_recall:
            return ""
        with self._prefetch_lock:
            text, self._prefetched = self._prefetched, ""
        return text

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Kick off background recall for the NEXT turn."""
        if not self.config.auto_recall or self._store is None:
            return
        if len(query.strip()) < 12 or is_trivial_prompt(query):
            # Host-owned classifier: "ok", "thanks", "vai" carry no retrievable
            # intent, and embedding them would return arbitrary neighbours.
            return

        def _work() -> None:
            try:
                hits = self._search(query, limit=self.config.recall_limit)
            except (EmbeddingError, StoreError, Exception) as exc:
                logger.debug("background recall failed: %s", exc)
                return
            if not hits:
                with self._prefetch_lock:
                    self._prefetched = ""
                self._last_recall_count = 0
                return
            lines = [f"- [{h['kind']}#{h['id']}] {h['content']}" for h in hits]
            with self._prefetch_lock:
                self._prefetched = "Relevant memories (retrieved from PostgreSQL):\n" + "\n".join(
                    lines
                )
            self._last_recall_count = len(hits)

        threading.Thread(target=_work, daemon=True, name="pgvec-recall").start()

    def recall_status(self) -> RecallStatus | None:
        if not self._last_recall_count:
            return None
        count, self._last_recall_count = self._last_recall_count, 0
        return RecallStatus(provider_label="pgvector", count=count)

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Optionally capture the turn, then queue recall for the next one."""
        if self.config.auto_capture_turns and self._store is not None:
            text = f"User: {user_content.strip()}\nAssistant: {assistant_content.strip()}"
            if len(text) >= self.config.min_turn_chars:
                threading.Thread(
                    target=self._store_safe,
                    args=(text,),
                    kwargs={"kind": "turn", "source": "turn", "session_id": session_id},
                    daemon=True,
                    name="pgvec-capture",
                ).start()
        self.queue_prefetch(user_content, session_id=session_id)

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mirror built-in MEMORY.md / USER.md writes into Postgres.

        MEMORY.md has a hard character budget; Postgres does not. Mirroring
        means a memory evicted from the file for space is still retrievable.
        """
        if not self.config.mirror_memory_tool or self._store is None:
            return
        if action not in ("add", "replace") or not content.strip():
            return
        kind = "preference" if target == "user" else "fact"
        threading.Thread(
            target=self._store_safe,
            args=(content,),
            kwargs={
                "kind": kind,
                "source": "memory_tool",
                "metadata": {"target": target, "action": action},
            },
            daemon=True,
            name="pgvec-mirror",
        ).start()

    # -- tools --------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [REMEMBER_SCHEMA, RECALL_SCHEMA, FORGET_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        if self._store is None:
            return "Error: pgvector-memory is not initialized."
        try:
            if tool_name == "pgvector_remember":
                return self._tool_remember(args)
            if tool_name == "pgvector_recall":
                return self._tool_recall(args)
            if tool_name == "pgvector_forget":
                return self._tool_forget(args)
        except EmbeddingError as exc:
            return f"Error: embedding failed — {exc}"
        except StoreError as exc:
            return f"Error: storage failed — {exc}"
        except Exception as exc:  # surface, never silently succeed
            logger.exception("pgvector-memory tool %s failed", tool_name)
            return f"Error: {type(exc).__name__}: {exc}"
        return f"Error: unknown tool {tool_name!r}"

    def _tool_remember(self, args: dict[str, Any]) -> str:
        content = (args.get("content") or "").strip()
        if not content:
            return "Error: 'content' is required."
        tags = [t.strip() for t in (args.get("tags") or "").split(",") if t.strip()]
        memory_id = self._store_now(
            content,
            kind=args.get("kind") or "observation",
            source="tool",
            metadata={"tags": tags} if tags else None,
        )
        if memory_id is None:
            return "Already stored (identical content) — nothing added."
        return f"Stored as memory #{memory_id}."

    def _tool_recall(self, args: dict[str, Any]) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return "Error: 'query' is required."
        hits = self._search(
            query,
            limit=int(args.get("limit") or 10),
            kind=args.get("kind") or "",
            # An explicit search is the agent looking for something specific:
            # do not apply the automatic-recall floor, let it see weak matches.
            min_similarity=0.0,
        )
        if not hits:
            return f"No memories found for {query!r}."
        lines = [
            f"#{h['id']} [{h['kind']}] (sim {h['similarity']:.2f}) {h['content']}" for h in hits
        ]
        return f"{len(hits)} memories:\n" + "\n".join(lines)

    def _tool_forget(self, args: dict[str, Any]) -> str:
        memory_id = args.get("memory_id")
        if not isinstance(memory_id, int):
            return "Error: 'memory_id' must be an integer."
        store, _ = self._require_ready()
        if store.delete(memory_id):
            return f"Deleted memory #{memory_id}."
        return f"No memory #{memory_id}."

    # -- internals ----------------------------------------------------------

    def _require_ready(self) -> tuple[MemoryStore, OllamaEmbedder]:
        """Return the live store/embedder, or fail loudly.

        Every call path below runs only after initialize(), but a provider
        whose init failed must not silently no-op: a memory that was never
        written is worse than an error the operator can see.
        """
        if self._store is None or self._embedder is None:
            raise StoreError("pgvector-memory is not initialized")
        return self._store, self._embedder

    def _search(
        self,
        query: str,
        *,
        limit: int = 10,
        kind: str = "",
        min_similarity: float | None = None,
    ) -> list[dict[str, Any]]:
        store, embedder = self._require_ready()
        vector = embedder.embed_one(query)
        return store.search(
            vector,
            query,
            limit=limit,
            kind=kind,
            agent_identity=self._agent_identity,
            min_similarity=(
                self.config.min_similarity if min_similarity is None else min_similarity
            ),
        )

    def _store_now(self, content: str, **kwargs) -> int | None:
        store, embedder = self._require_ready()
        vector = embedder.embed_one(content)
        return store.add(
            content,
            vector,
            session_id=kwargs.pop("session_id", self._session_id),
            agent_identity=self._agent_identity,
            **kwargs,
        )

    def _store_safe(self, content: str, **kwargs) -> None:
        """Background write. Logs failures instead of raising into a thread."""
        try:
            self._store_now(content, **kwargs)
        except Exception as exc:
            logger.warning("pgvector-memory background write failed: %s", exc)

    # -- setup surface ------------------------------------------------------

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "dsn",
                "description": "PostgreSQL DSN (libpq URI or key=value)",
                "default": "postgresql:///hermes_memory?host=/run/postgresql",
                "required": True,
            },
            {
                "key": "embed_model",
                "description": "Ollama embedding model",
                "default": "nomic-embed-text",
            },
            {
                "key": "ollama_host",
                "description": "Ollama base URL",
                "default": "http://127.0.0.1:11434",
            },
            {
                "key": "auto_recall",
                "description": "Inject relevant memories before each turn",
                "type": "boolean",
                "default": True,
            },
            {
                "key": "recall_limit",
                "description": "Max memories injected per turn",
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "maximum": 50,
            },
            {
                "key": "min_similarity",
                "description": "Cosine similarity floor for automatic recall",
                "type": "number",
                "default": 0.55,
                "minimum": 0.0,
                "maximum": 1.0,
                "step": 0.05,
            },
            {
                "key": "auto_capture_turns",
                "description": "Also store every conversation turn (noisy)",
                "type": "boolean",
                "default": False,
            },
            {
                "key": "mirror_memory_tool",
                "description": "Mirror MEMORY.md / USER.md writes into Postgres",
                "type": "boolean",
                "default": True,
            },
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        """Persist non-secret settings under plugins.pgvector-memory."""
        try:
            from hermes_cli.config import cfg_set
        except ImportError:
            logger.warning("hermes_cli.config unavailable; config not saved")
            return
        for key, value in values.items():
            cfg_set(f"plugins.pgvector-memory.{key}", value)


def register(ctx) -> None:
    """Plugin entry point called by Hermes' provider loader."""
    ctx.register_memory_provider(PgVectorMemoryProvider())


__all__ = ["PgVectorMemoryProvider", "register", "__version__"]
