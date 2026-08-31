"""PostgreSQL storage layer: schema management, writes, and hybrid retrieval."""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .embeddings import to_pgvector

logger = logging.getLogger(__name__)

# Retrieval knobs. Exposed here rather than buried in the query so the
# behaviour is reviewable in one place.
#
# RRF (Reciprocal Rank Fusion) merges the vector ranking and the lexical
# ranking without needing the two scores to be on a comparable scale --
# which cosine distance and ts_rank are not. k=60 is the constant from the
# original Cormack et al. paper and the one Postgres hybrid-search examples
# converged on; it damps the tail so rank 1 vs 2 matters much more than
# rank 40 vs 41.
RRF_K = 60


class StoreError(RuntimeError):
    """Raised for unrecoverable storage problems (bad config, missing deps)."""


def _sha256(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


class MemoryStore:
    """Thread-safe PostgreSQL-backed vector store.

    One connection guarded by a lock. The Hermes agent issues memory calls
    from the turn thread plus a background prefetch thread — two callers,
    short transactions. A pool would add a dependency and a failure mode for
    no measurable gain at this concurrency.
    """

    def __init__(self, dsn: str, dims: int, table: str = "hermes_memories") -> None:
        self.dsn = dsn
        self.dims = dims
        # Identifier, not a value: it cannot be a bound parameter. Validated
        # here so it can never carry SQL, then formatted into statements.
        if not table.replace("_", "").isalnum():
            raise StoreError(f"Invalid table name: {table!r}")
        self.table = table
        self._lock = threading.Lock()
        self._conn = None

    # -- connection ---------------------------------------------------------

    def connect(self) -> None:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise StoreError("psycopg (v3) is required: pip install 'psycopg[binary]'") from exc

        self._conn = psycopg.connect(self.dsn, autocommit=True)
        logger.debug("connected to %s", self.dsn)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def _require_conn(self):
        if self._conn is None:
            raise StoreError("MemoryStore is not connected; call connect() first")
        return self._conn

    # -- schema -------------------------------------------------------------

    def _read_schema(self) -> str:
        """Locate schema.sql for both layouts this package ships in.

        In the repo the file sits at ``<root>/sql/schema.sql`` next to the
        package. Installed into ``$HERMES_HOME/plugins/pgvector-memory/`` the
        package IS the plugin root, so the file sits at ``./sql/schema.sql``.
        Both are checked rather than making the installer reshape the tree.
        """
        here = Path(__file__).resolve().parent
        candidates = (
            here / "sql" / "schema.sql",  # installed as a plugin dir
            here.parent / "sql" / "schema.sql",  # repo layout
        )
        for path in candidates:
            if path.is_file():
                try:
                    return path.read_text(encoding="utf-8")
                except OSError as exc:
                    raise StoreError(f"Cannot read schema at {path}: {exc}") from exc
        raise StoreError(
            "schema.sql not found; looked in: " + ", ".join(str(p) for p in candidates)
        )

    def ensure_schema(self) -> None:
        """Create the table and indexes if absent, then verify dimensions.

        Idempotent: every statement in schema.sql is IF NOT EXISTS.
        """
        ddl = self._read_schema()

        # The schema ships one placeholder (vector width) and one fixed table
        # name; both are identifiers/type modifiers, which SQL does not allow
        # as bound parameters. dims is an int and table is validated above.
        ddl = ddl.replace("%(dims)s", str(int(self.dims)))
        if self.table != "hermes_memories":
            ddl = ddl.replace("hermes_memories", self.table)

        conn = self._require_conn()
        with self._lock, conn.cursor() as cur:
            cur.execute(ddl)
            cur.execute(
                """
                SELECT a.atttypmod
                  FROM pg_attribute a
                  JOIN pg_class c ON c.oid = a.attrelid
                 WHERE c.relname = %s AND a.attname = 'embedding'
                """,
                (self.table,),
            )
            row = cur.fetchone()

        if row and row[0] not in (-1, self.dims):
            raise StoreError(
                f"Table {self.table!r} stores {row[0]}-dim vectors but the configured "
                f"embedder produces {self.dims}. Changing embedder requires re-embedding: "
                f"see scripts/reembed.py."
            )

    # -- writes -------------------------------------------------------------

    def add(
        self,
        content: str,
        embedding: Sequence[float],
        *,
        kind: str = "observation",
        source: str = "tool",
        session_id: str = "",
        agent_identity: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> int | None:
        """Insert one memory. Returns its id, or None when it was a duplicate.

        Duplicates are decided by exact content hash per identity. Near-
        duplicates are intentionally kept: deciding that two differently worded
        memories are "the same" is a judgement call that belongs to the agent,
        not to an INSERT.
        """
        import json

        content = content.strip()
        if not content:
            return None

        conn = self._require_conn()
        with self._lock, conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self.table}
                    (content, embedding, kind, source, session_id,
                     agent_identity, metadata, content_sha256)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (content_sha256, coalesce(agent_identity, ''))
                DO NOTHING
                RETURNING id
                """,
                (
                    content,
                    to_pgvector(embedding),
                    kind,
                    source,
                    session_id or None,
                    agent_identity or None,
                    json.dumps(metadata or {}),
                    _sha256(content),
                ),
            )
            row = cur.fetchone()
        return row[0] if row else None

    def delete(self, memory_id: int) -> bool:
        conn = self._require_conn()
        with self._lock, conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table} WHERE id = %s", (memory_id,))
            return cur.rowcount > 0

    # -- retrieval ----------------------------------------------------------

    def search(
        self,
        query_embedding: Sequence[float],
        query_text: str = "",
        *,
        limit: int = 10,
        kind: str = "",
        agent_identity: str = "",
        min_similarity: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Hybrid search: DiskANN vector ranking fused with lexical ranking.

        Pure vector search misses exact tokens an embedder flattens -- error
        codes, flags, file paths. Pure lexical search misses paraphrase. RRF
        fuses both rankings; a row strong in either surfaces, and a row strong
        in both wins.
        """
        conn = self._require_conn()

        # Build the optional filter as a predicate (never a bare WHERE) so it
        # can be AND-ed into any position without rewriting the statement.
        preds, params = ["TRUE"], []
        if kind:
            preds.append("kind = %s")
            params.append(kind)
        if agent_identity:
            preds.append("agent_identity = %s")
            params.append(agent_identity)
        filter_sql = " AND ".join(preds)

        vec = to_pgvector(query_embedding)
        # Over-fetch per branch so fusion has candidates to work with: a row
        # ranked 20th by vector and 3rd lexically should still be reachable.
        pool = max(limit * 5, 50)

        if query_text.strip():
            sql = f"""
            WITH vector_hits AS (
                SELECT id, row_number() OVER (ORDER BY embedding <=> %s::vector) AS rank
                  FROM {self.table}
                 WHERE {filter_sql}
                 ORDER BY embedding <=> %s::vector
                 LIMIT {pool}
            ),
            lexical_hits AS (
                SELECT id, row_number() OVER (
                           ORDER BY ts_rank(to_tsvector('simple', content),
                                            plainto_tsquery('simple', %s)) DESC
                       ) AS rank
                  FROM {self.table}
                 WHERE {filter_sql}
                   AND to_tsvector('simple', content) @@ plainto_tsquery('simple', %s)
                 LIMIT {pool}
            ),
            fused AS (
                SELECT id, SUM(score) AS score FROM (
                    SELECT id, 1.0 / ({RRF_K} + rank) AS score FROM vector_hits
                    UNION ALL
                    SELECT id, 1.0 / ({RRF_K} + rank) AS score FROM lexical_hits
                ) s GROUP BY id
            )
            SELECT m.id, m.content, m.kind, m.source, m.created_at, m.metadata,
                   1 - (m.embedding <=> %s::vector) AS similarity,
                   f.score AS fused_score
              FROM fused f JOIN {self.table} m ON m.id = f.id
             WHERE 1 - (m.embedding <=> %s::vector) >= %s
             ORDER BY f.score DESC
             LIMIT %s
            """
            args = [
                vec,
                *params,  # vector_hits: rank window + filter
                vec,  #              ORDER BY
                query_text,
                *params,  # lexical_hits: ts_rank + filter
                query_text,  #               @@ predicate
                vec,
                vec,
                min_similarity,
                limit,  # outer select
            ]
        else:
            sql = f"""
            SELECT id, content, kind, source, created_at, metadata,
                   1 - (embedding <=> %s::vector) AS similarity,
                   NULL::float8 AS fused_score
              FROM {self.table}
             WHERE {filter_sql}
               AND 1 - (embedding <=> %s::vector) >= %s
             ORDER BY embedding <=> %s::vector
             LIMIT %s
            """
            args = [vec, *params, vec, min_similarity, vec, limit]

        with self._lock, conn.cursor() as cur:
            cur.execute(sql, args)
            rows = cur.fetchall()
            hits = [
                {
                    "id": r[0],
                    "content": r[1],
                    "kind": r[2],
                    "source": r[3],
                    "created_at": r[4],
                    "metadata": r[5],
                    "similarity": float(r[6]) if r[6] is not None else None,
                    "fused_score": float(r[7]) if r[7] is not None else None,
                }
                for r in rows
            ]
            if hits:
                cur.execute(
                    f"""
                    UPDATE {self.table}
                       SET accessed_at = now(), access_count = access_count + 1
                     WHERE id = ANY(%s)
                    """,
                    ([h["id"] for h in hits],),
                )
        return hits

    def recent(self, limit: int = 10, agent_identity: str = "") -> list[dict[str, Any]]:
        conn = self._require_conn()
        where, params = "", []
        if agent_identity:
            where, params = "WHERE agent_identity = %s", [agent_identity]
        with self._lock, conn.cursor() as cur:
            cur.execute(
                f"""SELECT id, content, kind, source, created_at
                      FROM {self.table} {where}
                     ORDER BY created_at DESC LIMIT %s""",
                params + [limit],
            )
            return [
                {"id": r[0], "content": r[1], "kind": r[2], "source": r[3], "created_at": r[4]}
                for r in cur.fetchall()
            ]

    def stats(self) -> dict[str, Any]:
        conn = self._require_conn()
        with self._lock, conn.cursor() as cur:
            cur.execute(
                f"""SELECT count(*), count(*) FILTER (WHERE kind = 'fact'),
                           count(*) FILTER (WHERE kind = 'preference'),
                           count(*) FILTER (WHERE kind = 'observation'),
                           count(*) FILTER (WHERE kind = 'turn'),
                           pg_size_pretty(pg_total_relation_size(%s))
                      FROM {self.table}""",
                (self.table,),
            )
            r = cur.fetchone()
        return {
            "total": r[0],
            "facts": r[1],
            "preferences": r[2],
            "observations": r[3],
            "turns": r[4],
            "size": r[5],
        }
