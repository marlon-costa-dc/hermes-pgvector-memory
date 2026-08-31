#!/usr/bin/env python3
"""Distil the operator's agent-history into Hermes memories.

Raw prompts are not memories. This pipeline is staged so each step is
checkable on its own:

    extract   parse claude/codex/opencode history -> distill_prompts (staging)
    embed     batch-embed staged prompts through Ollama
    distill   LLM reads prompt batches, proposes durable memories
              -> distilled_candidates (never touches the live table)
    promote   copy reviewed candidates into hermes_memories, skipping
              anything semantically close to an existing memory
    stats     row counts per stage, for verifying against known totals

Extraction is deterministic and idempotent: prompt text is hashed and the
staging table has a UNIQUE constraint, so re-running never duplicates.
Distillation is resumable: each batch key is recorded in distill_progress.

Counts on the reference install (2026-08-31) were measured before this
script existed: claude 1,012 / codex 1,513 / opencode 42,702 human prompts.
`stats` after `extract` should land on those numbers; a mismatch means the
filters changed and someone must say so out loud.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import urllib.request
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DSN_DEFAULT = "postgresql:///hermes_memory?host=/run/postgresql"
EMBED_MODEL_DEFAULT = "nomic-embed-text"
DISTILL_MODEL_DEFAULT = "qwen3:4b"
OLLAMA_DEFAULT = "http://127.0.0.1:11434"

# Prompts shorter than this carry no distillable intent ("ok", "?", "siga").
MIN_DISTILL_CHARS = 40
# Batch size for the LLM reader: enough context to judge, small enough that
# a 4B local model keeps producing valid JSON.
DISTILL_BATCH = 12
# A candidate within this cosine similarity of an existing memory is a
# duplicate and is not promoted.
PROMOTE_MAX_SIMILARITY = 0.90

SCHEMA = """
CREATE TABLE IF NOT EXISTS distill_prompts (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    origin      text        NOT NULL,
    source_file text        NOT NULL,
    session_key text        NOT NULL DEFAULT '',
    ts          timestamptz,
    prompt      text        NOT NULL,
    chars       int         NOT NULL,
    prompt_sha256 bytea     NOT NULL UNIQUE,
    embedding   vector(768),
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS distilled_candidates (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    content       text        NOT NULL,
    kind          text        NOT NULL,
    origin        text        NOT NULL,
    prompt_ids    bigint[]    NOT NULL,
    content_sha256 bytea      NOT NULL UNIQUE,
    model         text        NOT NULL,
    promoted_at   timestamptz,
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS distill_progress (
    batch_key text        PRIMARY KEY,
    done_at   timestamptz NOT NULL DEFAULT now()
);
"""


def _connect(dsn: str):
    import psycopg

    return psycopg.connect(dsn)


def _one(cur, sql: str, params: tuple = ()) -> Any:
    """fetchone() that fails loud on an empty result instead of on [0]."""
    cur.execute(sql, params)
    row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"query returned no rows: {sql[:80]}")
    return row


def _ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
    conn.commit()


def _sha(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


# --------------------------------------------------------------------------
# Source parsers. Each yields (source_file, session_key, ts_iso, prompt).
# Noise filters are deliberately conservative: anything uncertain stays in
# staging, because staging is cheap and deletion is forever.
# --------------------------------------------------------------------------

_CLAUDE_DROP_PREFIXES = (
    "caveat:",
    "[request interrupted",
    "[request cancelled",
)


def _iter_claude(root: Path) -> Iterable[tuple[str, str, str | None, str]]:
    for path in sorted(root.glob("*/*.jsonl")):
        session_key = path.stem
        for line in path.open(encoding="utf-8", errors="replace"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "user" or rec.get("isSidechain"):
                continue
            msg = rec.get("message") or {}
            content = msg.get("content")
            texts: list[str] = []
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(str(block.get("text", "")))
            prompt = "\n".join(t for t in texts if t).strip()
            if not prompt:
                continue
            if prompt.lower().startswith(_CLAUDE_DROP_PREFIXES):
                continue
            if prompt.startswith("<"):  # injected XML-ish wrappers
                continue
            yield str(path), session_key, rec.get("timestamp"), prompt


def _iter_codex(root: Path) -> Iterable[tuple[str, str, str | None, str]]:
    for path in sorted(root.rglob("rollout-*.jsonl")):
        session_key = path.stem
        for line in path.open(encoding="utf-8", errors="replace"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "response_item":
                continue
            payload = rec.get("payload") or {}
            if payload.get("type") != "message" or payload.get("role") != "user":
                continue
            texts = [
                str(part.get("text", ""))
                for part in payload.get("content") or []
                if isinstance(part, dict) and part.get("type") == "input_text"
            ]
            prompt = "\n".join(t for t in texts if t).strip()
            if not prompt or prompt.startswith("<"):
                continue
            yield str(path), session_key, rec.get("timestamp"), prompt


def _iter_opencode(db_path: Path) -> Iterable[tuple[str, str, str | None, str]]:
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = db.cursor()
        # LIKE prefilter keeps JSON parsing off the 466k-row message table.
        user_msgs: dict[str, tuple[str, int]] = {}
        for mid, data in cur.execute(
            'SELECT id, data FROM message WHERE data LIKE \'%"role":"user"%\''
        ):
            try:
                rec = json.loads(data)
            except json.JSONDecodeError:
                continue
            if rec.get("role") != "user":
                continue
            created = (rec.get("time") or {}).get("created", 0)
            user_msgs[mid] = (rec.get("session_id") or "", int(created))
        # Same prefilter on part: only text parts of known user messages.
        # SQLite has no ANY(%s): chunked IN (?,...) placeholders instead.
        id_list = list(user_msgs)

        def _parts() -> Iterable[tuple[str, str]]:
            for i in range(0, len(id_list), 500):
                chunk = id_list[i : i + 500]
                marks = ",".join("?" * len(chunk))
                yield from cur.execute(
                    "SELECT message_id, data FROM part"
                    f" WHERE message_id IN ({marks})"
                    ' AND data LIKE \'%"type":"text"%\'',
                    chunk,
                )

        for mid, data in (_parts() if id_list else iter(())):
            if mid not in user_msgs:
                continue
            try:
                part = json.loads(data)
            except json.JSONDecodeError:
                continue
            if part.get("type") != "text":
                continue
            prompt = str(part.get("text", "")).strip()
            if not prompt or prompt.startswith("<"):
                continue
            session_id, created = user_msgs[mid]
            ts = (
                datetime.fromtimestamp(created / 1000, tz=timezone.utc).isoformat()
                if created
                else None
            )
            yield str(db_path), session_id, ts, prompt
    finally:
        db.close()


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def cmd_extract(args: argparse.Namespace) -> None:
    conn = _connect(args.dsn)
    _ensure_schema(conn)
    home = Path.home()
    sources = {
        "claude": _iter_claude(home / ".claude" / "projects"),
        "codex": _iter_codex(home / ".codex" / "sessions"),
        "opencode": _iter_opencode(home / ".local/share/opencode/opencode.db"),
    }
    if args.origin:
        sources = {k: v for k, v in sources.items() if k == args.origin}
    total = 0
    inserted = 0
    for origin, it in sources.items():
        rows: list[tuple[Any, ...]] = []
        n_origin = 0
        for source_file, session_key, ts, prompt in it:
            n_origin += 1
            total += 1
            if len(prompt) > 8000:  # pasted dumps are not operator intent
                prompt = prompt[:8000]
            rows.append(
                (
                    origin,
                    source_file,
                    session_key,
                    ts,
                    prompt,
                    len(prompt),
                    _sha(prompt),
                )
            )
            if len(rows) >= 500:
                inserted += _copy_rows(conn, rows)
                rows = []
        if rows:
            inserted += _copy_rows(conn, rows)
        print(f"  {origin}: {n_origin} prompts found")
    conn.commit()
    print(f"extract: scanned {total}, inserted {inserted} new (rest duplicates)")


def _copy_rows(conn, rows: list[tuple[Any, ...]]) -> int:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO distill_prompts
                (origin, source_file, session_key, ts, prompt, chars, prompt_sha256)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (prompt_sha256) DO NOTHING
            """,
            rows,
        )
        return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0


def cmd_embed(args: argparse.Namespace) -> None:
    conn = _connect(args.dsn)
    _ensure_schema(conn)
    with conn.cursor() as cur:
        pending = _one(cur, "SELECT count(*) FROM distill_prompts WHERE embedding IS NULL")[0]
    print(f"embed: {pending} prompts without embedding")
    if not pending:
        return

    def embed_batch(texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": args.embed_model, "input": texts}).encode()
        req = urllib.request.Request(
            f"{args.ollama}/api/embed",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.load(resp)["embeddings"]

    done = 0
    while True:
        with conn.cursor(name="distill_embed_cursor") as cur:
            cur.itersize = 512
            cur.execute(
                """
                SELECT id, prompt FROM distill_prompts
                 WHERE embedding IS NULL ORDER BY id LIMIT 512
                """
            )
            batch = cur.fetchall()
        if not batch:
            break
        vectors = embed_batch([r[1] for r in batch])
        with conn.cursor() as cur:
            for (pid, _), vec in zip(batch, vectors, strict=True):
                cur.execute(
                    "UPDATE distill_prompts SET embedding = %s WHERE id = %s",
                    (vec, pid),
                )
        conn.commit()
        done += len(batch)
        print(f"  {done}/{pending}", flush=True)
    print("embed: done")


def _ollama_chat(base: str, model: str, prompt: str, json_mode: bool) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            **({"format": "json"} if json_mode else {}),
            "options": {
                "temperature": 0.1,
                # qwen3's thinking mode will happily emit 3000+ reasoning
                # tokens before the JSON (measured: 10+ min per batch at
                # 5 t/s). Cap the output hard; the JSON itself is small.
                "num_predict": 700,
            },
        }
    ).encode()
    req = urllib.request.Request(
        f"{base}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)["message"]["content"]


DISTILL_PROMPT = """Below are prompts the operator typed to coding agents.
Extract DURABLE facts about the operator, their environment, preferences and
conventions that would still matter months later. Ignore one-off task
details, file contents and anything re-derivable from the machine.

/no_think
Return JSON and NOTHING else: {"memories": [{"content": "...", "kind":
"fact|preference|observation", "prompt_indexes": [0,1]}]}
- content: ONE self-contained declarative sentence in the operator's language.
- kind: fact = objective/durable, preference = how they want things done,
  observation = weaker notice.
- prompt_indexes: indexes of the prompts supporting it.
- Maximum 4 memories. If nothing durable is present, return {"memories": []}.

Prompts:
{prompts}
"""


def cmd_distill(args: argparse.Namespace) -> None:
    conn = _connect(args.dsn)
    _ensure_schema(conn)
    with conn.cursor() as cur:
        total = _one(
            cur,
            "SELECT count(*) FROM distill_prompts WHERE chars >= %s",
            (MIN_DISTILL_CHARS,),
        )[0]
    print(f"distill: {total} eligible prompts (chars >= {MIN_DISTILL_CHARS})")

    offset = 0
    batch_no = 0
    proposed = 0
    while True:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, prompt FROM distill_prompts
                 WHERE chars >= %s
                 ORDER BY id
                 LIMIT %s OFFSET %s
                """,
                (MIN_DISTILL_CHARS, DISTILL_BATCH, offset),
            )
            batch = cur.fetchall()
        if not batch:
            break
        batch_no += 1
        offset += DISTILL_BATCH
        batch_key = f"{batch[0][0]}-{batch[-1][0]}"
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM distill_progress WHERE batch_key = %s", (batch_key,))
            already = cur.fetchone()
        if already:
            continue

        listing = "\n".join(f"[{i}] {p[:500]}" for i, (_pid, p) in enumerate(batch))
        raw = _ollama_chat(
            args.ollama,
            args.model,
            DISTILL_PROMPT.replace("{prompts}", listing),
            json_mode=True,
        )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  batch {batch_key}: model returned invalid JSON, skipped")
            with conn.cursor() as cur:
                cur.execute("INSERT INTO distill_progress VALUES (%s, now())", (batch_key,))
            conn.commit()
            continue

        kept = 0
        with conn.cursor() as cur:
            for mem in parsed.get("memories") or []:
                content = str(mem.get("content", "")).strip()
                kind = str(mem.get("kind", "observation"))
                if kind not in ("fact", "preference", "observation"):
                    kind = "observation"
                idxs = [
                    batch[i][0]
                    for i in mem.get("prompt_indexes") or []
                    if isinstance(i, int) and 0 <= i < len(batch)
                ]
                if not content or not idxs:
                    continue
                origin = "mixed"
                with conn.cursor() as c2:
                    c2.execute(
                        "SELECT origin FROM distill_prompts WHERE id = ANY(%s) LIMIT 1",
                        (idxs,),
                    )
                    row = c2.fetchone()
                    if row:
                        origin = row[0]
                try:
                    cur.execute(
                        """
                        INSERT INTO distilled_candidates
                            (content, kind, origin, prompt_ids, content_sha256, model)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (content_sha256) DO NOTHING
                        """,
                        (content, kind, origin, idxs, _sha(content), args.model),
                    )
                    kept += 1
                except Exception as exc:  # noqa: BLE001 — log, keep going
                    print(f"  candidate insert failed: {exc}")
            cur.execute("INSERT INTO distill_progress VALUES (%s, now())", (batch_key,))
        conn.commit()
        proposed += kept
        print(
            f"  batch {batch_no} ({batch_key}): {kept} candidates (total {proposed})",
            flush=True,
        )
        if args.batches and batch_no >= args.batches:
            print(f"distill: stopping after {args.batches} batches (--batches)")
            break
    print(f"distill: {proposed} new candidates")


def cmd_promote(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pgvector_memory.embeddings import OllamaEmbedder
    from pgvector_memory.store import MemoryStore

    conn = _connect(args.dsn)
    _ensure_schema(conn)
    embedder = OllamaEmbedder(args.ollama, args.embed_model)
    store = MemoryStore(args.dsn, 768, "hermes_memories")
    store.connect()
    store.ensure_schema()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, content, kind, origin, prompt_ids FROM distilled_candidates"
            " WHERE promoted_at IS NULL ORDER BY id"
        )
        pending = cur.fetchall()
    print(f"promote: {len(pending)} candidates waiting")
    promoted = skipped_dup = 0
    for cid, content, kind, origin, prompt_ids in pending:
        vec = embedder.embed_one(content)
        existing = store.search(vec, "", limit=1, min_similarity=PROMOTE_MAX_SIMILARITY)
        if existing:
            skipped_dup += 1
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE distilled_candidates SET promoted_at = now() WHERE id = %s",
                    (cid,),
                )
            continue
        meta = {"origin": origin, "prompt_ids": list(prompt_ids or [])}
        store.add(
            content,
            vec,
            kind=kind,
            source="distilled",
            session_id="distill",
            agent_identity="",
            metadata=meta,
        )
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE distilled_candidates SET promoted_at = now() WHERE id = %s",
                (cid,),
            )
        conn.commit()
        promoted += 1
    print(f"promote: {promoted} added, {skipped_dup} skipped as near-duplicates")


def cmd_stats(args: argparse.Namespace) -> None:
    conn = _connect(args.dsn)
    _ensure_schema(conn)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT origin, count(*), sum(chars)/1024
              FROM distill_prompts GROUP BY origin ORDER BY origin
            """
        )
        print("=== staged prompts ===")
        for origin, n, kb in cur.fetchall():
            print(f"  {origin:9s} {n:>7} prompts  {kb:>8} kB")
        embedded = _one(
            cur,
            "SELECT count(*) FROM distill_prompts WHERE embedding IS NOT NULL",
        )[0]
        print(f"  embedded: {embedded}")
        total_c, open_c = _one(
            cur,
            "SELECT count(*), count(*) FILTER (WHERE promoted_at IS NULL)"
            " FROM distilled_candidates",
        )
        print(f"=== candidates: {total_c} total, {open_c} awaiting promotion ===")
        cur.execute("SELECT source, count(*) FROM hermes_memories GROUP BY source")
        print("=== hermes_memories by source ===")
        for src, n in cur.fetchall():
            print(f"  {src}: {n}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dsn", default=DSN_DEFAULT)
    ap.add_argument("--ollama", default=OLLAMA_DEFAULT)
    ap.add_argument("--embed-model", default=EMBED_MODEL_DEFAULT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--origin", choices=["claude", "codex", "opencode"])
    sub.add_parser("embed")
    di = sub.add_parser("distill")
    di.add_argument("--model", default=DISTILL_MODEL_DEFAULT)
    di.add_argument("--batches", type=int, default=0, help="0 = all")
    sub.add_parser("promote")
    sub.add_parser("stats")
    args = ap.parse_args()
    {
        "extract": cmd_extract,
        "embed": cmd_embed,
        "distill": cmd_distill,
        "promote": cmd_promote,
        "stats": cmd_stats,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
