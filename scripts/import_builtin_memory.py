#!/usr/bin/env python3
"""Import Hermes' built-in memory files into PostgreSQL.

``mirror_memory_tool`` only captures FUTURE writes to MEMORY.md / USER.md.
Whatever those files already hold when the plugin is switched on stays behind,
invisible to semantic recall. This backfills them once.

Entries are separated by a line containing only ``§`` (the convention Hermes'
memory tool writes). MEMORY.md entries are stored as ``fact``, USER.md entries
as ``preference`` — matching what ``on_memory_write`` does for live writes, so
imported and mirrored rows are indistinguishable afterwards.

Idempotent: the content hash unique index means re-running imports nothing new.

    python scripts/import_builtin_memory.py --dry-run
    python scripts/import_builtin_memory.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pgvector_memory.config import DEFAULT_DSN  # noqa: E402
from pgvector_memory.embeddings import EmbeddingError, OllamaEmbedder  # noqa: E402
from pgvector_memory.store import MemoryStore, StoreError  # noqa: E402

SEPARATOR = "§"


def parse_entries(path: Path) -> list[str]:
    """Split a memory file into entries on lines containing only the separator."""
    if not path.is_file():
        return []
    raw = path.read_text(encoding="utf-8")
    entries = []
    for chunk in raw.split(f"\n{SEPARATOR}\n"):
        text = chunk.strip().strip(SEPARATOR).strip()
        if text:
            entries.append(text)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hermes-home",
        default=os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")),
        help="Hermes profile directory (default: $HERMES_HOME or ~/.hermes)",
    )
    parser.add_argument("--dsn", default=os.environ.get("PGVECTOR_MEMORY_DSN", DEFAULT_DSN))
    parser.add_argument(
        "--model", default=os.environ.get("PGVECTOR_MEMORY_EMBED_MODEL", "nomic-embed-text")
    )
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("PGVECTOR_MEMORY_OLLAMA_HOST", "http://127.0.0.1:11434"),
    )
    parser.add_argument(
        "--agent-identity",
        default="",
        help="Scope imported rows to a profile (dedupe is per identity)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be imported")
    args = parser.parse_args()

    memories_dir = Path(args.hermes_home) / "memories"
    sources = [
        (memories_dir / "MEMORY.md", "fact"),
        (memories_dir / "USER.md", "preference"),
    ]

    planned: list[tuple[str, str, str]] = []
    for path, kind in sources:
        entries = parse_entries(path)
        print(f"{path}: {len(entries)} entries")
        planned.extend((text, kind, path.name) for text in entries)

    if not planned:
        print("Nothing to import.")
        return 0

    if args.dry_run:
        print(f"\n--- dry run: {len(planned)} entries would be imported ---")
        for text, kind, origin in planned:
            preview = text.replace("\n", " ")[:90]
            print(f"  [{kind:10}] ({origin}) {preview}")
        return 0

    embedder = OllamaEmbedder(args.ollama_host, args.model)
    if not embedder.is_available():
        print(
            f"Ollama at {args.ollama_host} is not serving {args.model!r}.\n"
            f"Run: ollama pull {args.model}",
            file=sys.stderr,
        )
        return 1

    dims = embedder.dims or len(embedder.embed_one("dimension probe"))
    store = MemoryStore(args.dsn, dims)
    try:
        store.connect()
        store.ensure_schema()
    except StoreError as exc:
        print(f"Storage error: {exc}", file=sys.stderr)
        return 1

    imported = skipped = 0
    try:
        # Embed in one batch: 16 sequential HTTP round-trips is pointless when
        # the endpoint accepts a list.
        vectors = embedder.embed([text for text, _, _ in planned])
        for (text, kind, origin), vector in zip(planned, vectors, strict=True):
            memory_id = store.add(
                text,
                vector,
                kind=kind,
                source="memory_tool",
                agent_identity=args.agent_identity,
                metadata={"imported_from": origin},
            )
            if memory_id is None:
                skipped += 1
            else:
                imported += 1
    except EmbeddingError as exc:
        print(f"Embedding failed: {exc}", file=sys.stderr)
        return 1
    finally:
        stats = store.stats()
        store.close()

    print(f"\nImported {imported}, skipped {skipped} already present.")
    print(
        f"Table now holds {stats['total']} memories "
        f"({stats['facts']} facts, {stats['preferences']} preferences, "
        f"{stats['observations']} observations) using {stats['size']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
