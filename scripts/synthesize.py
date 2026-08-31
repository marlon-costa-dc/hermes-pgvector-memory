#!/usr/bin/env python3
"""Cross-memory synthesis: cluster live memories into themes, distill digests.

Where distill_prompts.py turns RAW history into memories, this script turns
EXISTING memories into higher-order ones — the local equivalent of
hindsight_reflect. Per theme (a greedy cosine cluster of live, non-synthetic
memories), an LLM writes one digest that:

- cites the member ids [n] supporting each claim (back-references for drill-down);
- preserves technical vocabulary verbatim (surviving-vocabulary rule);
- flags recurring PROCEDURES as "candidato a skill" — the operator consolidates
  learned workflows into skills at the end of heavy work.

Digests enter distilled_candidates like any candidate (origin='synthesis') and
go through the same promote path with dedupe and triple-key supersession.
Idempotent: an identical digest hash is a no-op, and synthesis never seeds a
new synthesis (source <> 'synthesis' in the seed query) so themes cannot cascade.

    python3 scripts/synthesize.py [--dsn ...] [--ollama ...] [--model phi4-mini]
                                  [--thresh 0.82] [--min-size 3] [--limit 400]
                                  [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from distill_prompts import (  # noqa: E402
    DSN_DEFAULT,
    EMBED_MODEL_DEFAULT,
    OLLAMA_DEFAULT,
    _ollama_chat,
    _strip_fences,
)

DISTILL_MODEL_DEFAULT = "phi4-mini"

THEME_SEEDS_SQL = """
WITH seeds AS (
    SELECT id, embedding FROM {table}
     WHERE superseded_at IS NULL
       AND kind IN ('fact', 'preference', 'observation')
       AND source <> 'synthesis'
       AND embedding IS NOT NULL
     ORDER BY created_at DESC
     LIMIT %s
)
SELECT s.id AS seed_id, m.id AS member_id, m.content, m.kind, m.specific_context
FROM seeds s
JOIN LATERAL (
    SELECT id, content, kind, specific_context
      FROM {table}
     WHERE superseded_at IS NULL AND id <> s.id AND embedding IS NOT NULL
       AND source <> 'synthesis'
       AND embedding <=> s.embedding < %s
     ORDER BY embedding <=> s.embedding
     LIMIT %s
) m ON true;
"""

SYNTH_PROMPT = """Você sintetiza memórias de um dev-operador num digest reutilizável.

Regras:
- Cite os ids [n] dos membros que suportam cada afirmação.
- Preserve termos técnicos verbatim (paths, flags, versões, comandos).
- Se o tema for um PROCEDIMENTO recorrente, inclua a frase "candidato a skill".
- Idioma: o das memórias. Máximo 200 palavras. Sem preâmbulo.

Memórias do tema:
{members}
"""


def _fetch_theme_rows(dsn: str, table: str, thresh: float, limit: int, neighbors: int):
    from typing import cast

    import psycopg
    from psycopg.abc import Query

    # table is an identifier interpolated by .format(); Query cast mirrors the
    # store's LiteralString handling for identifier-templated statements.
    sql = cast(Query, THEME_SEEDS_SQL.format(table=table))
    conn = psycopg.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (limit, thresh, neighbors))
            return cur.fetchall()
    finally:
        conn.close()


def _theme_clusters(
    rows: list[tuple[int, int, str, str, str]], *, min_size: int
) -> list[dict[str, Any]]:
    """Group lateral-join rows into themes keyed by seed_id.

    rows: (seed_id, member_id, content, kind, specific_context). The seed's own
    row is not returned by the SQL (id <> s.id), so the seed content is filled
    from its first member's perspective when building the member list — the
    seed id participates as a member id placeholder 0 (its content is not
    needed for synthesis; only member ids and texts are).
    """
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for seed_id, member_id, content, kind, specific_context in rows:
        by_seed.setdefault(seed_id, []).append(
            {
                "id": member_id,
                "content": content,
                "kind": kind,
                "specific_context": specific_context,
            }
        )
    themes: list[dict[str, Any]] = []
    claimed: set[int] = set()
    # Deterministic order: largest theme first, then by seed id.
    for seed_id in sorted(by_seed, key=lambda s: (-len(by_seed[s]), s)):
        if seed_id in claimed:
            continue
        members = by_seed[seed_id]
        if len(members) + 1 < min_size:
            continue
        member_ids = [seed_id] + [m["id"] for m in members]
        if any(mid in claimed for mid in member_ids):
            continue
        claimed.update(member_ids)
        themes.append({"seed_id": seed_id, "members": members, "member_ids": member_ids})
    return themes


def _synthesis_row(theme: dict[str, Any], digest_text: str) -> dict[str, Any]:
    return {
        "content": digest_text,
        "kind": "observation",
        "origin": "synthesis",
        "prompt_ids": [],
        "member_ids": theme["member_ids"],
        "theme_seed": theme["seed_id"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dsn", default=DSN_DEFAULT)
    ap.add_argument("--ollama", default=OLLAMA_DEFAULT)
    ap.add_argument("--embed-model", default=EMBED_MODEL_DEFAULT)
    ap.add_argument("--model", default=DISTILL_MODEL_DEFAULT)
    ap.add_argument("--table", default="hermes_memories")
    ap.add_argument("--thresh", type=float, default=0.25, help="cosine DISTANCE max")
    ap.add_argument("--min-size", type=int, default=3)
    ap.add_argument("--limit", type=int, default=400, help="max seeds to consider")
    ap.add_argument("--neighbors", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = _fetch_theme_rows(args.dsn, args.table, args.thresh, args.limit, args.neighbors)
    themes = _theme_clusters(rows, min_size=args.min_size)
    print(f"synthesize: {len(themes)} themes (min size {args.min_size})")
    if not themes:
        return

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if args.dry_run:
        for t in themes:
            listing = "\n".join(
                f"[{m['id']}] ({m['kind']}) {m['content'][:80]}" for m in t["members"]
            )
            print(f"--- theme seed={t['seed_id']} members={t['member_ids']}")
            print(listing)
        return

    from distill_prompts import _connect, _ensure_schema, _sha

    conn = _connect(args.dsn)
    _ensure_schema(conn)
    proposed = 0
    for t in themes:
        listing = "\n".join(f"[{m['id']}] ({m['kind']}) {m['content']}" for m in t["members"])
        raw = _ollama_chat(
            args.ollama,
            args.model,
            SYNTH_PROMPT.replace("{members}", listing),
            json_mode=False,
        )
        digest = _strip_fences(raw).strip()
        if len(digest) < 40:
            print(f"  theme seed={t['seed_id']}: digest too short, skipped")
            continue
        row = _synthesis_row(t, digest)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO distilled_candidates
                    (content, kind, origin, prompt_ids, content_sha256, model,
                     core, specific_context, tags)
                VALUES (%s, %s, %s, %s, %s, %s, %s, '', %s)
                ON CONFLICT (content_sha256) DO NOTHING
                """,
                (
                    row["content"],
                    row["kind"],
                    row["origin"],
                    row["prompt_ids"],
                    _sha(row["content"]),
                    args.model,
                    row["content"],
                    json.dumps({"member_ids": row["member_ids"], "theme_seed": row["theme_seed"]}),
                ),
            )
            # member_ids/theme_seed live in tags+metadata path: keep them also
            # in the tags column as machine-readable strings for promote to copy.
        conn.commit()
        proposed += 1
        print(f"  theme seed={t['seed_id']}: digest proposed ({len(digest)} chars)", flush=True)
    print(f"synthesize: {proposed} digests proposed (run promote to review/store)")


if __name__ == "__main__":
    main()
