# hermes-pgvector-memory

PostgreSQL memory provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent). Stores what the agent learns as vector embeddings, retrieves it by meaning with [pgvectorscale](https://github.com/timescale/pgvectorscale)'s DiskANN index, and computes every embedding locally through [Ollama](https://ollama.com).

**Nothing leaves your machine.** No embedding API, no hosted vector database, no telemetry.

## Why

Hermes ships with `MEMORY.md` and `USER.md` — plain files injected into every prompt. They work, but they have a hard character budget, and once it fills you must delete something to learn something. There is also no retrieval: everything is always in context, or gone.

This plugin adds a second tier. Memories live in PostgreSQL, retrieval is semantic, and only the handful relevant to the current turn enter the prompt. The built-in files keep working; their writes are mirrored here, so a memory evicted from `MEMORY.md` for space is still findable.

## How retrieval works

Two rankings, fused.

**Vector search** finds paraphrase. "gato dormindo" retrieves "A gata tira uma soneca no sofá" — almost no shared vocabulary, same meaning.

**Lexical search** finds literals that embedders flatten: error codes, CLI flags, file paths, identifiers. `E0609` is a nearly meaningless token to an embedding model and an exact match to `tsquery`.

The two are merged with Reciprocal Rank Fusion (`score = Σ 1/(k + rank)`, k=60). RRF needs no score calibration, which matters because cosine distance and `ts_rank` are not on comparable scales. A memory strong in either ranking surfaces; strong in both wins.

The DiskANN index from pgvectorscale keeps recall sublinear as the table grows — the difference between a few thousand memories and a few hundred thousand.

## Requirements

| | |
|---|---|
| PostgreSQL | 16+ |
| [pgvector](https://github.com/pgvector/pgvector) | any recent version |
| [pgvectorscale](https://github.com/timescale/pgvectorscale) | 0.9+ (the `diskann` access method) |
| [Ollama](https://ollama.com) | serving an embedding model |
| Python | 3.10+, `psycopg[binary]>=3.1` |

```bash
ollama pull nomic-embed-text     # 768 dims, ~274 MB
createdb hermes_memory
sudo -u postgres psql -d hermes_memory -c 'CREATE EXTENSION vectorscale CASCADE;'
```

`CREATE EXTENSION vectorscale` requires superuser and pulls in `vector` via `CASCADE`.

## Install

```bash
git clone https://github.com/marlon-costa-dc/hermes-pgvector-memory
cd hermes-pgvector-memory
./scripts/install.sh
hermes config set memory.provider pgvector-memory
```

The installer verifies the database, the extensions, and Ollama before copying anything, and tells you the exact command to run if something is missing. It never invokes `sudo` on your behalf.

Verify:

```bash
hermes memory status
```

## Configuration

`$HERMES_HOME/config.yaml`:

```yaml
memory:
  provider: pgvector-memory

plugins:
  pgvector-memory:
    dsn: "postgresql:///hermes_memory?host=/run/postgresql"
    embed_model: nomic-embed-text
    ollama_host: "http://127.0.0.1:11434"
    auto_recall: true
    recall_limit: 5
    min_similarity: 0.55
    auto_capture_turns: false
    mirror_memory_tool: true
```

| Key | Default | What it does |
|---|---|---|
| `dsn` | unix socket, `hermes_memory` | libpq connection string |
| `embed_model` | `nomic-embed-text` | Ollama embedding model |
| `ollama_host` | `http://127.0.0.1:11434` | Ollama base URL |
| `auto_recall` | `true` | Inject relevant memories before each turn |
| `recall_limit` | `5` | Max memories injected per turn |
| `min_similarity` | `0.55` | Cosine floor for automatic recall |
| `auto_capture_turns` | `false` | Also store every conversation turn |
| `mirror_memory_tool` | `true` | Mirror `MEMORY.md`/`USER.md` writes |
| `table` | `hermes_memories` | Table name |

Every key is overridable by environment variable: `PGVECTOR_MEMORY_DSN`, `PGVECTOR_MEMORY_EMBED_MODEL`, and so on. Environment wins over `config.yaml`, matching Hermes' own convention that credentials live in the environment and behaviour lives in config.

**On `auto_capture_turns`:** off by default and worth keeping off. Storing every turn grows the table fast and fills recall with conversational filler that crowds out curated facts. Explicit `pgvector_remember` calls and mirrored `MEMORY.md` writes are the signal; turn capture is the noise.

**On `min_similarity`:** applies only to automatic recall. Explicit `pgvector_recall` calls bypass the floor — when the agent is deliberately searching, it should see weak matches too.

## Tools

| Tool | Purpose |
|---|---|
| `pgvector_remember` | Store a durable memory (`fact`, `preference`, or `observation`) |
| `pgvector_recall` | Hybrid search over stored memories |
| `pgvector_forget` | Delete by id |

## Schema

```sql
CREATE TABLE hermes_memories (
    id              bigserial PRIMARY KEY,
    content         text NOT NULL,
    embedding       vector(768),
    kind            text NOT NULL DEFAULT 'observation',
    source          text NOT NULL DEFAULT 'tool',
    session_id      text,
    agent_identity  text,
    metadata        jsonb NOT NULL DEFAULT '{}',
    content_sha256  bytea NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    accessed_at     timestamptz,
    access_count    integer NOT NULL DEFAULT 0
);
```

Indexes: DiskANN on `embedding` (cosine), GIN on `to_tsvector('simple', content)`, plus btree on `created_at` and `kind`.

Exact-duplicate content is suppressed per `agent_identity` via a unique index on the hash — a conversation loop would otherwise store the same sentence hundreds of times. Near-duplicates are deliberately kept: deciding two differently-worded memories are "the same" is a judgement call that belongs to the agent, not to an `INSERT`.

The FTS configuration is `simple`, not a language-specific dictionary, because memories mix Portuguese, English, code identifiers and file paths — stemming any one of those hurts the others.

## Export

Plain PostgreSQL, no proprietary format:

```bash
pg_dump -t hermes_memories hermes_memory > memories.sql
psql -d hermes_memory -c "COPY (SELECT id, content, kind, created_at FROM hermes_memories) TO STDOUT WITH CSV HEADER" > memories.csv
```

## Tests

```bash
uv venv .venv && uv pip install --python .venv/bin/python pytest 'psycopg[binary]'
.venv/bin/python -m pytest tests/ -v
```

Three layers:

- **`test_unit.py`** — pure logic. No database, no Ollama.
- **`test_contract.py`** — asserts the provider satisfies Hermes' real `MemoryProvider` ABC, including override signatures. Catches host-side drift. Skips if Hermes is not installed.
- **`test_integration.py`** — real PostgreSQL, real DiskANN, real embeddings. Skips with a printed reason when the stack is absent.

The integration suite asserts semantic behaviour, not just that queries run: paraphrase must outrank word overlap, unrelated content must rank last, a similarity floor of 0.9 must return nothing for an unrelated query, and an exact literal (`E0609`) must be findable despite the embedder blurring it.

Tests resolve the host ABC from a real Hermes checkout when present (`HERMES_AGENT_PATH`, or the default install path) and fall back to a stub otherwise — the stub keeps the suite runnable, and `test_contract.py` skips rather than passing against it.

## Changing the embedding model

Vector width is fixed at table creation. Pointing the plugin at a model of a different dimension is refused at startup with an explicit error rather than silently corrupting recall. To switch, dump the content, drop the table, and re-embed.

## License

MIT
