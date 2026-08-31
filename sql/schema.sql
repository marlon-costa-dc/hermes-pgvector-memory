-- Schema for hermes-pgvector-memory.
--
-- Applied idempotently by the plugin on first initialize(); also usable
-- standalone:  psql -d hermes_memory -f sql/schema.sql
--
-- Requires: pgvector (vector type) and pgvectorscale (diskann index).
-- Both need superuser to CREATE EXTENSION, so that step is deliberately
-- NOT done here — see scripts/install.sh.

CREATE TABLE IF NOT EXISTS hermes_memories (
    id              bigserial PRIMARY KEY,
    content         text        NOT NULL,
    embedding       vector(%(dims)s),

    -- What kind of memory this is. Drives retrieval weighting and lets the
    -- operator prune one class without touching the others.
    kind            text        NOT NULL DEFAULT 'observation'
                    CHECK (kind IN ('observation', 'fact', 'preference', 'turn')),

    -- Where it came from: 'tool' (agent called pgvector_remember), 'turn'
    -- (auto-captured conversation), 'memory_tool' (mirrored from the built-in
    -- MEMORY.md/USER.md writes), 'import' (bulk ingest).
    source          text        NOT NULL DEFAULT 'tool',

    session_id      text,
    agent_identity  text,
    metadata        jsonb       NOT NULL DEFAULT '{}'::jsonb,

    -- Content hash for exact-duplicate suppression. A conversation loop can
    -- otherwise store the same sentence hundreds of times.
    content_sha256  bytea       NOT NULL,

    created_at      timestamptz NOT NULL DEFAULT now(),
    accessed_at     timestamptz,
    access_count    integer     NOT NULL DEFAULT 0
);

-- Exact-duplicate suppression, scoped per identity: two profiles may
-- legitimately hold the same fact.
CREATE UNIQUE INDEX IF NOT EXISTS hermes_memories_dedupe
    ON hermes_memories (content_sha256, coalesce(agent_identity, ''));

-- DiskANN (pgvectorscale). Cosine distance matches how nomic-embed-text and
-- most sentence embedders are trained. This is the index that makes recall
-- sublinear as the table grows past a few hundred thousand rows.
CREATE INDEX IF NOT EXISTS hermes_memories_diskann
    ON hermes_memories USING diskann (embedding vector_cosine_ops);

-- Lexical half of hybrid search. 'simple' rather than a language-specific
-- dictionary on purpose: memories mix Portuguese, English, code identifiers
-- and file paths, and stemming any one of those hurts the others.
CREATE INDEX IF NOT EXISTS hermes_memories_fts
    ON hermes_memories USING gin (to_tsvector('simple', content));

CREATE INDEX IF NOT EXISTS hermes_memories_created_at
    ON hermes_memories (created_at DESC);

CREATE INDEX IF NOT EXISTS hermes_memories_kind
    ON hermes_memories (kind, created_at DESC);
