# Changelog

All notable changes to this project are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-08-31

First release. Verified end to end against a live stack: PostgreSQL 18.6,
pgvector 0.8.6, pgvectorscale 0.9.0 (DiskANN), timescaledb 2.29.2, and Ollama
serving `nomic-embed-text`.

### Added

- Release-hygiene tests (`tests/test_release.py`): the version is declared in
  three files that nothing reconciles at runtime, so a release that bumps two
  of the three would ship a package whose self-reported version is wrong.
- `MemoryProvider` implementation for Hermes Agent, backed by PostgreSQL.
- Hybrid retrieval: DiskANN vector ranking fused with lexical `tsvector`
  ranking via Reciprocal Rank Fusion (k=60). Vector search alone misses exact
  literals such as error codes and file paths; lexical search alone misses
  paraphrase.
- Three agent tools: `pgvector_remember`, `pgvector_recall`, `pgvector_forget`.
- Local embeddings through Ollama — no embedding API, no hosted vector store,
  no telemetry.
- Mirroring of built-in `MEMORY.md` / `USER.md` writes, which have a hard
  character budget the database does not.
- Background prefetch, keeping embedding off the turn thread.
- Content-hash deduplication scoped per agent identity.
- Startup refusal on embedding-dimension mismatch, instead of silently
  corrupting recall.
- Three test layers: unit, host-ABC contract, and real-stack integration.
- `scripts/install.sh`, which verifies database, extensions, and Ollama before
  copying anything, and prints the exact superuser command when one is needed
  rather than invoking `sudo` on the operator's behalf.

### Fixed

- **The lexical half of hybrid search never matched a natural-language
  question.** `plainto_tsquery` ANDs every term and the `simple` config strips
  no stopwords, so *"quem e o dono da branch e do merge?"* compiled to
  `'quem' & 'e' & 'o' & 'dono' & 'da' & 'branch' & 'e' & 'do' & 'merge'` and
  matched nothing at all. Queries are now built as an OR of their own lexemes,
  with single-character tokens dropped — `'o'` alone matched 12 of 23 rows.
- **Slash- and dot-separated technical terms were unsearchable.** Postgres'
  parser reads `branch/worktree/gates/PR/merge` as a single `file` token, so
  searching "merge" or "worktree" missed it. Content is now indexed twice —
  verbatim plus a punctuation-normalised copy — and the query is normalised
  with the same expression, because splitting only the document side left
  `CLAUDE.md` in a query matching zero rows against a document holding
  `claude` + `md`.

- Defaults were not applied when the host's `cfg_get` returned `None`.
  Hermes' `cfg_get(key, default)` yields `None` for an unset key — it does not
  apply the default. Trusting the documented signature left `ollama_host=None`
  and crashed at agent startup with
  `AttributeError: 'NoneType' object has no attribute 'rstrip'`.
  Found by running `hermes memory status` against the real loader, not by
  reading the API.

### Changed

- `test_unrelated_content_ranks_last` asserted that a bicycle sentence would
  rank last among unrelated results. Measurement disproved it: for the query
  "gato dormindo" the bicycle scores 0.4207 against PostgreSQL's 0.3956, and
  for "gato" it scores 0.4687 against 0.4034. `nomic-embed-text` groups by
  linguistic register — everyday Portuguese prose versus technical jargon —
  not by human-perceived topical distance. The test now asserts the stable
  invariant (every relevant hit outranks every irrelevant one), which holds
  across all probe queries.

[Unreleased]: https://github.com/marlon-costa-dc/hermes-pgvector-memory/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/marlon-costa-dc/hermes-pgvector-memory/releases/tag/v0.1.0
