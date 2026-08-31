# Changelog

All notable changes to this project are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] — 2026-08-31

Memory pipeline v0.3: ingest, enrich, dedupe, synthesize, cross-memory.

### Added
- `capture_mode` config (`staging` default): `sync_turn` enqueues conversation
  turns into `distill_prompts` instead of writing raw turns into the live
  table; `live` preserves v0.2 behaviour; `off` discards.
- `on_pre_compress` hook: the transcript about to be compacted away is staged
  for distillation before the lossy summary (best-effort, idempotent by hash).
- Two-pass distillation: cheap classify pass, then structured enrichment
  (core + specific_context + tags, surviving-vocabulary rule), session-
  coherent batches, resumable `:enrich` batch keys.
- Intra-batch candidate clustering: same-topic candidates merge with united
  evidence before reaching `distilled_candidates`.
- Deterministic supersession (MemStrata, arXiv 2606.26511): mutable-value
  memories carry a (subject, relation, object) triple; a new object retires
  the old by key — never by similarity (cosine cannot separate contradiction
  from duplicate: AUROC 0.59). `superseded_at`/`superseded_by` keep the
  bi-temporal ledger queryable; recall filters retired memories by default.
- `pgvector_synthesize` tool + `scripts/synthesize.py`: theme clustering over
  live memories, digests with member back-references, procedures flagged as
  "candidato a skill".
- `identity` argument on remember/recall; `cross_identity` recall labels hits
  `[identity]`.

### Changed
- `store.add` accepts specific_context/tags/subject/relation/object.
- Version bumps to 0.3.0; fourth tool schema declared.

## [0.2.0] — 2026-08-31

Migrating the operator's real `MEMORY.md` into the database and then querying
it exposed two bugs that unit tests could not: the corpus is telegraphic,
mixes Portuguese with English and code identifiers, and asks questions in
prose. Recall on it went from 2/5 to 5/7.

### Added

- `scripts/import_builtin_memory.py` — backfills Hermes' built-in
  `MEMORY.md` / `USER.md` into PostgreSQL. `mirror_memory_tool` only captures
  *future* writes, so everything the files already held stayed invisible to
  semantic recall. Measured on a live install: 23 rows in the database, none
  with `source='memory_tool'`, while the files held 16 entries. Splits on the
  `§` separator, embeds in one batch, and relies on the content-hash index for
  idempotence (second run: `Imported 0, skipped 16`).

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
- **`install.sh` reported success over a broken install.** The Hermes venv is
  uv-managed and ships no pip, so `python -m pip install` failed with
  `No module named pip`; the error was swallowed by `|| warn` and the script
  printed "psycopg installed" while the provider was dead
  (`available: False`). It now uses `uv` when present and verifies by
  *importing* the module — an installer's exit code proves nothing.

### Known limitation

Two of seven probe queries still fail, and no code change fixes them: they
need causal inference rather than similarity. *"Por que o /home encheu?"*
should retrieve a memory stating `bd list --all` is 17.7 GB, with which it
shares no vocabulary. `bge-m3` (multilingual, 1024-dim) was measured on
exactly these cases and fails them too, so this is not an embedding-model
choice.

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

[Unreleased]: https://github.com/marlon-costa-dc/hermes-pgvector-memory/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/marlon-costa-dc/hermes-pgvector-memory/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/marlon-costa-dc/hermes-pgvector-memory/releases/tag/v0.1.0
