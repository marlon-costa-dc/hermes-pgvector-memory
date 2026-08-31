"""Integration tests against a real PostgreSQL + pgvectorscale + Ollama.

Skipped automatically when the stack is absent, so the suite stays green on a
laptop or in CI without a database. Nothing here is mocked: if these pass, the
plugin genuinely stores and retrieves vectors.

Run with the stack up:
    PGVECTOR_MEMORY_DSN="postgresql:///hermes_memory?host=/run/postgresql" \
        pytest tests/test_integration.py -v
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pgvector_memory.config import DEFAULT_DSN  # noqa: E402
from pgvector_memory.embeddings import OllamaEmbedder  # noqa: E402
from pgvector_memory.store import MemoryStore  # noqa: E402

DSN = os.environ.get("PGVECTOR_MEMORY_DSN", DEFAULT_DSN)
OLLAMA_HOST = os.environ.get("PGVECTOR_MEMORY_OLLAMA_HOST", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("PGVECTOR_MEMORY_EMBED_MODEL", "nomic-embed-text")


def _postgres_ready() -> tuple[bool, str]:
    try:
        import psycopg
    except ImportError:
        return False, "psycopg not installed"
    try:
        with psycopg.connect(DSN, connect_timeout=5) as conn, conn.cursor() as cur:
            cur.execute("SELECT extname FROM pg_extension")
            exts = {r[0] for r in cur.fetchall()}
            cur.execute("SELECT amname FROM pg_am WHERE amname = 'diskann'")
            has_diskann = cur.fetchone() is not None
    except Exception as exc:
        return False, f"cannot connect: {exc}"
    if "vector" not in exts:
        return False, "pgvector extension not installed in this database"
    if not has_diskann:
        return False, "pgvectorscale (diskann access method) not available"
    return True, ""


def _ollama_ready() -> tuple[bool, str]:
    if OllamaEmbedder(OLLAMA_HOST, EMBED_MODEL).is_available():
        return True, ""
    return False, f"Ollama at {OLLAMA_HOST} is not serving {EMBED_MODEL}"


PG_OK, PG_WHY = _postgres_ready()
OLLAMA_OK, OLLAMA_WHY = _ollama_ready()

requires_stack = pytest.mark.skipif(
    not (PG_OK and OLLAMA_OK),
    reason=f"stack unavailable — postgres: {PG_WHY or 'ok'}; ollama: {OLLAMA_WHY or 'ok'}",
)


@pytest.fixture(scope="module")
def embedder():
    return OllamaEmbedder(OLLAMA_HOST, EMBED_MODEL)


@pytest.fixture
def store(embedder):
    """A throwaway table per test, dropped afterwards."""
    table = f"test_mem_{uuid.uuid4().hex[:12]}"
    s = MemoryStore(DSN, embedder.dims or 768, table)
    s.connect()
    s.ensure_schema()
    yield s
    try:
        conn = s._require_conn()
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table}")
    finally:
        s.close()


@requires_stack
class TestSchema:
    def test_creates_table_and_diskann_index(self, store):
        conn = store._require_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT indexdef FROM pg_indexes WHERE tablename = %s", (store.table,))
            defs = " ".join(r[0] for r in cur.fetchall())
        assert "diskann" in defs, "DiskANN index missing — pgvectorscale unused"
        assert "gin" in defs.lower(), "FTS index missing — hybrid search degraded"

    def test_ensure_schema_is_idempotent(self, store):
        store.ensure_schema()
        store.ensure_schema()  # must not raise

    def test_dimension_mismatch_is_refused(self, store, embedder):
        from pgvector_memory.store import StoreError

        wrong = MemoryStore(DSN, (embedder.dims or 768) + 1, store.table)
        wrong.connect()
        try:
            with pytest.raises(StoreError, match="re-embedding"):
                wrong.ensure_schema()
        finally:
            wrong.close()


@requires_stack
class TestWriteAndRead:
    def test_add_returns_id_and_row_is_readable(self, store, embedder):
        text = "O operador usa CachyOS com systemd-boot"
        memory_id = store.add(text, embedder.embed_one(text), kind="fact")
        assert isinstance(memory_id, int)
        assert any(r["content"] == text for r in store.recent(limit=5))

    def test_identical_content_is_deduplicated(self, store, embedder):
        text = "PostgreSQL roda na porta 5432"
        vec = embedder.embed_one(text)
        first = store.add(text, vec)
        second = store.add(text, vec)
        assert isinstance(first, int)
        assert second is None, "duplicate should be suppressed, not stored twice"

    def test_same_content_under_different_identity_is_kept(self, store, embedder):
        text = "prefere respostas curtas"
        vec = embedder.embed_one(text)
        assert store.add(text, vec, agent_identity="default") is not None
        assert store.add(text, vec, agent_identity="coder") is not None

    def test_delete_removes_the_row(self, store, embedder):
        memory_id = store.add("efemero", embedder.embed_one("efemero"))
        assert store.delete(memory_id) is True
        assert store.delete(memory_id) is False


@requires_stack
class TestSemanticSearch:
    """The claim that matters: retrieval by MEANING, not by shared words."""

    CORPUS = [
        "O gato dorme no sofa da sala",
        "A gata tira uma soneca no sofa",
        "PostgreSQL e um banco de dados relacional",
        "O indice DiskANN acelera busca vetorial no Postgres",
        "A bicicleta quebrou no meio da estrada de terra",
    ]

    def _seed(self, store, embedder):
        for text, vec in zip(self.CORPUS, embedder.embed(self.CORPUS), strict=True):
            store.add(text, vec)

    def test_paraphrase_beats_word_overlap(self, store, embedder):
        self._seed(store, embedder)
        hits = store.search(embedder.embed_one("gato dormindo"), limit=3)
        top = hits[0]["content"]
        assert "gat" in top.lower(), f"expected a cat sentence first, got {top!r}"
        # The two cat sentences share almost no vocabulary ("sofa" aside);
        # both surfacing above the bicycle proves the ranking is semantic.
        cats = [h for h in hits if "gat" in h["content"].lower()]
        assert len(cats) == 2, f"both cat sentences should rank high: {hits}"

    def test_relevant_outranks_everything_irrelevant(self, store, embedder):
        """The invariant that matters: relevant content beats irrelevant content.

        Note what is NOT asserted: which irrelevant sentence lands last.
        Measured against nomic-embed-text, that order is not stable —
        for "gato dormindo" the bicycle scores 0.4207 and PostgreSQL 0.3956,
        but for "gato" the bicycle scores 0.4687 and PostgreSQL 0.4034.
        The embedder groups by linguistic register (everyday Portuguese prose
        vs technical jargon), not by human-perceived topical distance, so the
        bicycle sentence drifts. Asserting a total ordering would be encoding
        a coincidence; the group separation is the real contract.
        """
        self._seed(store, embedder)
        hits = store.search(embedder.embed_one("gato dormindo"), limit=5)
        assert len(hits) == len(self.CORPUS)

        cats = [h["similarity"] for h in hits if "gat" in h["content"].lower()]
        others = [h["similarity"] for h in hits if "gat" not in h["content"].lower()]
        assert len(cats) == 2 and len(others) == 3
        assert min(cats) > max(others), f"cat sentences {cats} must all outrank non-cat {others}"

    def test_similarity_is_a_sane_cosine(self, store, embedder):
        self._seed(store, embedder)
        hits = store.search(embedder.embed_one(self.CORPUS[0]), limit=5)
        assert 0.99 <= hits[0]["similarity"] <= 1.0001, "self-match should be ~1.0"
        assert all(-1.0 <= h["similarity"] <= 1.0001 for h in hits)

    def test_hybrid_finds_exact_token_a_vector_would_blur(self, store, embedder):
        # Embedders flatten rare literals like identifiers and error codes;
        # the lexical half of the fusion is what makes these findable.
        text = "O erro E0609 vem do bindgen com libclang 22"
        store.add(text, embedder.embed_one(text))
        self._seed(store, embedder)
        hits = store.search(embedder.embed_one("E0609"), "E0609", limit=5)
        assert any("E0609" in h["content"] for h in hits)

    def test_lexical_half_actually_contributes(self, store, embedder):
        """The lexical branch must change the outcome, not merely exist.

        This is the discriminating test: it compares the SAME vector query
        with and without query_text. If the lexical half is dead -- as it was
        when plainto_tsquery AND-ed every stopword and matched nothing -- both
        calls return the identical ranking and this fails.

        Written after three earlier regression tests turned out to pass
        against the broken code: on a small corpus vector search alone put the
        right document in the top 3, so they proved nothing.
        """
        # Deliberately unlike the corpus semantically, but sharing rare words
        # with the query below.
        text = "gc ROTEIA; o REPOSITORIO e dono de branch/worktree/gates/PR/merge"
        store.add(text, embedder.embed_one(text))
        # Pad so a lone vector ranking cannot trivially surface it.
        for i in range(12):
            filler = f"nota de preenchimento numero {i} sobre assuntos diversos"
            store.add(filler, embedder.embed_one(filler))
        self._seed(store, embedder)

        question = "quem e o dono da branch e do merge?"
        qvec = embedder.embed_one(question)

        vector_only = [h["id"] for h in store.search(qvec, "", limit=5)]
        hybrid = [h["id"] for h in store.search(qvec, question, limit=5)]

        assert hybrid != vector_only, (
            "hybrid ranking is identical to vector-only: the lexical branch "
            f"contributed nothing. vector={vector_only} hybrid={hybrid}"
        )
        target = next(h for h in store.recent(limit=50) if "ROTEIA" in h["content"])
        assert target["id"] in hybrid, (
            f"lexically matching memory absent from hybrid results: {hybrid}"
        )

    def test_slash_separated_terms_are_searchable(self, store, embedder):
        """Postgres reads "a/b/c" as ONE `file` token; "worktree" must hit it.

        Asserted through the lexical branch alone (an unrelated vector) so a
        passing result cannot come from semantic similarity.
        """
        text = "o REPOSITORIO e dono de branch/worktree/gates/PR/merge"
        store.add(text, embedder.embed_one(text))
        self._seed(store, embedder)
        # Vector points at cats; only the lexical half can find "worktree".
        unrelated = embedder.embed_one("gato dormindo no sofa")
        hits = store.search(unrelated, "worktree", limit=3)
        assert any("worktree" in h["content"] for h in hits), (
            f"slash-separated token unreachable lexically: {hits}"
        )

    def test_dotted_identifier_in_query_matches_document(self, store, embedder):
        """The query must be normalised exactly like the document.

        Regression: the indexed side split "CLAUDE.md" into 'claude'+'md'
        while the query kept 'claude.md' whole, so the lexeme matched zero
        rows. Uses an unrelated vector so only the lexical half can succeed.
        """
        text = "AGENTS.md/CLAUDE.md protegidos: prompt expirado = BLOQUEIO"
        store.add(text, embedder.embed_one(text))
        self._seed(store, embedder)
        unrelated = embedder.embed_one("gato dormindo no sofa")
        hits = store.search(unrelated, "CLAUDE.md", limit=3)
        assert any("CLAUDE.md" in h["content"] for h in hits), (
            f"dotted identifier unreachable lexically: {hits}"
        )

    def test_min_similarity_filters_weak_matches(self, store, embedder):
        self._seed(store, embedder)
        strict = store.search(
            embedder.embed_one("assunto totalmente sem relacao: culinaria tailandesa"),
            limit=10,
            min_similarity=0.9,
        )
        assert strict == [], "0.9 similarity floor must exclude unrelated content"

    def test_kind_filter_restricts_results(self, store, embedder):
        store.add("um fato", embedder.embed_one("um fato"), kind="fact")
        store.add("uma pref", embedder.embed_one("uma pref"), kind="preference")
        hits = store.search(embedder.embed_one("fato"), limit=10, kind="fact")
        assert all(h["kind"] == "fact" for h in hits)


@requires_stack
class TestStats:
    def test_counts_by_kind(self, store, embedder):
        store.add("f", embedder.embed_one("f"), kind="fact")
        store.add("p", embedder.embed_one("p"), kind="preference")
        stats = store.stats()
        assert stats["total"] == 2
        assert stats["facts"] == 1 and stats["preferences"] == 1
        assert stats["size"]  # pg_size_pretty string
