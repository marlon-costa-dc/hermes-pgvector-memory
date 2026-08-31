"""Unit tests that need no PostgreSQL and no Ollama.

Everything here is pure logic: config precedence, vector rendering, hashing,
SQL shape. Tests that require the real stack live in test_integration.py and
skip themselves when it is absent.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pgvector_memory.config import Config, load_config  # noqa: E402
from pgvector_memory.embeddings import KNOWN_DIMS, OllamaEmbedder, to_pgvector  # noqa: E402
from pgvector_memory.store import MemoryStore, StoreError, _sha256  # noqa: E402


class TestToPgvector:
    def test_renders_bracketed_csv(self):
        assert to_pgvector([1.0, 2.0, 3.0]) == "[1.000000,2.000000,3.000000]"

    def test_negative_and_small_values_keep_sign_and_scale(self):
        out = to_pgvector([-0.5, 0.000001, -0.0000001])
        assert out.startswith("[-0.500000,")
        # 1e-7 rounds to zero at 6 decimals; that is intentional (pgvector
        # stores float4 anyway) but must not become "-0.000000e-07" garbage.
        assert "e" not in out

    def test_empty_vector(self):
        assert to_pgvector([]) == "[]"


class TestConfig:
    def test_defaults_are_conservative(self):
        cfg = Config()
        # Turn capture defaults OFF: it is lossy and grows the table fast.
        assert cfg.auto_capture_turns is False
        assert cfg.auto_recall is True
        assert cfg.mirror_memory_tool is True

    def test_env_overrides_config_yaml(self, monkeypatch):
        monkeypatch.setenv("PGVECTOR_MEMORY_DSN", "postgresql://from-env/db")
        cfg = load_config(cfg_get=lambda key, default=None: "postgresql://from-yaml/db")
        assert cfg.dsn == "postgresql://from-env/db"

    def test_config_yaml_used_when_env_absent(self, monkeypatch):
        monkeypatch.delenv("PGVECTOR_MEMORY_DSN", raising=False)

        def fake_cfg(key, default=None):
            return "postgresql://from-yaml/db" if key.endswith(".dsn") else default

        assert load_config(cfg_get=fake_cfg).dsn == "postgresql://from-yaml/db"

    def test_falls_back_to_defaults_without_host_config(self, monkeypatch):
        for var in list(os.environ):
            if var.startswith("PGVECTOR_MEMORY_"):
                monkeypatch.delenv(var, raising=False)
        cfg = load_config(cfg_get=None)
        assert cfg.dsn.endswith("host=/run/postgresql")
        assert cfg.embed_model == "nomic-embed-text"

    def test_broken_cfg_get_does_not_crash_startup(self, monkeypatch):
        monkeypatch.delenv("PGVECTOR_MEMORY_DSN", raising=False)

        def exploding(key, default=None):
            raise RuntimeError("config backend down")

        # A memory plugin must not take the whole agent down over config.
        assert load_config(cfg_get=exploding).dsn == Config().dsn

    def test_cfg_get_returning_none_falls_back_to_default(self, monkeypatch):
        """Regression: Hermes' cfg_get returns None for an unset key.

        Measured, not assumed — cfg_get('missing.key', 'default') yields None,
        the default is NOT applied by the host. Trusting it produced
        ollama_host=None and an AttributeError on .rstrip() at agent startup.
        """
        for var in list(os.environ):
            if var.startswith("PGVECTOR_MEMORY_"):
                monkeypatch.delenv(var, raising=False)

        cfg = load_config(cfg_get=lambda key, default=None: None)
        assert cfg.ollama_host == "http://127.0.0.1:11434"
        assert cfg.dsn == Config().dsn
        assert cfg.embed_model == "nomic-embed-text"
        assert cfg.recall_limit == 5
        assert cfg.auto_recall is True

    def test_blank_config_value_is_treated_as_absent(self, monkeypatch):
        monkeypatch.delenv("PGVECTOR_MEMORY_DSN", raising=False)
        # A blank DSN is never a deliberate choice; falling back beats
        # failing to connect to "".
        cfg = load_config(cfg_get=lambda key, default=None: "   ")
        assert cfg.dsn == Config().dsn


class TestStoreValidation:
    def test_rejects_table_name_carrying_sql(self):
        # Table is an identifier, so it cannot be a bound parameter; the
        # guard is the only thing standing between config and injection.
        with pytest.raises(StoreError):
            MemoryStore("postgresql:///x", 768, table="users; DROP TABLE x")

    def test_rejects_quoted_identifier(self):
        with pytest.raises(StoreError):
            MemoryStore("postgresql:///x", 768, table='a" OR "1"="1')

    def test_accepts_plain_identifier(self):
        assert MemoryStore("postgresql:///x", 768, table="my_memories").table == "my_memories"

    def test_operations_before_connect_fail_loudly(self):
        store = MemoryStore("postgresql:///x", 768)
        with pytest.raises(StoreError, match="not connected"):
            store.recent()


class TestHashing:
    def test_same_content_same_hash(self):
        assert _sha256("hello") == _sha256("hello")

    def test_different_content_different_hash(self):
        assert _sha256("hello") != _sha256("hello ")

    def test_unicode_is_stable(self):
        assert len(_sha256("memória vetorial 🇧🇷")) == 32


class TestEmbedderContract:
    def test_known_dims_match_documented_models(self):
        assert KNOWN_DIMS["nomic-embed-text"] == 768
        assert KNOWN_DIMS["all-minilm"] == 384

    def test_unknown_model_has_no_preset_dims(self):
        assert OllamaEmbedder(model="some-future-model").dims is None

    def test_known_model_preloads_dims(self):
        assert OllamaEmbedder(model="nomic-embed-text").dims == 768

    def test_host_trailing_slash_normalised(self):
        assert OllamaEmbedder(host="http://localhost:11434/").host == "http://localhost:11434"

    def test_embed_empty_list_short_circuits(self):
        # Must not perform a request for an empty batch.
        assert OllamaEmbedder().embed([]) == []

    def test_none_host_and_model_fall_back_to_defaults(self):
        # Regression: config that resolved to None reached the constructor and
        # crashed on None.rstrip() during agent startup.
        embedder = OllamaEmbedder(host=None, model=None)  # type: ignore[arg-type]
        assert embedder.host == "http://127.0.0.1:11434"
        assert embedder.model == "nomic-embed-text"
        assert embedder.dims == 768


class TestStagingEnqueue:
    """capture_mode='staging': turns go to distill_prompts, never live."""

    class _SentinelStore:
        """Type-compatible stand-in: capture only checks initialization."""

    def _provider(self, tmp_path, **cfg_overrides):
        from pgvector_memory import PgVectorMemoryProvider
        from pgvector_memory.config import Config

        cfg = Config(**cfg_overrides)
        provider = PgVectorMemoryProvider(cfg)
        # Type-true uninitialized instance: capture gates on _store not None.
        provider._store = MemoryStore.__new__(MemoryStore)
        return provider

    def test_sync_turn_enqueues_to_staging(self, monkeypatch, tmp_path):
        provider = self._provider(
            tmp_path, auto_capture_turns=True, capture_mode="staging", dsn="postgresql:///x"
        )
        recorded = []
        monkeypatch.setattr(
            provider, "_enqueue_staging",
            lambda prompt, *, origin="hermes", session_id="": recorded.append((prompt, origin, session_id)),
        )
        provider.sync_turn(
            "Como configuro a porta do gateway?",
            "A porta e 8372, configurada no city.toml.",
            session_id="s1",
        )
        assert len(recorded) == 1
        prompt, origin, sid = recorded[0]
        assert origin == "hermes"
        assert sid == "s1"
        assert "8372" in prompt and "gateway" in prompt

    def test_sync_turn_trivial_user_text_is_not_enqueued(self, monkeypatch, tmp_path):
        provider = self._provider(
            tmp_path, auto_capture_turns=True, capture_mode="staging", dsn="postgresql:///x"
        )
        recorded = []
        monkeypatch.setattr(
            provider, "_enqueue_staging",
            lambda prompt, *, origin="hermes", session_id="": recorded.append(prompt),
        )
        provider.sync_turn("ok continue", "Right, moving on to the next step.", session_id="s1")
        assert recorded == []

    def test_sync_turn_live_mode_keeps_v02_behaviour(self, monkeypatch, tmp_path):
        provider = self._provider(
            tmp_path, auto_capture_turns=True, capture_mode="live", dsn="postgresql:///x"
        )
        enqueued, stored = [], []
        monkeypatch.setattr(
            provider, "_enqueue_staging",
            lambda prompt, *, origin="hermes", session_id="": enqueued.append(prompt),
        )
        monkeypatch.setattr(provider, "_store_safe", lambda content, **kw: stored.append(content))
        provider.sync_turn(
            "Configure o systemd unit do hermes-gateway agora",
            "Feito, o unit esta ativo e habilitado no boot.",
            session_id="s1",
        )
        # live mode writes from a daemon thread; give it a moment.
        for _ in range(50):
            if stored:
                break
            time.sleep(0.02)
        assert enqueued == []
        assert len(stored) == 1

    def test_capture_off_enqueues_nothing(self, monkeypatch, tmp_path):
        provider = self._provider(
            tmp_path, auto_capture_turns=True, capture_mode="off", dsn="postgresql:///x"
        )
        enqueued, stored = [], []
        monkeypatch.setattr(
            provider, "_enqueue_staging",
            lambda prompt, *, origin="hermes", session_id="": enqueued.append(prompt),
        )
        monkeypatch.setattr(provider, "_store_safe", lambda content, **kw: stored.append(content))
        provider.sync_turn("Long enough user message about infra.", "Long enough assistant reply.", session_id="s")
        assert enqueued == [] and stored == []

    def test_on_pre_compress_enqueues_transcript(self, monkeypatch, tmp_path):
        provider = self._provider(tmp_path, dsn="postgresql:///x")
        recorded = []
        monkeypatch.setattr(
            provider, "_enqueue_staging",
            lambda prompt, *, origin="hermes", session_id="": recorded.append((prompt, origin)),
        )
        msgs = [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "qual a porta do api server?"},
            {"role": "assistant", "content": "8642, autenticada."},
            {"role": "tool", "content": "tool output"},
        ]
        provider.on_pre_compress(msgs)
        assert any("8642" in p for p, _ in recorded)
        assert all(o == "hermes" for _, o in recorded)

    def test_on_pre_compress_never_raises(self, monkeypatch, tmp_path):
        provider = self._provider(tmp_path, dsn="postgresql:///x")
        def exploding(*a, **kw):
            raise RuntimeError("pg down")
        monkeypatch.setattr(provider, "_enqueue_staging", exploding)
        # Best-effort contract: compression must proceed regardless.
        assert provider.on_pre_compress([{"role": "user", "content": "x"}]) == ""

    def test_short_transcript_not_enqueued(self, monkeypatch, tmp_path):
        provider = self._provider(tmp_path, dsn="postgresql:///x", min_turn_chars=500)
        recorded = []
        monkeypatch.setattr(
            provider, "_enqueue_staging",
            lambda prompt, *, origin="hermes", session_id="": recorded.append(prompt),
        )
        provider.on_pre_compress([{"role": "user", "content": "tiny"}])
        assert recorded == []


class TestNoiseFilter:
    def test_trivial_affirmations_filtered(self):
        from pgvector_memory import _is_noisy_user_text

        assert _is_noisy_user_text("ok")
        assert _is_noisy_user_text("Ok, continue.")
        assert _is_noisy_user_text("continue")
        assert _is_noisy_user_text("?")
        assert _is_noisy_user_text("valeu")

    def test_substantive_text_passes(self):
        from pgvector_memory import _is_noisy_user_text

        assert not _is_noisy_user_text("Como configuro o gateway?")
        assert not _is_noisy_user_text("roda os testes de integracao")
        assert not _is_noisy_user_text("why did the deploy fail?")
