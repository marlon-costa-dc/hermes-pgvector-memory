"""Contract tests: the provider must satisfy the host's real ABC.

These are the tests that catch a Hermes-side signature change. They skip when
only the stub ABC is available (see conftest), because passing against a stub
we wrote ourselves would prove nothing.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import HERMES_AVAILABLE  # noqa: E402

from pgvector_memory import PgVectorMemoryProvider, register  # noqa: E402

requires_hermes = pytest.mark.skipif(
    not HERMES_AVAILABLE,
    reason="real Hermes not installed; set HERMES_AGENT_PATH to run contract tests",
)


class TestProviderContract:
    def test_instantiable(self):
        """An ABC with an unimplemented abstractmethod raises on construction."""
        assert PgVectorMemoryProvider() is not None

    def test_name_is_stable_identifier(self):
        # The name is the value of memory.provider in config.yaml; changing it
        # silently orphans every existing installation's configuration.
        assert PgVectorMemoryProvider().name == "pgvector-memory"

    def test_declares_three_tools(self):
        names = {s["name"] for s in PgVectorMemoryProvider().get_tool_schemas()}
        assert names == {"pgvector_remember", "pgvector_recall", "pgvector_forget"}

    def test_tool_schemas_are_valid_openai_functions(self):
        for schema in PgVectorMemoryProvider().get_tool_schemas():
            assert set(schema) >= {"name", "description", "parameters"}
            params = schema["parameters"]
            assert params["type"] == "object"
            assert isinstance(params["properties"], dict)
            for required in params.get("required", []):
                assert required in params["properties"], (
                    f"{schema['name']}: required field {required!r} is not declared"
                )

    def test_config_schema_fields_are_well_formed(self):
        for field in PgVectorMemoryProvider().get_config_schema():
            assert "key" in field and "description" in field
            if field.get("type") in ("integer", "number"):
                if "minimum" in field and "maximum" in field:
                    assert field["minimum"] <= field["maximum"]
                if "default" in field and "minimum" in field:
                    assert field["default"] >= field["minimum"]

    def test_register_hands_provider_to_host(self):
        captured = []

        class Ctx:
            def register_memory_provider(self, provider):
                captured.append(provider)

        register(Ctx())
        assert len(captured) == 1
        assert isinstance(captured[0], PgVectorMemoryProvider)


@requires_hermes
class TestAgainstRealAbc:
    def test_is_a_real_memory_provider(self):
        from agent.memory_provider import MemoryProvider

        assert isinstance(PgVectorMemoryProvider(), MemoryProvider)

    def test_no_abstract_method_left_unimplemented(self):
        from agent.memory_provider import MemoryProvider

        missing = {
            name
            for name in getattr(MemoryProvider, "__abstractmethods__", set())
            if not hasattr(PgVectorMemoryProvider, name)
        }
        assert not missing, f"unimplemented abstract methods: {missing}"

    @pytest.mark.parametrize(
        "method",
        [
            "initialize",
            "prefetch",
            "queue_prefetch",
            "sync_turn",
            "handle_tool_call",
            "on_memory_write",
            "save_config",
            "recall_status",
            "system_prompt_block",
            "unavailable_reason",
        ],
    )
    def test_override_signature_matches_host(self, method):
        """A host signature change must fail here, not at runtime in the agent."""
        from agent.memory_provider import MemoryProvider

        base = getattr(MemoryProvider, method, None)
        if base is None:
            pytest.skip(f"host ABC has no {method}")
        ours = getattr(PgVectorMemoryProvider, method)

        base_params = inspect.signature(base).parameters
        our_params = inspect.signature(ours).parameters

        # Accepting **kwargs makes us forward-compatible with new host args.
        if any(p.kind is p.VAR_KEYWORD for p in our_params.values()):
            positional = [
                n
                for n, p in base_params.items()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            for name in positional:
                assert name in our_params, (
                    f"{method}: host passes {name!r} positionally; we do not accept it"
                )
        else:
            assert list(our_params) == list(base_params), (
                f"{method}: signature drifted from host ABC"
            )

    def test_recall_status_returns_host_dataclass(self):
        from agent.memory_provider import RecallStatus

        provider = PgVectorMemoryProvider()
        assert provider.recall_status() is None  # nothing recalled yet

        provider._last_recall_count = 3
        status = provider.recall_status()
        assert isinstance(status, RecallStatus)
        assert status.count == 3
        # Must reset, or a stale count would be reported on the next turn.
        assert provider.recall_status() is None
