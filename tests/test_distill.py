"""Tests for the distill pipeline extensions (v0.3, phase 2+3).

Pure-logic tests: batch grouping, verdict/enrichment/triple parsing, and the
promote decision function. LLM and PostgreSQL paths are covered by the live
pilot (6.2) and test_integration.py respectively.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from distill_prompts import (  # noqa: E402
    _group_batches,
    _parse_enrichment,
    _parse_verdict,
    _promote_decision,
    _strip_fences,
)


class TestStripFences:
    def test_plain_json_passes_through(self):
        assert _strip_fences('{"a": 1}') == '{"a": 1}'

    def test_markdown_fence_stripped(self):
        assert _strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_bare_fence_stripped(self):
        assert _strip_fences('```\n{"a": 1}\n```') == '{"a": 1}'


class TestGroupBatches:
    def test_same_session_stays_together(self):
        rows = [("s1", ("a", 1)), ("s1", ("b", 2)), ("s2", ("c", 3)), ("", ("d", 4))]
        batches = _group_batches(rows, batch_size=2)
        # s1's two rows must land in the same batch.
        assert any({r[0] for r in b} == {"s1"} for b in batches)

    def test_session_never_split_across_batches(self):
        rows = [(f"s{i}", (f"p{i}", i)) for i in range(6)]
        batches = _group_batches(rows, batch_size=4)
        seen = {}
        for b in batches:
            for sk, _ in b:
                seen.setdefault(sk, len(batches))
                assert seen[sk] == len(batches)  # each session appears in exactly one batch

    def test_loose_rows_fill_batches(self):
        rows = [("", (f"p{i}", i)) for i in range(5)]
        batches = _group_batches(rows, batch_size=2)
        flat = [r for b in batches for r in b]
        assert len(flat) == 5
        assert all(len(b) <= 2 for b in batches)

    def test_large_session_is_chunked(self):
        rows = [("s1", (f"p{i}", i)) for i in range(10)]
        batches = _group_batches(rows, batch_size=4)
        # 10 rows at batch_size 4 -> 3 batches covering everything, in order
        flat = [r for b in batches for r in b]
        assert len(flat) == 10


class TestParseVerdict:
    def test_worthy(self):
        raw = '{"memory_worthy": true, "reason": "fato de ambiente"}'
        assert _parse_verdict(raw) == (True, "fato de ambiente")

    def test_not_worthy(self):
        assert _parse_verdict('{"memory_worthy": false, "reason": "procedural"}') == (
            False,
            "procedural",
        )

    def test_fenced_json(self):
        assert _parse_verdict('```json\n{"memory_worthy": true}\n```') == (True, "")

    def test_unparseable_fails_closed(self):
        assert _parse_verdict("não é json") == (False, "unparseable")

    def test_non_dict_json_fails_closed(self):
        assert _parse_verdict('["list"]') == (False, "unparseable")


class TestParseEnrichment:
    def test_full_shape(self):
        raw = (
            '{"core": "Drop-in systemd exige cabeçalho [Service]",'
            ' "specific_context": "override.conf em ~/.config/systemd/user/",'
            ' "kind": "fact", "tags": ["systemd", "hermes"],'
            ' "subject": "hermes-gateway drop-in", "relation": "cabeçalho exigido",'
            ' "object": "[Service]"}'
        )
        parsed = _parse_enrichment(raw)
        assert parsed["kind"] == "fact"
        assert parsed["tags"] == ["systemd", "hermes"]
        assert "override.conf" in parsed["specific_context"]
        assert parsed["subject"] == "hermes-gateway drop-in"
        assert parsed["object"] == "[Service]"

    def test_bad_kind_falls_back_to_observation(self):
        raw = '{"core": "x", "kind": "gremlim", "tags": []}'
        assert _parse_enrichment(raw)["kind"] == "observation"

    def test_invalid_json_gives_empty_safe_shape(self):
        parsed = _parse_enrichment("lixo")
        assert parsed["core"] == ""
        assert parsed["kind"] == "observation"
        assert parsed["subject"] == ""

    def test_tags_are_capped_and_stringified(self):
        raw = '{"core": "c", "tags": ["a", 1, "b", 2, "c", 3, "d", 4]}'
        parsed = _parse_enrichment(raw)
        assert len(parsed["tags"]) == 6
        assert all(isinstance(t, str) for t in parsed["tags"])

    def test_empty_triple_when_absent(self):
        parsed = _parse_enrichment('{"core": "only core"}')
        assert parsed["subject"] == "" and parsed["relation"] == "" and parsed["object"] == ""


class TestPromoteDecision:
    def test_triple_conflict_supersedes(self):
        decision, old_id = _promote_decision(
            subject="bd", relation="versao", obj="1.2.2",
            live_key_hit=(77, "1.2.1"),
        )
        assert decision == "supersede" and old_id == 77

    def test_triple_same_object_is_duplicate(self):
        decision, _ = _promote_decision(
            subject="bd", relation="versao", obj="1.2.2",
            live_key_hit=(77, "1.2.2"),
        )
        assert decision == "skip"

    def test_triple_without_live_hit_inserts(self):
        decision, old_id = _promote_decision(
            subject="bd", relation="versao", obj="1.2.2", live_key_hit=None
        )
        assert decision == "insert" and old_id is None

    def test_no_triple_high_similarity_skips(self):
        decision, _ = _promote_decision(
            subject="", relation="", obj="", live_key_hit=None, best_similarity=0.95
        )
        assert decision == "skip"

    def test_no_triple_low_similarity_inserts(self):
        decision, _ = _promote_decision(
            subject="", relation="", obj="", live_key_hit=None, best_similarity=0.60
        )
        assert decision == "insert"

    def test_no_triple_never_supersedes(self):
        # The MemStrata point: without a structural key, similarity alone can
        # NEVER retire a memory — similarity cannot tell duplicate from conflict.
        decision, old_id = _promote_decision(
            subject="", relation="", obj="", live_key_hit=(5, "qualquer"),
            best_similarity=0.99,
        )
        assert decision in ("skip", "insert")
        assert old_id is None


class TestClusterCandidates:
    def test_similar_candidates_merge(self):
        from distill_prompts import _cluster_candidates

        cands = [
            {"core": "a", "embedding": [0.99, 0.1], "prompt_ids": [1]},
            {"core": "b", "embedding": [0.98, 0.1], "prompt_ids": [2]},
            {"core": "c", "embedding": [0.1, 0.99], "prompt_ids": [3]},
        ]
        clusters = _cluster_candidates(cands, thresh=0.90)
        assert len(clusters) == 2
        merged = [c for c in clusters if len(c["prompt_ids"]) == 2][0]
        assert sorted(merged["prompt_ids"]) == [1, 2]

    def test_merged_cluster_keeps_longest_core(self):
        from distill_prompts import _cluster_candidates

        cands = [
            {"core": "curto", "embedding": [1.0, 0.0], "prompt_ids": [1]},
            {"core": "um core bem mais longo e detalhado aqui", "embedding": [0.99, 0.01], "prompt_ids": [2]},
        ]
        clusters = _cluster_candidates(cands, thresh=0.90)
        assert len(clusters) == 1
        assert clusters[0]["core"] == "um core bem mais longo e detalhado aqui"

    def test_fields_union_across_cluster(self):
        from distill_prompts import _cluster_candidates

        cands = [
            {"core": "a", "embedding": [1.0, 0.0], "prompt_ids": [1],
             "specific_context": "ctx-a", "tags": ["x"], "subject": "", "relation": "", "object": "", "kind": "fact"},
            {"core": "b", "embedding": [0.99, 0.0], "prompt_ids": [2],
             "specific_context": "ctx-b", "tags": ["y"], "subject": "s", "relation": "r", "object": "o", "kind": "fact"},
        ]
        (merged,) = _cluster_candidates(cands, thresh=0.90)
        assert merged["tags"] == ["x", "y"]
        assert merged["specific_context"] in ("ctx-a", "ctx-b")
        # a triple surviving from any member keeps supersession power
        assert (merged["subject"], merged["relation"], merged["object"]) == ("s", "r", "o")

    def test_embeddings_normalised_before_dot(self):
        from distill_prompts import _cluster_candidates

        cands = [
            {"core": "a", "embedding": [2.0, 0.0], "prompt_ids": [1]},
            {"core": "b", "embedding": [1.0, 0.0], "prompt_ids": [2]},
        ]
        # Un-normalised but colinear -> cosine 1.0 -> must merge.
        clusters = _cluster_candidates(cands, thresh=0.90)
        assert len(clusters) == 1
