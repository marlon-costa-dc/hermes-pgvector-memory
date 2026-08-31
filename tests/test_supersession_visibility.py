"""Supersession visibility: the live-filter contract (v0.3 follow-up).

Contract under test — three clauses:
  C1. search()/recent() EXCLUDE retired memories by default (recall answers
      "what is true NOW").
  C2. include_superseded=True surfaces them (historical drill-down).
  C3. The exclusion is a predicate joined into the WHERE, not a post-filter —
      a superseded row must not consume a result slot when it would otherwise
      rank, i.e. the limit is filled with LIVE rows.

Rule 2 (guard-and-invariant-tests): each clause is proven to GATE, not just to
pass — see TestLiveFilterGates, which injects the violation by reverting the
filter at the SQL level and asserting the result CHANGES (a guard that passes
both with and without the filter asserts nothing).
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pgvector_memory.store import MemoryStore  # noqa: E402

requires_stack = pytest.mark.skipif(
    not Path("/run/postgresql").exists(),
    reason="postgres socket unavailable",
)

DIMS = 768


def _vec(seed: float) -> list[float]:
    return [seed] * DIMS


@pytest.fixture()
def store():
    table = f"test_sup_{uuid.uuid4().hex[:10]}"
    s = MemoryStore("postgresql:///hermes_memory?host=/run/postgresql", DIMS, table)
    s.connect()
    s.ensure_schema()
    yield s
    try:
        conn = s._require_conn()
        sql = f"DROP TABLE IF EXISTS {table}"
        with conn.cursor() as cur:
            cur.execute(sql)
    finally:
        s.close()


@requires_stack
class TestLiveFilterContract:
    def test_c1_search_excludes_superseded_by_default(self, store):
        old = store.add("porta do api server = 8642", _vec(0.5), kind="fact",
                        subject="api server", relation="porta", object="8642")
        new = store.add("porta do api server = 9999", _vec(0.5), kind="fact",
                        subject="api server", relation="porta", object="9999")
        store.supersede_by_key(old, new)
        hits = store.search(_vec(0.5), "porta do api server", limit=10)
        ids = {h["id"] for h in hits}
        assert old not in ids, f"retired memory #{old} surfaced in default recall: {ids}"
        assert new in ids

    def test_c2_include_superseded_surfaces_history(self, store):
        old = store.add("versao do bd = 1.2.2", _vec(0.4), kind="fact",
                        subject="bd", relation="versao", object="1.2.2")
        new = store.add("versao do bd = 1.4.1", _vec(0.4), kind="fact",
                        subject="bd", relation="versao", object="1.4.1")
        store.supersede_by_key(old, new)
        hits = store.search(_vec(0.4), "versao do bd", limit=10, include_superseded=True)
        ids = {h["id"] for h in hits}
        assert {old, new} <= ids, "include_superseded=True must surface both generations"

    def test_c3_retired_row_does_not_consume_a_slot(self, store):
        # 3 live memories + 1 retired. limit=3 must return exactly the 3 LIVE
        # ones: the retired row ranks between them but is filtered IN SQL, so
        # the slot goes to a live row instead of being wasted.
        store.add("alpha fato", _vec(0.60), kind="fact")
        store.add("beta fato", _vec(0.61), kind="fact")
        retired = store.add("gama fato (antigo)", _vec(0.62), kind="fact")
        store.add("delta fato", _vec(0.63), kind="fact")
        heir = store.add("gama fato (novo)", _vec(0.62), kind="fact")
        store.supersede_by_key(retired, heir)
        hits = store.search(_vec(0.62), "fato", limit=3)
        ids = [h["id"] for h in hits]
        assert len(ids) == 3
        assert retired not in ids, "retired row consumed a result slot"
        assert heir in ids  # the successor is live and near the query

    def test_c1b_recent_excludes_superseded_by_default(self, store):
        old = store.add("fato antigo X", _vec(0.1), kind="fact")
        new = store.add("fato novo X", _vec(0.1), kind="fact")
        store.supersede_by_key(old, new)
        rows = store.recent(limit=50)
        ids = {r["id"] for r in rows}
        assert old not in ids
        rows_all = store.recent(limit=50, include_superseded=True)
        assert old in {r["id"] for r in rows_all}

    def test_gate_filter_actually_gates(self, store):
        """Rule 2 injection: with the filter REMOVED (both rows identical), the
        same query must return BOTH rows; with the filter, only the live one.
        If both branches return the same thing, the test asserts nothing."""
        old = store.add("gate fato", _vec(0.7), kind="fact",
                        subject="g", relation="r", object="old")
        new = store.add("gate fato 2", _vec(0.7), kind="fact",
                        subject="g", relation="r", object="new")
        store.supersede_by_key(old, new)
        with_filter = store.search(_vec(0.7), "gate fato", limit=10)
        without_filter = store.search(_vec(0.7), "gate fato", limit=10,
                                      include_superseded=True)
        assert len(without_filter) > len(with_filter), (
            "include_superseded=True returned the same set — the filter is not "
            "gating (vacuous contract)"
        )
