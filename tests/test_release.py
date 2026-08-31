"""Release-hygiene tests.

The version is declared in three places that a release must keep in step:
``pyproject.toml`` (what pip installs), ``plugin.yaml`` (what Hermes reads),
and ``__version__`` (what the code reports). Nothing enforces agreement at
runtime, so a release that bumps two of the three ships a package whose
self-reported version is a lie. These tests are that enforcement.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pgvector_memory  # noqa: E402

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def _pyproject_version() -> str:
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def _plugin_yaml_version() -> str:
    # Parsed with a regex rather than PyYAML: the plugin must stay dependency
    # free beyond psycopg, and the test suite should not need more than the
    # package itself.
    text = (REPO_ROOT / "plugin.yaml").read_text(encoding="utf-8")
    match = re.search(r'^version:\s*"?([^"\s]+)"?', text, re.MULTILINE)
    assert match, "plugin.yaml declares no version"
    return match.group(1)


class TestVersionConsistency:
    def test_all_three_declarations_agree(self):
        versions = {
            "pyproject.toml": _pyproject_version(),
            "plugin.yaml": _plugin_yaml_version(),
            "__init__.__version__": pgvector_memory.__version__,
        }
        assert len(set(versions.values())) == 1, f"version drift: {versions}"

    def test_version_is_semver(self):
        assert SEMVER.match(pgvector_memory.__version__)


class TestChangelog:
    def test_changelog_documents_the_current_version(self):
        changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        assert f"## [{pgvector_memory.__version__}]" in changelog, (
            f"CHANGELOG.md has no section for {pgvector_memory.__version__}"
        )

    def test_changelog_keeps_an_unreleased_section(self):
        # Without it, the next change has nowhere to land and the habit dies.
        changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        assert "## [Unreleased]" in changelog


class TestPackagedFiles:
    def test_schema_ships_with_the_package(self):
        # package-data in pyproject must match reality, or an installed copy
        # raises StoreError("schema.sql not found") on first run.
        assert (REPO_ROOT / "sql" / "schema.sql").is_file()

    def test_schema_declares_the_diskann_index(self):
        ddl = (REPO_ROOT / "sql" / "schema.sql").read_text(encoding="utf-8")
        assert "USING diskann" in ddl, "pgvectorscale index missing from schema"
        assert "USING gin" in ddl, "lexical index missing; hybrid search degrades"
