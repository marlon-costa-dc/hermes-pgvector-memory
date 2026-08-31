#!/usr/bin/env bash
# Install hermes-pgvector-memory into a Hermes profile.
#
#   ./scripts/install.sh                 # default profile (~/.hermes)
#   HERMES_HOME=~/.hermes/profiles/work ./scripts/install.sh
#
# Database extensions need superuser, so that step prints the exact command
# instead of silently invoking sudo.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
PLUGIN_DIR="$HERMES_HOME/plugins/pgvector-memory"
DB="${PGVECTOR_MEMORY_DB:-hermes_memory}"

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m!!\033[0m %s\n' "$*" >&2; exit 1; }

[ -d "$HERMES_HOME" ] || die "HERMES_HOME not found: $HERMES_HOME"

info "Checking prerequisites"
command -v psql >/dev/null || die "psql not found — install postgresql"
psql -d "$DB" -tAc 'SELECT 1' >/dev/null 2>&1 \
  || die "cannot connect to database '$DB' — create it first: createdb $DB"

missing_ext=""
psql -d "$DB" -tAc "SELECT 1 FROM pg_extension WHERE extname='vector'" | grep -q 1 \
  || missing_ext="$missing_ext vector"
psql -d "$DB" -tAc "SELECT 1 FROM pg_am WHERE amname='diskann'" | grep -q 1 \
  || missing_ext="$missing_ext vectorscale"

if [ -n "$missing_ext" ]; then
  warn "Missing database extensions:$missing_ext"
  warn "They require superuser. Run this, then re-run install.sh:"
  echo
  echo "    sudo -u postgres psql -d $DB -c 'CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;'"
  echo
  exit 1
fi

if curl -fsS --max-time 5 "${PGVECTOR_MEMORY_OLLAMA_HOST:-http://127.0.0.1:11434}/api/tags" \
     | grep -q "${PGVECTOR_MEMORY_EMBED_MODEL:-nomic-embed-text}"; then
  info "Ollama is serving the embedding model"
else
  warn "Ollama is not serving ${PGVECTOR_MEMORY_EMBED_MODEL:-nomic-embed-text}"
  warn "Run: ollama pull ${PGVECTOR_MEMORY_EMBED_MODEL:-nomic-embed-text}"
fi

info "Installing plugin to $PLUGIN_DIR"
mkdir -p "$PLUGIN_DIR"
cp -r "$REPO_DIR/pgvector_memory/." "$PLUGIN_DIR/"
cp "$REPO_DIR/plugin.yaml" "$PLUGIN_DIR/"
# The store looks for sql/schema.sql beside the package too (see
# MemoryStore._read_schema), which is the layout a flattened plugin has.
cp -r "$REPO_DIR/sql" "$PLUGIN_DIR/sql"

info "Installing the psycopg driver into the Hermes venv"
HERMES_PY="$HOME/.hermes/hermes-agent/venv/bin/python"
if [ -x "$HERMES_PY" ]; then
  # The Hermes venv is uv-managed and ships no pip, so `python -m pip` fails
  # with "No module named pip". Try uv first, fall back to pip for installs
  # that do have it.
  if command -v uv >/dev/null 2>&1; then
    uv pip install --quiet --python "$HERMES_PY" "psycopg[binary]>=3.1" || true
  else
    "$HERMES_PY" -m pip install --quiet "psycopg[binary]>=3.1" || true
  fi

  # Verify by IMPORTING, never by trusting the installer's exit code: the
  # previous version printed "psycopg installed" while the venv had no pip
  # and nothing had been installed at all.
  if "$HERMES_PY" -c 'import psycopg' 2>/dev/null; then
    info "psycopg verified: $("$HERMES_PY" -c 'import psycopg; print(psycopg.__version__)')"
  else
    die "psycopg is still not importable by $HERMES_PY.
    Install it manually:  uv pip install --python $HERMES_PY 'psycopg[binary]'"
  fi
else
  warn "Hermes venv not found at $HERMES_PY — install 'psycopg[binary]' yourself"
fi

cat <<EOF

Installed. Activate it with:

    hermes config set memory.provider pgvector-memory

Then verify:

    hermes memory status

EOF
