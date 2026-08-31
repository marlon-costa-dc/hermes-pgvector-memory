# Contributing

## Branching model

This project follows git-flow.

| Branch | Purpose | Merges into |
|---|---|---|
| `main` | Released code only. Every commit is a tagged release. | — |
| `develop` | Integration branch. Default branch; PRs land here. | `main` via release |
| `feature/*` | New capability or refactor. | `develop` |
| `fix/*` | Bug fix that is not urgent. | `develop` |
| `release/x.y.z` | Version freeze: bump, changelog, final checks. | `main` **and** `develop` |
| `hotfix/x.y.z` | Urgent fix against a release. | `main` **and** `develop` |

Nothing is committed directly to `main` or `develop` — both take merges only.

```bash
git checkout develop && git pull
git checkout -b feature/my-change
# ... work, commit ...
git push -u origin feature/my-change
gh pr create --base develop
```

### Releases

```bash
git checkout -b release/0.2.0 develop
# bump version in pyproject.toml + plugin.yaml + pgvector_memory/__init__.py
# update CHANGELOG.md
gh pr create --base main --title "release: 0.2.0"
# after merge:
git tag -a v0.2.0 -m "0.2.0" && git push origin v0.2.0
git checkout develop && git merge --no-ff main && git push
```

Merging the release back into `develop` is not optional: skipping it makes
`main` diverge and the next release reintroduce old code.

## Definition of done

A change is done when all of these hold. No exceptions, no "will fix later".

1. **`pytest tests/` passes with exit code 0.** Run it without piping through
   `tail`/`head` — a pipe reports the pager's exit code, not pytest's, and
   turns a red suite green.
2. **The integration suite actually ran.** It skips itself when PostgreSQL or
   Ollama is absent. A skip is not a pass; check with `-rs`.
3. **`ruff check .` and `ruff format --check .` are clean.**
4. **New behaviour has a test that fails without the fix.** Verify it: stash
   the source change, keep the test, watch it fail. A test that passes both
   before and after proves nothing.
5. **Anything touching the host contract was exercised against real Hermes**,
   not only against the stub ABC in `tests/conftest.py`.

## Testing

```bash
uv venv .venv
uv pip install --python .venv/bin/python pytest 'psycopg[binary]'
.venv/bin/python -m pytest tests/ -v
```

Three layers, in increasing cost:

- `test_unit.py` — pure logic. Runs anywhere.
- `test_contract.py` — asserts the provider satisfies Hermes' real
  `MemoryProvider` ABC, including override signatures. Skips without Hermes.
- `test_integration.py` — real PostgreSQL + pgvectorscale + Ollama. Skips,
  with a printed reason, when the stack is unavailable.

For the full stack:

```bash
createdb hermes_memory
sudo -u postgres psql -d hermes_memory -c 'CREATE EXTENSION vectorscale CASCADE;'
sudo -u postgres psql -d hermes_memory -c 'GRANT ALL ON SCHEMA public TO '"$USER"';'
ollama pull nomic-embed-text
.venv/bin/python -m pytest tests/ -v -rs
```

## Conventions

Commits follow [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `style:`, `chore:`.

The body explains **why**, with evidence. A commit that fixes a bug found at
runtime should quote the actual error, not paraphrase it.

### Comments

Comments explain reasoning that is not visible in the code: why a constant has
that value, why a seemingly redundant guard exists, what was measured. They do
not narrate what the next line does.

Where behaviour was determined empirically rather than from documentation, say
so — the next reader needs to know which claims were verified and which were
assumed.
