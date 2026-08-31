## What and why

<!-- What changes, and what problem it solves. Link the issue if there is one. -->

## Evidence

<!--
Paste real output, not a description of it. For a bug fix, include the error
as it actually appeared.
-->

```
$ .venv/bin/python -m pytest tests/
```

## Checklist

- [ ] `pytest tests/` exits 0 (run without piping through `tail`/`head` — a
      pipe reports the pager's exit code, not pytest's)
- [ ] Integration tests actually ran, not skipped (`-rs` shows skip reasons)
- [ ] `ruff check .` and `ruff format --check .` are clean
- [ ] New behaviour has a test that **fails without the change** — verified by
      reverting the source and watching it fail
- [ ] Host-contract changes exercised against real Hermes, not only the stub ABC
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] Base branch is `develop` (only releases and hotfixes target `main`)
