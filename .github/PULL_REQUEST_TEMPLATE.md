## What changed

## Checks

- [ ] `uv run ruff check .` and `uv run ruff format --check .`
- [ ] `uv run pytest -m "not slow and not network and not data"`
- [ ] `uv run tsfm docs check`

Do not hand-edit `results/`, `locks/`, `PREREGISTRATION*.md`, `DEVIATIONS.md`, or `uv.lock`. Numbers in the generated docs come from `uv run tsfm docs build`.
