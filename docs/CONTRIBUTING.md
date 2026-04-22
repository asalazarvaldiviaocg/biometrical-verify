# Contributing

Thanks for your interest in biometrical-verify.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
make seed
make up
```

## Running checks

```bash
make lint    # ruff + mypy
make test    # pytest with coverage
make fmt     # auto-format
```

## Pull requests

- Keep changes focused. One feature / fix per PR.
- Add or update tests for behaviour you change.
- Run `make lint test` locally before pushing.
- Reference any related issue (`Closes #123`).

## Reporting security issues

Do **not** open a public issue. Email `security@biometrical.org` with a
description and reproduction steps. We will acknowledge within 72h.

## Project layout

See [README.md](../README.md#layout).
