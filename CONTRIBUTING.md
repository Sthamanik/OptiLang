# Contributing to OptiLang

Thank you for considering contributing to OptiLang!

## Local Setup

```bash
git clone https://github.com/Sthamanik/optilang.git
cd optilang
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

## Workflow

1. Fork the repository and create a branch for your change.
2. Make the smallest coherent change that solves the problem.
3. Add or update tests when behavior changes.
4. Update `README.md`, `docs/`, and `CHANGELOG.md` when public behavior or APIs change.
5. Run the quality checks before opening a pull request.

## Quality Checks

Run these commands before submitting:

```bash
python3 -m pytest
black optilang tests
mypy optilang
flake8 optilang
```

## Code Guidelines

- Follow existing project structure and naming patterns.
- Prefer small, focused patches over broad rewrites.
- Keep public APIs typed and documented.
- Preserve user-facing behavior unless the change explicitly updates it.
- Do not leave README or docs examples behind when signatures change.

## Documentation Guidelines

- Keep examples executable against the current API.
- Document new optimizer patterns, scoring behavior, or runtime options when added.
- Use the `docs/` directory for user-facing guides and keep the README high signal.

## Commit Messages

Conventional commits are preferred:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions or updates
- `refactor:` for internal restructuring

## Pull Requests

- Explain what changed and why.
- Call out user-visible behavior changes.
- Mention any follow-up work that remains out of scope.

## Questions

Open an issue or contact the maintainers.
