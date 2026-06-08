# Format, lint project
validate: format lint test

# Format project code
format:
    uv run ruff format

# Check project code for type and linting errors, auto-fix if possible
lint:
    uv run ruff check --fix
    uv run ty check

# Run tests/
test:
    uv run pytest -v --color=yes
