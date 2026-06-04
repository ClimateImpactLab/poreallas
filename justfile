# Format, lint project
validate: format lint

# Format project code
format:
    uv run ruff format

# Check project code for type and linting errors, auto-fix if possible
lint:
    uv run ruff check --fix
    uv run ty check
