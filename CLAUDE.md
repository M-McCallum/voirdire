# voirdire

The project name is voirdire, one word, lowercase. The CLI is vd.

## Rules

- Python 3.12, uv, ruff, pytest; service in core/, clients in clients/, packs in packs/; traces are OTel spans everywhere, never backend-native formats; no network calls in tests, mock the chat model and the backends; every story ends by running the thing and showing output, never by adding scope; never generate human labels.
- Never add Co-Authored-By or "Generated with Claude Code" lines to commits or pull requests.
- The service package is core/, not engine/: LangSmith has a product called Engine and the names must not collide.
