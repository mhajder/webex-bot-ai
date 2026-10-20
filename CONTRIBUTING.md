# Contributing

Thanks for helping out. Bug reports and feature requests go through the [issue templates](https://github.com/mhajder/webex-bot-ai/issues/new/choose). For security problems, follow [SECURITY.md](SECURITY.md) instead.

## Development setup

```sh
git clone https://github.com/mhajder/webex-bot-ai.git
cd webex-bot-ai
uv sync --group dev
uv run prek install
cp .env.example .env
```

`prek install` sets up the git hooks, which run ruff, ty and typos on commit and check the commit message. Use a separate Webex bot and test space for development, not your production bot.

## Before opening a pull request

```sh
uv run ruff check .
uv run ruff format .
uv run ty check .
```

- Try the change against a real Webex space, including a follow-up in a thread.
- Update the README when you add a setting or change behavior.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/), for example `feat(mcp): retry failed tool calls` or `fix(conversation): keep thread summaries per room`. The changelog and the next version are generated from them.

## License

By contributing, you agree that your contributions are licensed under the project's MIT license.
