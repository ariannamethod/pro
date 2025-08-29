# pro-nano
P.R.O. Pure Recursive Organism

## Telegram Bot

1. Copy `.env.example` to `.env` and replace the placeholder token.
2. Run `python pro_tg.py` to start the bot. It echoes incoming messages using long polling.

## Deployment

Platforms such as [Railway](https://railway.app) use Nixpacks to detect a project's
language and dependencies. An empty `requirements.txt` file is included so the
app is recognized as a Python project during deployment. Add any runtime
dependencies to that file as your project evolves.
