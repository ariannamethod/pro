"""Simple asynchronous Telegram bot using aiohttp for HTTP requests."""

import os
import asyncio
import logging
import aiohttp

TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN environment variable not set")

API_URL = f"https://api.telegram.org/bot{TOKEN}"


async def get_updates(
    session: aiohttp.ClientSession, offset=None
):
    params = {"timeout": 30}
    if offset is not None:
        params["offset"] = offset
    url = f"{API_URL}/getUpdates"
    async with session.get(url, params=params) as resp:
        data = await resp.json()
    return data.get("result", [])


async def send_message(
    session: aiohttp.ClientSession, chat_id: int, text: str
) -> None:
    url = f"{API_URL}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    async with session.post(url, json=payload) as resp:
        await resp.json()


logger = logging.getLogger(__name__)

MAX_RETRIES = 3


async def main() -> None:
    from pro_engine import ProEngine

    engine = ProEngine()
    await engine.setup()
    offset = None
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        retries = 0
        try:
            while True:
                try:
                    updates = await get_updates(session, offset)
                    for update in updates:
                        offset = update["update_id"] + 1
                        message = update.get("message") or {}
                        text = message.get("text")
                        chat = message.get("chat") or {}
                        chat_id = chat.get("id")
                        if chat_id and text:
                            response, _ = await engine.process_message(text)
                            await send_message(session, chat_id, response)
                    retries = 0
                except Exception as exc:  # pragma: no cover - network handling
                    logger.exception("Error handling update: %s", exc)
                    retries += 1
                    if retries >= MAX_RETRIES:
                        logger.error("Max retries exceeded")
                        raise
        finally:
            await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
