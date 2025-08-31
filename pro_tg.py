"""Simple asynchronous Telegram bot using aiohttp for HTTP requests."""

import os
import asyncio
import aiohttp

TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN environment variable not set")

API_URL = f"https://api.telegram.org/bot{TOKEN}"


async def get_updates(
    session: aiohttp.ClientSession, offset: int | None = None
) -> list[dict]:
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


async def main() -> None:
    from pro_engine import ProEngine

    engine = ProEngine()
    await engine.setup()
    offset = None
    async with aiohttp.ClientSession() as session:
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
                except Exception as exc:  # pragma: no cover - logging only
                    print(f"Error: {exc}")
                    await asyncio.sleep(1)
        finally:
            await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
