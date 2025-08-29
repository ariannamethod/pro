"""Simple asynchronous Telegram bot using urllib for HTTP requests."""

import asyncio
import json
import os
from urllib import parse, request

TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN environment variable not set")

API_URL = f"https://api.telegram.org/bot{TOKEN}"


def _sync_request(req: request.Request) -> dict:
    with request.urlopen(req) as resp:  # type: ignore[arg-type]
        return json.loads(resp.read().decode())


async def _request(req: request.Request) -> dict:
    return await asyncio.to_thread(_sync_request, req)


async def get_updates(offset: int | None = None) -> list[dict]:
    params = {"timeout": 30}
    if offset is not None:
        params["offset"] = offset
    url = f"{API_URL}/getUpdates?{parse.urlencode(params)}"
    req = request.Request(url)
    data = await _request(req)
    return data.get("result", [])


async def send_message(chat_id: int, text: str) -> None:
    url = f"{API_URL}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": text}).encode()
    req = request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    await _request(req)


async def main() -> None:
    offset = None
    while True:
        try:
            updates = await get_updates(offset)
            for update in updates:
                offset = update["update_id"] + 1
                message = update.get("message") or {}
                text = message.get("text")
                chat = message.get("chat") or {}
                chat_id = chat.get("id")
                if chat_id and text:
                    await send_message(chat_id, text)
        except Exception as exc:  # pragma: no cover - logging only
            print(f"Error: {exc}")
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
