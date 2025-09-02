"""Simple asynchronous Telegram bot using aiohttp for HTTP requests."""

import os
import asyncio
import logging
import aiohttp

# Настройка логирования для Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    try:
        async with session.get(url, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("result", [])
            else:
                logger.warning(f"Telegram API returned status {resp.status}")
                return []
    except asyncio.TimeoutError:
        logger.warning("Telegram API timeout, continuing...")
        return []
    except Exception as e:
        logger.error(f"Error in get_updates: {e}")
        return []


async def send_message(
    session: aiohttp.ClientSession, chat_id: int, text: str
) -> bool:
    url = f"{API_URL}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status == 200:
                await resp.json()
                return True
            else:
                logger.warning(f"Failed to send message, status: {resp.status}")
                return False
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        return False


MAX_RETRIES = 5
RETRY_DELAY = 2


async def main() -> None:
    logger.info("Starting Telegram bot...")
    
    try:
        from pro_engine import ProEngine
        engine = ProEngine()
        await engine.setup()
        logger.info("ProEngine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ProEngine: {e}")
        return
    
    offset = None
    # Увеличиваем таймаут для Railway
    timeout = aiohttp.ClientTimeout(total=60, connect=30)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        consecutive_errors = 0
        
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
                            logger.info(f"Processing message from chat {chat_id}")
                            try:
                                response, _ = await engine.process_message(text)
                                success = await send_message(session, chat_id, response)
                                if success:
                                    logger.info("Message sent successfully")
                                else:
                                    logger.warning("Failed to send message")
                            except Exception as e:
                                logger.error(f"Error processing message: {e}")
                                # Отправляем простой ответ об ошибке
                                await send_message(session, chat_id, "Sorry, I encountered an error processing your message.")
                    
                    consecutive_errors = 0
                    
                except Exception as exc:
                    consecutive_errors += 1
                    logger.exception(f"Error in main loop (attempt {consecutive_errors}): {exc}")
                    
                    if consecutive_errors >= MAX_RETRIES:
                        logger.error("Too many consecutive errors, restarting...")
                        break
                    
                    # Exponential backoff
                    delay = min(RETRY_DELAY * (2 ** (consecutive_errors - 1)), 60)
                    logger.info(f"Waiting {delay} seconds before retry...")
                    await asyncio.sleep(delay)
                    
        finally:
            logger.info("Shutting down engine...")
            await engine.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
