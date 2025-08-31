import asyncio
import hashlib
import json
from typing import Dict, Tuple


class P2PResonance:
    """Simple P2P node exchanging gradient hashes using asyncio streams."""

    def __init__(self) -> None:
        self.params: Dict[str, float] = {}
        self._pending: Dict[str, float] = {}
        self._server: asyncio.AbstractServer | None = None
        self._host: str = "127.0.0.1"
        self._port: int = 0

    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Start listening for peer connections."""
        self._server = await asyncio.start_server(self._handle, self._host, 0)
        sock = self._server.sockets[0]
        addr = sock.getsockname()
        self._host, self._port = addr[0], addr[1]

    @property
    def server_address(self) -> Tuple[str, int]:
        return self._host, self._port

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    # Gradient management -------------------------------------------------
    def queue_update(self, grads: Dict[str, float]) -> None:
        """Queue gradients for broadcasting and apply locally."""
        for k, v in grads.items():
            self.params[k] = self.params.get(k, 0.0) + v
            self._pending[k] = self._pending.get(k, 0.0) + v

    def apply_gradients(self, grads: Dict[str, float]) -> None:
        for k, v in grads.items():
            self.params[k] = self.params.get(k, 0.0) + v

    def pop_pending(self) -> Dict[str, float]:
        pend = self._pending
        self._pending = {}
        return pend

    def current_hash(self) -> str:
        data = json.dumps(self.params, sort_keys=True).encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    # Networking ----------------------------------------------------------
    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        data = await reader.read(65536)
        if not data:
            writer.close()
            await writer.wait_closed()
            return
        msg = json.loads(data.decode("utf-8"))
        update = msg.get("update", {})
        self.apply_gradients(update)
        payload = {
            "hash": self.current_hash(),
            "update": self.pop_pending(),
        }
        writer.write(json.dumps(payload).encode("utf-8"))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def exchange(self, host: str, port: int) -> Dict[str, float]:
        """Send queued gradients to ``host`` and receive its update."""
        update = self.pop_pending()
        msg = json.dumps({"update": update, "hash": self.current_hash()}).encode(
            "utf-8"
        )
        reader, writer = await asyncio.open_connection(host, port)
        writer.write(msg)
        await writer.drain()
        resp = await reader.read(65536)
        writer.close()
        await writer.wait_closed()
        payload = json.loads(resp.decode("utf-8"))
        self.apply_gradients(payload.get("update", {}))
        return payload.get("update", {})
