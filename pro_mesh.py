"""Simple mesh gossip protocol for exchanging adapter deltas."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Dict, Tuple, Optional


# -- Encryption helpers -------------------------------------------------

def _xor_bytes(data: bytes, key: bytes) -> bytes:
    """Return *data* XORed with *key* (repeating as needed)."""
    if not key:
        return data
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


def encrypt(payload: Dict, key: bytes) -> bytes:
    raw = json.dumps(payload).encode()
    return base64.b64encode(_xor_bytes(raw, key))


def decrypt(data: bytes, key: bytes) -> Dict:
    raw = _xor_bytes(base64.b64decode(data), key)
    return json.loads(raw.decode())


# -- Mesh node ----------------------------------------------------------

class MeshNode(asyncio.DatagramProtocol):
    """Lightweight gossip node exchanging encrypted adapter deltas."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0, key: Optional[bytes] = None) -> None:
        self.host = host
        self.port = port
        self.key = key or b"mesh-secret"
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.peers: set[Tuple[str, int]] = set()
        self.adapters: Dict[str, Dict] = {}
        self._pong_waiters: Dict[Tuple[str, int], asyncio.Future] = {}

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self.transport, _ = await loop.create_datagram_endpoint(lambda: self, local_addr=(self.host, self.port))
        sockname = self.transport.get_extra_info("sockname")
        if sockname:
            self.port = sockname[1]

    def connection_made(self, transport: asyncio.BaseTransport) -> None:  # pragma: no cover - provided by asyncio
        self.transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:  # pragma: no cover - network callbacks
        message = decrypt(data, self.key)
        mtype = message.get("type")
        if mtype == "gossip":
            self._handle_gossip(message.get("adapters", {}))
        elif mtype == "ping":
            self._send({"type": "pong"}, addr)
        elif mtype == "pong":
            fut = self._pong_waiters.get(addr)
            if fut and not fut.done():
                fut.set_result(True)
        elif mtype == "join":
            peer = tuple(message.get("peer", []))
            if len(peer) == 2:
                self.peers.add((peer[0], int(peer[1])))
        elif mtype == "leave":
            peer = tuple(message.get("peer", []))
            if len(peer) == 2:
                self.peers.discard((peer[0], int(peer[1])))

    def _send(self, payload: Dict, addr: Tuple[str, int]) -> None:
        if self.transport is None:
            raise RuntimeError("MeshNode not started")
        self.transport.sendto(encrypt(payload, self.key), addr)

    def join(self, host: str, port: int) -> None:
        self.peers.add((host, port))

    def leave_peer(self, host: str, port: int) -> None:
        self.peers.discard((host, port))

    async def leave(self) -> None:
        if self.transport:
            self.transport.close()
        self.peers.clear()

    async def health_check(self, host: str, port: int, timeout: float = 1.0) -> bool:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._pong_waiters[(host, port)] = fut
        self._send({"type": "ping"}, (host, port))
        try:
            await asyncio.wait_for(fut, timeout)
            return True
        except asyncio.TimeoutError:
            return False
        finally:
            self._pong_waiters.pop((host, port), None)

    def update_adapter(self, name: str, weights: Dict[str, float], version: int) -> None:
        self.adapters[name] = {"weights": weights, "version": version}

    async def gossip(self) -> None:
        payload = {"type": "gossip", "adapters": self.adapters}
        for peer in list(self.peers):
            self._send(payload, peer)

    def _handle_gossip(self, remote: Dict[str, Dict]) -> None:
        for name, data in remote.items():
            r_version = data.get("version", 0)
            r_weights = data.get("weights", {})
            local = self.adapters.get(name)
            if local is None or r_version > local.get("version", 0):
                self.adapters[name] = {"version": r_version, "weights": r_weights}
            elif r_version == local.get("version", 0):
                # conflict resolution: prefer higher metric (sum of weights)
                r_metric = sum(r_weights.values())
                l_metric = sum(local.get("weights", {}).values())
                if r_metric > l_metric:
                    self.adapters[name] = {"version": r_version, "weights": r_weights}


# -- CLI helpers --------------------------------------------------------

async def send_command(
    command: str,
    target_host: str,
    target_port: int,
    peer: Optional[Tuple[str, int]] = None,
    key: bytes = b"mesh-secret",
) -> Optional[bool]:
    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(asyncio.DatagramProtocol, local_addr=("0.0.0.0", 0))
    try:
        if command == "health":
            fut = loop.create_future()

            class P(asyncio.DatagramProtocol):  # pragma: no cover - network callbacks
                def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
                    if addr == (target_host, target_port):
                        fut.set_result(True)

            transport2, _ = await loop.create_datagram_endpoint(P, local_addr=("0.0.0.0", 0))
            try:
                transport2.sendto(encrypt({"type": "ping"}, key), (target_host, target_port))
                try:
                    await asyncio.wait_for(fut, 1)
                    return True
                except asyncio.TimeoutError:
                    return False
            finally:
                transport2.close()
        else:
            payload = {"type": command}
            if peer:
                payload["peer"] = list(peer)
            transport.sendto(encrypt(payload, key), (target_host, target_port))
            return None
    finally:
        transport.close()
