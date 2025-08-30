import hashlib
import json
import socket
import socketserver
from threading import Thread
from typing import Dict


class _ResonanceHandler(socketserver.BaseRequestHandler):
    """Handle incoming gradient hashes and exchange updates."""

    def handle(self) -> None:  # type: ignore[override]
        data = self.request.recv(65536)
        if not data:
            return
        msg = json.loads(data.decode("utf-8"))
        update = msg.get("update", {})
        self.server.peer.apply_gradients(update)  # type: ignore[attr-defined]
        payload = {
            "hash": self.server.peer.current_hash(),  # type: ignore[attr-defined]
            "update": self.server.peer.pop_pending(),  # type: ignore[attr-defined]
        }
        self.request.sendall(json.dumps(payload).encode("utf-8"))


class P2PResonance(socketserver.ThreadingTCPServer):
    """Simple P2P node exchanging gradient hashes over TCP."""

    allow_reuse_address = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _ResonanceHandler)
        self.peer = self  # for handler access
        self.params: Dict[str, float] = {}
        self._pending: Dict[str, float] = {}
        self._thread = Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

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
    def exchange(self, host: str, port: int) -> Dict[str, float]:
        """Send queued gradients to ``host`` and receive its update."""
        update = self.pop_pending()
        msg = json.dumps({"update": update, "hash": self.current_hash()}).encode(
            "utf-8"
        )
        with socket.create_connection((host, port), timeout=1.0) as sock:
            sock.sendall(msg)
            resp = sock.recv(65536)
        payload = json.loads(resp.decode("utf-8"))
        self.apply_gradients(payload.get("update", {}))
        return payload.get("update", {})

    def stop(self) -> None:
        self.shutdown()
        self.server_close()
        self._thread.join(timeout=1.0)
