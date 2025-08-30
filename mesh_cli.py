"""Command line utility for interacting with the mesh network."""

import argparse
import asyncio
from typing import Tuple

from pro_mesh import send_command


async def main() -> None:
    parser = argparse.ArgumentParser(description="Mesh control utility")
    sub = parser.add_subparsers(dest="cmd", required=True)

    join_p = sub.add_parser("join", help="ask a node to join a peer")
    join_p.add_argument("target_host")
    join_p.add_argument("target_port", type=int)
    join_p.add_argument("peer_host")
    join_p.add_argument("peer_port", type=int)
    join_p.add_argument("key", help="shared secret")

    leave_p = sub.add_parser("leave", help="ask a node to drop a peer")
    leave_p.add_argument("target_host")
    leave_p.add_argument("target_port", type=int)
    leave_p.add_argument("peer_host")
    leave_p.add_argument("peer_port", type=int)
    leave_p.add_argument("key")

    health_p = sub.add_parser("health", help="check if a node responds")
    health_p.add_argument("target_host")
    health_p.add_argument("target_port", type=int)
    health_p.add_argument("key")

    args = parser.parse_args()

    if args.cmd == "health":
        ok = await send_command("health", args.target_host, args.target_port, key=args.key.encode())
        print("healthy" if ok else "unreachable")
    else:
        peer: Tuple[str, int] = (args.peer_host, args.peer_port)
        await send_command(args.cmd, args.target_host, args.target_port, peer, args.key.encode())


if __name__ == "__main__":
    asyncio.run(main())
