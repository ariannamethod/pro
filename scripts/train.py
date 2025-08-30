"""Training entry point for toy transformer models."""

from __future__ import annotations

import argparse
import numpy as np

from models.transformer import Transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny transformer")
    parser.add_argument("--dim", type=int, default=16, help="Hidden dimension")
    parser.add_argument(
        "--use-fractal-adapter",
        action="store_true",
        help="Enable fractal resonance adapter",
    )
    parser.add_argument(
        "--fractal-depth",
        type=int,
        default=1,
        help="Depth of the fractal adapter",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = Transformer(
        args.dim,
        use_fractal_adapter=args.use_fractal_adapter,
        fractal_depth=args.fractal_depth,
    )
    dummy = np.zeros(args.dim, dtype=np.float32)
    out = model(dummy)
    print("output", out)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
