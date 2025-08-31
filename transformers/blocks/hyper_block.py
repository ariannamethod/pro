import numpy as np
import numpy.typing as npt


class HyperBlock:
    """Generate a weight matrix from a context vector."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim

    def weights(
        self, context: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        flat = np.resize(
            context.astype(np.float32), self.out_dim * self.in_dim
        )
        return flat.reshape(self.out_dim, self.in_dim)

    def __call__(
        self, x: npt.NDArray[np.float32], context: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Apply generated weights to *x* using *context*."""
        w = self.weights(context)
        return w @ x
