import numpy as np
from quantum.quant4 import compress_weights
from transformers import load_quant4

# Create random weights and compress them
w = np.random.randn(4, 3).astype(np.float32)
packed, scale, shape = compress_weights(w)
np.savez("toy_quant4.npz", packed=packed, scale=scale, shape=shape)

# Load layer and run a forward pass
layer = load_quant4("toy_quant4.npz")
x = np.random.randn(2, 4).astype(np.float32)
print(layer(x))
