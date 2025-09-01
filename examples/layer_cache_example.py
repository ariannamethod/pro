"""Example demonstrating layer construction and caching."""

from memory import cache_layers

macros = {"dense": {"in": 2, "out": 2}}

if __name__ == "__main__":
    layers = cache_layers(macros)
    print("Constructed layers:", layers)
