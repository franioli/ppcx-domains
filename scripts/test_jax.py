import time

import jax
import jax.numpy as jnp
from jax import device_put, random


def test_gpu_computation():
    # 1. Check if JAX sees the GPU
    print("------------------------------------------------")
    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend platform: {jax.default_backend()}")
    devices = jax.devices()
    print(f"Available devices: {devices}")

    gpu_available = any(d.platform == "gpu" for d in devices)
    if not gpu_available:
        print("WARNING: No GPU detected! JAX will run on CPU.")
    else:
        print("SUCCESS: GPU detected.")
    print("------------------------------------------------")

    # 2. Perform a heavy matrix multiplication
    size = 5000  # Large enough to make GPU advantage obvious
    print(f"\nCreating two {size}x{size} random matrices...")

    key = random.PRNGKey(0)
    # Generate data (initially on host/CPU usually)
    x = random.normal(key, (size, size), dtype=jnp.float32)
    y = random.normal(key, (size, size), dtype=jnp.float32)

    print("Moving data to device (GPU if available)...")
    # JAX usually puts data on the default device (GPU) automatically,
    # but device_put ensures it.
    x = device_put(x)
    y = device_put(y)

    print("Starting matrix multiplication (dot product)...")
    start_time = time.time()

    # Perform computation
    # JAX operations are asynchronous, so the call returns immediately.
    # We use .block_until_ready() to measure actual execution time.
    z = jnp.dot(x, y).block_until_ready()

    end_time = time.time()
    duration = end_time - start_time

    print(f"Computation completed in {duration:.4f} seconds.")
    print(f"Result shape: {z.shape}")
    print(f"Result mean: {jnp.mean(z)}")

    if gpu_available:
        print("\nTest PASSED: Operations ran on GPU.")
    else:
        print("\nTest ran on CPU (GPU not found).")


if __name__ == "__main__":
    try:
        test_gpu_computation()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
