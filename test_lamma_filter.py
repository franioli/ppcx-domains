import time

import matplotlib.pyplot as plt
import numpy as np
from lamma_filter_original import vector_field_filter_opt
from ppcluster.lamma_simple import vector_field_filter_simple



def compare_filters():
    print("Generating synthetic field...")
    values, nodes = generate_synthetic_field(N=20)
    print(f"Testing on {nodes.shape[0]} points...")

    # --- 1. Run Simple (Old Logic) ---
    start = time.time()
    u_s, v_s, _, _ = vector_field_filter_simple(values, nodes, k=8)
    time_simple = time.time() - start
    print(f"Simple (Legacy Logic): {time_simple:.4f} s")

    # --- 2. Run Optimized (Vectorized) ---
    start = time.time()
    u_o, v_o, _, _ = vector_field_filter_opt(values, nodes, k=8)
    time_opt = time.time() - start
    print(f"Optimized (Vectorized): {time_opt:.4f} s")

    speedup = time_simple / time_opt
    print(f"Speedup: {speedup:.1f}x")

    # --- 3. Statistical Check ---
    mse_u = np.mean((u_s - u_o) ** 2)
    mse_v = np.mean((v_s - v_o) ** 2)
    print(f"MSE between implementations (U): {mse_u:.2e}")
    print(f"MSE between implementations (V): {mse_v:.2e}")

    if mse_u < 1e-10:
        print(">> VERDICT: Implementations are statistically IDENTICAL.")
    else:
        print(">> VERDICT: Differences detected (check float precision).")

    # --- 4. Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original noisy
    mag_orig = np.sqrt(values[0] ** 2 + values[1] ** 2)
    sc0 = axes[0].scatter(nodes[:, 0], nodes[:, 1], c=mag_orig, s=1)
    axes[0].set_title("Input (Noisy)")
    plt.colorbar(sc0, ax=axes[0])

    # Filtered
    mag_filt = np.sqrt(u_o**2 + v_o**2)
    sc1 = axes[1].scatter(nodes[:, 0], nodes[:, 1], c=mag_filt, s=1)
    axes[1].set_title("Filtered (Optimized)")
    plt.colorbar(sc1, ax=axes[1])

    # Difference
    diff = np.abs(mag_orig - mag_filt)
    sc2 = axes[2].scatter(nodes[:, 0], nodes[:, 1], c=diff, s=1, cmap="inferno")
    axes[2].set_title("Difference (Removed Outliers)")
    plt.colorbar(sc2, ax=axes[2])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_filters()
