"""
Filters for displacement/vector fields taken from LAMMA v.2024.10.03.

Major upgrades:
- Vectorized pairwise distance calculations using NumPy broadcasting.
- Replaced joblib parallelization with list mapping to eliminate serialization overhead for small neighborhoods.
- Optimized dense core selection using `np.argpartition` (O(N) complexity).
- Improved numerical robustness using `np.isclose` for floating-point inlier detection.
"""

import logging
from time import perf_counter
from typing import Literal

import numpy as np
from sklearn.neighbors import KDTree

logger = logging.getLogger("ppcx")


def vector_field_filter(
    values: list[np.ndarray],
    nodes: np.ndarray,
    method: Literal["Neighbours", "Radius"] = "Neighbours",
    k: int | float = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | list, np.ndarray]:
    """
    Main entry point for filtering displacement/vector fields.

    This function dispatches the filtering to the scattered data loop. It currently
    supports 'Neighbours' and 'Radius' search methods.

    Args:
        values: A list of numpy arrays, e.g., [U_components, V_components].
        nodes: A (N, 2) or (N, 3) numpy array of spatial coordinates.
        method: The search strategy ('Neighbours' or 'Radius'). Defaults to "Neighbours".
        k: The number of neighbors or the radius for search. Defaults to 4.

    Returns:
        A tuple containing (U_filtered, V_filtered, W_filtered, optimized_nodes).
        W_filtered is an empty list if data is 2D.

    Examples:
        >>> nodes = np.array([[0, 0], [1, 1], [0.1, 0.1], [5, 5]])
        >>> u = np.array([1.0, 1.1, 0.9, 10.0])  # 10.0 is an outlier
        >>> v = np.array([0.0, 0.1, -0.1, 5.0])
        >>> uf, vf, _, _ = vector_field_filter([u, v], nodes, k=2)
    """
    # -check inputs
    if method not in ["Neighbours", "Radius"]:
        raise NameError("Method must be 'Neighbours' or 'Radius'")

    for i in values:
        if type(i) is not np.ndarray:
            raise NameError("Values must be a list of numpy arrays")

    if type(nodes) is not np.ndarray:
        raise NameError("Nodes must a numpy array")
    if nodes.shape[1] != 2:
        raise NameError("Nodes must be a Nx2 numpy array")

    if len(values) == 2:
        X, Y = values[0], values[1]
        if X.shape != Y.shape:
            raise NameError("The sizes of the values arrays must be the same")
    elif len(values) == 3:
        X, Y, Z = values[0], values[1], values[2]
        if (X.shape != Y.shape) or (X.shape != Z.shape):
            raise NameError("The sizes of the values arrays must be the same")

    # Simply redirect to scattered data loop to keep code structure similar to original
    val_matrix = np.column_stack([v.ravel() for v in values])
    return loopScattered(val_matrix, np.asarray(nodes), method, k)


# -##########################################


def loopScattered(
    values: np.ndarray, nodes: np.ndarray, method: str, k: int | float
) -> tuple[np.ndarray, np.ndarray, np.ndarray | list, np.ndarray]:
    """
    Applies the LAMMA filter to scattered data points.

    Handles NaN removal, neighbor searching via KDTree, and orchestrates the
    point-wise filtering via list mapping.

    Args:
        values: (N, D) array of vector components (e.g., U, V).
        nodes: (N, 2) array of spatial coordinates.
        method: Search method ('Neighbours' or 'Radius').
        k: Search parameter (count or radius).

    Returns:
        Tuple of (U_filtered, V_filtered, W_filtered, nannodes).
    """
    # -check whether there are NaNs in values/nodes
    temp = np.sum(values, axis=1)
    nanpun = np.argwhere(np.isnan(temp)).flatten()
    realpun = np.argwhere(~np.isnan(temp)).flatten()
    sz = nodes.shape[0]

    # Work only with valid nodes
    v_nodes = nodes[realpun, :] if len(nanpun) > 0 else nodes
    v_values = values[realpun, :] if len(nanpun) > 0 else values

    v_nodes = np.ascontiguousarray(v_nodes, dtype=np.float64)
    tree = KDTree(v_nodes)

    if method == "Radius":
        ind = tree.query_radius(v_nodes, r=k)
    else:  # "Neighbours"
        # query k neighbors + 1 (including the point itself)
        ind = tree.query(v_nodes, k=int(k) + 1, return_distance=False)

    # Neighborhood values extraction
    z = [v_values[i, :] for i in ind]
    Zi = v_values

    # -apply filter using list mapping (replaces joblib parallel loop)
    start = perf_counter()
    O_list = [loc_filter(Zi[i], z[i]) for i in range(len(Zi))]
    O = np.array(O_list)
    O = O.squeeze()

    # -reinsert NaNs if they were present
    if len(nanpun) > 0:
        nannodes = np.full((sz, 2), np.nan)
        nanValues = np.full((sz, values.shape[1]), np.nan)
        nannodes[realpun, :] = v_nodes
        nanValues[realpun, :] = O
    else:
        nannodes = nodes
        nanValues = O

    # Prepare outputs
    U = nanValues[:, 0]
    V = nanValues[:, 1]
    W = nanValues[:, 2] if values.shape[1] > 2 else []

    logger.info(f"Filtering completed in {perf_counter() - start:.2f} seconds")
    return U, V, W, nannodes


def loc_filter(Zi: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Core median-based outlier filter for a single neighborhood.

    Computes pairwise distances within the neighborhood 'z', selects the 50%
    vectors with the lowest average distance to others (dense core), and
    compares 'Zi' to this core.

    Args:
        Zi: The vector (displacement) at the current point, shape (D,).
        z: The vectors in the neighborhood, shape (K, D).

    Returns:
        The filtered vector (either Zi or the median of the densest core).
    """
    # Optimization: Vectorized pairwise distance calculation
    # (K, 1, D) - (1, K, D) -> (K, K, D)
    diff = z[:, np.newaxis, :] - z[np.newaxis, :, :]
    di = np.sqrt(np.sum(diff**2, axis=-1))

    # Determine 50% of vectors with lowest reciprocal distance
    num = max(int(np.round(z.shape[0] / 2)), 1)
    # argpartition is O(N), much faster than sort
    mean_distances = np.mean(di, axis=1)
    ptr = np.argpartition(mean_distances, num - 1)[:num]

    selected_z = z[ptr, :]

    # Check if Zi is an inlier (belongs to the selected dense core)
    # np.isclose is safer for floating point comparisons
    is_inlier = np.any(np.all(np.isclose(selected_z, Zi, atol=1e-8), axis=1))

    if is_inlier:
        return Zi
    else:
        return np.median(selected_z, axis=0)
