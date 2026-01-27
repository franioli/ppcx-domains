import logging
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image
from scipy.spatial.distance import cdist

logger = logging.getLogger("ppcx")

COLORMAP = plt.get_cmap("tab10")

""" PRIOR """


def _validate_and_normalize_prior_vecs(
    prior_probs: dict[str, list[float]], sector_names: list[str]
) -> dict[str, np.ndarray]:
    """Validate prior_probs dict and return normalized numpy arrays per sector."""
    n_sectors = len(sector_names)
    if set(prior_probs.keys()) != set(sector_names):
        raise ValueError(
            "assign_spatial_priors: sector names in polygons and prior_probs do not match.\n"
            f"polygons: {sector_names}\nprior_probs: {list(prior_probs.keys())}"
        )
    prior_vecs: dict[str, np.ndarray] = {}
    for name, vec in prior_probs.items():
        arr = np.asarray(vec, dtype=float)
        if arr.ndim != 1 or arr.size != n_sectors:
            raise ValueError(
                f"assign_spatial_priors: prior vector for '{name}' must be 1D and length {n_sectors}."
            )
        if np.any(arr < 0) or arr.sum() <= 0:
            raise ValueError(
                f"assign_spatial_priors: probabilities for '{name}' must be non-negative and sum > 0."
            )
        prior_vecs[name] = arr / arr.sum()
    return prior_vecs


def _polygon_contains_mask(
    poly: Any, pts: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Return boolean mask of points inside polygon (try common APIs)."""
    try:
        # common matplotlib Path-like API: contains_points(pts)
        return np.asarray(poly.contains_points(pts), dtype=bool)
    except Exception:
        try:
            # some objects accept separate x,y arrays
            return np.asarray(poly.contains_points(x, y), dtype=bool)
        except Exception:
            # fallback: can't test membership -> no points inside
            return np.zeros(len(x), dtype=bool)


def _compute_sector_centroids(polygons: dict[str, Any]) -> dict[str, np.ndarray]:
    """Compute centroid for each polygon sector."""
    centroids = {}
    for name, poly in polygons.items():
        if hasattr(poly, "path") and hasattr(poly.path, "vertices"):
            vertices = np.asarray(poly.path.vertices)
        else:
            # Fallback for other polygon types
            vertices = np.asarray(poly)
        centroids[name] = vertices.mean(axis=0)
    return centroids


def _apply_distance_fade_per_cluster(
    prior_weights: np.ndarray,
    points: np.ndarray,
    sector_mask: np.ndarray,
    centroid: np.ndarray,
    sector_prior_vec: np.ndarray,  # The base probabilities for this sector
    method: str = "idw",
    method_kws: dict | None = None,
) -> np.ndarray:
    """Apply distance-based fading to all cluster probabilities within a sector."""

    available_methods = ("idw", "linear", "exponential")
    if method not in available_methods:
        raise ValueError(
            f"Unknown fading method: {method}. Available methods: {available_methods}"
        )

    if method_kws is None:
        method_kws = {}

    if not np.any(sector_mask):
        return prior_weights

    # Only compute distances for points INSIDE the sector
    sector_points = points[sector_mask]
    distances = cdist(sector_points, centroid.reshape(1, -1)).ravel()

    if method == "idw":
        power = method_kws.get("power", 2.0)
        eps = 1e-6
        weights = 1.0 / (distances + eps) ** power

    elif method == "linear":
        max_dist = method_kws.get("max_distance")
        if max_dist is not None:
            max_dist = distances.max() if len(distances) > 0 else 1.0
        weights = 1.0 - (distances / max_dist)

    elif method == "exponential":
        decay_rate = method_kws.get("decay_rate", 0.001)
        weights = np.exp(-decay_rate * distances)

    else:
        raise ValueError(f"Unknown fading method: {method}")

    # Normalize weights to [0, 1]
    if weights.max() > 0:
        weights = weights / weights.max()

    # Apply fading to ALL cluster probabilities within this sector
    faded_weights = prior_weights.copy()

    # For points in this sector, interpolate between uniform and sector-specific probabilities
    uniform = np.ones(len(sector_prior_vec)) / len(sector_prior_vec)

    for i, point_weight in enumerate(weights):
        point_idx = np.where(sector_mask)[0][i]  # Get original point index
        # Interpolate: weight=1 -> sector_prior_vec, weight=0 -> uniform
        faded_weights[point_idx, :] = (
            point_weight * sector_prior_vec + (1 - point_weight) * uniform
        )

    return faded_weights


def assign_spatial_priors(
    x: np.ndarray,
    y: np.ndarray,
    polygons: dict[str, Any],
    prior_probs: dict[str, list[float]],
    *,
    fade_method: str = "constant",
    fade_options: dict | None = None,
) -> np.ndarray:
    """Assign spatial prior probabilities to each point."""

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    pts = np.column_stack((x, y))
    sector_names = list(polygons.keys())
    n_sectors = len(sector_names)
    ndata = len(x)

    prior_vecs = _validate_and_normalize_prior_vecs(prior_probs, sector_names)

    # Initialize with uniform probabilities
    uniform = np.ones(n_sectors, dtype=float) / n_sectors
    prior_probs_arr = np.tile(uniform, (ndata, 1))

    if fade_method == "constant":
        # Original binary assignment
        for name, polygon in polygons.items():
            mask = _polygon_contains_mask(polygon, pts, x, y)
            prior_probs_arr[mask, :] = prior_vecs[name]

    else:
        # Distance-based fading methods
        centroids = _compute_sector_centroids(polygons)

        # Apply fading for each sector independently
        for name, polygon in polygons.items():
            mask = _polygon_contains_mask(polygon, pts, x, y)
            if np.any(mask):
                centroid = centroids[name]
                sector_prior_vec = prior_vecs[name]
                prior_probs_arr = _apply_distance_fade_per_cluster(
                    prior_probs_arr,
                    pts,
                    mask,
                    centroid,
                    sector_prior_vec,
                    method=fade_method,
                    method_kws=fade_options,
                )

    # Ensure rows sum to 1
    row_sums = prior_probs_arr.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    prior_probs_arr = prior_probs_arr / row_sums

    return prior_probs_arr


def plot_spatial_priors(
    df: pd.DataFrame,
    prior_probs: np.ndarray | Image.Image,
    img: np.ndarray | None = None,
    cmap: str = "Reds",
    point_size: float = 1,
    alpha: float = 0.7,
    figsize_per_panel: tuple[float, float] = (4, 4),
) -> tuple[Figure, np.ndarray | Axes]:
    """Plot spatial prior probability maps for each cluster.

    Args:
        df: DataFrame with columns ``x`` and ``y`` for point coordinates.
        prior_probs: Array of shape ``(n_points, k)`` with prior probabilities per cluster.
        img: Optional background image array to show under the scatter.
        cmap: Colormap name for priors (default ``'Reds'``).
        point_size: Scatter point size.
        alpha: Scatter alpha.
        figsize_per_panel: Tuple ``(width, height)`` per panel in inches.

    Returns:
        Tuple[Figure, Union[np.ndarray, Axes]]: Matplotlib figure and axes grid.
    """

    prior_probs = np.asarray(prior_probs)
    n_points, k = prior_probs.shape

    nrows = int(np.ceil(np.sqrt(k)))
    ncols = int(np.ceil(k / nrows))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
    )
    # Normalize axes to flat list for easy indexing
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for cluster in range(k):
        ax = axes_flat[cluster]
        if img is not None:
            ax.imshow(img, alpha=0.3)
        scatter = ax.scatter(
            df["x"],
            df["y"],
            c=prior_probs[:, cluster],
            cmap=cmap,
            s=point_size,
            alpha=alpha,
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"Prior for Cluster {cluster}")
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.colorbar(scatter, ax=ax)

    # Hide any unused axes
    total_panels = nrows * ncols
    if total_panels > k:
        for ia in range(k, total_panels):
            axes_flat[ia].axis("off")

    plt.tight_layout()
    return fig, axes
