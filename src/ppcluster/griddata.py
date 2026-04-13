import logging
from pathlib import Path
from typing import Literal, overload

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, label

logger = logging.getLogger("ppcx")

# === GRID from/to scattered points ===


def _infer_grid_spacing(x_vals: np.ndarray, y_vals: np.ndarray) -> float:
    """Infer a reasonable grid spacing from scattered points."""
    x_unique = np.unique(x_vals)
    y_unique = np.unique(y_vals)

    def _min_positive_diff(arr: np.ndarray) -> float:
        if arr.size < 2:
            return 1.0
        diffs = np.diff(arr)
        diffs = diffs[diffs > 0]
        return float(diffs.min()) if diffs.size else 1.0

    spacing = float(
        np.mean([_min_positive_diff(x_unique), _min_positive_diff(y_unique)])
    )
    return spacing if spacing > 0 else 1.0


@overload
def create_2d_grid(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray | None = None,
    grid_spacing: float | None = None,
    *,
    return_axes: Literal[False] = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


@overload
def create_2d_grid(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray | None = None,
    grid_spacing: float | None = None,
    *,
    return_axes: Literal[True],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


def create_2d_grid(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray | None = None,
    grid_spacing: float | None = None,
    *,
    return_axes: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    if grid_spacing is None:
        grid_spacing = _infer_grid_spacing(np.asarray(x, float), np.asarray(y, float))
        logger.info(f"Estimated grid spacing: {grid_spacing:.2f}")

    x = np.asarray(x, float)
    y = np.asarray(y, float)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_grid = np.arange(x_min, x_max + grid_spacing, grid_spacing)
    y_grid = np.arange(y_min, y_max + grid_spacing, grid_spacing)

    X, Y = np.meshgrid(x_grid, y_grid)
    label_grid = np.full(X.shape, np.nan)

    if labels is not None:
        labels_arr = np.asarray(labels)
        for xi, yi, lab in zip(x, y, labels_arr, strict=False):
            ix = np.argmin(np.abs(x_grid - xi))
            iy = np.argmin(np.abs(y_grid - yi))
            label_grid[iy, ix] = lab

    if return_axes:
        return X, Y, label_grid, x_grid, y_grid
    return X, Y, label_grid


def create_2d_grid_from_df(
    df: pd.DataFrame, grid_spacing: float | None = None
) -> tuple:
    """
    Create a 2D grid from scattered DIC points.

    Args:
        df: DataFrame with columns 'x', 'y', 'u', 'v', 'V'
        grid_spacing: Spacing between grid points. If None, estimated from data

    Returns:
        tuple: (x_grid, y_grid, u_grid, v_grid, v_mag_grid, valid_mask)
    """
    X, Y, _, x_grid, y_grid = create_2d_grid(
        df["x"].to_numpy(),
        df["y"].to_numpy(),
        labels=None,
        grid_spacing=grid_spacing,
        return_axes=True,
    )

    u_grid = np.full(X.shape, np.nan, dtype=float)
    v_grid = np.full(X.shape, np.nan, dtype=float)
    v_mag_grid = np.full(X.shape, np.nan, dtype=float)

    for xi, yi, ui, vi, vmag in df[["x", "y", "u", "v", "V"]].itertuples(
        index=False, name=None
    ):
        ix = int(np.argmin(np.abs(x_grid - xi)))
        iy = int(np.argmin(np.abs(y_grid - yi)))
        u_grid[iy, ix] = ui
        v_grid[iy, ix] = vi
        v_mag_grid[iy, ix] = vmag

    valid_mask = ~np.isnan(v_mag_grid)
    logger.info(f"Created 2D grid: {X.shape}, {int(np.sum(valid_mask))} valid points")

    return X, Y, u_grid, v_grid, v_mag_grid, valid_mask


def map_grid_to_points(
    X,
    Y,
    label_grid,
    x_points,
    y_points,
    keep_nan=True,
    nan_fill=-1,
):
    """
    Map values from a 2D grid (label_grid) back to query points.

    Parameters:
    -----------
    X, Y : numpy.ndarray
        Meshgrid arrays returned by create_2d_grid (X.shape == label_grid.shape)
    label_grid : numpy.ndarray
        2D array with labels (np.nan = empty cells)
    x_points, y_points : numpy.ndarray
        1D arrays of query point coordinates
    keep_nan : bool, default=True
        If True, keep NaN points and fill with nan_fill value
        If False, filter out NaN points and return filtered coordinates
    nan_fill : int, default=-1
        Value to use when grid cell is NaN (only used if keep_nan=True)

    Returns:
    --------
        tuple : (labels, x, y)
            - label: Array of mapped labels (excluding NaN points if keep_nan=False)
            - x: Array of x coordinates (excluding NaN points if keep_nan=False)
            - y: Array of y coordinates (excluding NaN points if keep_nan=False)
    """
    # Ensure inputs are numpy arrays
    x_points = np.asarray(x_points)
    y_points = np.asarray(y_points)

    if len(x_points) != len(y_points):
        raise ValueError("x_points and y_points must have the same length")

    # Extract grid coordinates
    x_grid = X[0, :]
    y_grid = Y[:, 0]

    # Initialize output array
    n_points = len(x_points)
    labels = np.full(n_points, nan_fill, dtype=int)

    # Map each point to nearest grid cell
    for i, (xi, yi) in enumerate(zip(x_points, y_points, strict=False)):
        # Find nearest grid indices
        ix = np.argmin(np.abs(x_grid - xi))
        iy = np.argmin(np.abs(y_grid - yi))

        # Get grid value
        val = label_grid[iy, ix]
        labels[i] = nan_fill if np.isnan(val) else int(val)

    if not keep_nan:
        # Filter out NaN points
        valid_mask = labels != nan_fill
        labels = labels[valid_mask]
        x_points = x_points[valid_mask]
        y_points = y_points[valid_mask]

    return labels, x_points, y_points


# === GRID data filtering ===


def apply_cluster_grid_cleaning(
    X: np.ndarray,
    Y: np.ndarray,
    clusters: np.ndarray,
    config: dict,
    output_dir: Path,
    base_name: str,
    img: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply morphological operations and cleaning to the cluster grid.
    Refines clusters, removing noise and small components.

    Args:
        X, Y: 2D meshgrid arrays of point coordinates
        clusters: 2D array of cluster labels on the grid
        config: post-processing configuration parameters
        output_dir: directory to save outputs
        base_name: base name for output files
        img: optional background image for plotting

    Returns:
        Refined cluster grid as 2D numpy array.
    """

    # Retrieve post-processing parameters
    do_split = config.get("split_disconnected_components", True)
    erosion_iters = config.get("erosion_iterations", 0)
    dilation_iters = config.get("dilation_iterations", 0)
    connectivity = config.get("connectivity", 8)
    min_cluster_size = config.get("min_cluster_size", 0)
    keep_only_largest_n = config.get("keep_only_largest_n", 0)

    logger.info(
        f"Post-proc params: erosion={erosion_iters}, "
        f"dilation={dilation_iters}, min_size={min_cluster_size}"
    )

    # Store pre-postprocessing grid for comparison
    cluster_before = clusters.copy()

    # Split disconnected components first
    if do_split:
        clusters, _ = split_disconnected_components(
            clusters,
            connectivity=connectivity,
            start_label=0,
        )

    # Remove very small components and merge to nearest neighbor
    clusters = remove_small_grid_components(
        label_grid=clusters,
        min_size=20,  # initial removal threshold to clean noise (hard-coded)
        connectivity=connectivity,
        merge_strategy="merge",  # merge small components to nearest neighbor
    )

    # Apply morphological operations (erosion + dilation)
    if erosion_iters > 0 or dilation_iters > 0:
        clusters = apply_morphological_operations(
            cluster_grid=clusters,
            erosion_iterations=erosion_iters,
            dilation_iterations=dilation_iters,
            min_cluster_size=min_cluster_size,
            connectivity=connectivity,
        )
    # Remove small components again after morph operations (do not merge)
    if min_cluster_size > 0:
        clusters = remove_small_grid_components(
            label_grid=clusters,
            min_size=min_cluster_size,
            connectivity=connectivity,
            merge_strategy="remove",  # or "merge" to assign to nearest neighbor
        )

    # Keep only N largest clusters (on grid)
    if keep_only_largest_n > 0:
        clusters = keep_only_largest_clusters(
            label_grid=clusters,
            n_largest=keep_only_largest_n,
            connectivity=connectivity,
        )

    # Plot comparison before/after post-processing
    if img is not None:
        fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(12, 6))
        plot_clustering_grid(
            ax=ax_before,
            img=img,
            cluster_grid=cluster_before,
            X=X,
            Y=Y,
            title="Before Post-Processing",
            show_legend=True,
            show_stats=True,
            alpha=0.5,
        )
        plot_clustering_grid(
            ax=ax_after,
            img=img,
            cluster_grid=clusters,
            X=X,
            Y=Y,
            title="After Post-Processing",
            show_legend=True,
            show_stats=True,
            alpha=0.5,
        )
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{base_name}_kinematic_clustering_postproc.jpg",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    return clusters


def remove_small_grid_components(
    label_grid: np.ndarray,
    min_size: int = 5,
    connectivity: int = 4,
    merge_strategy: str = "remove",
) -> np.ndarray:
    """
    Remove small connected components from a 2D label grid.

    This function identifies disconnected components within each cluster label
    and removes those smaller than min_size.

    Args:
        label_grid: 2D numpy array with cluster labels (negative values = unassigned)
        min_size: Minimum size (in grid cells) to keep a component
        connectivity: 4 or 8 (neighbor connectivity for component detection)
        merge_strategy: 'remove' (set to -1) or 'merge' (assign to nearest neighbor)

    Returns:
        cleaned_grid: 2D array with small components removed/merged
    """
    from scipy import ndimage

    cleaned = label_grid.copy().astype(float)

    # Define connectivity structure
    if connectivity == 8:
        structure = np.ones((3, 3), dtype=bool)
    else:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    # Get unique cluster labels (excluding negative/unassigned)
    unique_labels = np.unique(label_grid)
    unique_labels = unique_labels[unique_labels >= 0]

    if len(unique_labels) == 0:
        logger.warning("No valid clusters found in grid")
        return cleaned

    total_removed = 0

    # Process each cluster separately to find disconnected components
    for lab in unique_labels:
        mask = label_grid == lab
        if not np.any(mask):
            continue

        # Find connected components for this cluster
        components, n_components = ndimage.label(mask, structure=structure)

        # Process each component
        for comp_id in range(1, n_components + 1):
            comp_mask = components == comp_id
            comp_size = comp_mask.sum()

            if comp_size < min_size:
                if merge_strategy == "merge":
                    # Try to merge with neighboring cluster
                    # Dilate component to find neighbors
                    neighbor_mask = ndimage.binary_dilation(
                        comp_mask, structure=np.ones((3, 3))
                    ) & (~comp_mask)

                    neighbor_labels = cleaned[neighbor_mask]
                    # Filter: exclude current label and negative values
                    neighbor_labels = neighbor_labels[neighbor_labels >= 0]
                    neighbor_labels = neighbor_labels[neighbor_labels != lab]

                    if neighbor_labels.size > 0:
                        # Assign to most common neighbor
                        unique, counts = np.unique(neighbor_labels, return_counts=True)
                        new_label = unique[np.argmax(counts)]
                        cleaned[comp_mask] = new_label
                        total_removed += comp_size
                        logger.debug(
                            f"Merged small component (size={comp_size}) "
                            f"from cluster {lab} to cluster {new_label}"
                        )
                    else:
                        # No neighbors, remove it
                        cleaned[comp_mask] = -1
                        total_removed += comp_size
                else:
                    # Remove strategy: set to unassigned (-1)
                    cleaned[comp_mask] = -1
                    total_removed += comp_size
                    logger.debug(
                        f"Removed small component (size={comp_size}) from cluster {lab}"
                    )

    # Log summary
    final_unique = np.unique(cleaned)
    final_unique = final_unique[final_unique >= 0]
    n_unassigned = np.sum(cleaned < 0)

    logger.info(
        f"Removed {total_removed} grid cells from small components. "
        f"Result: {len(final_unique)} clusters, {n_unassigned} unassigned cells"
    )

    return cleaned


def keep_only_largest_clusters(
    label_grid: np.ndarray,
    n_largest: int,
    connectivity: int = 4,
) -> np.ndarray:
    """
    Keep only the N largest clusters in a 2D label grid.

    This function calculates the total size of each cluster (sum of all connected
    components for that label) and keeps only the N largest by total grid cell count.

    Args:
        label_grid: 2D numpy array with cluster labels (negative values = unassigned)
        n_largest: Number of largest clusters to keep
        connectivity: 4 or 8 (neighbor connectivity, used for logging only)

    Returns:
        filtered_grid: 2D array with only the N largest clusters retained
    """
    if n_largest <= 0:
        logger.warning("n_largest must be > 0, returning unchanged grid")
        return label_grid.copy()

    cleaned = label_grid.copy().astype(float)

    # Get unique cluster IDs (excluding negative/unassigned)
    unique_labels = np.unique(label_grid)
    unique_labels = unique_labels[unique_labels >= 0]

    if len(unique_labels) == 0:
        logger.warning("No valid clusters found in grid")
        return cleaned

    if len(unique_labels) <= n_largest:
        logger.info(
            f"Found {len(unique_labels)} clusters, no filtering needed "
            f"(n_largest={n_largest})"
        )
        return cleaned

    # Calculate total size for each cluster
    cluster_sizes = {}
    for cluster_id in unique_labels:
        cluster_sizes[cluster_id] = np.sum(label_grid == cluster_id)

    # Sort by size (descending) and keep largest N
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)

    largest_clusters = set([cid for cid, _ in sorted_clusters[:n_largest]])

    # Remove all clusters not in the largest N
    for cluster_id in unique_labels:
        if cluster_id not in largest_clusters:
            cleaned[label_grid == cluster_id] = -1

    # Calculate statistics
    n_removed_clusters = len(unique_labels) - n_largest
    total_removed_cells = sum(size for cid, size in sorted_clusters[n_largest:])
    total_kept_cells = sum(size for cid, size in sorted_clusters[:n_largest])

    logger.info(
        f"Kept {n_largest} largest clusters ({total_kept_cells} cells), "
        f"removed {n_removed_clusters} clusters ({total_removed_cells} cells)"
    )

    # Log details of kept clusters
    for idx, (cid, size) in enumerate(sorted_clusters[:n_largest], 1):
        logger.info(f"  #{idx}: Cluster {cid} with {size} grid cells")

    # Log details of removed clusters
    if n_removed_clusters > 0:
        logger.debug("Removed clusters:")
        for cid, size in sorted_clusters[n_largest:]:
            logger.debug(f"  Cluster {cid} with {size} grid cells")

    return cleaned


def close_small_holes(
    label_grid,
    max_hole_size=10,
    connectivity=8,
    require_single_neighbor=True,
):
    """
    Close small NaN-holes in a 2D label grid.

    Rules:
      - Only close holes whose size (number of NaN cells) <= max_hole_size.
      - Only close if the dilated border of the hole contains no NaNs
        (i.e. data all around the hole).
      - If require_single_neighbor is True the border must contain a single unique
        label; otherwise the most common border label is used.

    Args:
      label_grid: 2D ndarray with labels (NaN for empty cells).
      max_hole_size: maximum hole area (in grid cells) to fill.
      connectivity: 4 or 8 connectivity for labeling/dilation.
      require_single_neighbor: if True, require single neighbor label on border.

    Returns:
      new_grid: copy of label_grid with selected holes filled.
    """
    new_grid = label_grid.copy().astype(float)
    if connectivity == 8:
        structure = np.ones((3, 3), dtype=bool)
    else:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    # mask of holes (NaNs)
    hole_mask_all = np.isnan(label_grid)
    if not np.any(hole_mask_all):
        logger.debug("close_small_holes: no holes found (no NaNs in grid).")
        return new_grid

    # label each hole component
    comp_labels, ncomp = ndimage.label(hole_mask_all, structure=structure)
    filled_count = 0
    skipped_count = 0
    for comp_id in range(1, ncomp + 1):
        comp_mask = comp_labels == comp_id
        comp_size = int(comp_mask.sum())
        reason = None
        if comp_size > max_hole_size:
            reason = f"size>{max_hole_size}"
            skipped_count += 1
            logger.debug(f"hole {comp_id}: size={comp_size} skipped ({reason})")
            continue

        # dilate to get border cells (use same connectivity structure)
        dilated = ndimage.binary_dilation(comp_mask, structure=structure)
        border_mask = dilated & (~comp_mask)

        # border values
        border_vals = new_grid[border_mask]
        if border_vals.size == 0:
            reason = "no border cells"
            skipped_count += 1
            logger.debug(f"hole {comp_id}: size={comp_size} skipped ({reason})")
            continue

        # if any border cell is NaN -> not fully surrounded by data
        if np.any(np.isnan(border_vals)):
            reason = "border_has_nans"
            skipped_count += 1
            logger.debug(
                f"hole {comp_id}: size={comp_size} skipped ({reason}) border_nan_fraction={np.isnan(border_vals).mean():.3f}"
            )
            continue

        # get unique neighbor labels and counts
        unique_neighbors, counts = np.unique(border_vals, return_counts=True)
        if unique_neighbors.size == 0:
            reason = "no_neighbors"
            skipped_count += 1
            logger.debug(f"hole {comp_id}: size={comp_size} skipped ({reason})")
            continue

        if require_single_neighbor:
            # require border to be all same label
            if unique_neighbors.size == 1:
                fill_label = unique_neighbors[0]
            else:
                reason = f"multiple_neighbors({unique_neighbors.tolist()})"
                skipped_count += 1
                logger.debug(f"hole {comp_id}: size={comp_size} skipped ({reason})")
                continue
        else:
            # pick most common neighbor label
            fill_label = unique_neighbors[np.argmax(counts)]

        # sanity: ensure fill_label is finite
        if not np.isfinite(fill_label):
            reason = "fill_label_not_finite"
            skipped_count += 1
            logger.debug(f"hole {comp_id}: size={comp_size} skipped ({reason})")
            continue

        # coerce label to int-like if appropriate (avoid filling with floats like 1.0)
        if np.allclose(fill_label, np.round(fill_label)):
            fill_label = float(int(np.round(fill_label)))
        else:
            fill_label = float(fill_label)

        # fill hole with chosen label
        new_grid[comp_mask] = fill_label
        filled_count += 1
        logger.debug(
            f"hole {comp_id}: size={comp_size} filled with label={fill_label} (require_single_neighbor={require_single_neighbor})"
        )

    logger.info(
        f"close_small_holes: filled={filled_count} skipped={skipped_count} total={ncomp}"
    )
    return new_grid


def split_disconnected_components(label_grid, connectivity=8, start_label=0):
    """
    Split disconnected components of each label into unique labels.

    Args:
      label_grid: 2D array with labels (NaN = empty).
      connectivity: 4 or 8 connectivity for ndimage.label.
      start_label: integer to start new labels from.
      debug: if True logs summary.

    Returns:
      new_grid: 2D array with disconnected pieces assigned unique integer labels.
      mapping: dict original_label -> list of new labels assigned for its pieces.
    """
    if connectivity == 8:
        structure = np.ones((3, 3), dtype=bool)
    else:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    new_grid = np.full_like(label_grid, np.nan, dtype=float)
    mapping = {}
    next_label = int(start_label)

    unique_labels = np.unique(label_grid[~np.isnan(label_grid)])
    for lab in unique_labels:
        lab = float(lab)
        mask = label_grid == lab
        if not np.any(mask):
            continue
        comp, ncomp = ndimage.label(mask, structure=structure)
        mapping[int(lab)] = []
        for cid in range(1, ncomp + 1):
            comp_mask = comp == cid
            new_grid[comp_mask] = float(next_label)
            mapping[int(lab)].append(int(next_label))
            next_label += 1

    total_new = next_label - int(start_label)
    logger.debug(
        f"split_disconnected_components: original_labels={unique_labels.size} new_pieces={total_new} mapping={mapping}",
    )
    return new_grid, mapping


def apply_morphological_operations(
    cluster_grid: np.ndarray,
    erosion_iterations: int = 2,
    dilation_iterations: int = 2,
    min_cluster_size: int = 100,
    connectivity: int = 8,
) -> np.ndarray:
    """
    Apply morphological operations (erosion + dilation) to disconnect narrow bridges
    and remove small clusters.

    Operations performed per cluster:
    1. Erosion: disconnect narrow bridges and remove small protrusions
    2. Component labeling: identify separated parts after erosion
    3. Size filtering: remove components smaller than threshold
    4. Dilation: restore cluster size without reconnecting separated parts

    Args:
        cluster_grid: 2D array with cluster labels (negative values = unassigned)
        erosion_iterations: Number of erosion iterations to disconnect narrow bridges
        dilation_iterations: Number of dilation iterations to restore cluster size
        min_cluster_size: Minimum number of pixels for a cluster component to survive
        structure: Structuring element for morphological operations (default: 3x3 ones)

    Returns:
        Processed cluster grid with same shape as input
    """

    if connectivity == 8:
        structure = np.ones((3, 3), dtype=bool)
    else:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    logger.info(
        f"Applying morphological operations: erosion={erosion_iterations}, "
        f"dilation={dilation_iterations}, min_size={min_cluster_size}"
    )

    # Get unique cluster labels (excluding negative = unassigned)
    unique_labels = np.unique(cluster_grid)
    unique_labels = unique_labels[unique_labels >= 0]

    # Process each cluster separately
    processed_grid = np.full_like(cluster_grid, -1, dtype=int)

    for cluster_id in sorted(unique_labels):
        # Create binary mask for this cluster
        cluster_mask = (cluster_grid == cluster_id).astype(np.uint8)

        # Step 1: Erosion to disconnect narrow bridges
        if erosion_iterations > 0:
            eroded_mask = binary_erosion(
                cluster_mask, iterations=erosion_iterations, structure=structure
            ).astype(np.uint8)
        else:
            eroded_mask = cluster_mask

        # Step 2: Label connected components after erosion
        labeled_eroded, n_components = label(eroded_mask, structure=structure)

        if n_components == 0:
            continue

        # Step 4: Dilation to restore size (but not reconnect separated parts)
        if dilation_iterations > 0:
            # Dilate each component separately to avoid reconnection
            dilated_mask = np.zeros_like(labeled_eroded, dtype=np.uint8)
            remaining_components = np.unique(labeled_eroded)
            remaining_components = remaining_components[remaining_components > 0]

            for comp_id in remaining_components:
                comp_mask = (labeled_eroded == comp_id).astype(np.uint8)
                dilated_comp = binary_dilation(
                    comp_mask, iterations=dilation_iterations, structure=structure
                ).astype(np.uint8)
                # Add to final mask (max to handle overlaps)
                dilated_mask = np.maximum(dilated_mask, dilated_comp)

            final_mask = dilated_mask
        else:
            final_mask = (labeled_eroded > 0).astype(np.uint8)

        # Add processed cluster to output grid
        processed_grid[final_mask > 0] = cluster_id

    n_clusters_before = len(unique_labels)
    n_clusters_after = len(np.unique(processed_grid[processed_grid >= 0]))
    logger.info(
        f"Morphological operations: {n_clusters_before} → {n_clusters_after} clusters"
    )

    return processed_grid


# === Visualization ===


def plot_clustering(
    ax,
    img: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    cluster_labels: np.ndarray,
    title: str = "",
    colormap_name: str = "tab10",
    show_legend: bool = True,
    show_stats: bool = False,
    point_size: int = 10,
    alpha: float = 0.7,
) -> None:
    """
    Plot clustering results on a given matplotlib axis.

    Args:
        ax: Matplotlib axis to plot on
        img: Background image to display
        x: X coordinates of points
        y: Y coordinates of points
        cluster_labels: Cluster assignments for each point
        title: Plot title
        colormap_name: Name of matplotlib colormap to use
        show_legend: Whether to show legend
        show_stats: Whether to show statistics text box
        point_size: Size of scatter plot points
        alpha: Transparency of scatter points
    """
    # Display background image
    ax.imshow(img, alpha=0.5, cmap="gray")

    # Get colormap
    colormap = plt.get_cmap(colormap_name)

    # Get unique cluster labels (excluding negative = unassigned)
    unique_labels = np.unique(cluster_labels)
    unique_labels = unique_labels[unique_labels >= 0]

    # Plot each cluster
    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        ax.scatter(
            x[mask],
            y[mask],
            color=colormap(i % colormap.N),
            label=f"Cluster {label}",
            s=point_size,
            alpha=alpha,
        )

    # Add legend if requested
    if show_legend and len(unique_labels) > 0:
        ax.legend(loc="upper right", framealpha=0.9, fontsize=10)

    # Add statistics text box if requested
    if show_stats:
        n_clusters = len(unique_labels)
        n_points = len(x)
        stats_text = f"Clusters: {n_clusters}\nPoints: {n_points}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Set title and formatting
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_clustering_grid(
    ax,
    img: np.ndarray,
    cluster_grid: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    title: str = "",
    colormap_name: str = "tab10",
    show_legend: bool = True,
    show_stats: bool = False,
    alpha: float = 0.7,
) -> None:
    """
    Plot gridded clustering results on a matplotlib axis.

    Args:
        ax: Matplotlib axis to plot on
        img: Background image to display
        cluster_grid: 2D array with cluster IDs (negative = unassigned)
        X: 2D meshgrid of X coordinates
        Y: 2D meshgrid of Y coordinates
        title: Plot title
        colormap_name: Name of matplotlib colormap to use
        show_legend: Whether to show legend
        show_stats: Whether to show statistics text box
        alpha: Transparency of colored overlay
    """

    # Display background image
    ax.imshow(img, alpha=0.5, cmap="gray")

    # Get unique cluster labels (excluding negative = unassigned)
    unique_labels = np.unique(cluster_grid)
    unique_labels = unique_labels[unique_labels >= 0]
    unique_labels = sorted(unique_labels)

    if len(unique_labels) == 0:
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # Get base colormap
    base_cmap = plt.get_cmap(colormap_name)

    # Create a masked array for visualization
    # Mask out negative values (unassigned)
    cluster_masked = np.ma.masked_where(cluster_grid < 0, cluster_grid)

    # Create color mapping: map each cluster_id to a color index
    # This ensures consistent colors across plots
    n_colors = len(unique_labels)
    colors = [base_cmap(i % base_cmap.N) for i in range(n_colors)]

    # Create a custom colormap
    custom_cmap = ListedColormap(colors)

    # Map cluster IDs to sequential indices for color mapping
    cluster_to_idx = {cid: idx for idx, cid in enumerate(unique_labels)}
    cluster_indexed = np.full_like(cluster_grid, -1, dtype=float)
    for cid, idx in cluster_to_idx.items():
        cluster_indexed[cluster_grid == cid] = idx

    cluster_indexed_masked = np.ma.masked_where(cluster_indexed < 0, cluster_indexed)

    # Plot the grid with colors
    im = ax.imshow(
        cluster_indexed_masked,
        cmap=custom_cmap,
        alpha=alpha,
        interpolation="nearest",
        vmin=0,
        vmax=n_colors - 1,
        extent=[X.min(), X.max(), Y.max(), Y.min()],  # Y reversed for image coords
    )

    # Add legend if requested
    if show_legend:
        legend_patches = []
        for cid in unique_labels:
            idx = cluster_to_idx[cid]
            color = colors[idx]
            patch = mpatches.Patch(color=color, label=f"Cluster {cid}", alpha=alpha)
            legend_patches.append(patch)

        ax.legend(
            handles=legend_patches,
            loc="upper right",
            framealpha=0.9,
            fontsize=9,
        )

    # Add statistics text box if requested
    if show_stats:
        n_clusters = len(unique_labels)
        n_cells = np.sum(cluster_grid >= 0)
        stats_text = f"Clusters: {n_clusters}\nCells: {n_cells}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Set title and formatting
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.max(), Y.min())  # Y reversed for image coordinates
    ax.set_xticks([])
    ax.set_yticks([])
