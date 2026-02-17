import logging

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.stats import binned_statistic
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger("ppcx")


def _compute_bins(
    x,
    y,
    horizontal_bins=None,
    vertical_bins=None,
    bin_edges_h=None,
    bin_centers_h=None,
    bin_edges_v=None,
    bin_centers_v=None,
):
    """Create horizontal / vertical bin edges and centers if not provided."""
    if bin_edges_h is None or bin_centers_h is None:
        if horizontal_bins is None:
            raise ValueError("Either horizontal_bins or bin_edges_h must be provided")
        bin_edges_h = np.linspace(x.min(), x.max(), horizontal_bins + 1)
        bin_centers_h = 0.5 * (bin_edges_h[:-1] + bin_edges_h[1:])
    if bin_edges_v is None or bin_centers_v is None:
        if vertical_bins is None:
            raise ValueError("Either vertical_bins or bin_edges_v must be provided")
        bin_edges_v = np.linspace(y.min(), y.max(), vertical_bins + 1)
        bin_centers_v = 0.5 * (bin_edges_v[:-1] + bin_edges_v[1:])
    return bin_edges_h, bin_centers_h, bin_edges_v, bin_centers_v


def _collect_column_detections(
    x,
    y,
    v,
    bin_edges_h,
    bin_centers_h,
    bin_edges_v,
    bin_centers_v,
    vertical_bins,
    min_points_per_bin_col,
    smoothing_sigma_1d,
    gradient_threshold_factor,
    min_strength,
):
    """
    Scan each horizontal column, compute a smoothed vertical profile and find
    local negative-gradient peaks (candidate discontinuities). Returns arrays
    of detection x,y,strength and the column index they came from.
    """
    all_x, all_y, all_strengths, all_col_indices = [], [], [], []
    v_bin_width = np.mean(np.diff(bin_centers_v))

    for col in range(len(bin_centers_h)):
        # select points inside horizontal bin
        x0, x1 = bin_edges_h[col], bin_edges_h[col + 1]
        if col == len(bin_centers_h) - 1:
            mask_col = (x >= x0) & (x <= x1)
        else:
            mask_col = (x >= x0) & (x < x1)

        # require enough raw points in the column
        if np.sum(mask_col) < min_points_per_bin_col:
            continue

        # vertical binned count and mean velocity
        col_counts = binned_statistic(
            y[mask_col], np.ones_like(y[mask_col]), statistic="count", bins=bin_edges_v
        ).statistic.astype(int)
        col_mean = binned_statistic(
            y[mask_col], v[mask_col], statistic="mean", bins=bin_edges_v
        ).statistic
        valid_mask_col = col_counts >= 1

        if not np.any(valid_mask_col):
            continue

        # fill invalid bins by nearest valid for smoothing stability
        col_profile = np.copy(col_mean)
        valid_idx = np.where(valid_mask_col)[0]
        for i in range(vertical_bins):
            if not valid_mask_col[i]:
                nearest = valid_idx[np.argmin(np.abs(valid_idx - i))]
                col_profile[i] = col_profile[nearest]

        # smooth and compute gradient dv/dy
        col_profile_smooth = ndimage.gaussian_filter1d(
            col_profile, sigma=smoothing_sigma_1d, mode="nearest"
        )
        grad = np.gradient(col_profile_smooth) / v_bin_width

        # focus negative gradients and find strict local minima
        min_grad = np.min(grad)
        if min_grad >= 0:
            continue
        threshold = gradient_threshold_factor * min_grad  # negative value

        for i in range(1, len(grad) - 1):
            if (
                valid_mask_col[i]
                and grad[i] < threshold
                and grad[i] < grad[i - 1]
                and grad[i] < grad[i + 1]
            ):
                pos_y = bin_centers_v[i]
                strength = abs(grad[i])
                if strength >= min_strength:
                    all_y.append(pos_y)
                    all_x.append(bin_centers_h[col])
                    all_strengths.append(strength)
                    all_col_indices.append(col)

    # Return numpy arrays for convenience
    return (
        np.array(all_x),
        np.array(all_y),
        np.array(all_strengths),
        np.array(all_col_indices),
    )


def _normalize_border(border, bin_edges_h, bin_edges_v):
    """Normalize border input into inner rectangle coordinates (left,right,bottom,top)."""
    if np.isscalar(border):
        left_b = right_b = bottom_b = top_b = float(border)
    else:
        b = list(border)
        if len(b) == 1:
            left_b = right_b = bottom_b = top_b = float(b[0])
        elif len(b) == 4:
            left_b, right_b, bottom_b, top_b = map(float, b)
        else:
            raise ValueError(
                "border must be scalar or length-4 sequence [left,right,bottom,top]"
            )
    x_min_inner = bin_edges_h[0] + left_b
    x_max_inner = bin_edges_h[-1] - right_b
    y_max_inner = bin_edges_v[-1] - bottom_b
    y_min_inner = bin_edges_v[0] + top_b
    return x_min_inner, x_max_inner, y_min_inner, y_max_inner


def _polygon_contains(poly, pts):
    """Robustly test polygon membership for pts using common APIs."""
    try:
        return np.asarray(poly.contains_points(pts), dtype=bool)
    except Exception:
        try:
            return np.asarray(poly.contains_points(pts[:, 0], pts[:, 1]), dtype=bool)
        except Exception:
            # cannot test membership -> no points inside
            return np.zeros(pts.shape[0], dtype=bool)


def _filter_individuals_and_map_indices(
    XY, strengths, cols, eps, cluster_min_samples, min_strength
):
    """
    Conservative per-detection filtering: strength vs median + local neighbor support
    and minimal column support. Returns filtered arrays and original indices relative
    to the input XY (or None if nothing kept).
    """
    labels_all = -1 * np.ones(XY.shape[0], dtype=int)

    # neighbors within eps for local density check
    nbrs = NearestNeighbors(radius=eps).fit(XY)
    neighbors = nbrs.radius_neighbors(XY, return_distance=False)
    neighbor_counts = np.array([len(nb) for nb in neighbors])

    median_strength = np.median(strengths) if strengths.size else 0.0
    strength_thresh = max(min_strength, 0.3 * median_strength)
    required_neighbors = max(1, cluster_min_samples)

    # column-wise support: ensure detection has at least two detections from same column
    col_support_counts = np.array([np.sum(cols == c) for c in cols])

    keep_mask = (
        (strengths >= strength_thresh) & (neighbor_counts >= required_neighbors)
    ) | (col_support_counts >= 2)

    if not np.any(keep_mask):
        # relax thresholds once
        strength_thresh_relaxed = max(min_strength, 0.2 * median_strength)
        keep_mask = (
            (strengths >= strength_thresh_relaxed) & (neighbor_counts >= 1)
        ) | (col_support_counts >= 2)

    if not np.any(keep_mask):
        return None, None, None, None, labels_all

    original_indices = np.where(keep_mask)[0]
    return (
        XY[keep_mask],
        strengths[keep_mask],
        cols[keep_mask],
        original_indices,
        labels_all,
    )


def _assign_labels_by_sectors_or_dbscan(XY_filtered, sectors, eps, cluster_min_samples):
    """
    If sectors are provided, assign labels by simple polygon containment.
    Remove tiny sector groups (< cluster_min_samples). If no sector labels
    produced, fallback to DBSCAN.
    """
    labels_filtered = -1 * np.ones(XY_filtered.shape[0], dtype=int)
    used_sectors = False

    if sectors is not None and len(sectors) > 0:
        used_sectors = True
        sector_names = list(sectors.keys())
        pts = XY_filtered
        for si, name in enumerate(sector_names):
            poly = sectors[name]
            mask = _polygon_contains(poly, pts)
            to_assign = mask & (labels_filtered == -1)
            labels_filtered[to_assign] = si

        # remove tiny sector groups (not enough support)
        for lab in range(len(sector_names)):
            idxs = np.where(labels_filtered == lab)[0]
            if len(idxs) < cluster_min_samples:
                labels_filtered[idxs] = -1

        if not np.any(labels_filtered != -1):
            used_sectors = False  # nothing assigned -> fallback

    if not used_sectors:
        # DBSCAN fallback
        clustering = DBSCAN(eps=eps, min_samples=cluster_min_samples).fit(XY_filtered)
        labels_filtered = clustering.labels_

    return labels_filtered


def _build_cluster_metadata(
    labels_all, XY, all_strengths, all_cols_arr, cluster_min_samples
):
    """Build cluster summaries (mean position, strength, support columns, raw pts)."""
    clustered_boundaries = []
    present_labels = sorted(set(labels_all[np.where(labels_all != -1)[0]]))
    for lab in present_labels:
        if lab == -1:
            continue
        idxs = np.where(labels_all == lab)[0]
        if idxs.size == 0:
            continue

        pts = XY[idxs]
        strg = np.array(all_strengths)[idxs]
        cols = all_cols_arr[idxs]

        # require cluster to span at least two columns for reliability
        support_cols = np.unique(cols).size
        if support_cols < 2:
            continue

        mean_y = float(np.mean(pts[:, 1]))
        mean_x = float(np.mean(pts[:, 0]))
        mean_strength = float(np.mean(strg))

        clustered_boundaries.append(
            {
                "label": int(lab),
                "position": mean_y,
                "x_mean": mean_x,
                "strength": mean_strength,
                "support": int(support_cols),
                "raw_xy": XY[idxs],
                "raw_strengths": np.array(all_strengths)[idxs],
                "columns": all_cols_arr[idxs],
            }
        )
    return clustered_boundaries


def find_vertical_discontinuities(
    x: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    vertical_bins: int | None = None,
    horizontal_bins: int | None = None,
    bin_edges_h: np.ndarray | None = None,
    bin_centers_h: np.ndarray | None = None,
    bin_edges_v: np.ndarray | None = None,
    bin_centers_v: np.ndarray | None = None,
    min_points_per_bin_col: int = 5,
    gradient_threshold_factor: float = 0.3,
    smoothing_sigma_1d: float = 1.0,
    min_strength: float = 1e-3,
    cluster_eps_factor: float = 1.5,
    cluster_min_samples: int = 3,
    border: float | list[float] | tuple[float] = 0.0,
    sectors: dict | None = None,
):
    """
    Wrapper that detects per-column discontinuities and clusters them.
    Added logging on crucial steps: detection counts, border filtering, individual
    filtering, clustering method and final cluster retention.
    """
    logger.info(
        "Start find_vertical_discontinuities: horizontal_bins=%s vertical_bins=%s min_points_per_bin_col=%d",
        horizontal_bins,
        vertical_bins,
        min_points_per_bin_col,
    )

    # prepare bins
    bin_edges_h, bin_centers_h, bin_edges_v, bin_centers_v = _compute_bins(
        x,
        y,
        horizontal_bins,
        vertical_bins,
        bin_edges_h,
        bin_centers_h,
        bin_edges_v,
        bin_centers_v,
    )
    v_bin_width = np.mean(np.diff(bin_centers_v))
    if vertical_bins is None:
        vertical_bins = len(bin_centers_v)

    # collect per-column candidate detections
    all_x, all_y, all_strengths, all_col_indices = _collect_column_detections(
        x,
        y,
        v,
        bin_edges_h,
        bin_centers_h,
        bin_edges_v,
        bin_centers_v,
        vertical_bins,
        min_points_per_bin_col,
        smoothing_sigma_1d,
        gradient_threshold_factor,
        min_strength,
    )
    logger.info("Per-column detection candidates: %d", all_x.size)

    if all_x.size == 0:
        logger.info("No per-column detections found -> returning empty result.")
        return {
            "individual": {
                "x": np.array([]),
                "y": np.array([]),
                "strength": np.array([]),
                "col_idx": np.array([]),
                "labels": np.array([], dtype=int),
            },
            "clustered": [],
        }

    # prepare arrays and border mask
    XY = np.column_stack((all_x, all_y))
    all_strengths_arr = np.array(all_strengths)
    all_cols_arr = np.array(all_col_indices)

    x_min_inner, x_max_inner, y_min_inner, y_max_inner = _normalize_border(
        border, bin_edges_h, bin_edges_v
    )
    inside_mask = (
        (XY[:, 0] >= x_min_inner)
        & (XY[:, 0] <= x_max_inner)
        & (XY[:, 1] >= y_min_inner)
        & (XY[:, 1] <= y_max_inner)
    )

    n_inside = int(np.sum(inside_mask))
    logger.info(
        "Detections inside inner border: %d (inner box: x[%0.1f,%0.1f] y[%0.1f,%0.1f])",
        n_inside,
        x_min_inner,
        x_max_inner,
        y_min_inner,
        y_max_inner,
    )

    labels_all = -1 * np.ones(XY.shape[0], dtype=int)
    if not np.any(inside_mask):
        logger.info("No detections inside border -> returning all as outliers.")
        return {
            "individual": {
                "x": all_x,
                "y": all_y,
                "strength": all_strengths,
                "col_idx": all_col_indices,
                "labels": labels_all,
            },
            "clustered": [],
        }

    # keep only inside detections for filtering & clustering
    XY_in = XY[inside_mask]
    strengths_in = all_strengths_arr[inside_mask]
    cols_in = all_cols_arr[inside_mask]
    original_indices_in = np.where(inside_mask)[0]

    # clustering radius
    h_bin_width = np.median(np.diff(bin_centers_h))
    eps = max(v_bin_width * cluster_eps_factor, h_bin_width * cluster_eps_factor)

    # conservative individual filtering
    XY_filtered, strengths_filtered, cols_filtered, original_indices, labels_all = (
        _filter_individuals_and_map_indices(
            XY_in, strengths_in, cols_in, eps, cluster_min_samples, min_strength
        )
    )
    if XY_filtered is None:
        logger.info("All detections filtered out as weak/noisy -> returning.")
        return {
            "individual": {
                "x": all_x,
                "y": all_y,
                "strength": all_strengths,
                "col_idx": all_col_indices,
                "labels": labels_all,
            },
            "clustered": [],
        }

    logger.info("Detections after individual filtering: %d", XY_filtered.shape[0])

    # clustering: try sectors first (if provided), otherwise DBSCAN
    use_sectors = bool(sectors)
    if use_sectors:
        logger.info("Attempting sector-based labeling using %d sectors", len(sectors))
    else:
        logger.info("No sectors provided -> using DBSCAN clustering")

    labels_filtered = _assign_labels_by_sectors_or_dbscan(
        XY_filtered, sectors, eps, cluster_min_samples
    )

    # if we used DBSCAN, log number of clusters found (exclude noise -1)
    if not use_sectors or (use_sectors and not np.any(labels_filtered != -1)):
        unique_labels = set(labels_filtered.tolist())
        n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
        logger.info(
            "DBSCAN produced %d clusters (noise label -1 may be present)",
            n_clusters_found,
        )
    else:
        # sectors used: log how many detections got assigned
        assigned = np.sum(labels_filtered != -1)
        logger.info(
            "Sector labeling assigned %d / %d kept detections",
            int(assigned),
            XY_filtered.shape[0],
        )

    # map labels back to the original detection array (inside-only indices)
    labels_all[original_indices] = labels_filtered
    full_original_indices = original_indices_in[original_indices]
    labels_all_full = -1 * np.ones(XY.shape[0], dtype=int)
    labels_all_full[full_original_indices] = labels_filtered

    # Build cluster metadata and apply support filtering (>=2 columns)
    clustered_boundaries = _build_cluster_metadata(
        labels_all_full, XY, all_strengths_arr, all_cols_arr, cluster_min_samples
    )
    logger.info(
        "Clusters retained after support filtering: %d", len(clustered_boundaries)
    )

    # If no cluster survives, try relaxing min_samples (one step) and retry
    if len(clustered_boundaries) == 0 and cluster_min_samples > 2:
        logger.info(
            "No clusters retained; retrying with relaxed cluster_min_samples=%d",
            max(2, cluster_min_samples - 1),
        )
        return find_vertical_discontinuities(
            x,
            y,
            v,
            vertical_bins=vertical_bins,
            horizontal_bins=horizontal_bins,
            bin_edges_h=bin_edges_h,
            bin_centers_h=bin_centers_h,
            bin_edges_v=bin_edges_v,
            bin_centers_v=bin_centers_v,
            min_points_per_bin_col=min_points_per_bin_col,
            gradient_threshold_factor=gradient_threshold_factor,
            smoothing_sigma_1d=smoothing_sigma_1d,
            min_strength=min_strength,
            cluster_eps_factor=cluster_eps_factor,
            cluster_min_samples=max(2, cluster_min_samples - 1),
            border=border,
            sectors=sectors,
        )

    # sort clusters top->bottom for consistency (reverse True: bottom first)
    clustered_boundaries.sort(key=lambda b: b["position"], reverse=True)

    logger.info(
        "Finished find_vertical_discontinuities: %d detections, %d clusters",
        XY.shape[0],
        len(clustered_boundaries),
    )

    return {
        "individual": {
            "x": all_x,
            "y": all_y,
            "strength": all_strengths,
            "col_idx": all_col_indices,
            "labels": labels_all_full,
        },
        "clustered": clustered_boundaries,
    }


def plot_discontinuities(
    x: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    discontinuities: dict,
    img=None,
    ax=None,
):
    inds = discontinuities["individual"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    if img is not None:
        ax.imshow(img, alpha=0.5, cmap="gray")

    # plot all input points colored by full velocity field for context
    scatter = ax.scatter(x, y, c=v, cmap="viridis", s=5, alpha=0.6)
    plt.colorbar(scatter, label="Velocity")

    # per-column detections: color by DBSCAN label (if available)
    det_x = inds["x"]
    det_y = inds["y"]
    det_strength = inds["strength"]
    det_labels = inds.get("labels", -1 * np.ones_like(det_x, dtype=int))

    # plot noise (label == -1) as small red crosses
    noise_mask = det_labels == -1
    if np.any(noise_mask):
        ax.scatter(
            det_x[noise_mask],
            det_y[noise_mask],
            c="red",
            s=18,
            marker="x",
            label="noise",
        )

    # plot clustered detections with distinct colors
    cluster_labels = np.unique(det_labels[det_labels != -1])
    cmap = plt.get_cmap("tab10")
    for i, cl in enumerate(cluster_labels):
        mask = det_labels == cl
        color = cmap(i % 10)
        ax.scatter(
            det_x[mask],
            det_y[mask],
            c=[color],
            s=35,
            edgecolor="k",
            alpha=0.9,
            label=f"cluster {cl}",
        )

    # also plot cluster centroids and support
    for b in discontinuities.get("clustered", []):
        ax.plot(
            b["x_mean"],
            b["position"],
            marker="o",
            markersize=10,
            color="white",
            markeredgecolor="k",
        )
        ax.text(
            b["x_mean"],
            b["position"],
            f" {b['label']} (s={b['support']})",
            color="white",
            fontsize=9,
            va="center",
        )

    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        f"Per-column discontinuities: {len(det_y)} detections, {len(discontinuities['clustered'])} clusters"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.show()


# # 1-D binning approach

# vertical_bins = 50
# min_points_per_bin = 10
# gradient_threshold = 0.4
# smoothing_sigma = 1.0

# # Use the smoothed dataframe finding morpho-kinematic boundaries
# df_smooth = apply_2d_gaussian_filter(df, sigma=3)
# x = df_smooth["x"].to_numpy()
# y = df_smooth["y"].to_numpy()
# v = df_smooth["V"].to_numpy()

# # Compute bin statistics
# y_min, y_max = y.min(), y.max()
# bin_edges = np.linspace(y_min, y_max, vertical_bins + 1)
# bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# count_binned = binned_statistic(y, np.ones_like(y), statistic="count", bins=bin_edges)
# mean_binned = binned_statistic(y, v, statistic="mean", bins=bin_edges)
# points_per_bin = count_binned.statistic.astype(int)
# valid_bins = points_per_bin >= min_points_per_bin
# mean_velocities = np.where(valid_bins, mean_binned.statistic, 0.0)

# # Apply smoothing to velocity profile for valid bins only
# if np.sum(valid_bins) > 3:
#     # Create continuous array for smoothing
#     valid_indices = np.where(valid_bins)[0]
#     valid_centers = bin_centers[valid_indices]
#     valid_velocities = mean_velocities[valid_indices]

#     # Fill invalid bins with nearest neighbor values for smoothing
#     smoothed_velocities = mean_velocities.copy()
#     for i in range(vertical_bins):
#         if not valid_bins[i] and len(valid_indices) > 0:
#             nearest_idx = valid_indices[np.argmin(np.abs(valid_indices - i))]
#             smoothed_velocities[i] = smoothed_velocities[nearest_idx]

#     # Apply Gaussian smoothing
#     smoothed_velocities = ndimage.gaussian_filter1d(
#         smoothed_velocities, sigma=smoothing_sigma
#     )
# else:
#     smoothed_velocities = mean_velocities.copy()

# # Calculate velocity gradient (derivative)
# bin_width = np.mean(mean_binned.bin_edges[1:] - mean_binned.bin_edges[:-1])
# vel_gradient = np.gradient(smoothed_velocities) / bin_width

# # Calculate second derivative
# vel_second_derivative = np.gradient(vel_gradient) / bin_width

# # Identify significant velocity transitions using gradient threshold
# # Focusing on negative gradients (velocity decreases moving upward)
# gradient_threshold_value = gradient_threshold * np.min(vel_gradient)
# transition_indices = []
# transition_strengths = []
# for i in range(1, len(vel_gradient) - 1):
#     # Skip bins with too few points
#     if not valid_bins[i]:
#         continue

#     # Check for significant negative gradient (velocity decrease)
#     # Verify this is a peak in the gradient (local minimum)
#     if (
#         (vel_gradient[i] < gradient_threshold_value)
#         and (vel_gradient[i] < vel_gradient[i - 1])
#         and (vel_gradient[i] < vel_gradient[i + 1])
#     ):
#         transition_indices.append(i)
#         transition_strengths.append(abs(vel_gradient[i]))

# # Convert transition indices to y-coordinates
# bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
# boundaries = [bin_centers[i] for i in transition_indices]

# # If no transitions found, raise a warning
# if len(boundaries) < 1:
#     logger.warning("No morpho boundaries detected. Assigning single domain.")

#     # Assign all points to a single domain
#     morpho_domains = np.zeros_like(x, dtype=int)
#     morpho_domains_map = {"m1": 0}

# # Sort boundaries from bottom to top (highest y to lowest y)
# sorted_boundaries = sorted(boundaries, reverse=True)

# # Assign domain labels with letters starting from A at the bottom upwards
# morpho_domains = np.zeros_like(x, dtype=int)
# morpho_domains_map = {}

# # Domain m1 is at the bottom (highest y values)
# morpho_domains[(y >= sorted_boundaries[0])] = 0  # Domain m1 (bottom)
# morpho_domains_map["m1"] = 0

# # Middle domains if multiple boundaries
# for i in range(len(sorted_boundaries) - 1):
#     morpho_domains[(y < sorted_boundaries[i]) & (y >= sorted_boundaries[i + 1])] = i + 1
#     morpho_domains_map[f"m{i + 2}"] = i + 1  # m2, m3, m4...

# # Upper domain (top of image, lowest y values)
# morpho_domains[(y < sorted_boundaries[-1])] = len(sorted_boundaries)
# morpho_domains_map[f"m{len(sorted_boundaries) + 1}"] = len(sorted_boundaries)

# colormap = plt.get_cmap("tab10")

# # Create figure with shared y-axis
# fig = plt.figure(figsize=(12, 7))

# gs = fig.add_gridspec(1, 4, width_ratios=[2, 1, 1, 1], wspace=0.01)

# # Create axes with shared y-axis
# ax1 = fig.add_subplot(gs[0, 0])  # Domain visualization
# ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)  # Velocity profile
# ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)  # Velocity gradient
# ax4 = fig.add_subplot(gs[0, 3], sharey=ax1)  # Second derivative

# # 1. Domains visualization

# # 1. Domains visualization
# ax1.imshow(img, alpha=0.5, cmap="gray")
# for i, (label, domain_id) in enumerate(morpho_domains_map.items()):
#     mask = morpho_domains == domain_id
#     ax1.scatter(
#         x[mask],
#         y[mask],
#         color=colormap(i),
#         label=f"Domain {label}",
#         s=10,
#         alpha=0.7,
#     )

# # Draw horizontal boundary lines on all panels
# for i, bound in enumerate(boundaries):
#     strength = (
#         transition_strengths[i] / max(transition_strengths)
#         if transition_strengths
#         else 1.0
#     )
#     for ax in [ax1, ax2, ax3, ax4]:
#         ax.axhline(
#             y=bound,
#             color="red",
#             linestyle="--",
#             linewidth=1 + 2 * strength,
#             alpha=0.7,
#         )

# # 2. Velocity profile with transitions
# ax2.plot(
#     mean_velocities[valid_bins],
#     bin_centers[valid_bins],
#     "o-",
#     color="gray",
#     markersize=4,
#     linewidth=1,
#     label="Raw",
# )
# ax2.plot(
#     smoothed_velocities[valid_bins],
#     bin_centers[valid_bins],
#     "b-",
#     linewidth=2,
#     label="Smoothed",
# )

# # 3. Velocity gradient (first derivative)
# ax3.plot(vel_gradient[valid_bins], bin_centers[valid_bins], "g-", linewidth=2)
# ax3.axvline(x=gradient_threshold_value, color="r", linestyle=":", label="Threshold")

# # 4. Second derivative
# ax4.plot(vel_second_derivative[valid_bins], bin_centers[valid_bins], "m-", linewidth=2)
# # Highlight zero crossings that indicate inflection points
# zero_line = ax4.axvline(x=0, color="k", linestyle="-.", linewidth=1, label="Zero")

# # Set titles and labels
# ax1.set_title("Morphological Domains")
# ax1.set_xlabel("")
# ax1.set_xticks([])
# ax1.set_ylabel("Height (y coordinate)")

# ax2.set_title("Velocity Profile")
# ax2.set_xlabel("Velocity")
# ax2.tick_params(axis="y", labelright=False, labelleft=False)

# ax3.set_title("First Derivative")
# ax3.set_xlabel("dv/dy")
# ax3.tick_params(axis="y", labelright=False, labelleft=False)

# ax4.set_title("Second Derivative")
# ax4.set_xlabel("d²v/dy²")
# ax4.tick_params(axis="y", labelright=False, labelleft=False)

# # Add legends
# ax1.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)
# ax2.legend(loc="upper left", fontsize=9)
# ax3.legend(loc="upper left", fontsize=9)
# ax4.legend(loc="upper left", fontsize=9)

# # Adjust layout
# plt.tight_layout()

# # Save the plot
# plt.savefig(
#     output_dir / f"{reference_start_date}_{reference_end_date}_velocity_analysis.png",
#     dpi=300,
#     bbox_inches="tight",
# )


if __name__ == "__main__":
    df = ...  # load your dataframe with x,y,V columns
    img = ...  # load your background image if available
    sectors = ...  # optionally define sectors as dict of matplotlib.patches.Polygon

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    v = df["V"].to_numpy()

    discontinuity_results = find_vertical_discontinuities(
        x=x,
        y=y,
        v=v,
        vertical_bins=50,
        horizontal_bins=10,
        min_points_per_bin_col=20,  # try 5-20 depending on data density
        gradient_threshold_factor=0.3,  # # Threshold for significant gradient (as fraction of max) adjust to be more/less sensitive
        smoothing_sigma_1d=1.0,
        min_strength=1e-3,
        cluster_eps_factor=2.0,
        cluster_min_samples=3,
        border=[500, 500, 1000, 500],  # left, right, bottom, top in px units
        sectors=sectors,  # optional: use predefined sectors instead of DBSCAN
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_discontinuities(
        x,
        y,
        v,
        discontinuities=discontinuity_results,
        img=img,
        ax=ax,
    )
