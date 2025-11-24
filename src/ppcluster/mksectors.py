import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from string import ascii_uppercase
from typing import Any, Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from shapely.geometry import MultiPoint, MultiPolygon, Polygon, box
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

try:
    from skimage.draw import polygon as sk_polygon
except ImportError:
    sk_polygon = None

logger = logging.getLogger("ppcx")


Numeric = float | int
ImageShape = tuple[int, int] | tuple[int, int, int]
ArrayLike = Sequence[Numeric] | np.ndarray
LabelArray = Sequence[str] | np.ndarray


@dataclass
class SectorPolygons:
    """Container for sector polygon coordinate arrays and their Shapely geometries."""

    data: dict[str, np.ndarray] = field(default_factory=dict)
    geometries: dict[str, Polygon] = field(default_factory=dict)

    # Dict-like interface
    def __iter__(self):
        return iter(self.data)

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def __getitem__(self, key: str) -> np.ndarray:
        return self.data[key]

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        self.data[key] = value

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def to_dict(self) -> dict[str, np.ndarray]:
        return dict(self.data)

    def __len__(self) -> int:
        return len(self.data)


def vectorize_clusters(
    cluster_grid: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    method: Literal[
        "convex_hull", "cell_union", "marching_squares", "alphashape"
    ] = "convex_hull",
    buffer_distance: float = 2.0,
    simplify_tolerance: float = 2.0,
    alpha: float = 0.0,
) -> SectorPolygons:
    """
    Create polygons from gridded cluster data using various methods.

    This is the main interface for polygon vectorization. It supports multiple
    backend algorithms suitable for different use cases.

    Args:
        cluster_grid: 2D array with cluster IDs (negative = unassigned)
        X: 2D meshgrid of X coordinates
        Y: 2D meshgrid of Y coordinates
        method: Vectorization method to use:
            - 'convex_hull': Simple convex hull (fastest, convex shapes only)
            - 'cell_union': Union of grid cells (good for blocky shapes)
            - 'marching_squares': Contour tracing (best for regular grids)
            - 'alphashape': Concave hull with alpha shapes (requires alphashape library)
        buffer_distance: Distance to buffer/smooth polygon boundaries (pixels)
        simplify_tolerance: Tolerance for simplifying polygon geometry (pixels)
        alpha: Alpha parameter for alphashape method (0 = auto, >0 = tighter fit)

    Returns:
        SectorPolygons containing boundary coordinates and Shapely geometries

    Raises:
        ValueError: If method is not recognized
        ImportError: If required library is not installed (alphashape)

    Examples:
        >>> # Simple convex hull (fastest)
        >>> polygons = vectorize_clusters(grid, X, Y, method="convex_hull")

        >>> # Marching squares for smooth boundaries (recommended for grids)
        >>> polygons = vectorize_clusters(
        ...     grid,
        ...     X,
        ...     Y,
        ...     method="marching_squares",
        ...     buffer_distance=3.0,
        ...     simplify_tolerance=2.0,
        ... )

        >>> # Alpha shapes for concave boundaries
        >>> polygons = vectorize_clusters(
        ...     grid, X, Y, method="alphashape", alpha=0.03, buffer_distance=2.0
        ... )
    """
    # Validate method
    valid_methods = {"convex_hull", "cell_union", "marching_squares", "alphashape"}
    if method not in valid_methods:
        raise ValueError(
            f"Invalid method '{method}'. Must be one of: {', '.join(valid_methods)}"
        )

    # Log method selection
    logger.info(f"Vectorizing clusters using method: {method}")

    # Dispatch to appropriate backend
    if method == "convex_hull":
        return _vectorize_convex_hull(
            cluster_grid,
            X,
            Y,
            buffer_distance=buffer_distance,
            simplify_tolerance=simplify_tolerance,
        )
    elif method == "cell_union":
        return _vectorize_cell_union(
            cluster_grid,
            X,
            Y,
            buffer_distance=buffer_distance,
            simplify_tolerance=simplify_tolerance,
        )
    elif method == "marching_squares":
        return _vectorize_marching_squares(
            cluster_grid,
            X,
            Y,
            buffer_distance=buffer_distance,
            simplify_tolerance=simplify_tolerance,
        )
    elif method == "alphashape":
        return _vectorize_alphashape(
            cluster_grid,
            X,
            Y,
            alpha=alpha,
            buffer_distance=buffer_distance,
            simplify_tolerance=simplify_tolerance,
        )


def remove_polygon_overlaps(
    polygons: SectorPolygons,
    ordered_labels: Sequence[str] | None = None,
    buffer_after_difference: float = 0.0,
) -> SectorPolygons:
    """
    Remove overlaps between polygons by computing differences in order.

    Each polygon is subtracted from all subsequent polygons in the order.
    This ensures that earlier polygons (e.g., sector A) maintain their full
    extent, while later polygons (B, C, D...) are trimmed where they overlap.

    Args:
        polygons: SectorPolygons object with geometries
        ordered_labels: Order to process labels (first has priority).
                    If None, uses sorted(polygons.keys())
        buffer_after_difference: Small buffer to apply after difference operations
                                to smooth jagged edges (default: 0.0)

    Returns:
        SectorPolygons with non-overlapping geometries
    """
    if not polygons.geometries:
        logger.warning("No geometries in polygons, returning empty")
        return SectorPolygons()

    # Determine processing order
    if ordered_labels is None:
        ordered_labels = sorted(polygons.keys())
    else:
        # Ensure all labels exist
        ordered_labels = [lab for lab in ordered_labels if lab in polygons.geometries]

    if not ordered_labels:
        logger.warning("No valid labels to process")
        return SectorPolygons()

    logger.info(f"Removing overlaps in order: {', '.join(ordered_labels)}")

    # Create output containers
    cleaned_polygons = SectorPolygons()
    cleaned_geometries: dict[str, Polygon] = {}

    # Process each polygon in order
    for idx, current_label in enumerate(ordered_labels):
        current_poly = polygons.geometries.get(current_label)

        if current_poly is None or current_poly.is_empty or current_poly.area <= 0:
            logger.warning(f"Skipping invalid polygon for label {current_label}")
            continue

        # Make a copy to work with
        result_poly = current_poly

        # Subtract all previously processed polygons from current one
        if idx > 0:
            # Get all polygons that have higher priority (processed earlier)
            previous_polys = [
                cleaned_geometries[lab]
                for lab in ordered_labels[:idx]
                if lab in cleaned_geometries
            ]

            if previous_polys:
                # Union all previous polygons
                previous_union = unary_union(previous_polys)

                # Subtract from current polygon
                result_poly = result_poly.difference(previous_union)

                # Apply small buffer if requested (smooths jagged edges)
                if buffer_after_difference > 0:
                    result_poly = result_poly.buffer(
                        buffer_after_difference, cap_style=1, join_style=1
                    )
                    result_poly = result_poly.buffer(
                        -buffer_after_difference, cap_style=1, join_style=1
                    )

                # Handle resulting geometry
                result_poly = _largest_polygon(result_poly)

                if result_poly is None or result_poly.area <= 0:
                    logger.warning(
                        f"Polygon {current_label} completely removed by overlap subtraction"
                    )
                    continue

                area_lost = current_poly.area - result_poly.area
                pct_lost = 100 * area_lost / current_poly.area
                logger.debug(
                    f"Polygon {current_label}: removed {area_lost:.1f} px² "
                    f"({pct_lost:.1f}%) due to overlap"
                )

        # Store cleaned polygon
        exterior_coords = np.asarray(result_poly.exterior.coords)[:-1]
        cleaned_polygons[current_label] = exterior_coords
        cleaned_geometries[current_label] = result_poly

        logger.debug(
            f"Polygon {current_label}: {len(exterior_coords)} vertices, "
            f"area={result_poly.area:.1f} px²"
        )

    cleaned_polygons.geometries = cleaned_geometries

    return cleaned_polygons


def validate_no_overlaps(polygons: SectorPolygons, tolerance: float = 1e-6) -> bool:
    """
    Validate that polygons do not overlap (within tolerance).

    Args:
        polygons: SectorPolygons to validate
        tolerance: Small tolerance for floating point comparisons

    Returns:
        True if no overlaps detected, False otherwise
    """
    labels = list(polygons.geometries.keys())

    for i, lab1 in enumerate(labels):
        poly1 = polygons.geometries[lab1]

        for lab2 in labels[i + 1 :]:
            poly2 = polygons.geometries[lab2]

            if poly1.intersects(poly2):
                intersection = poly1.intersection(poly2)
                overlap_area = intersection.area

                if overlap_area > tolerance:
                    logger.warning(
                        f"Overlap detected between {lab1} and {lab2}: "
                        f"{overlap_area:.2f} px²"
                    )
                    return False

    logger.info("No overlaps detected")
    return True


def assign_sector_labels(
    cluster_grid: np.ndarray,
    Y: np.ndarray,
    polygons: SectorPolygons,
    *,
    order_by: Literal["y_position", "area", "cluster_id"] = "y_position",
    reverse_order: bool = True,
    label_prefix: str = "",
) -> dict[str, Any]:
    """
    Assign letter labels (A, B, C...) to clusters based on ordering criteria.

    This function:
    1. Orders clusters by specified criterion (default: Y position, bottom-to-top)
    2. Assigns sequential letters (A, B, C, D...)
    3. Maps cluster IDs to sector letters
    4. Creates sector polygons dictionary with letter keys

    Args:
        cluster_grid: 2D array with cluster IDs (negative = unassigned)
        Y: 2D meshgrid of Y coordinates
        polygons: SectorPolygons with cluster_id as keys
        order_by: Ordering criterion:
            - 'y_position': Order by median Y coordinate (default)
            - 'area': Order by cluster size (number of pixels)
            - 'cluster_id': Order by cluster ID number
        reverse_order: If True, sort descending (bottom-to-top for Y, largest-first for area)
        label_prefix: Prefix for sector labels (default: "", use "S" for S0, S1, S2...)

    Returns:
        Dictionary containing:
            - 'cluster_to_letter': Mapping {cluster_id: letter}
            - 'letter_to_cluster': Reverse mapping {letter: cluster_id}
            - 'ordered_cluster_ids': List of cluster IDs in order
            - 'sector_polygons': SectorPolygons with letter keys
            - 'cluster_stats': Dict with statistics for each cluster

    Examples:
        >>> # Order by Y position (bottom = A, top = D)
        >>> result = assign_sector_labels(grid, Y, polygons, order_by="y_position")
        >>> cluster_to_letter = result["cluster_to_letter"]

        >>> # Order by area (largest = A)
        >>> result = assign_sector_labels(
        ...     grid, Y, polygons, order_by="area", reverse_order=True
        ... )
    """

    # Get unique cluster IDs
    unique_ids = np.unique(cluster_grid)
    unique_ids = unique_ids[unique_ids >= 0]

    if len(unique_ids) == 0:
        logger.warning("No valid clusters found in grid")
        return {
            "cluster_to_letter": {},
            "letter_to_cluster": {},
            "ordered_cluster_ids": [],
            "sector_polygons": SectorPolygons(),
            "cluster_stats": {},
        }

    # Compute ordering criterion for each cluster
    cluster_stats = {}
    for cluster_id in unique_ids:
        mask = cluster_grid == cluster_id
        n_pixels = int(np.sum(mask))

        stats = {
            "cluster_id": int(cluster_id),
            "n_pixels": n_pixels,
        }

        # Compute median Y position
        y_coords = Y[mask]
        if len(y_coords) > 0:
            stats["median_y"] = float(np.median(y_coords))
        else:
            stats["median_y"] = 0.0

        cluster_stats[int(cluster_id)] = stats

    # Order clusters based on criterion
    if order_by == "y_position":
        # Order by median Y coordinate
        ordered_cluster_ids = sorted(
            cluster_stats.keys(),
            key=lambda c: cluster_stats[c]["median_y"],
            reverse=reverse_order,
        )
    elif order_by == "area":
        # Order by cluster size
        ordered_cluster_ids = sorted(
            cluster_stats.keys(),
            key=lambda c: cluster_stats[c]["n_pixels"],
            reverse=reverse_order,
        )
    elif order_by == "cluster_id":
        # Order by cluster ID
        ordered_cluster_ids = sorted(cluster_stats.keys(), reverse=reverse_order)
    else:
        raise ValueError(
            f"Invalid order_by='{order_by}'. Must be 'y_position', 'area', or 'cluster_id'"
        )

    # Log ordering results
    logger.info(
        f"Ordered {len(ordered_cluster_ids)} clusters by '{order_by}' "
        f"({'descending' if reverse_order else 'ascending'})"
    )
    for cid in ordered_cluster_ids:
        stats = cluster_stats[cid]
        logger.info(
            f"  Cluster {cid}: {stats['n_pixels']} pixels, "
            f"median_y={stats['median_y']:.1f}"
        )

    # Assign letter labels
    cluster_to_letter = {}
    letter_to_cluster = {}
    sector_polygons = SectorPolygons()
    sector_geometries = {}

    for idx, cluster_id in enumerate(ordered_cluster_ids):
        # Generate letter label
        if label_prefix:
            letter = f"{label_prefix}{idx}"
        else:
            letter = ascii_uppercase[idx] if idx < len(ascii_uppercase) else f"S{idx}"

        cluster_to_letter[cluster_id] = letter
        letter_to_cluster[letter] = cluster_id

        # Map polygon from cluster_id to letter
        cluster_id_str = str(cluster_id)
        if cluster_id_str in polygons:
            sector_polygons[letter] = polygons[cluster_id_str]
            if cluster_id_str in polygons.geometries:
                sector_geometries[letter] = polygons.geometries[cluster_id_str]

        logger.info(f"Cluster {cluster_id} → Sector {letter}")

    sector_polygons.geometries = sector_geometries

    return {
        "cluster_to_letter": cluster_to_letter,
        "letter_to_cluster": letter_to_cluster,
        "ordered_cluster_ids": ordered_cluster_ids,
        "sector_polygons": sector_polygons,
        "cluster_stats": cluster_stats,
    }


def draw_polygon(
    ax_draw: plt.Axes,
    poly_coords: np.ndarray | None,
    color_hex: str,
    *,
    fill_alpha: float = 0.1,
    zorder: int = 1,
) -> None:
    if poly_coords is None or len(poly_coords) < 3:
        return
    ax_draw.fill(
        poly_coords[:, 0],
        poly_coords[:, 1],
        color=color_hex,
        alpha=fill_alpha,
        lw=0,
        zorder=zorder,
    )
    ax_draw.plot(
        np.r_[poly_coords[:, 0], poly_coords[0, 0]],
        np.r_[poly_coords[:, 1], poly_coords[0, 1]],
        color=color_hex,
        lw=2,
        zorder=zorder + 1,
    )
    cx, cy = np.mean(poly_coords, axis=0)


def classify_points_by_sectors(
    polygons: SectorPolygons,
    x: ArrayLike,
    y: ArrayLike,
) -> np.ndarray:
    """
    Classify points into sectors based on spatial containment.

    For each point, determines which sector polygon contains it using
    efficient spatial queries.

    Args:
        polygons: SectorPolygons with geometries
        x, y: Point coordinates

    Returns:
        Array of sector labels (str) for each point. Empty string if unassigned.

    Examples:
        >>> labels = classify_points_by_sectors(sector_polygons, x, y)
        >>> assigned_mask = labels != ""
        >>> print(f"Assigned {assigned_mask.sum()} / {len(x)} points")
    """
    from shapely.geometry import Point
    from shapely.prepared import prep

    # Normalize inputs
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have same shape")

    if not polygons.geometries:
        logger.warning("No geometries in polygons")
        return np.full(len(x_arr), "", dtype=object)

    # Initialize labels
    labels = np.full(len(x_arr), "", dtype=object)
    points = np.column_stack([x_arr, y_arr])

    # Process each sector
    for sector_label, geom in polygons.geometries.items():
        if geom is None or geom.is_empty or geom.area <= 0:
            continue

        # Use prepared geometry for speed
        prepared_geom = prep(geom)

        # Find points in this sector (only check unassigned points)
        unassigned_mask = labels == ""
        unassigned_indices = np.where(unassigned_mask)[0]

        if len(unassigned_indices) == 0:
            break  # All points assigned

        # Check containment for unassigned points
        for idx in unassigned_indices:
            if prepared_geom.contains(Point(points[idx])):
                labels[idx] = sector_label

    # Log results
    n_assigned = np.sum(labels != "")
    pct_assigned = 100 * n_assigned / len(labels) if len(labels) > 0 else 0
    logger.info(
        f"Classified {n_assigned} / {len(labels)} points "
        f"({pct_assigned:.1f}%) into {len(polygons.geometries)} sectors"
    )

    # Log per-sector counts
    unique_labels, counts = np.unique(labels[labels != ""], return_counts=True)
    for label, count in zip(unique_labels, counts, strict=False):
        logger.debug(f"  Sector {label}: {count} points")

    return labels

def compute_sector_stats(
    polygons: SectorPolygons,
    point_labels: ArrayLike,
    *,
    x: ArrayLike,
    y: ArrayLike,
    v: ArrayLike | None = None,
    rasterize: bool = False,
    img_shape: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """
    Compute descriptive statistics for sector polygons.

    For each sector, computes geometric properties and statistics on
    the points assigned to that sector.

    Args:
        polygons: SectorPolygons with geometries
        point_labels: 1D array of sector labels (same length as x/y)
        x, y: Point coordinates
        v: Optional per-point values (e.g., velocity) for stats
        rasterize: If True, count pixels covered by polygon
        img_shape: (H, W) for pixel counting

    Returns:
        DataFrame with one row per sector

    Examples:
        >>> # Classify points first
        >>> labels = classify_points_by_sectors(polygons, x, y)
        >>> # Then compute stats
        >>> stats = compute_sector_stats(polygons, labels, x=x, y=y, v=velocity)
    """
    # Normalize inputs
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    labels_arr = np.asarray(point_labels, dtype=object)
    v_arr = np.asarray(v, dtype=float) if v is not None else None

    # Validate
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have same shape")
    if labels_arr.shape[0] != x_arr.shape[0]:
        raise ValueError("point_labels must have same length as x/y")

    if not polygons.geometries:
        logger.warning("No geometries in polygons")
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for sector_label, geom in polygons.geometries.items():
        if geom is None or geom.is_empty or geom.area <= 0:
            logger.debug(f"Skipping sector {sector_label}: invalid geometry")
            continue

        # Find points in this sector
        mask = labels_arr == sector_label
        n_points = int(np.sum(mask))

        # Geometric properties
        area = float(geom.area)
        perimeter = float(geom.length)
        centroid_x, centroid_y = geom.centroid.coords[0]
        compactness = (
            (4.0 * np.pi * area) / (perimeter**2 + 1e-12)
            if perimeter > 0
            else np.nan
        )
        point_density = n_points / area if area > 0 else 0.0

        # Value statistics
        v_mean = v_std = v_median = v_min = v_max = np.nan
        if v_arr is not None and n_points > 0:
            v_sel = v_arr[mask]
            if v_sel.size > 0:
                v_mean = float(np.mean(v_sel))
                v_std = float(np.std(v_sel))
                v_median = float(np.median(v_sel))
                v_min = float(np.min(v_sel))
                v_max = float(np.max(v_sel))

        # Pixel count (optional)
        pixel_count = np.nan
        if rasterize and img_shape is not None:
            coords = polygons.get(sector_label)
            if coords is not None:
                try:
                    from skimage.draw import polygon as sk_polygon

                    h, w = img_shape[:2]
                    rr, cc = sk_polygon(coords[:, 1], coords[:, 0], shape=(h, w))
                    pixel_count = int(rr.size)
                except Exception as exc:
                    logger.debug(f"Rasterization failed for {sector_label}: {exc}")

        rows.append(
            {
                "label": sector_label,
                "n_points": n_points,
                "area_px2": area,
                "perimeter_px": perimeter,
                "compactness": compactness,
                "centroid_x": float(centroid_x),
                "centroid_y": float(centroid_y),
                "pixel_count": pixel_count,
                "point_density_pts_per_px2": point_density,
                "v_mean": v_mean,
                "v_std": v_std,
                "v_median": v_median,
                "v_min": v_min,
                "v_max": v_max,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    logger.info(
        f"Computed stats for {len(df)} sectors "
        f"(total points: {df['n_points'].sum()} / {len(x_arr)})"
    )

    return df.sort_values("label").reset_index(drop=True)
)


# ============================================================================
# Vectorization bBackend implementations
# ============================================================================


def _vectorize_convex_hull(
    cluster_grid: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    buffer_distance: float,
    simplify_tolerance: float,
) -> SectorPolygons:
    """Backend: Simple convex hull method (fastest)."""
    unique_ids = np.unique(cluster_grid)
    unique_ids = unique_ids[unique_ids >= 0]

    polygons = SectorPolygons()
    geometries: dict[str, Polygon] = {}

    for cluster_id in unique_ids:
        label_str = str(int(cluster_id))
        mask = cluster_grid == cluster_id
        y_indices, x_indices = np.where(mask)

        if len(x_indices) < 3:
            logger.warning(f"Cluster {cluster_id} has < 3 points, skipping")
            continue

        x_coords = X[y_indices, x_indices]
        y_coords = Y[y_indices, x_indices]
        points = np.column_stack([x_coords, y_coords])

        try:
            multi_pt = MultiPoint(points)
            poly = multi_pt.convex_hull

            if poly.geom_type != "Polygon":
                logger.warning(f"Cluster {cluster_id} hull is not a polygon")
                continue

            poly = _apply_smoothing(poly, buffer_distance, simplify_tolerance)

            if poly is None or poly.area <= 0:
                continue

            exterior_coords = np.asarray(poly.exterior.coords)[:-1]
            polygons[label_str] = exterior_coords
            geometries[label_str] = poly

        except Exception as exc:
            logger.warning(f"Failed for cluster {cluster_id}: {exc}")
            continue

    polygons.geometries = geometries
    logger.info(f"Created {len(polygons)} polygons using convex hull")
    return polygons


def _vectorize_cell_union(
    cluster_grid: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    buffer_distance: float,
    simplify_tolerance: float,
) -> SectorPolygons:
    """Backend: Grid cell union method (good for blocky shapes)."""
    # Infer grid spacing
    dx = _infer_grid_spacing(X, axis=1)
    dy = _infer_grid_spacing(Y, axis=0)

    logger.debug(f"Grid spacing: dx={dx:.1f}, dy={dy:.1f}")

    unique_ids = np.unique(cluster_grid)
    unique_ids = unique_ids[unique_ids >= 0]

    polygons = SectorPolygons()
    geometries: dict[str, Polygon] = {}

    for cluster_id in unique_ids:
        label_str = str(int(cluster_id))
        mask = cluster_grid == cluster_id
        y_indices, x_indices = np.where(mask)

        if len(x_indices) < 3:
            logger.warning(f"Cluster {cluster_id} has < 3 points, skipping")
            continue

        x_coords = X[y_indices, x_indices]
        y_coords = Y[y_indices, x_indices]

        try:
            # Create box for each grid cell
            cell_boxes = [
                box(x - dx / 2, y - dy / 2, x + dx / 2, y + dy / 2)
                for x, y in zip(x_coords, y_coords, strict=False)
            ]

            # Union all cells
            if len(cell_boxes) == 1:
                poly = cell_boxes[0]
            else:
                poly = unary_union(cell_boxes)

            # Handle MultiPolygon
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda p: p.area)
            elif poly.geom_type != "Polygon":
                logger.warning(f"Cluster {cluster_id} has unexpected geometry")
                continue

            poly = _apply_smoothing(poly, buffer_distance, simplify_tolerance)

            if poly is None or poly.area <= 0:
                continue

            exterior_coords = np.asarray(poly.exterior.coords)[:-1]
            polygons[label_str] = exterior_coords
            geometries[label_str] = poly

        except Exception as exc:
            logger.warning(f"Failed for cluster {cluster_id}: {exc}")
            continue

    polygons.geometries = geometries
    logger.info(f"Created {len(polygons)} polygons using cell union")
    return polygons


def _vectorize_marching_squares(
    cluster_grid: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    buffer_distance: float,
    simplify_tolerance: float,
) -> SectorPolygons:
    """Backend: Marching squares method (best for regular grids)."""
    from skimage import measure

    unique_ids = np.unique(cluster_grid)
    unique_ids = unique_ids[unique_ids >= 0]

    polygons = SectorPolygons()
    geometries: dict[str, Polygon] = {}

    for cluster_id in unique_ids:
        label_str = str(int(cluster_id))
        mask = (cluster_grid == cluster_id).astype(np.uint8)

        # Find contours
        contours = measure.find_contours(mask, level=0.5)

        if not contours:
            logger.warning(f"No contours for cluster {cluster_id}")
            continue

        # Take longest contour (outer boundary)
        longest_contour = max(contours, key=len)

        try:
            # Convert grid indices to real coordinates
            coords = _grid_to_coords(longest_contour, X, Y)

            poly = ShapelyPolygon(coords)

            if not poly.is_valid:
                poly = poly.buffer(0)

            poly = _apply_smoothing(poly, buffer_distance, simplify_tolerance)

            if poly is None or poly.area <= 0:
                continue

            exterior_coords = np.asarray(poly.exterior.coords)[:-1]
            polygons[label_str] = exterior_coords
            geometries[label_str] = poly

        except Exception as exc:
            logger.warning(f"Failed for cluster {cluster_id}: {exc}")
            continue

    polygons.geometries = geometries
    logger.info(f"Created {len(polygons)} polygons using marching squares")
    return polygons


def _vectorize_alphashape(
    cluster_grid: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float,
    buffer_distance: float,
    simplify_tolerance: float,
) -> SectorPolygons:
    """Backend: Alpha shape method (concave hulls, requires alphashape)."""
    try:
        import alphashape
    except ImportError as exc:
        raise ImportError(
            "alphashape library required for this method. "
            "Install with: pip install alphashape"
        ) from exc

    unique_ids = np.unique(cluster_grid)
    unique_ids = unique_ids[unique_ids >= 0]

    polygons = SectorPolygons()
    geometries: dict[str, Polygon] = {}

    for cluster_id in unique_ids:
        label_str = str(int(cluster_id))
        mask = cluster_grid == cluster_id
        y_indices, x_indices = np.where(mask)

        if len(x_indices) < 3:
            logger.warning(f"Cluster {cluster_id} has < 3 points, skipping")
            continue

        x_coords = X[y_indices, x_indices]
        y_coords = Y[y_indices, x_indices]
        points = np.column_stack([x_coords, y_coords])

        # Add small jitter to break perfect grid alignment
        np.random.seed(42)
        points = points + np.random.randn(*points.shape) * 0.5

        try:
            # Compute alpha shape
            if alpha == 0:
                poly = alphashape.alphashape(points)  # Auto-optimize
            else:
                poly = alphashape.alphashape(points, alpha)

            # Handle edge cases
            if poly is None or poly.is_empty:
                logger.warning(
                    f"Alphashape failed for cluster {cluster_id}, using convex hull"
                )
                poly = MultiPoint(points).convex_hull

            # Handle MultiPolygon
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda p: p.area)
            elif poly.geom_type != "Polygon":
                logger.warning(
                    f"Unexpected geometry for cluster {cluster_id}, using convex hull"
                )
                poly = MultiPoint(points).convex_hull

            if not poly.is_valid:
                poly = poly.buffer(0)

            poly = _apply_smoothing(poly, buffer_distance, simplify_tolerance)

            if poly is None or poly.area <= 0:
                continue

            exterior_coords = np.asarray(poly.exterior.coords)[:-1]
            polygons[label_str] = exterior_coords
            geometries[label_str] = poly

        except Exception as exc:
            logger.warning(f"Failed for cluster {cluster_id}: {exc}")
            continue

    polygons.geometries = geometries
    logger.info(f"Created {len(polygons)} polygons using alphashape")
    return polygons


# ============================================================================
# Helper functions
# ============================================================================


def _largest_polygon(geom: Polygon | MultiPolygon | None) -> Polygon | None:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda g: g.area)
    if geom.geom_type != "Polygon" or geom.area <= 0:
        return None
    return geom


def _apply_smoothing(
    poly: Polygon,
    buffer_distance: float,
    simplify_tolerance: float,
) -> Polygon | None:
    """Apply buffering and simplification to polygon."""
    if buffer_distance > 0:
        poly = poly.buffer(buffer_distance, cap_style=1, join_style=1)
        poly = poly.buffer(-buffer_distance, cap_style=1, join_style=1)

    if simplify_tolerance > 0:
        poly = poly.simplify(simplify_tolerance, preserve_topology=True)

    return _largest_polygon(poly)


def _infer_grid_spacing(grid: np.ndarray, axis: int) -> float:
    """Infer grid spacing from coordinate grid."""
    unique_vals = np.unique(grid)
    if len(unique_vals) > 1:
        return float(np.median(np.diff(unique_vals)))
    return 64.0  # Fallback


def _grid_to_coords(
    contour: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    """Convert grid indices to real coordinates using bilinear interpolation."""
    row_idx = contour[:, 0]
    col_idx = contour[:, 1]

    # Floor and ceiling indices
    row_floor = np.clip(np.floor(row_idx).astype(int), 0, Y.shape[0] - 1)
    col_floor = np.clip(np.floor(col_idx).astype(int), 0, X.shape[1] - 1)
    row_ceil = np.clip(row_floor + 1, 0, Y.shape[0] - 1)
    col_ceil = np.clip(col_floor + 1, 0, X.shape[1] - 1)

    # Fractional parts
    row_frac = row_idx - row_floor
    col_frac = col_idx - col_floor

    # Bilinear interpolation
    x_coords = (
        (1 - row_frac) * (1 - col_frac) * X[row_floor, col_floor]
        + (1 - row_frac) * col_frac * X[row_floor, col_ceil]
        + row_frac * (1 - col_frac) * X[row_ceil, col_floor]
        + row_frac * col_frac * X[row_ceil, col_ceil]
    )

    y_coords = (
        (1 - row_frac) * (1 - col_frac) * Y[row_floor, col_floor]
        + (1 - row_frac) * col_frac * Y[row_floor, col_ceil]
        + row_frac * (1 - col_frac) * Y[row_ceil, col_floor]
        + row_frac * col_frac * Y[row_ceil, col_ceil]
    )

    return np.column_stack([x_coords, y_coords])


## --- DEPRECATED FUNCTION ---


def assign_sectors_simple(
    x: np.ndarray,
    y: np.ndarray,
    kin_cluster: np.ndarray,
    ordered_clusters_ids: list[int],
    polygon_kwargs: dict | None = None,
) -> dict[str, Any]:
    """
    Simple morpho-kinematic sector assignment based on centroid Y position.
    Assigns letters A, B, C, D... from bottom to top of the image.

    Args:
        x: X coordinates of points
        y: Y coordinates of points
        kin_cluster: Cluster labels for each point
        ordered_clusters_ids: List of cluster IDs (already sorted by median Y descending)
        img: Optional background image for visualization
        polygon_kwargs: Optional kwargs for polygon computation

    Returns:
        Dictionary with sector assignments and polygons
    """
    from string import ascii_uppercase

    # Default polygon options
    poly_opts = dict(
        prevent_overlap=True,
        containment_strategy="difference",
        polygon_mode="boundary",
    )
    if polygon_kwargs:
        poly_opts.update(polygon_kwargs)

    # Assign letters sequentially from bottom to top (A = bottom, B, C, D = top)
    cluster_to_letter = {}
    mk_label_str = np.full_like(kin_cluster, "", dtype=object)
    mk_label_id = np.full_like(kin_cluster, -1, dtype=int)

    for idx, cluster_id in enumerate(ordered_clusters_ids):
        letter = ascii_uppercase[idx] if idx < len(ascii_uppercase) else f"S{idx}"
        cluster_to_letter[cluster_id] = letter
        mask = kin_cluster == cluster_id
        mk_label_str[mask] = letter
        mk_label_id[mask] = idx

        logger.info(f"Cluster {cluster_id} → Sector {letter}")

    # Compute polygons for each sector
    polygons_dict = compute_sector_polygons(
        x=x,
        y=y,
        mk_label_str=mk_label_str,
        **poly_opts,
    )

    return {
        "mk_label_str": mk_label_str,
        "mk_label_id": mk_label_id,
        "cluster_to_letter": cluster_to_letter,
        "polygons": dict(polygons_dict),
        "geometries": polygons_dict.geometries,
    }


def assign_sectors_major_minor(
    x: ArrayLike,
    y: ArrayLike,
    kin_cluster: ArrayLike,
    ordered_clusters_ids: Sequence[int],
    *,
    overlap_threshold: float = 0.6,
    convex_kwargs: Mapping[str, Any] | None = None,
    major_polygon_kwargs: Mapping[str, Any] | None = None,
    minor_polygon_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Automatically assign major/minor MK sectors based on polygon overlap."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    clusters = np.asarray(kin_cluster, dtype=int)
    ordered_clusters = [int(c) for c in ordered_clusters_ids]

    cluster_labels_str = clusters.astype(str)
    convex_opts: dict[str, Any] = dict(
        smooth_iters=0,
        prevent_overlap=False,
        containment_strategy="none",
        polygon_mode="convex",
    )
    if convex_kwargs:
        convex_opts.update(convex_kwargs)

    raw_polygons = compute_mk_sector_polygons(
        x_arr,
        y_arr,
        cluster_labels_str,
        **convex_opts,
    )

    cluster_shapes: dict[int, Polygon] = {}
    for lab_str, shape in raw_polygons.geometries.items():
        try:
            cluster_id = int(float(lab_str))
        except ValueError:
            continue
        if shape.is_empty or shape.area <= 0:
            continue
        cluster_shapes[cluster_id] = shape

    if not cluster_shapes:
        raise RuntimeError(
            "Unable to build polygons for any cluster; aborting assignment."
        )

    major_candidates = set(cluster_shapes.keys())
    minor_parent: dict[int, tuple[int, float]] = {}

    for child_id, child_poly in cluster_shapes.items():
        if child_poly.area <= 0:
            continue
        best_parent: int | None = None
        best_overlap = 0.0
        for parent_id, parent_poly in cluster_shapes.items():
            if parent_id == child_id or parent_poly.area <= child_poly.area:
                continue
            if not parent_poly.intersects(child_poly):
                continue
            overlap_ratio = parent_poly.intersection(child_poly).area / child_poly.area
            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_parent = parent_id
        if best_parent is not None and best_overlap >= overlap_threshold:
            minor_parent[child_id] = (best_parent, best_overlap)
            major_candidates.discard(child_id)

    ordered_major_clusters = [
        cid for cid in ordered_clusters if cid in major_candidates
    ]
    if not ordered_major_clusters:
        ordered_major_clusters = [
            cid for cid in ordered_clusters if cid in cluster_shapes
        ]
        major_candidates = set(ordered_major_clusters)
        minor_parent.clear()

    mk_label_str = np.full_like(clusters, "", dtype=object)
    mk_label_id = -1 * np.ones_like(clusters, dtype=int)

    label_to_index: dict[str, int] = {}
    major_label_map: dict[int, str] = {}
    cluster_to_label: dict[int, str] = {}
    minor_labels_map: dict[int, str] = {}
    minor_counts: defaultdict[str, int] = defaultdict(int)

    def assign_label(cluster_id: int, label: str) -> None:
        idx = label_to_index.setdefault(label, len(label_to_index))
        mask = clusters == cluster_id
        mk_label_str[mask] = label
        mk_label_id[mask] = idx
        cluster_to_label[cluster_id] = label

    for idx, cluster_id in enumerate(ordered_major_clusters):
        letter = ascii_uppercase[idx] if idx < len(ascii_uppercase) else f"S{idx}"
        major_label_map[cluster_id] = letter
        assign_label(cluster_id, letter)

    for cluster_id in ordered_clusters:
        if cluster_id not in minor_parent:
            continue
        parent_id, overlap_ratio = minor_parent[cluster_id]
        parent_label = major_label_map.get(parent_id)
        if parent_label is None:
            logger.warning(
                "Cluster %d expected as minor but parent %d unavailable; promoting to major.",
                cluster_id,
                parent_id,
            )
            letter = (
                ascii_uppercase[len(major_label_map)]
                if len(major_label_map) < len(ascii_uppercase)
                else f"S{len(major_label_map)}"
            )
            major_label_map[cluster_id] = letter
            assign_label(cluster_id, letter)
            major_candidates.add(cluster_id)
            continue
        minor_counts[parent_label] += 1
        label = f"{parent_label}{minor_counts[parent_label]}"
        assign_label(cluster_id, label)
        minor_labels_map[cluster_id] = label
        logger.info(
            "Cluster %d assigned to %s (minor of %s, overlap %.2f)",
            cluster_id,
            label,
            parent_label,
            overlap_ratio,
        )

    for cluster_id, label in major_label_map.items():
        logger.info("Cluster %d assigned to %s (major)", cluster_id, label)

    major_mask = np.isin(clusters, list(major_label_map.keys()))
    major_opts: dict[str, Any] = dict(
        smooth_iters=4,
        prevent_overlap=True,
        containment_strategy="difference",
    )
    if major_polygon_kwargs:
        major_opts.update(major_polygon_kwargs)
    polygons_major_dict: dict[str, np.ndarray] = {}
    if major_mask.any():
        polygons_major = compute_mk_sector_polygons(
            x_arr[major_mask],
            y_arr[major_mask],
            mk_label_str[major_mask],
            **major_opts,
        )
        polygons_major_dict = dict(polygons_major)

    minor_clusters = [cid for cid in ordered_clusters if cid in minor_parent]
    polygons_minor_dict: dict[str, np.ndarray] = {}
    if minor_clusters:
        minor_mask = np.isin(clusters, minor_clusters)
        minor_opts: dict[str, Any] = dict(
            smooth_iters=2,
            prevent_overlap=False,
            containment_strategy="none",
            polygon_mode="convex",
        )
        if minor_polygon_kwargs:
            minor_opts.update(minor_polygon_kwargs)
        polygons_minor = compute_mk_sector_polygons(
            x_arr[minor_mask],
            y_arr[minor_mask],
            mk_label_str[minor_mask],
            **minor_opts,
        )
        polygons_minor_dict = dict(polygons_minor)

    return {
        "mk_label_str": mk_label_str,
        "mk_label_id": mk_label_id,
        "major_clusters": ordered_major_clusters,
        "major_label_map": major_label_map,
        "minor_parent": minor_parent,
        "label_to_index": label_to_index,
        "cluster_to_label": cluster_to_label,
        "minor_labels_map": minor_labels_map,
        "raw_polygons": dict(raw_polygons),
        "raw_geometries": raw_polygons.geometries,
        "polygons_major": polygons_major_dict,
        "polygons_minor": polygons_minor_dict,
    }
