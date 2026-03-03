import logging
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
from affine import Affine
from shapely import MultiPolygon
from shapely.geometry import Polygon, shape
from smoothify import smoothify

logger = logging.getLogger("ppcx")

# === Sector Vectorization and Cleaning Functions ===#


def vectorize_gridded_sectors(
    cluster_grid: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
) -> gpd.GeoDataFrame:
    """
    Vectorize a grid of cluster IDs into a GeoDataFrame using smoothify.

    Args:
        cluster_grid: 2D array of IDs (int or float).
        X, Y: 2D meshgrids of coordinates.

    Returns:
        GeoDataFrame with columns ['cluster_id', 'geometry', 'label']
    """
    # 1. Setup Transform
    dx = float(X[0, 1] - X[0, 0]) if X.shape[1] > 1 else 1.0  # pixel size in x
    dy = float(Y[1, 0] - Y[0, 0]) if Y.shape[0] > 1 else 1.0  # pixel size in y

    # Origin (Center of top-left pixel minus half-step)
    x_origin = X[0, 0] - (dx / 2.0)
    y_origin = Y[0, 0] - (dy / 2.0)
    transform = Affine(dx, 0.0, x_origin, 0.0, dy, y_origin)

    # 2. Vectorize using Rasterio
    mask = ~np.isnan(cluster_grid) & (cluster_grid >= 0)
    # Ensure int32 for rasterio
    grid_int = cluster_grid.astype(np.int32)

    try:
        shapes_gen = rasterio.features.shapes(grid_int, mask=mask, transform=transform)
        records = [
            {"geometry": shape(geom), "cluster_id": int(val)}
            for geom, val in shapes_gen
        ]
    except Exception as e:
        logger.error(f"Rasterio vectorization failed: {e}")
        return gpd.GeoDataFrame(columns=["cluster_id", "geometry"])

    if not records:
        return gpd.GeoDataFrame(columns=["cluster_id", "geometry"])

    gdf = gpd.GeoDataFrame(records)

    # Dissolve to get one MultiPolygon per cluster ID
    gdf = gdf.dissolve(by="cluster_id", as_index=False)

    # Drop empty or invalid geometries
    gdf = gdf[~gdf.is_empty & gdf.geometry.notna()].copy()

    return gdf


def clean_vector_sectors(
    gdf_sectors: gpd.GeoDataFrame,
    df_points: pd.DataFrame,
    min_area_px2: float = 100000.0,
    isolation_buffer: float = 30.0,
    velocity_merge_threshold: float = 1.6,
    target_number_of_sectors: int = 4,
    force_minimum_sectors: bool = True,
    fill_holes_area: float = 0.0,
    smooth_geometries: bool = True,
    smooth_method: Literal["smoothify", "simplify"] = "smoothify",
    smooth_iterations: int = 4,
    raster_res: float | None = None,
) -> gpd.GeoDataFrame:
    """
    Clean polygon sectors by removing isolated ones, merging contained ones,
    and merging small ones based on robust velocity statistics (NMAD Z-test).

    Args:
        gdf_sectors: GeoDataFrame of sector polygons.
        df_points: DataFrame of points with coordinates and velocities.
        min_area_px2: Minimum area threshold for sectors.
        isolation_buffer: Buffer distance for isolation removal.
        velocity_merge_threshold: Z-score threshold for merging by velocity.
        target_number_of_sectors: Target number of sectors after cleaning.
        force_minimum_sectors: If True, force reduction to target_number_of_sectors.
        fill_holes_area: Area threshold for filling holes in polygons.
        smooth_geometries: Whether to smooth geometries.
        smooth_method: Smoothing method ('smoothify' or 'simplify').
        smooth_iterations: Number of smoothing iterations.
        raster_res: Raster resolution for smoothing.

    Returns:
        Cleaned GeoDataFrame of sectors.
    """
    if gdf_sectors.empty:
        return gdf_sectors

    logger.info(f"Starting sector cleaning with {len(gdf_sectors)} sectors.")

    # 0. Ensure we work with single polygons
    gdf = split_disconnected_polygons(gdf_sectors)
    logger.info(f"After exploding multipolygons: {len(gdf)} sectors.")

    # 1. Merge Contained Polygons
    try:
        gdf = merge_contained_sectors(gdf)
    except Exception as e:
        logger.error(f"Error during contained polygon merging: {e}")

    # 2. Filter Small Sectors
    gdf = filter_small_sectors(gdf, min_area_px2)

    # 3. Remove Isolated Polygons
    try:
        gdf = remove_isolated_sectors(gdf, isolation_buffer)
    except Exception as e:
        logger.error(f"Error during isolated sector removal: {e}")

    # 4. Iterative Statistical Merging
    try:
        gdf = merge_sectors_by_velocity(
            gdf, df_points, velocity_merge_threshold, target_number_of_sectors
        )
    except Exception as e:
        logger.error(f"Error during velocity-based merging: {e}")

    # 5. Final Force Limit
    if force_minimum_sectors:
        gdf = enforce_sector_limit(gdf, target_number_of_sectors)

    # 6. Fill Holes (if requested)
    if fill_holes_area > 0.0:
        try:
            gdf = fill_polygon_holes(gdf, fill_holes_area)
        except Exception as e:
            logger.error(f"Error during hole filling: {e}")

    # 7. Smooth Geometries (optional)
    if smooth_geometries:
        # Auto-select raster_res from points if needed
        if smooth_method == "smoothify" and raster_res is None:
            raster_res = max(
                np.sqrt(df_points["x"].ptp() * df_points["y"].ptp() / len(df_points)),
                5.0,
            )
            logger.info(f"Auto-selected raster_res={raster_res:.2f} for smoothing.")

        gdf = smooth_polygons(
            gdf,
            smooth_method=smooth_method,
            raster_res=raster_res,
            smooth_iterations=smooth_iterations,
            merge_collection=False,
            area_tolerance=0.01,  # % of original area allowed as error
            num_cores=1,  # Keep it single-threaded to avoid overhead on small datasets
        )

    return gdf


def classify_points_by_polygons(
    polygons: gpd.GeoDataFrame,
    points: pd.DataFrame | gpd.GeoDataFrame,
    x_col: str = "x",
    y_col: str = "y",
    keep_unclassified: bool = False,
) -> gpd.GeoDataFrame:
    """
    Classify points based on which polygon they fall within.

    Args:
        polygons: GeoDataFrame with polygon geometries and sector info.
        points: DataFrame or GeoDataFrame with point coordinates.
        x_col: Name of the column with X coordinates (if points is DataFrame).
        y_col: Name of the column with Y coordinates (if points is DataFrame).
        keep_unclassified: If True, points outside any polygon are kept with NaN sector.

    Returns:
        GeoDataFrame of points with assigned sector info.
    """

    # Convert points to GeoDataFrame if necessary
    if isinstance(points, pd.DataFrame):
        if x_col not in points.columns or y_col not in points.columns:
            raise ValueError(
                f"Points DataFrame must contain '{x_col}' and '{y_col}' columns."
            )
        points_df = points.copy()
        gdf_pts = gpd.GeoDataFrame(
            points_df,
            geometry=gpd.points_from_xy(points_df[x_col], points_df[y_col]),
        )
    else:
        gdf_pts = points.copy()

    # Check if any of the GeoDataFrames has CRS defined
    if gdf_pts.crs is not None or polygons.crs is not None:
        logger.info("Checking CRS consistency between points and polygons...")
        if gdf_pts.crs is None and polygons.crs is not None:
            logger.info(
                f"Points GeoDataFrame has no CRS defined. Setting to match polygons CRS: {polygons.crs}"
            )
            gdf_pts.set_crs(polygons.crs, allow_override=True)
        elif polygons.crs is None and gdf_pts.crs is not None:
            logger.info(
                f"Polygons GeoDataFrame has no CRS defined. Setting to match points CRS: {gdf_pts.crs}"
            )
            polygons.set_crs(gdf_pts.crs, allow_override=True)
        # CRS consistency
        if gdf_pts.crs != polygons.crs:
            logger.info(
                f"CRS mismatch detected. Transforming points from {gdf_pts.crs} to {polygons.crs}."
            )
            gdf_pts = gdf_pts.to_crs(polygons.crs)

    # Spatial Join
    # 'inner' will drop points outside any sector.
    # 'left' keeps them with NaN sector.
    how = "left" if keep_unclassified else "inner"
    joined = gpd.sjoin(gdf_pts, polygons, how=how, predicate="within")

    # Handle duplicates (if polygons overlap) - take the first match or specific logic
    joined = joined[~joined.index.duplicated(keep="first")]

    # Remove the rigth index added by sjoin
    out_gdf = joined.drop(columns=["index_right"])

    return out_gdf


def assign_sector_labels(
    gdf: gpd.GeoDataFrame,
    order_by: Literal["y_centroid", "area"] = "y_centroid",
    ascending: bool = True,
    label_prefix: str = "",
) -> gpd.GeoDataFrame:
    """
    Assign letter labels (A, B, C...) to the GeoDataFrame based on geometry.

    Args:
        gdf: GeoDataFrame of polygons.
        order_by: Sort by 'y_centroid' or 'area'.
        ascending: Sort order.
        label_prefix: Optional prefix for labels.

    Returns:
        GeoDataFrame with 'sector' column assigned.
    """
    from string import ascii_uppercase

    def get_label(idx):
        if label_prefix:
            return f"{label_prefix}{idx}"
        if idx < len(ascii_uppercase):
            return ascii_uppercase[idx]
        return f"S{idx}"

    if gdf.empty:
        return gdf

    # Calculate sort key
    if order_by == "y_centroid":
        gdf["sort_key"] = gdf.geometry.centroid.y
    elif order_by == "area":
        gdf["sort_key"] = gdf.geometry.area

    gdf = gdf.sort_values("sort_key", ascending=ascending).reset_index(drop=True)

    # Assign letters
    gdf["sector"] = [get_label(i) for i in range(len(gdf))]

    # Cleanup
    return gdf.drop(columns=["sort_key"])


def compute_distribution_stats(
    df: pd.DataFrame | gpd.GeoDataFrame,
    value_col: str = "V",
    group_col: str | None = None,
) -> pd.DataFrame | pd.Series:
    """
    Compute robust distribution statistics for a value column, optionally grouped.

    Args:
        df: Input DataFrame or GeoDataFrame containing points.
        value_col: Column name to aggregate (e.g. "V").
        group_col: Column name to group by (e.g. "sector").

    Returns:
        DataFrame or Series with statistics (mean, std, median, nmad, etc.).
        Columns/Index are prefixed with 'v_' based on value_col name logic.
    """

    def _stats_func(x):
        if len(x) == 0:
            return pd.Series(
                {
                    "mean": np.nan,
                    "std": np.nan,
                    "median": np.nan,
                    "mad": np.nan,
                    "nmad": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "percentile_5": np.nan,
                    "percentile_95": np.nan,
                    "quartile_1": np.nan,
                    "quartile_3": np.nan,
                    "iqr": np.nan,
                    "n_points": 0,
                }
            )

        med = np.median(x)
        abs_dev = np.abs(x - med)
        mad = np.median(abs_dev)
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)

        return pd.Series(
            {
                "mean": np.mean(x),
                "std": np.std(x),
                "median": med,
                "mad": mad,
                "nmad": 1.4826 * mad,
                "min": np.min(x),
                "max": np.max(x),
                "percentile_5": np.percentile(x, 5),
                "percentile_95": np.percentile(x, 95),
                "quartile_1": q1,
                "quartile_3": q3,
                "iqr": q3 - q1,
                "n_points": int(len(x)),
            }
        )

    # Filter NaNs in value column
    clean_df = df.dropna(subset=[value_col])

    if group_col:
        # Grouped stats -> DataFrame (rows=groups, columns=stats)
        out = clean_df.groupby(group_col)[value_col].apply(_stats_func)
        if isinstance(out, pd.Series):
            # If only one group or weird unstack behavior
            out = out.unstack()
        else:
            out = out.unstack()

        # Rename columns: v_mean, v_std ... but keep n_points columns
        out.columns = [f"v_{c}" if c != "n_points" else c for c in out.columns]
        return out
    else:
        # Flat stats -> Series
        if clean_df.empty:
            out = _stats_func(np.array([]))
        else:
            out = _stats_func(clean_df[value_col].to_numpy())
        out.index = [f"v_{c}" if c != "n_points" else c for c in out.index]
        return out


def compute_sector_stats(
    sector_gdf: gpd.GeoDataFrame,
    points_df: pd.DataFrame,
    value_col: str = "V",
    group_col: str = "sector",
    inplace: bool = False,
) -> gpd.GeoDataFrame | None:
    """
    Compute geometric properties and value distribution statistics per sector.
    Appends statistics to the sector GeoDataFrame.

    Args:
        sector_gdf: GeoDataFrame containing sector polygons.
        points_df: DataFrame containing point data with group_col.
        value_col: Column in points_df for statistics (e.g. velocity).
        group_col: Column to link sectors and points.
        inplace: Whether to modify sector_gdf in place.

    Returns:
        GeoDataFrame with appended statistics if inplace=False, else None.
    """
    if not inplace:
        sector_gdf = sector_gdf.copy()

    # 1. Compute Point Statistics (grouped) using the simpler function
    if group_col in points_df.columns:
        stats_df = compute_distribution_stats(
            points_df, value_col=value_col, group_col=group_col
        )
    else:
        logger.warning(
            f"Group column '{group_col}' missing in points DF. Skipping point stats."
        )
        stats_df = pd.DataFrame()

    # 2. Compute Geometry Statistics
    if not sector_gdf.empty:
        # Initialize columns
        sector_gdf["area_px2"] = sector_gdf.geometry.area
        sector_gdf["perimeter_px"] = sector_gdf.geometry.length
        sector_gdf["centroid_x"] = sector_gdf.geometry.centroid.x
        sector_gdf["centroid_y"] = sector_gdf.geometry.centroid.y
        # Compactness: (4 * pi * A) / P^2
        sector_gdf["compactness"] = (4.0 * np.pi * sector_gdf["area_px2"]) / (
            sector_gdf["perimeter_px"] ** 2 + 1e-12
        )

    # 3. Merge Point Stats
    if not stats_df.empty:
        if group_col not in sector_gdf.columns:
            logger.warning(
                f"Group column '{group_col}' missing in sector GDF. Cannot join stats."
            )
        else:
            # Merge stats into sector_gdf
            # stats_df index is the group label. We left join.
            merged = sector_gdf.merge(
                stats_df, left_on=group_col, right_index=True, how="left"
            )

            # Assign mixed columns back
            for col in stats_df.columns:
                sector_gdf[col] = merged[col]

    # 4. Derived stats
    if "n_points" in sector_gdf.columns:
        sector_gdf["n_points"] = sector_gdf["n_points"].fillna(0)
        sector_gdf["point_density_pts_per_px2"] = (
            sector_gdf["n_points"] / sector_gdf["area_px2"].replace(0, np.nan)
        ).fillna(0)
    else:
        sector_gdf["n_points"] = 0
        sector_gdf["point_density_pts_per_px2"] = 0.0

    logger.info(f"Computed stats for {len(sector_gdf)} sectors.")

    if not inplace:
        return sector_gdf
    return None


# ================= Helper Functions for Cleaning =================#
def split_disconnected_polygons(gdf_in: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Explodes MultiPolygons into individual Polygons and assigns unique cluster IDs.

    Args:
        gdf_in: Input GeoDataFrame with (possibly) MultiPolygon geometries.

    Returns:
        GeoDataFrame with only single Polygon geometries and unique cluster IDs.
    """
    # Explode multipolygons to single polygons
    out = gdf_in.explode(index_parts=False).reset_index(drop=True)

    # Remove empty or invalid geometries
    out = out[~out.is_empty & out.geometry.notna()].copy()

    # Re-assign unique cluster IDs
    out["cluster_id"] = range(len(out))
    return out.reset_index(drop=True)


def filter_small_sectors(
    gdf: gpd.GeoDataFrame, min_area_px2: float
) -> gpd.GeoDataFrame:
    """
    Remove polygons smaller than a minimum area threshold.

    Args:
        gdf: GeoDataFrame of polygons.
        min_area_px2: Minimum area threshold.

    Returns:
        Filtered GeoDataFrame.
    """
    # Remove very small noise polygons
    gdf["area"] = gdf.geometry.area
    n_before = len(gdf)
    gdf_clean = gdf[gdf["area"] >= min_area_px2].reset_index(drop=True)

    if len(gdf_clean) < n_before:
        logger.info(
            f"Removed {n_before - len(gdf_clean)} small noise polygons (< {min_area_px2} px²)"
        )
    return gdf_clean


def remove_isolated_sectors(
    gdf: gpd.GeoDataFrame, isolation_buffer: float
) -> gpd.GeoDataFrame:
    """
    Remove polygons that do not intersect with any other polygon (buffered).

    Args:
        gdf: GeoDataFrame of polygons.
        isolation_buffer: Buffer distance for intersection check.

    Returns:
        Filtered GeoDataFrame.
    """
    keep_indices = []
    for idx in gdf.index:
        geom = gdf.geometry[idx]
        buffered = geom.buffer(isolation_buffer)
        others = gdf.drop(idx)

        if others.empty or others.geometry.intersects(buffered).any():
            keep_indices.append(idx)
        else:
            logger.info(f"Removing isolated sector ID {idx} (area={geom.area:.0f})")

    return gdf.loc[keep_indices].reset_index(drop=True)


def merge_contained_sectors(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Merge polygons that are contained within others or fill holes.

    Args:
        gdf: GeoDataFrame of polygons.

    Returns:
        GeoDataFrame with merged polygons.
    """
    gdf = gdf.copy()
    changed = True
    while changed:
        changed = False
        for i in gdf.index:
            geom_i = gdf.geometry[i]
            for j in gdf.index:
                if i == j:
                    continue

                geom_j = gdf.geometry[j]

                # Check strict containment or hole filling
                is_contained = geom_i.within(geom_j)
                is_hole = any(
                    Polygon(interior).equals(geom_i) for interior in geom_j.interiors
                )

                if is_contained or is_hole:
                    logger.info(f"Merging polygon {i} into container {j}")
                    gdf.at[j, "geometry"] = geom_j.union(geom_i)
                    gdf = gdf.drop(i).reset_index(drop=True)
                    changed = True
                    break
            if changed:
                break
    return gdf


def merge_sectors_by_velocity(
    gdf: gpd.GeoDataFrame,
    df_points: pd.DataFrame,
    threshold: float,
    target_n: int,
) -> gpd.GeoDataFrame:
    """
    Iteratively merge sectors based on velocity similarity until target is reached.

    Args:
        gdf: GeoDataFrame of polygons.
        df_points: DataFrame of points with velocities.
        threshold: Z-score threshold for merging.
        target_n: Target number of sectors.

    Returns:
        GeoDataFrame with merged sectors.
    """

    def _get_robust_stats(values: np.ndarray) -> tuple[float, float]:
        """Calculate median and NMAD for robust statistics."""
        if len(values) == 0:
            return 0.0, 1.0
        med = np.median(values)
        nmad = 1.4826 * np.median(np.abs(values - med))
        return med, nmad

    gdf = gdf.copy()

    # Prepare point data
    gdf_pts = gpd.GeoDataFrame(
        df_points, geometry=gpd.points_from_xy(df_points["x"], df_points["y"])
    )
    # Initial Spatial Join
    pts_joined = gpd.sjoin(gdf_pts, gdf[["geometry"]], how="inner", predicate="within")

    while len(gdf) > target_n:
        gdf["area"] = gdf.geometry.area
        idx_small = gdf["area"].idxmin()
        geom_small = gdf.geometry[idx_small]

        # Get stats for candidate
        vals_small = pts_joined.loc[pts_joined["index_right"] == idx_small, "V"].values
        med_small, nmad_small = _get_robust_stats(vals_small)

        logger.info(
            f"Checking smallest sector {idx_small}: Area={gdf.at[idx_small, 'area']:.0f}, V={med_small:.2f}±{nmad_small:.2f}"
        )

        # Search Neighbors
        search_geom = geom_small.buffer(np.sqrt(gdf.at[idx_small, "area"]) * 0.05)
        best_neighbor = None
        min_score = float("inf")

        for idx_other in gdf.index:
            if idx_other == idx_small:
                continue
            if not search_geom.intersects(gdf.geometry[idx_other]):
                continue

            # Compare stats
            vals_other = pts_joined.loc[
                pts_joined["index_right"] == idx_other, "V"
            ].values
            med_other, nmad_other = _get_robust_stats(vals_other)

            sigma_comb = np.sqrt(nmad_small**2 + nmad_other**2) + 1e-6
            z_score = abs(med_small - med_other) / sigma_comb

            logger.info(
                f" -> Neighbor {idx_other}: V={med_other:.2f}, Z-score={z_score:.2f}"
            )

            if z_score < threshold and z_score < min_score:
                min_score = z_score
                best_neighbor = idx_other

        # Merge or Stop
        if best_neighbor is not None:
            logger.info(
                f"Merging {idx_small} -> {best_neighbor} (Score={min_score:.2f})"
            )
            # Union
            gdf.at[best_neighbor, "geometry"] = gdf.geometry[best_neighbor].union(
                geom_small
            )

            # Update Point references used for stats calculation
            mask_pts = pts_joined["index_right"] == idx_small
            pts_joined.loc[mask_pts, "index_right"] = best_neighbor

            # Handle index shift due to drop
            old_indices = gdf.index.tolist()
            old_indices.remove(idx_small)
            gdf = gdf.drop(idx_small).reset_index(drop=True)

            idx_map = {old: new for new, old in enumerate(old_indices)}
            pts_joined["index_right"] = pts_joined["index_right"].map(idx_map)
            pts_joined = pts_joined.dropna(subset=["index_right"])
        else:
            logger.info("No compatible neighbor found. Stopping iterative merge.")
            break

    return gdf


def enforce_sector_limit(gdf: gpd.GeoDataFrame, limit: int) -> gpd.GeoDataFrame:
    """
    Force reduction to N sectors by keeping the largest by area.

    Args:
        gdf: GeoDataFrame of polygons.
        limit: Maximum number of sectors to keep.

    Returns:
        GeoDataFrame with at most 'limit' sectors.
    """
    if len(gdf) > limit:
        logger.info(f"Forcing reduction to {limit} sectors (keeping largest).")
        gdf["area"] = gdf.geometry.area
        return gdf.nlargest(limit, columns="area").reset_index(drop=True)
    return gdf


def fill_polygon_holes(gdf: gpd.GeoDataFrame, threshold: float) -> gpd.GeoDataFrame:
    """
    Fills holes within polygons that are smaller than the threshold area.

    Args:
        gdf: GeoDataFrame of polygons.
        threshold: Area threshold for filling holes.

    Returns:
        GeoDataFrame with holes filled.
    """

    def _fill(geom):
        if geom is None or geom.is_empty:
            return geom

        if geom.geom_type == "Polygon":
            # Keep interiors (holes) only if they are larger than threshold
            # Smaller holes are removed (effectively filled)
            new_interiors = [i for i in geom.interiors if Polygon(i).area > threshold]
            return Polygon(geom.exterior, new_interiors)

        elif geom.geom_type == "MultiPolygon":
            parts = [_fill(p) for p in geom.geoms]
            return MultiPolygon(parts)

        return geom

    logger.info(f"Filling polygon holes smaller than {threshold}")

    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(_fill)
    return gdf


def smooth_polygons(
    gdf: gpd.GeoDataFrame,
    smooth_method: Literal["smoothify", "simplify"] = "smoothify",
    raster_res: float | None = None,
    smooth_iterations: int = 3,
    **kwargs,
) -> gpd.GeoDataFrame:
    """
    Smooth polygon geometries using specified method.

    Args:
        gdf: GeoDataFrame of polygons.
        smooth_method: Smoothing method ('smoothify' or 'simplify').
        raster_res: Raster resolution for smoothing.
        smooth_iterations: Number of smoothing iterations.
        **kwargs: Additional keyword arguments for smoothing.

    Returns:
        GeoDataFrame with smoothed geometries.
    """
    logger.info(f"Smoothing polygons using method '{smooth_method}'")

    if smooth_method == "smoothify":
        gdf = smoothify(
            gdf,
            segment_length=raster_res,
            smooth_iterations=smooth_iterations,
            **kwargs,
        )
    elif smooth_method == "simplify":
        logger.warning("This approach has not been tested yet. Use with caution.")
        gdf["geometry"] = gdf.geometry.simplify(
            tolerance=raster_res if raster_res else 1.0, preserve_topology=True
        )
    else:
        logger.warning(f"Unknown smooth_method '{smooth_method}'; skipping smoothing.")

    return gdf
