import logging
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
from affine import Affine
from shapely.geometry import Polygon, shape

logger = logging.getLogger("ppcx")


def vectorize_grid_to_gdf(
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


def split_disconnected_polygons(gdf_in: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Explodes MultiPolygons into individual Polygons and assigns unique cluster IDs.
    """
    # Explode multipolygons to single polygons
    out = gdf_in.explode(index_parts=False).reset_index(drop=True)

    # Remove empty or invalid geometries
    out = out[~out.is_empty & out.geometry.notna()].copy()

    # Re-assign unique cluster IDs
    out["cluster_id"] = range(len(out))
    return out.reset_index(drop=True)


def clean_morphokinematic_sectors(
    gdf_sectors: gpd.GeoDataFrame,
    df_points: pd.DataFrame,
    min_area_px2: float = 100000.0,
    isolation_buffer: float = 30.0,
    velocity_merge_threshold: float = 1.6,
    target_number_of_sectors: int = 4,
    force_minimum_sectors: bool = True,
) -> gpd.GeoDataFrame:
    """
    Clean polygon sectors by removing isolated ones, merging contained ones,
    and merging small ones based on robust velocity statistics (NMAD Z-test).
    """
    if gdf_sectors.empty:
        return gdf_sectors

    gdf = gdf_sectors.copy()
    logger.info(f"Starting sector cleaning with {len(gdf)} sectors.")

    # ---------------------------------------------------------
    # Step 0: Pre-processing
    # ---------------------------------------------------------
    # Ensure we work with single polygons (no MultiPolygons)
    gdf = split_disconnected_polygons(gdf)
    logger.info(f"After exploding multipolygons: {len(gdf)} sectors.")

    # Remove very small noise polygons immediately
    gdf["area"] = gdf.geometry.area
    n_before = len(gdf)
    gdf = gdf[gdf["area"] >= min_area_px2].reset_index(drop=True)
    if len(gdf) < n_before:
        logger.info(
            f"Removed {n_before - len(gdf)} small noise polygons (< {min_area_px2} px²)"
        )

    # ---------------------------------------------------------
    # Step 1: Remove Isolated Polygons
    # ---------------------------------------------------------
    # Logic: If a buffered polygon doesn't touch any other, it's an outlier.
    keep_indices = []
    for idx in gdf.index:
        geom = gdf.geometry[idx]
        # Buffer slightly to check proximity
        buffered = geom.buffer(isolation_buffer)

        others = gdf.drop(idx)
        if others.empty:
            keep_indices.append(idx)  # Only one exists, keep it
        elif others.geometry.intersects(buffered).any():
            keep_indices.append(idx)  # Connected
        else:
            logger.info(f"Removing isolated sector ID {idx} (area={geom.area:.0f})")

    gdf = gdf.loc[keep_indices].reset_index(drop=True)

    # ---------------------------------------------------------
    # Step 2: Merge Contained Polygons (Hole-Filling)
    # ---------------------------------------------------------
    # Logic: If A is inside B (or fills a hole in B), merge A into B.
    # We use a while loop to handle nested cases safely.
    changed = True
    while changed:
        changed = False
        for i in gdf.index:
            geom_i = gdf.geometry[i]

            # Check against all other polygons
            for j in gdf.index:
                if i == j:
                    continue

                geom_j = gdf.geometry[j]

                # Check 1: Strict containment
                is_contained = geom_i.within(geom_j)
                # Check 2: Hole filling (matches one of the interiors)
                is_hole = any(
                    Polygon(interior).equals(geom_i) for interior in geom_j.interiors
                )

                if is_contained or is_hole:
                    logger.info(f"Merging polygon {i} into container {j}")
                    # Update geometry of J to include I
                    gdf.at[j, "geometry"] = geom_j.union(geom_i)
                    # Remove I
                    gdf = gdf.drop(i).reset_index(drop=True)
                    changed = True
                    break  # Break inner loop, restart outer loop
            if changed:
                break

    # ---------------------------------------------------------
    # Step 3: Iterative Statistical Merging
    # ---------------------------------------------------------
    # Prepare point data for velocity stats
    gdf_pts = gpd.GeoDataFrame(
        df_points, geometry=gpd.points_from_xy(df_points["x"], df_points["y"])
    )

    # Initial Spatial Join (Expensive, done once)
    # We track points by index to update their cluster assignment cheaply later
    pts_joined = gpd.sjoin(gdf_pts, gdf[["geometry"]], how="inner", predicate="within")
    # index_right corresponds to the dataframe index in `gdf`

    def get_robust_stats(values):
        if len(values) == 0:
            return 0.0, 1.0  # default/safe return
        med = np.median(values)
        # NMAD = 1.4826 * median(|x - median|)
        nmad = 1.4826 * np.median(np.abs(values - med))
        return med, nmad

    while len(gdf) > target_number_of_sectors:
        gdf["area"] = gdf.geometry.area

        # Identify candidate: The smallest remaining sector
        idx_small = gdf["area"].idxmin()
        geom_small = gdf.geometry[idx_small]

        # Get stats for candidate
        # Note: pts_joined['index_right'] matches gdf index
        vals_small = pts_joined.loc[pts_joined["index_right"] == idx_small, "V"].values
        med_small, nmad_small = get_robust_stats(vals_small)

        logger.info(
            f"Checking smallest sector {idx_small}: Area={gdf.at[idx_small, 'area']:.0f}, V={med_small:.2f}±{nmad_small:.2f}"
        )

        # Find Neighbors
        # Small buffer proportional to size to find touching polygons
        search_geom = geom_small.buffer(np.sqrt(gdf.at[idx_small, "area"]) * 0.05)

        best_neighbor = None
        min_score = float("inf")

        for idx_other in gdf.index:
            if idx_other == idx_small:
                continue

            # Must be spatially adjacent
            if not search_geom.intersects(gdf.geometry[idx_other]):
                continue

            # Get neighbor stats
            vals_other = pts_joined.loc[
                pts_joined["index_right"] == idx_other, "V"
            ].values
            med_other, nmad_other = get_robust_stats(vals_other)

            # Robust Z-Score: |Diff| / Combined_Sigma
            sigma_comb = np.sqrt(nmad_small**2 + nmad_other**2) + 1e-6
            z_score = abs(med_small - med_other) / sigma_comb

            logger.info(
                f" -> Neighbor {idx_other}: V={med_other:.2f}, Z-score={z_score:.2f}"
            )

            # Check threshold
            if z_score < velocity_merge_threshold:
                if z_score < min_score:
                    min_score = z_score
                    best_neighbor = idx_other

        # Perform Merge if compatible neighbor found
        if best_neighbor is not None:
            logger.info(
                f"Merging {idx_small} -> {best_neighbor} (Score={min_score:.2f})"
            )

            # Union Geometries
            geom_new = gdf.geometry[best_neighbor].union(geom_small)
            gdf.at[best_neighbor, "geometry"] = geom_new

            # Update points reference: move points from small -> neighbor
            # This avoids re-running sjoin
            mask_pts = pts_joined["index_right"] == idx_small
            pts_joined.loc[mask_pts, "index_right"] = best_neighbor

            # Remove small sector
            # Note: Drop changes indices, so we must update pts_joined indices too!
            # IMPORTANT: Resetting index usually breaks external references.
            # Strategy: Map old indices to new indices.

            old_indices = gdf.index.tolist()
            old_indices.remove(idx_small)  # This index is gone

            gdf = gdf.drop(idx_small).reset_index(drop=True)

            # Create map: old_index -> new_index
            # Since we reset index, the remaining rows just shift up.
            # old_indices[0] becomes 0, old_indices[1] becomes 1...
            idx_map = {old: new for new, old in enumerate(old_indices)}

            # Update points dataframe to match new gdf indices
            pts_joined["index_right"] = pts_joined["index_right"].map(idx_map)

            # Clean up any points that belonged to dropped sector but weren't reassigned (shouldn't happen here)
            pts_joined = pts_joined.dropna(subset=["index_right"])

        else:
            logger.info("No compatible neighbor found. Stopping iterative merge.")
            break

    # ---------------------------------------------------------
    # Step 4: Final Force Limit (Optional)
    # ---------------------------------------------------------
    logger.info(f"Sectors remaining: {len(gdf)}")
    if force_minimum_sectors and len(gdf) > target_number_of_sectors:
        logger.info(
            f"Forcing reduction to {target_number_of_sectors} sectors (keeping largest)."
        )
        gdf["area"] = gdf.geometry.area
        gdf = gdf.nlargest(target_number_of_sectors, columns="area").reset_index(
            drop=True
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
    order_by: Literal["y_position", "area"] = "y_position",
    ascending: bool = True,
    label_prefix: str = "",
) -> gpd.GeoDataFrame:
    """
    Assign letter labels (A, B, C...) to the GeoDataFrame based on geometry.
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
    if order_by == "y_position":
        gdf["sort_key"] = gdf.geometry.centroid.y
    elif order_by == "area":
        gdf["sort_key"] = gdf.geometry.area

    gdf = gdf.sort_values("sort_key", ascending=ascending).reset_index(drop=True)

    # Assign letters
    gdf["label"] = [get_label(i) for i in range(len(gdf))]

    # Cleanup
    return gdf.drop(columns=["sort_key"])


def compute_sector_stats(
    sector_gdf: gpd.GeoDataFrame,
    points_df: pd.DataFrame,  # This df has 'sector', 'V', etc.
    value_col: str = "V",
) -> pd.DataFrame:
    """
    Compute comprehensive statistics per sector.

    Calculates geometric properties (area, perimeter, compactness) and
    distribution statistics of the values (e.g. velocity) for points falling
    within each sector.
    """

    def _compute_distribution_stats(arr: np.ndarray) -> dict[str, float]:
        """Helper to compute distribution statistics."""
        if len(arr) == 0:
            return {}

        median_val = np.median(arr)
        mad = np.median(np.abs(arr - median_val))
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)

        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(median_val),
            "mad": float(mad),
            "nmad": float(mad * 1.4826),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "percentile_5": float(np.percentile(arr, 5)),
            "percentile_95": float(np.percentile(arr, 95)),
            "quartile_1": float(q1),
            "quartile_3": float(q3),
            "iqr": float(q3 - q1),
        }

    stats_list = []

    for _, row in sector_gdf.iterrows():
        label = row["label"]
        geom = row["geometry"]

        if geom is None or geom.is_empty:
            continue

        # Points in this sector
        points_in = points_df[points_df["sector"] == label]
        n_points = len(points_in)

        # Geometric properties
        area = float(geom.area)
        perimeter = float(geom.length)
        centroid = geom.centroid
        centroid_x, centroid_y = centroid.x, centroid.y

        # Compactness (Isoperimetric quotient: 1 for circle, <1 for others)
        compactness = (
            (4.0 * np.pi * area) / (perimeter**2 + 1e-12) if perimeter > 0 else np.nan
        )
        density = n_points / area if area > 0 else 0.0

        # Value statistics
        v_stats = {}
        vals = points_in[value_col].dropna().values

        if len(vals) > 0:
            raw_stats = _compute_distribution_stats(vals)
            # Prefix keys with v_
            v_stats = {f"v_{k}": v for k, v in raw_stats.items()}
        else:
            # Populate with NaNs if no points
            empty_keys = [
                "mean",
                "std",
                "median",
                "mad",
                "nmad",
                "min",
                "max",
                "percentile_5",
                "percentile_95",
                "quartile_1",
                "quartile_3",
                "iqr",
            ]
            v_stats = {f"v_{k}": np.nan for k in empty_keys}

        stats_list.append(
            {
                "label": label,
                "n_points": n_points,
                "area_px2": area,
                "point_density_pts_per_px2": density,
                "perimeter_px": perimeter,
                "compactness": compactness,
                "centroid_x": float(centroid_x),
                "centroid_y": float(centroid_y),
                **v_stats,
            }
        )

    if not stats_list:
        return pd.DataFrame()

    df = pd.DataFrame(stats_list)

    # Sort by label
    df = df.sort_values("label").reset_index(drop=True)

    logger.info(
        f"Computed stats for {len(df)} sectors (total points: {df['n_points'].sum()})"
    )

    return df
