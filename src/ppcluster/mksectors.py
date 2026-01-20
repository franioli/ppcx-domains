import logging
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
from affine import Affine
from shapely.geometry import shape
from smoothify import smoothify

logger = logging.getLogger("ppcx")


def vectorize_grid_to_gdf(
    cluster_grid: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    smooth_iterations: int = 3,
    merge_collection: bool = True,
    merge_field: str = "cluster_id",
    **kwargs,
) -> gpd.GeoDataFrame:
    """
    Vectorize a grid of cluster IDs into a GeoDataFrame using smoothify.

    Args:
        cluster_grid: 2D array of IDs (int or float).
        X, Y: 2D meshgrids of coordinates.
        smooth_iterations: Number of smoothing iterations.


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

    # 3. Apply Smoothify
    if smooth_iterations > 0:
        try:
            gdf = smoothify(
                gdf,
                segment_length=abs(dx),
                smooth_iterations=smooth_iterations,
                merge_collection=merge_collection,
                merge_field=merge_field,
                **kwargs,
            )
        except Exception as e:
            logger.warning(f"Smoothify failed ({e}), using raw polygons.")

    # Drop empty or invalid geometries
    gdf = gdf[~gdf.is_empty & gdf.geometry.notna()].copy()

    # Explode MultiPolygons to keep only largest component per cluster?
    # Usually we want one row per cluster loc.
    # For sector assignment, we treat the whole cluster ID as one entity.

    return gdf


def assign_sector_labels(
    gdf: gpd.GeoDataFrame,
    order_by: Literal["y_position", "area"] = "y_position",
    ascending: bool = True,
    label_prefix: str = "",
) -> gpd.GeoDataFrame:
    """
    Assign letter labels (A, B, C...) to the GeoDataFrame based on geometry.
    """
    if gdf.empty:
        return gdf

    # Calculate sort key
    if order_by == "y_position":
        gdf["sort_key"] = gdf.geometry.centroid.y
    elif order_by == "area":
        gdf["sort_key"] = gdf.geometry.area

    gdf = gdf.sort_values("sort_key", ascending=ascending).reset_index(drop=True)

    # Assign letters
    from string import ascii_uppercase

    def get_label(idx):
        if label_prefix:
            return f"{label_prefix}{idx}"
        if idx < len(ascii_uppercase):
            return ascii_uppercase[idx]
        return f"S{idx}"

    gdf["label"] = [get_label(i) for i in range(len(gdf))]

    # Cleanup
    return gdf.drop(columns=["sort_key"])


def classify_points(
    gdf_sectors: gpd.GeoDataFrame,
    points_df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
) -> pd.DataFrame:
    """
    Classify a dataframe of points into sectors using the GeoDataFrame polygons.
    Returns the original dataframe with a new 'sector' column.
    """
    # 1. Convert points to GeoDataFrame
    gdf_pts = gpd.GeoDataFrame(
        points_df, geometry=gpd.points_from_xy(points_df[x_col], points_df[y_col])
    )

    # 2. Spatial Join
    # 'inner' will drop points outside any sector.
    # 'left' keeps them with NaN sector.
    joined = gpd.sjoin(
        gdf_pts, gdf_sectors[["geometry", "label"]], how="left", predicate="within"
    )

    # 3. Handle duplicates (if polygons overlap) - take the first match or specific logic
    # smoothify with merge_collection=True usually prevents overlap,
    # but sjoin might still find edge cases.
    joined = joined[~joined.index.duplicated(keep="first")]

    # Restore to DataFrame
    out_df = pd.DataFrame(joined.drop(columns="geometry"))
    out_df["sector"] = out_df["label"].fillna("")
    if "label" in out_df.columns:
        out_df = out_df.drop(columns=["label"])

    return out_df


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
