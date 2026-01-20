import logging
from pathlib import Path
from typing import Literal

import geopandas as gpd
import joblib
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import rasterio.features
from affine import Affine
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from shapely import MultiPolygon
from shapely.geometry import Polygon, shape
from smoothify import smoothify

logger = logging.getLogger("ppcx")


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
    raster_res: float | None = None,
    smooth_iterations: int = 4,
    merge_collection: bool = True,
    merge_field: str = "cluster_id",
    area_tolerance: float = 0.5,
) -> gpd.GeoDataFrame:
    """
    Clean polygon sectors by removing isolated ones, merging contained ones,
    and merging small ones based on robust velocity statistics (NMAD Z-test).
    """
    if gdf_sectors.empty:
        return gdf_sectors

    logger.info(f"Starting sector cleaning with {len(gdf_sectors)} sectors.")

    # 0. Ensure we work with single polygons
    gdf = _split_disconnected_polygons(gdf_sectors)
    logger.info(f"After exploding multipolygons: {len(gdf)} sectors.")

    # 3. Merge Contained Polygons (Hole-Filling)
    gdf = _merge_contained_sectors(gdf)

    # 1. Pre-processing
    gdf = _filter_small_sectors(gdf, min_area_px2)

    # 2. Remove Isolated Polygons
    gdf = _remove_isolated_sectors(gdf, isolation_buffer)

    # 4. Iterative Statistical Merging
    gdf = _merge_sectors_by_velocity(
        gdf, df_points, velocity_merge_threshold, target_number_of_sectors
    )

    # 5. Final Force Limit
    if force_minimum_sectors:
        gdf = _enforce_sector_limit(gdf, target_number_of_sectors)

    # 6. Fill Holes (if requested)
    if fill_holes_area > 0.0:
        gdf = _fill_polygon_holes(gdf, fill_holes_area)

    # 7. Smooth Geometries (optional)
    if smooth_geometries:
        gdf = smoothify(
            gdf,
            segment_length=raster_res,
            smooth_iterations=smooth_iterations,
            merge_collection=merge_collection,
            merge_field=merge_field,
            num_cores=target_number_of_sectors,
            area_tolerance=area_tolerance,
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


def plot_kinematic_sectors(
    velocity_df: pd.DataFrame,
    sector_gdf: gpd.GeoDataFrame,
    sector_stats: pd.DataFrame,
    img: np.ndarray,
    sector_colors: dict | None,
    output_dir: Path,
    base_name: str,
    figsize: tuple = (18, 7),
    dpi: int = 300,
) -> Path:
    """
    Plot morpho-kinematic sectors summary with velocity field and statistics table.
    """

    from matplotlib import colors as mcolors

    # Configuration
    label_column = "label"
    min_cbar_percentile = 5.0
    max_cbar_percentile = 95.0
    stat_cols = [
        "label",
        "v_mean",
        "v_std",
        "v_median",
        "v_mad",
        "n_points",
        "area_px2",
        "compactness",
    ]
    max_labels_in_table = 12

    if sector_colors is None:
        sector_colors = {}

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.3, 1.0], "wspace": 0.25},
    )
    ax_sectors, ax_table = axes

    # 1) Velocity field with sectors overlay
    mags = velocity_df["V"].to_numpy()
    vmin = np.percentile(mags, min_cbar_percentile)
    vmax = np.percentile(mags, max_cbar_percentile)

    ax_sectors.imshow(img, cmap="gray")
    norm = Normalize(vmin=vmin, vmax=vmax)
    q = ax_sectors.quiver(
        velocity_df["x"].to_numpy(),
        velocity_df["y"].to_numpy(),
        velocity_df["u"].to_numpy(),
        velocity_df["v"].to_numpy(),
        mags,
        norm=norm,
        scale=None,
        scale_units="xy",
        angles="xy",
        cmap="viridis",
        width=0.006,
        headwidth=2.0,
    )
    ax_sectors.set_aspect("equal")
    ax_sectors.set_xticks([])
    ax_sectors.set_yticks([])
    cbar = plt.colorbar(q, ax=ax_sectors, fraction=0.046, pad=0.03)
    cbar.set_label("Velocity [px/day]", rotation=270, labelpad=12, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax_sectors.set_title("Velocity Field", fontsize=11)

    # Sectors colors mapping
    present_labels = sorted(sector_gdf[label_column].unique())
    colors = {}
    fallback_cmap = plt.get_cmap("tab10")
    for i, label in enumerate(present_labels):
        if label in sector_colors:
            colors[label] = sector_colors[label]
        else:
            # Fallback color based on index
            colors[label] = mcolors.to_hex(fallback_cmap(i % 10))

    # Plot sectors
    plot_gdf = sector_gdf.copy()
    plot_gdf["color"] = plot_gdf["label"].map(colors)

    if not plot_gdf.empty:
        # 1. Plot the fill with high transparency to show velocity vectors
        plot_gdf.plot(
            ax=ax_sectors,
            color=plot_gdf["color"],
            alpha=0.1,
            linewidth=0,
        )
        # 2. Plot the edges with full opacity to define limits
        plot_gdf.plot(
            ax=ax_sectors,
            facecolor="none",
            edgecolor=plot_gdf["color"],
            linewidth=2.5,
            alpha=1.0,
        )

    # Add legend manually to match style
    legend_patches = [
        mpatches.Patch(color=colors[label], label=label, alpha=0.8)
        for label in present_labels
        if label in colors
    ]
    if legend_patches:
        ax_sectors.legend(
            handles=legend_patches,
            loc="upper right",
            fontsize=8,
            framealpha=0.9,
        )

    # Add letters to sectors
    for _, row in plot_gdf.iterrows():
        cent = row.geometry.centroid
        ax_sectors.text(
            cent.x,
            cent.y,
            row["label"],
            fontsize=12,
            weight="bold",
            color="white",
            ha="center",
            va="center",
        )

    ax_sectors.set_aspect("equal")
    ax_sectors.set_xticks([])
    ax_sectors.set_yticks([])
    ax_sectors.set_title("Morpho-Kinematic Sectors", fontsize=11)

    # 3) Statistics table
    available = [c for c in stat_cols if c in sector_stats.columns]
    if "label" not in available:
        logger.warning("sector_stats has no 'label' column; skipping table.")
        display_df = pd.DataFrame()
    else:
        display_df = sector_stats[available].copy()

    ax_table.axis("off")

    if not display_df.empty:
        # Formatting
        for c in display_df.columns:
            if c == "label":
                continue
            if c in {"n_points", "area_px2"}:
                display_df[c] = display_df[c].round(0).astype(int)
            else:
                display_df[c] = display_df[c].round(2)

        # Limit rows
        if display_df.shape[0] > max_labels_in_table:
            display_df = display_df.iloc[:max_labels_in_table, :]

        table_df = display_df.set_index("label").T
        table = ax_table.table(
            cellText=table_df.values,
            colLabels=list(table_df.columns),
            rowLabels=list(table_df.index),
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.05, 1.6)

        # Style table headers
        for (i, j), cell in table.get_celld().items():
            if i == 0 or j == -1:
                cell.set_facecolor("#E8E8E8")
                cell.set_text_props(weight="bold", size=7)
            else:
                cell.set_facecolor("white")

    ax_table.set_title("Sector Statistics", fontsize=11, pad=6)

    fig.suptitle(base_name, fontsize=13, weight="bold", y=0.985)

    out_path = output_dir / f"{base_name}_sectors_summary.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved summary figure to {out_path}")
    return out_path


def save_sector_results(
    output_dir: Path,
    base_name: str,
    *,
    gdf_sectors: gpd.GeoDataFrame,
    mk_stats: pd.DataFrame,
    posterior_probs: np.ndarray | None = None,
    cluster_pred: np.ndarray | None = None,
    uncertainty: np.ndarray | None = None,
    img_shape: tuple[int, int] | None = None,
    export_geojson: bool = True,
    export_shapefile: bool = False,
) -> dict[str, Path]:
    """
    Persist final sector results.

    Saves:
    - Bundle (.joblib): python dict with dataframes
    - Arrays (.npz): raw numpy arrays
    - GIS (.geojson/.shp): Merged geometry + stats, optionally with Y-inversion.
    """
    artifacts: dict[str, Path] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Pythonized bundle (Simplified: just dict of DF/GDF/Arrays)
    try:
        # Convert GDF to list-dict for plain pickling if needed, or just keep GDF
        bundle = {
            "gdf_sectors": gdf_sectors,
            "mk_stats": mk_stats,
            "posterior_probs": posterior_probs,
            "cluster_pred": cluster_pred,
            "uncertainty": uncertainty,
        }
        py_path = output_dir / f"{base_name}_sectors_bundle.joblib"
        joblib.dump(bundle, py_path)
        artifacts["python_bundle"] = py_path
    except Exception as exc:
        logger.warning(f"Failed saving joblib bundle: {exc}")

    # 2) NumPy arrays
    try:
        np_path = output_dir / f"{base_name}_sectors_arrays.npz"
        np.savez_compressed(
            np_path,
            cluster_pred=cluster_pred if cluster_pred is not None else np.array([]),
            posterior_probs=posterior_probs
            if posterior_probs is not None
            else np.array([]),
            uncertainty=uncertainty if uncertainty is not None else np.array([]),
        )
        artifacts["numpy_arrays"] = np_path
    except Exception as exc:
        logger.warning(f"Failed saving npz arrays: {exc}")

    # 3) GIS export
    if export_geojson or export_shapefile:
        if gdf_sectors.empty:
            logger.warning("No geometries to export for GIS.")
            return artifacts

        try:
            # Merge stats into GDF for a rich export file
            export_gdf = gdf_sectors.merge(mk_stats, on="label", how="left")

            # Handle Y-Inversion for GIS compatibility (Image Origin vs Map Origin)
            H = img_shape[0] if (img_shape and len(img_shape) >= 1) else None

            if H is not None:
                # Invert Y coordinate: y_new = H - 1 - y_old
                def invert_y(geom):
                    if geom.is_empty:
                        return geom
                    # shapely scalar transform
                    return scale(
                        translate(geom, yoff=-(H - 1)), yfact=-1, origin=(0, 0)
                    )
                    # Or manual rebuild to be safe without imports:
                    # return Polygon([(x, H - 1 - y) for x, y in geom.exterior.coords])

                # Manual reliable inversion loop for polygons
                new_geoms = []
                for geom in export_gdf.geometry:
                    if geom.geom_type == "Polygon":
                        new_g = Polygon(
                            [(x, H - 1 - y) for x, y in geom.exterior.coords]
                        )
                        new_geoms.append(new_g)
                    elif geom.geom_type == "MultiPolygon":
                        # Handle multi-polygons if present
                        parts = [
                            Polygon([(x, H - 1 - y) for x, y in p.exterior.coords])
                            for p in geom.geoms
                        ]
                        from shapely.ops import unary_union

                        new_geoms.append(unary_union(parts))
                    else:
                        new_geoms.append(geom)  # Fallback

                export_gdf.geometry = new_geoms

            if export_geojson:
                geojson_path = output_dir / f"{base_name}_sectors.geojson"
                export_gdf.to_file(geojson_path, driver="GeoJSON")
                artifacts["geojson"] = geojson_path

            if export_shapefile:
                shp_dir = output_dir / f"{base_name}_shp"
                shp_dir.mkdir(exist_ok=True)
                shp_path = shp_dir / f"{base_name}_sectors.shp"
                export_gdf.to_file(shp_path)
                artifacts["shapefile"] = shp_path

        except Exception as exc:
            logger.warning(f"Failed GIS export: {exc}")

    return artifacts


# ================= Helper Functions for Cleaning =================#
def _split_disconnected_polygons(gdf_in: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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


def _filter_small_sectors(
    gdf: gpd.GeoDataFrame, min_area_px2: float
) -> gpd.GeoDataFrame:
    """Explode multipolygons and remove those smaller than threshold."""
    # Remove very small noise polygons
    gdf["area"] = gdf.geometry.area
    n_before = len(gdf)
    gdf_clean = gdf[gdf["area"] >= min_area_px2].reset_index(drop=True)

    if len(gdf_clean) < n_before:
        logger.info(
            f"Removed {n_before - len(gdf_clean)} small noise polygons (< {min_area_px2} px²)"
        )
    return gdf_clean


def _remove_isolated_sectors(
    gdf: gpd.GeoDataFrame, isolation_buffer: float
) -> gpd.GeoDataFrame:
    """Remove polygons that do not intersect with any other polygon (buffered)."""
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


def _merge_contained_sectors(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Diffusely merge polygons contained within others or filling holes."""
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


def _merge_sectors_by_velocity(
    gdf: gpd.GeoDataFrame,
    df_points: pd.DataFrame,
    threshold: float,
    target_n: int,
) -> gpd.GeoDataFrame:
    """Iteratively merge sectors based on velocity similarity until target is reached."""

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


def _enforce_sector_limit(gdf: gpd.GeoDataFrame, limit: int) -> gpd.GeoDataFrame:
    """Force reduction to N sectors by keeping the largest by area."""
    if len(gdf) > limit:
        logger.info(f"Forcing reduction to {limit} sectors (keeping largest).")
        gdf["area"] = gdf.geometry.area
        return gdf.nlargest(limit, columns="area").reset_index(drop=True)
    return gdf


def _fill_polygon_holes(gdf: gpd.GeoDataFrame, threshold: float) -> gpd.GeoDataFrame:
    """Fills holes within polygons that are smaller than the threshold area."""

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
