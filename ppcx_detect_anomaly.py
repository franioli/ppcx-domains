import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
from PIL import Image

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import DictConfig, ListConfig, OmegaConf

from ppcluster import Timer, load_config, setup_logger
from ppcluster.cvat import (
    filter_dataframe_by_polygons,
)
from ppcluster.data import (
    find_ensemble_files,
    read_data_from_pylamma_nc,
)
from ppcluster.griddata import create_2d_grid
from ppcluster.mcmc.clustering import (
    clusterize_gaussian_mixture,
    save_sampling_summary,
)
from ppcluster.mcmc.priors import plot_spatial_priors
from ppcluster.preprocessing import (
    apply_dic_filters,
    spatial_subsample,
    transform_and_scale_features,
)
from ppcluster.sectors import (
    classify_points_by_polygons,
    compute_sector_stats,
    vectorize_gridded_sectors,
)
from ppcluster.visualization import (
    plot_dic_vectors,
)

matplotlib.use("Agg")

from smoothify import smoothify

logger = setup_logger(level=logging.INFO, name="ppcx")

CONFIG_PATH = "config_anomaly.yaml"  # Path to the config file. Can be overwritten by --config argument in CLI.
HEADLESS = True  # set to True when running in non-GUI environment

if HEADLESS:
    plt.switch_backend("Agg")


def parse_arguments():
    p = argparse.ArgumentParser(
        description="Run Anomaly Detection on existing Domain Classification results."
    )
    p.add_argument(
        "--date",
        "-d",
        help="Reference date (YYYY-MM-DD). Overrides config.",
        default=None,
    )
    p.add_argument(
        "--sectors-file",
        "-s",
        help="Path to the sectors GeoJSON file (output of domain classification). If not provided, tries to find it automatically.",
        default=None,
    )
    p.add_argument(
        "--config",
        "-c",
        type=str,
        default="config_anomaly.yaml",
        help="Path to anomaly config file.",
    )
    p.add_argument(
        "--output_dir",
        "-o",
        help="Output directory override.",
        default=None,
    )
    # Robust generic override mechanism using dotlist (e.g., data.dt_min=24)
    # This captures any leftover arguments in the format key=value
    p.add_argument(
        "overrides",
        nargs="*",
        help="Configuration overrides in dotlist format (e.g., data.dt_min=24 mcmc.sample_options.draws=500).",
    )
    return p.parse_args()


def run_anomaly_detection(
    dic_df: pd.DataFrame,
    sectors_gdf: gpd.GeoDataFrame,
    target_sector: str,
    config: DictConfig | ListConfig,
    output_dir: Path,
    img: Any,
    base_name: str,
    sector_buffer: float = 50.0,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Perform a second-pass clustering on a specific sector (e.g., Sector A)
    to detect sub-anomalies (e.g., small high-velocity clusters).

    Args:
        dic_df: The full preprocessed DIC DataFrame.
        sectors_gdf: The GeoDataFrame of sectors from the main clustering step.
        target_sector: The name of the sector to refine (e.g., "Sector A").
        config: The full configuration object for parameters.
        output_dir: Directory to save outputs.
        img: The original image for plotting.
        base_name: Base name for output files.
        sector_buffer: Buffer distance in pixels to apply around the target sector for point selection.

    Returns:
        gpd.GeoDataFrame: The refined geometries for the target sector with anomaly sub-sectors.
        gpd.GeoDataFrame: The classified points for the target sector with anomaly labels.
    """
    anomaly_config = config.anomaly_detection

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    anomaly_base_name = f"{base_name}_sector{target_sector}_anomaly"

    # 1. Filter data for the target sector (Add buffer to include boundary points)
    # We use the already classified points or geometric filtering
    sector_row = sectors_gdf[sectors_gdf["sector"] == target_sector]
    if sector_row.empty:
        logger.warning(f"Sector {target_sector} not found. Skipping refinement.")
        return sectors_gdf, dic_df
    sector_poly = sector_row.geometry.iloc[0]

    # Create a buffer to include frontier points that might have been smoothed out
    # 64px is standard grid spacing, so 100px ensures we capture immediate neighbors
    logger.info(
        f"Applying {sector_buffer}px buffer to Sector {target_sector} for anomaly search."
    )
    sector_poly_buffered = sector_poly.buffer(sector_buffer)

    df_sub = filter_dataframe_by_polygons(dic_df, polygon=sector_poly_buffered)
    if len(df_sub) < 50:  # Arbitrary minimum points
        logger.warning(
            f"Not enough points in Sector {target_sector} ({len(df_sub)}) for refinement."
        )
        return sectors_gdf, dic_df

    logger.info(f"Refining Sector {target_sector} with {len(df_sub)} points...")

    # 2. Compute velocity components along and across the dominant flow direction to use as features for anomaly detection.

    # Compute Median Flow Direction of the Background (Base)
    # Use the median of u and v to find the dominant flow vector
    u_med = df_sub["u"].median()
    v_med = df_sub["v"].median()
    flow_angle = np.arctan2(v_med, u_med)

    # Rotate velocities to align with flow
    # Rotation matrix logic:
    # V_long = u * cos(a) + v * sin(a)
    # V_trans = -u * sin(a) + v * cos(a)
    cos_a = np.cos(flow_angle)
    sin_a = np.sin(flow_angle)
    df_sub["V_long"] = df_sub["u"] * cos_a + df_sub["v"] * sin_a
    df_sub["V_trans"] = -df_sub["u"] * sin_a + df_sub["v"] * cos_a

    # 3. Select Features for Clustering
    # We use Magnitude (V) to detect speed, and V_trans to detect directional anomalies.
    # We usually don't need V_long if we have V, but V_long is cleaner than V because it allows negative values (noise).
    # Let's use [V, V_trans] or [V_long, V_trans].
    # [V, V_trans] is often best: V separates by speed, V_trans separates by direction.
    refine_features = ["V"]  # TODO: We can experiment with adding "V_trans"

    # 3. Setup Anomalous/Background Priors based on Velocity Percentiles
    # We assume 2 classes: 0=Background, 1=Fast/Anomaly
    v_data = df_sub["V"].to_numpy()
    percentile = anomaly_config.prior_percentile_threshold
    threshold = np.percentile(v_data, percentile)

    n_components = 2
    prior_probs = np.zeros((len(df_sub), n_components))

    # Initialize priors: High V points get high probability for Class 1
    for i, v in enumerate(v_data):
        if v > threshold:
            prior_probs[i] = [0.3, 0.7]
        else:
            prior_probs[i] = [0.7, 0.3]

    # 4. Scale Features
    data_array, scaler, _, _ = transform_and_scale_features(
        df_input=df_sub,
        variables_names=refine_features,
        transform_velocity="none",  # We might want strict raw velocity here
    )

    # --- Debugging Plot: Scaled vs Original Distributions ---
    n_feats = data_array.shape[1]
    fig, axes = plt.subplots(n_feats, 1, figsize=(10, 4 * n_feats), squeeze=False)
    for i, var_name in enumerate(config.preprocessing.variables_names):
        ax = axes[i, 0]
        scaled_data = data_array[:, i]

        # Plot distribution on primary axis (Scaled)
        ax.hist(scaled_data, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
        ax.set_xlabel(f"{var_name} (Scaled / Z-score)")
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle="--", alpha=0.6)

        # Add secondary axis for original values
        ax2 = ax.twiny()
        # Scale the secondary axis limits by inverting the primary limits
        ax2.set_xlim(
            scaler.inverse_transform(
                np.array([ax.get_xlim()]).T.repeat(n_feats, axis=1)
            )[:, i]
        )
        ax2.set_xlabel(f"{var_name} (Original Units)")
    plt.tight_layout()
    fig.savefig(output_dir / f"{base_name}_mcmc_feature_distributions.jpg", dpi=150)
    plt.close(fig)

    # 5. Run MCMC Clustering
    # We construct a dummy sectors dict for the function signature,
    # though spatially they are all in the same "Sector A" envelope.
    # The GMM will separate them based on Feature (Velocity) + Spatial cohesion (MRF).
    dummy_sectors = {"Base": sector_poly, "Anomaly": sector_poly}
    result = clusterize_gaussian_mixture(
        data_array_scaled=data_array,
        prior_probs=prior_probs,
        sectors=dummy_sectors,
        sample_args=anomaly_config.sample_options,
        mu_params=anomaly_config.model_options.mu_params,
        sigma_params=anomaly_config.model_options.sigma_params,
        enforce_ordered_means=True,  # Enforce that Class 1 (Anomaly) has higher mean velocity than Class 0 (Base)
        apply_mrf_regularization=True,  # Critical for spatial coherence of the anomaly
        x_pos=df_sub["x"].to_numpy(),
        y_pos=df_sub["y"].to_numpy(),
        mrf_kwargs=anomaly_config.mrf_options,  # Use specific MRF settings for anomaly detection
        random_seed=config.random_seed,
        force_cpu=config.mcmc.force_cpu,
    )

    # 6. Save Diagnostics
    save_sampling_summary(
        convergence_flag=result.convergence_flag,
        idata=result.idata,
        output_dir=output_dir,
        base_name=f"{anomaly_base_name}_mcmc",
        make_plots=True,
        df_input=df_sub,
        cluster_pred=result.cluster_pred,
        posterior_probs=result.posterior_probs,
        scaler=scaler,
        img=img,
    )

    # 6. Plot spatial priors for the anomaly detection step (before-after MRF)
    try:
        fig, ax = plot_spatial_priors(
            df=df_sub,
            prior_probs=prior_probs,
            img=img,
        )
        fig.savefig(
            output_dir / f"{anomaly_base_name}_spatial_priors_beforeMRF.jpg",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
        fig, ax = plot_spatial_priors(
            df=df_sub,
            prior_probs=result.idata.constant_data.prior_w.data,
            img=img,
        )
        fig.savefig(
            output_dir / f"{anomaly_base_name}_spatial_priors_afterMRF.jpg",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Failed to plot spatial priors for anomaly detection: {e}")

    # 7. Post-Processing: Vectorization

    # Map classes: 0 -> base, 1 -> anomaly
    labels_map = ["base", "anomaly"]
    df_sub["sector"] = [labels_map[p] for p in result.cluster_pred]

    # Save Sub-sector GeoJSON (points)
    df_sub.to_csv(output_dir / f"{anomaly_base_name}_points.csv", index=False)

    # 1. Separate points by class
    # We are specifically interested in the 'Anomaly' cluster
    anomaly_df = df_sub[df_sub["sector"] == "anomaly"]
    base_df = df_sub[df_sub["sector"] == "base"]
    logger.info(f"Refinement Stats: Base={len(base_df)}, Anomaly={len(anomaly_df)}")

    if len(anomaly_df) < 10:
        logger.info("Anomaly cluster too small or non-existent.")
        return gpd.GeoDataFrame(), df_sub

    logger.info(f"Vectorizing anomaly cluster with {len(anomaly_df)} points...")

    # Vectorize points to polygon
    x_a = anomaly_df["x"].to_numpy()
    y_a = anomaly_df["y"].to_numpy()
    X_sub, Y_sub, label_grid_sub = create_2d_grid(
        x=x_a, y=y_a, labels=np.ones(len(x_a))
    )
    anomaly_gdf = vectorize_gridded_sectors(label_grid_sub, X_sub, Y_sub)

    # Check validity and non-emptiness
    if anomaly_gdf.empty or not anomaly_gdf.is_valid.all():
        logger.warning(
            "Anomaly geometry is empty or invalid after vectorization. Skipping smoothing and refinement."
        )
        anomaly_gdf = gpd.GeoDataFrame(geometry=[])
        return anomaly_gdf, df_sub

    # Explode MultiPolygons into individual Polygon rows
    anomaly_gdf = anomaly_gdf.explode(index_parts=False, ignore_index=True)

    # Remove small anomalies that are likely noise (e.g., smaller than 4 points)
    pixel_area_threshold = (
        abs(X_sub[0, 1] - X_sub[0, 0]) * abs(Y_sub[1, 0] - Y_sub[0, 0])
    ) * 4
    anomaly_gdf["area"] = anomaly_gdf.area
    anomaly_gdf = anomaly_gdf[anomaly_gdf["area"] >= pixel_area_threshold].copy()

    # Smooth the anomaly geometry slightly to make it more visually coherent, but keep it tight
    logger.info("Smoothing anomaly geometry...")
    sub_raster_res = abs(
        float(X_sub[0, 1] - X_sub[0, 0]) if X_sub.shape[1] > 1 else 1.0
    )
    anomaly_gdf = smoothify(
        anomaly_gdf,
        segment_length=sub_raster_res,  # Use the local grid resolution as a reference for smoothing
        smooth_iterations=2,  # Light smoothing to preserve details
        merge_collection=True,  # Merge adjacent polygons to avoid fragmentation
        merge_multipolygons=False,  # Don't merge separate MultiPolygons to preserve distinct anomalies if they exist
        num_cores=1,  # Avoid parallelism for small geometries to prevent overhead
    )

    # Prepare the 'anomaly' and base geometries for combination:
    # Keep only geometry and add new labels
    anoms = anomaly_gdf[["geometry"]].copy()
    anoms["sector"] = "anomaly"
    sector_geom = sector_row.geometry.iloc[0]
    base_geom = sector_geom.difference(anoms.geometry.union_all())
    base = gpd.GeoDataFrame(
        {"geometry": [base_geom], "sector": ["base"]}, crs=anomaly_gdf.crs
    )
    gdf_refined = pd.concat([base, anoms], ignore_index=True)

    # Reclassify points by the new anomaly geometry to see how many points fall into the refined anomaly sector vs base
    anomaly_pots = classify_points_by_polygons(
        gdf_refined, dic_df, x_col="x", y_col="y", keep_unclassified=False
    )

    # Compute statistics for the anomaly and base geometries
    gdf_refined["area_px2"] = gdf_refined.area
    gdf_refined = compute_sector_stats(
        gdf_refined, anomaly_pots, value_col="V", group_col="sector"
    )

    # Save refined geometries, stats and classified points
    logger.info("Saving refined geometries and stats...")
    gdf_refined.to_file(
        output_dir / f"{anomaly_base_name}_vector.geojson",
        driver="GeoJSON",
    )
    gdf_refined.drop(columns="geometry").to_csv(
        output_dir / f"{anomaly_base_name}_stats.csv",
        index=False,
    )
    anomaly_pots.to_csv(
        output_dir / f"{anomaly_base_name}_classified_points.csv",
        index=False,
    )

    if img is not None:
        plot_anomaly_map(
            gdf_refined, anomaly_df, img, target_sector, output_dir, anomaly_base_name
        )

    logger.info(f"Anomaly detection completed. Results in {output_dir}")

    return gdf_refined, anomaly_pots


def plot_anomaly_map(gdf, points, img, sector_name, out_dir, base_name):
    """Helper to plot the result."""
    if img is not None:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, cmap="gray")

        # Plot Base
        base_row = gdf[gdf["sector"] == "base"]
        if not base_row.empty:
            base_row.plot(
                ax=ax, facecolor="blue", alpha=0.3, edgecolor="blue", label="Base"
            )

        # Plot Anomaly (Highlighted)
        anom_row = gdf[gdf["sector"] == "anomaly"]
        if not anom_row.empty:
            anom_row.plot(
                ax=ax,
                facecolor="red",
                alpha=0.6,
                edgecolor="red",
                linewidth=2,
                label="Anomaly",
            )
            # Add text label
            if not anom_row.geometry.is_empty.all():
                c_x = anom_row.geometry.centroid.x.values[0]
                c_y = anom_row.geometry.centroid.y.values[0]
                ax.text(
                    c_x, c_y, "ANOMALY", color="white", fontweight="bold", ha="center"
                )

        # scatter points on top for detail
        ax.scatter(points["x"], points["y"], c="yellow", s=1, alpha=0.5)

        plt.title(f"Sector {sector_name} Refinement: Anomaly Detection")
        plt.legend()
        fig.savefig(out_dir / f"{base_name}_anomaly_map.jpg", dpi=150)
        plt.close(fig)


def find_sectors_file(config, provided_path=None):
    """Locates the domain classification GeoJSON output."""
    # 1. Check CLI argument
    if provided_path and Path(provided_path).exists():
        return Path(provided_path)

    # 2. Check Config
    if (
        "input_sectors_path" in config.anomaly_detection
        and config.anomaly_detection.input_sectors_path
    ):
        path = Path(config.anomaly_detection.input_sectors_path)
        if path.exists():
            return path

    # 3. Automatic Discovery based on directory structure
    # Expected: {base_output_dir}/kinematic_sectors_geojson/{date}_sectors_polygon.geojson
    base_dir = Path(config.data.base_output_dir)
    ref_date = config.data.reference_date

    # Common standard paths
    candidates = [
        base_dir / "kinematic_sectors_geojson" / f"{ref_date}_sectors_polygon.geojson",
        base_dir
        / ref_date
        / "kinematic_sectors_geojson"
        / f"{ref_date}_sectors_polygon.geojson",
        base_dir / "kinematic_sectors" / f"{ref_date}_sectors_polygon.geojson",
    ]

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Could not find sectors file for {ref_date}. Tried: {candidates}"
    )


def load_best_dic_map(
    ref_date_dt: datetime,
    file_path: str | None,
    search_dir: Path,
    search_pattern: str,
    dt_min: int,
    dt_max: int,
    base_image_dir: Path | None = None,
    min_global_mad_threshold: float | None = None,
    min_ensemble_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Image.Image | None]:
    """
    Locate, read and select the best DIC NetCDF file for a given reference date.

    Args:
        ref_date_dt: reference date as datetime (used to search day-before maps).
        file_path: explicit file path to a single NetCDF file (if provided, search is skipped).
        search_dir: directory to search for candidate NetCDF files.
        search_pattern: filename glob/pattern used by find_ensemble_files.
        dt_min, dt_max: accepted temporal difference limits passed to find_ensemble_files.
        base_image_dir: optional directory containing background images (used to load an image if metadata lacks one).
        min_global_mad_threshold: optional MAD threshold to reject noisy maps (None disables this check).
        min_ensemble_size: optional minimum ensemble size to accept a map (None disables this check).

    Returns:
        tuple of (dic_df, dic_analyses_meta, image_or_none)

    Raises:
        FileNotFoundError: if no candidate files are found.
        RuntimeError: if no candidate passes the optional quality filters.
    """
    # discover candidate files
    if file_path and Path(file_path).is_file():
        nc_paths = [Path(file_path)]
    else:
        date_day_before = (ref_date_dt - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        nc_paths = find_ensemble_files(
            search_dir, date_day_before, dt_min, dt_max, search_pattern
        )

    if not nc_paths:
        raise FileNotFoundError(
            "No DIC files found for the specified date and criteria."
        )

    # read candidates
    candidates: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for p in nc_paths:
        try:
            df, meta = read_data_from_pylamma_nc(p, base_image_dir=base_image_dir)
            if not df.empty:
                candidates[p.stem] = (df, meta)
        except Exception as exc:
            logger.warning(f"Failed to read {p}: {exc}")

    if not candidates:
        raise FileNotFoundError("No readable DIC candidates found.")

    # evaluate and filter candidates (skip checks when thresholds are None)
    valid: list[dict] = []
    for name, (df, meta) in candidates.items():
        mean_mad = float(df["mad"].mean()) if "mad" in df.columns else None
        if mean_mad is None:
            logger.warning(
                f"Source {name}: 'mad' column not available; skipping MAD-based filtering."
            )

        min_ens = (
            int(df["ensemble_size"].min()) if "ensemble_size" in df.columns else None
        )
        if min_ens is None:
            logger.warning(
                f"Source {name}: 'ensemble_size' column not available; skipping ensemble-size-based filtering."
            )

        if (
            (min_global_mad_threshold is not None)
            and (mean_mad is not None)
            and (mean_mad > min_global_mad_threshold)
        ):
            logger.warning(
                f"Rejecting {name}: MAD {mean_mad:.2f} > {min_global_mad_threshold}"
            )
            continue

        if (
            (min_ensemble_size is not None)
            and (min_ens is not None)
            and (min_ens < min_ensemble_size)
        ):
            logger.warning(
                f"Rejecting {name}: Ensemble size {min_ens} < {min_ensemble_size}"
            )
            continue

        valid.append(
            {
                "name": name,
                "df": df,
                "meta": meta,
                "mad": mean_mad,
                "ens": min_ens,
                "dt": meta.iloc[0]["dt_hours"],
            }
        )

    if not valid:
        raise RuntimeError("No DIC candidates passed quality filters.")

    # prefer entries with MAD available (lowest MAD), otherwise keep first valid
    with_mad = [v for v in valid if v["mad"] is not None]
    if with_mad:
        with_mad.sort(key=lambda x: (x["mad"], -(x["ens"] or 0), -x["dt"]))
        best = with_mad[0]
    else:
        logger.warning(
            "MAD not available for any valid candidate. Selecting the first valid candidate."
        )
        best = valid[0]

    dic_df = best["df"]
    dic_meta = best["meta"]

    # try to load an associated image (prefer metadata path, fallback to base_image_dir)
    img: Image.Image | None = None
    img_path = dic_meta.iloc[0].get("image_path")
    if img_path and not pd.isna(img_path):
        try:
            img = Image.open(img_path)
        except Exception as exc:
            logger.warning(f"Could not load image from metadata {img_path}: {exc}")

    if img is None and base_image_dir is not None:
        try:
            ref_date_str = ref_date_dt.strftime("%Y_%m_%d")
            img_candidates = sorted(Path(base_image_dir).glob(f"*{ref_date_str}*.jpg"))
            if img_candidates:
                img = Image.open(img_candidates[len(img_candidates) // 2])
                logger.info(
                    f"Loaded background image from {img_candidates[len(img_candidates) // 2]}"
                )
        except Exception as exc:
            logger.warning(f"Could not locate/load fallback image: {exc}")

    return dic_df, dic_meta, img


def load_dic_from_nc_file(
    ref_date_dt: datetime,
    file_path: str | None,
    search_dir: Path,
    search_pattern: str,
    dt_min: int,
    dt_max: int,
    min_global_mad_threshold: float | None = None,
    min_ensemble_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Locate, read and select the best DIC NetCDF file for a given reference date.

    Args:
        ref_date_dt: reference date as datetime (used to search day-before maps).
        file_path: explicit file path to a single NetCDF file (if provided, search is skipped).
        search_dir: directory to search for candidate NetCDF files.
        search_pattern: filename glob/pattern used by find_ensemble_files.
        dt_min, dt_max: accepted temporal difference limits passed to find_ensemble_files.
        min_global_mad_threshold: optional MAD threshold to reject noisy maps (None disables this check).
        min_ensemble_size: optional minimum ensemble size to accept a map (None disables this check).

    Returns:
        tuple of (dic_df, dic_analyses_meta)

    Raises:
        FileNotFoundError: if no candidate files are found.
        RuntimeError: if no candidate passes the optional quality filters.
    """
    # discover candidate files
    if file_path and Path(file_path).is_file():
        nc_paths = [Path(file_path)]
    else:
        date_day_before = (ref_date_dt - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        nc_paths = find_ensemble_files(
            search_dir, date_day_before, dt_min, dt_max, search_pattern
        )

    if not nc_paths:
        raise FileNotFoundError(
            "No DIC files found for the specified date and criteria."
        )

    # read candidates
    candidates: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for p in nc_paths:
        try:
            df, meta = read_data_from_pylamma_nc(p)
            if not df.empty:
                candidates[p.stem] = (df, meta)
        except Exception as exc:
            logger.warning(f"Failed to read {p}: {exc}")

    if not candidates:
        raise FileNotFoundError("No readable DIC candidates found.")

    # evaluate and filter candidates (skip checks when thresholds are None)
    valid: list[dict] = []
    for name, (df, meta) in candidates.items():
        mean_mad = float(df["mad"].mean()) if "mad" in df.columns else None
        if mean_mad is None:
            logger.warning(
                f"Source {name}: 'mad' column not available; skipping MAD-based filtering."
            )

        min_ens = (
            int(df["ensemble_size"].min()) if "ensemble_size" in df.columns else None
        )
        if min_ens is None:
            logger.warning(
                f"Source {name}: 'ensemble_size' column not available; skipping ensemble-size-based filtering."
            )

        if (
            (min_global_mad_threshold is not None)
            and (mean_mad is not None)
            and (mean_mad > min_global_mad_threshold)
        ):
            logger.warning(
                f"Rejecting {name}: MAD {mean_mad:.2f} > {min_global_mad_threshold}"
            )
            continue

        if (
            (min_ensemble_size is not None)
            and (min_ens is not None)
            and (min_ens < min_ensemble_size)
        ):
            logger.warning(
                f"Rejecting {name}: Ensemble size {min_ens} < {min_ensemble_size}"
            )
            continue

        valid.append(
            {
                "name": name,
                "df": df,
                "meta": meta,
                "mad": mean_mad,
                "ens": min_ens,
                "dt": meta.iloc[0]["dt_hours"],
            }
        )

    if not valid:
        raise RuntimeError("No DIC candidates passed quality filters.")

    # prefer entries with MAD available (lowest MAD), otherwise keep first valid
    with_mad = [v for v in valid if v["mad"] is not None]
    if with_mad:
        with_mad.sort(key=lambda x: (x["mad"], -(x["ens"] or 0), -x["dt"]))
        best = with_mad[0]
    else:
        logger.warning(
            "MAD not available for any valid candidate. Selecting the first valid candidate."
        )
        best = valid[0]

    dic_df = best["df"]
    dic_meta = best["meta"]

    return dic_df, dic_meta


def run_anomaly_pipeline(
    config: DictConfig | ListConfig,
    sectors_file_path: Path | str,
    buffer: float = 100.0,
) -> bool:
    """
    Main execution pipeline taking a fully merged configuration object.
    """

    timer = Timer()

    if not isinstance(config, DictConfig | ListConfig):
        raise ValueError("config must be an OmegaConf DictConfig or ListConfig object.")

    ref_date = config.data.reference_date
    if not ref_date:
        raise ValueError("reference_date must be provided via CLI or config.")
    ref_date_dt = datetime.strptime(ref_date, "%Y-%m-%d")

    target_sector = config.anomaly_detection.target_sector

    # Determine output directory for anomaly detection results.
    output_base_dir = Path(config.data.base_output_dir)
    run_output_subdir = config.data.get("run_output_subdir")
    if run_output_subdir:
        output_dir = output_base_dir / run_output_subdir
    else:
        output_dir = output_base_dir
    output_dir = output_dir / f"anomaly_{target_sector}"  # Dedicated subfolder
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Anomaly detection outputs will be saved to: {output_dir}")

    # Define base names for outputs
    base_name = f"{ref_date}"

    # Save a copy of the used config in the output dir with omegaconfig dump
    config_path = output_dir / f"{base_name}_config.yaml"
    OmegaConf.save(config, config_path)

    # 1. Load Sectors
    sectors_path = find_sectors_file(config, sectors_file_path)
    logger.info(f"Loading sectors from: {sectors_path}")
    sectors_gdf = gpd.read_file(sectors_path)

    # 2. Extract ROI (Target Sector) for optimizing data loading
    if target_sector not in sectors_gdf["sector"].values:
        logger.error(f"Sector {target_sector} not present in loaded file.")
        return False

    target_poly = sectors_gdf[
        sectors_gdf["sector"] == target_sector
    ].geometry.union_all()
    # Use a safe buffer to ensure we load enough data for the anomaly detection context (MRF needs neighbors)
    roi_poly = target_poly.buffer(buffer)

    # 3. Load DIC Data (Partial Load using ROI)
    # Using existing data loading logic
    source = config.data.get("source", "database")
    img = None  # Initialize img variable to None; it will be set if an image is successfully loaded later.

    if source == "database ":
        raise NotImplementedError(
            "Currently only file-based loading is implemented for anomaly detection."
        )

    # TODO: Use the refractored function instead of this code here.

    file_path = config.data.get("file_path")
    if file_path and Path(file_path).is_file():
        nc_paths = [Path(file_path)]
    else:  # Auto-discover mode
        search_dir = Path(config.data.get("search_dir"))
        pattern = config.data.get("search_pattern")

        # NOTE: We use the day before the reference date to search for the DIC map of the day before with dt=1 day (i-1, i-2)
        date_day_before = (ref_date_dt - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info(
            f"Using {config.data.reference_date} for DIC loading (searching for maps from the day before)."
        )
        nc_paths = find_ensemble_files(
            search_dir,
            date_day_before,
            config.data.dt_min,
            config.data.dt_max,
            pattern,
        )
        logger.info(f"Found {len(nc_paths)} candidate DIC files for loading.")

    if not nc_paths:
        logger.error("No DIC files found for the specified date and criteria.")
        return False

    # 1. Read all candidates
    base_img_dir = Path(config.data.image_dir) if config.data.get("image_dir") else None
    candidates = {}
    for p in nc_paths:
        try:
            df, meta = read_data_from_pylamma_nc(p, base_image_dir=base_img_dir)
            if not df.empty:
                candidates[p.stem] = (df, meta)
        except Exception as e:
            logger.warning(f"Failed to read {p}: {e}")

    # Filter DIC maps by quality
    mean_mad_threshold = config.preprocessing.min_global_mad_threshold
    min_ensemble_size = config.preprocessing.min_ensemble_size
    valid_results = []
    for name, (df, meta) in candidates.items():
        # Compute mean MAD only if column exists and a threshold is provided
        if "mad" in df.columns:
            mean_mad = float(df["mad"].mean())
        else:
            mean_mad = None
            logger.warning(
                f"Source {name}: 'mad' column not available; skipping MAD-based filtering for this source."
            )
        # Compute min ensemble size only if column exists and a threshold is provided
        if "ensemble_size" in df.columns:
            min_ens = int(df["ensemble_size"].min())
        else:
            min_ens = None
            logger.warning(
                f"Source {name}: 'ensemble_size' column not available; skipping ensemble-size-based filtering for this source."
            )

        # Apply MAD threshold check only when a threshold is configured and MAD is available
        if (
            mean_mad_threshold is not None
            and mean_mad is not None
            and mean_mad > mean_mad_threshold
        ):
            logger.warning(
                f"Rejecting {name}: MAD {mean_mad:.2f} > {mean_mad_threshold}"
            )
            continue

        # Apply ensemble size check only when a threshold is configured and ensemble info is available
        if (
            min_ensemble_size is not None
            and min_ens is not None
            and min_ens < min_ensemble_size
        ):
            logger.warning(
                f"Rejecting {name}: Ensemble size {min_ens} < {min_ensemble_size}"
            )
            continue

        valid_results.append(
            {
                "name": name,
                "df": df,
                "meta": meta,
                "mad": mean_mad,
                "ens": min_ens,
                "dt": meta.iloc[0]["dt_hours"],
            }
        )

    if not valid_results:
        logger.error("No DIC results passed quality filters.")
        return False

    # If some entries have MAD available prefer them; otherwise pick the first element.
    with_mad = [v for v in valid_results if v["mad"] is not None]
    if with_mad:
        # Sort by MAD (ascending), then largest ensemble, then largest dt
        with_mad.sort(key=lambda x: (x["mad"], -(x["ens"] or 0), -x["dt"]))
        best = with_mad[0]
    else:
        logger.warning(
            "MAD not available for any candidate. Selecting the first available result."
        )
        # keep original order: take first valid result
        best = valid_results[0]

    logger.info(
        f"Selected best DIC map: {best['name']} (MAD: {best['mad'] if best['mad'] is not None else 'N/A'}, DT: {best['dt']:.1f}h)"
    )

    # Extract the DIC dataframe and metadata from the selected best result for further processing
    dic_df = best["df"]
    dic_analyses = best["meta"]

    # 5. Try to load the background image from the selected metadata
    img_path = dic_analyses.iloc[0].get("image_path")
    if img_path and not pd.isna(img_path):
        try:
            img = Image.open(img_path)
        except Exception as e:
            logger.warning(f"Could not load image {img_path}: {e}")

    # If the image is still None, try to find it in the base image directory using the reference date and camera name
    if img is None:
        ref_date_str = ref_date_dt.strftime("%Y_%m_%d")
        img_candidates = sorted(base_img_dir.glob(f"*{ref_date_str}*.jpg"))

        # Take middle one if multiple candidates found
        if img_candidates:
            img_path = img_candidates[len(img_candidates) // 2]
            try:
                img = Image.open(img_path)
                logger.info(f"Loaded background image from {img_path}")
            except Exception as e:
                logger.warning(f"Could not load image {img_path}: {e}")

    date_start = dic_analyses.iloc[0]["master_timestamp"].strftime("%Y-%m-%d")
    date_end = dic_analyses.iloc[0]["slave_timestamp"].strftime("%Y-%m-%d")
    dic_analyses.to_csv(
        output_dir / f"{base_name}_dic_analyses-master{date_start}_slave{date_end}.csv",
        index=False,
    )

    num_points = len(dic_df)

    # Filter only points inside the spatial priors sectors
    if roi_poly is not None:
        dic_df = filter_dataframe_by_polygons(dic_df, polygon=roi_poly)

    # Apply MAD filtering if max_point_mad is specified
    max_point_mad = config.preprocessing.max_point_mad
    if max_point_mad is not None and "mad" in dic_df.columns:
        dic_df = dic_df[dic_df["mad"] <= max_point_mad]
        logger.info(
            f"Applied point MAD filtering with threshold {max_point_mad}. Points before: {num_points}, after: {len(dic_df)}."
        )

    # Apply other DIC filters if any
    dic_df = apply_dic_filters(dic_df, **config.preprocessing.filter_kwargs)
    if dic_df.empty:
        raise RuntimeError("No dataframes left after filtering.")

    # Apply subsampling
    if config.preprocessing.subsample_factor > 1:
        dic_df = spatial_subsample(
            dic_df,
            n_subsample=config.preprocessing.subsample_factor,
            method=config.preprocessing.subsample_method,
        )
        logger.info(f"Data shape after subsampling: {dic_df.shape}")

    dic_df.to_csv(output_dir / f"{base_name}_preprocessed_dic_data.csv", index=False)

    # Plot the preprocessed DIC data for visual inspection
    try:
        dic_plot_result = plot_dic_vectors(
            x=dic_df["x"].to_numpy(),
            y=dic_df["y"].to_numpy(),
            u=dic_df["u"].to_numpy(),
            v=dic_df["v"].to_numpy(),
            magnitudes=dic_df["V"].to_numpy(),
            background_image=img,
            cmap_name="OrRd",
            figsize=(10, 8),
            title=f"{date_start} - {date_end}",
        )
        fig, _, _ = dic_plot_result
        fig.savefig(output_dir / f"{base_name}_preprocessed_dic_vectors.jpg", dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Failed to plot preprocessed DIC vectors: {e}")

    # 5. Run Anomaly Detection
    anomaly_gdf, anomaly_pts = run_anomaly_detection(
        dic_df=dic_df,
        sectors_gdf=sectors_gdf,
        target_sector=target_sector,
        config=config,
        output_dir=output_dir,
        img=img,
        base_name=ref_date,
    )

    # Save anomaly polygons and points in the common anomaly folder for easier access
    anomaly_dir = output_base_dir / "anomaly_A_geojson"
    anomaly_dir.mkdir(parents=True, exist_ok=True)
    anomaly_gdf.to_file(
        anomaly_dir / f"{base_name}_anomaly_A_polygons.geojson",
        driver="GeoJSON",
    )
    anomaly_pts.to_file(
        anomaly_dir / f"{base_name}_anomaly_A_points.geojson",
        driver="GeoJSON",
    )

    logger.info("Anomaly detection pipeline completed successfully.")
    timer.print()

    return True


def main():
    args = parse_arguments()

    # 1. Load Base Config
    config_path = args.config if args.config else None
    config = load_config(config_path)

    # CLI Overrides
    if args.date:
        config.data.reference_date = args.date
    if args.output_dir:
        config.data.base_output_dir = args.output_dir
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))

    # 4. Dynamic update 'year' based on 'reference_date'
    # This must happen before resolution so that paths using ${data.year} are correct
    if not config.data.reference_date:
        raise ValueError("reference_date must be provided via CLI or config.")
    ref_dt = datetime.strptime(config.data.reference_date, "%Y-%m-%d")
    config.data.year = str(ref_dt.year)

    # 5. Resolve Configuration
    OmegaConf.resolve(config)

    # Force CPU for MCMC if specified in config to avoid potential GPU-related issues with JAX in some environments. This should be set before any JAX imports.
    if config.mcmc.force_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"

    try:
        run_anomaly_pipeline(config, sectors_file_path=args.sectors_file)
    except Exception as e:
        logger.error(f"Anomaly Pipeline Failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
