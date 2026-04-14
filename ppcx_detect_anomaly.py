"""Detect kinematic anomalies within an existing classified domain (sector).

This script performs a second-pass MCMC clustering on a specific sector
(e.g., Sector A) identified by the domain-classification pipeline
(``ppcx_identify_domains.py``). It separates anomalous high-velocity
sub-regions from the background flow and outputs vectorized anomaly polygons
with velocity statistics.

Configuration is loaded from ``config_anomaly.yaml`` by default. Any value
can be overridden at the command line using OmegaConf dot-list syntax
(e.g. ``data.dt_min=24``).

------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------

    python ppcx_detect_anomaly.py [OPTIONS] [OVERRIDES ...]

OPTIONS
    -d, --date DATE          Reference date to process (YYYY-MM-DD).
    -s, --sectors-file PATH  Path to the sectors GeoJSON from domain
                             classification. Auto-discovered if omitted.
    -o, --output_dir DIR     Override the output directory from config.
    -c, --config PATH        Path to a custom config_anomaly.yaml file.

OVERRIDES
    Any number of dot-list key=value pairs forwarded to OmegaConf, e.g.:
        data.dt_min=24
        mcmc.sample_options.draws=500
        data.subset_name="2024_24mp"
        anomaly_detection.force_cpu=true

------------------------------------------------------------------------
EXAMPLES
------------------------------------------------------------------------

1. Run anomaly detection for a single date (sectors file auto-discovered):
    python ppcx_detect_anomaly.py --date 2024-06-06

2. Provide the sectors file explicitly:
    python ppcx_detect_anomaly.py --date 2024-06-06 \\
        --sectors-file output/2024-06-06_sectors_polygon.geojson

3. Use a custom config file with overrides:
    python ppcx_detect_anomaly.py --date 2024-06-06 \\
        --config config_anomaly.yaml data.subset_name="2024_18mp"

4. Generate and run a batch job file for a date range:
    python ppcx_prepare_job_file.py ppcx_detect_anomaly.py \\
        --date-range 2024-06-01 2024-10-30 --output jobs_anomaly.txt \\
        data.subset_name="2024_24mp" anomaly_detection.force_cpu=true
    parallel -j 4 --bar --joblog run.log --resume < jobs_anomaly.txt
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image
from smoothify import smoothify

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
)
from ppcluster.preprocessing import (
    transform_and_scale_features,
)
from ppcluster.sectors import (
    classify_points_by_polygons,
    compute_sector_stats,
    fill_polygon_holes,
    filter_small_sectors,
    vectorize_gridded_sectors,
)
from ppcluster.visualization import (
    plot_dic_vectors,
)

logger = setup_logger(level=logging.INFO, name="ppcx")

CONFIG_PATH = "config_anomaly.yaml"  # Path to the config file. Can be overwritten by --config argument in CLI.

HEADLESS = True  # set to True when running in non-GUI environment

if HEADLESS:
    plt.switch_backend("Agg")


class SectorNotFoundError(Exception):
    """Custom exception raised when the specified sector is not found in the sectors GeoDataFrame."""

    pass


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
        default=CONFIG_PATH,
        help=f"Path to anomaly config file. Default: {CONFIG_PATH}",
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


def run_anomaly_pipeline(
    sectors_file_path: Path | str,
    config: DictConfig | ListConfig,
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
        raise SectorNotFoundError(f"Sector {target_sector} not found.")

    target_poly = sectors_gdf[
        sectors_gdf["sector"] == target_sector
    ].geometry.union_all()
    # Use a safe buffer to ensure we load enough data for the anomaly detection context (MRF needs neighbors)
    buffer = config.anomaly_detection.sector_buffer
    if buffer is not None and buffer > 0:
        roi_poly = target_poly.buffer(buffer)
    else:
        roi_poly = target_poly

    # 3. Load DIC Data (Partial Load using ROI)
    source = config.data.get("source", "database")
    if source == "database":
        raise NotImplementedError(
            "Currently only file-based loading is implemented for anomaly detection."
        )

    base_img_dir = Path(config.data.image_dir) if config.data.get("image_dir") else None
    logger.info(
        f"Using {config.data.reference_date} for DIC loading (searching for maps from the day before)."
    )
    try:
        dic_df, dic_analyses, img = load_best_dic_map(
            ref_date_dt=ref_date_dt,
            file_path=config.data.get("file_path"),
            search_dir=Path(config.data.get("search_dir")),
            search_pattern=config.data.get("search_pattern"),
            dt_min=config.data.dt_min,
            dt_max=config.data.dt_max,
            base_image_dir=base_img_dir,
            mean_global_mad_threshold=config.preprocessing.mean_global_mad_threshold,
            min_ensemble_size=config.preprocessing.min_ensemble_size,
        )
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(str(e))
        return False

    date_start = dic_analyses.iloc[0]["master_timestamp"].strftime("%Y-%m-%d")
    date_end = dic_analyses.iloc[0]["slave_timestamp"].strftime("%Y-%m-%d")
    dic_analyses.to_csv(
        output_dir / f"{base_name}_dic_analyses-master{date_start}_slave{date_end}.csv",
        index=False,
    )

    # Filter only points inside the spatial priors sectors
    if roi_poly is not None:
        dic_df = filter_dataframe_by_polygons(dic_df, polygon=roi_poly)

    # Apply MAD filtering if max_point_mad is specified
    num_points = len(dic_df)
    max_point_mad = config.preprocessing.max_point_mad
    if max_point_mad is not None and "mad" in dic_df.columns:
        dic_df = dic_df[dic_df["mad"] <= max_point_mad]
        logger.info(
            f"Applied point MAD filtering with threshold {max_point_mad}. Points before: {num_points}, after: {len(dic_df)}."
        )

    dic_df.to_csv(output_dir / f"{base_name}_preprocessed_dic_data.csv", index=False)

    if len(dic_df) < 50:  # Arbitrary minimum points #TODO: make this configurable
        logger.warning(
            f"Not enough points in Sector {target_sector} ({len(dic_df)}) for refinement."
        )
        return False

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
    anomaly_gdf, anomaly_pts = detect_anomaly(
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
    if isinstance(anomaly_gdf, gpd.GeoDataFrame) and not anomaly_gdf.empty:
        anomaly_gdf.to_file(
            anomaly_dir / f"{base_name}_anomaly_A_polygons.geojson",
            driver="GeoJSON",
        )
    if isinstance(anomaly_pts, gpd.GeoDataFrame) and not anomaly_pts.empty:
        anomaly_pts.to_file(
            anomaly_dir / f"{base_name}_anomaly_A_points.geojson",
            driver="GeoJSON",
        )

    # Save the summary figure with the anomaly map and velocity field for easier access
    try:
        anomaly_plot_dir = output_base_dir / "anomaly_A_plots"
        anomaly_plot_dir.mkdir(parents=True, exist_ok=True)
        plot_anomaly_with_velocity(
            anomaly_gdf=anomaly_gdf,
            sectors_gdf=sectors_gdf,
            dic_df=dic_df,
            img=img,
            sector_name=target_sector,
            out_dir=anomaly_plot_dir,
            base_name=base_name,
        )
    except Exception as e:
        logger.warning(f"Failed to plot anomaly with velocity: {e}")

    logger.info("Anomaly detection pipeline completed successfully.")
    timer.print()

    return True


def detect_anomaly(
    dic_df: pd.DataFrame,
    sectors_gdf: gpd.GeoDataFrame,
    target_sector: str,
    config: DictConfig | ListConfig,
    output_dir: Path,
    img: Any,
    base_name: str,
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

    # 1. Select points within the target sector (with optional buffer)
    sector_row = sectors_gdf[sectors_gdf["sector"] == target_sector]
    if sector_row.empty:
        logger.warning(f"Sector {target_sector} not found. Skipping refinement.")
        return sectors_gdf, dic_df
    sector_poly = sector_row.geometry.iloc[0]

    logger.info(f"Refining Sector {target_sector} with {len(dic_df)} points...")

    # 2. Compute velocity components along and across the dominant flow direction to use as features for anomaly detection.

    # Compute Median Flow Direction of the Background (Base)
    # Use the median of u and v to find the dominant flow vector
    u_med = dic_df["u"].median()
    v_med = dic_df["v"].median()
    flow_angle = np.arctan2(v_med, u_med)

    # Rotate velocities to align with flow
    # Rotation matrix logic:
    # V_long = u * cos(a) + v * sin(a)
    # V_trans = -u * sin(a) + v * cos(a)
    cos_a = np.cos(flow_angle)
    sin_a = np.sin(flow_angle)
    dic_df["V_long"] = dic_df["u"] * cos_a + dic_df["v"] * sin_a
    dic_df["V_trans"] = -dic_df["u"] * sin_a + dic_df["v"] * cos_a

    # 3. Select and scale features for clustering
    # We use Magnitude (V) to detect speed, and V_trans to detect directional anomalies.
    # We usually don't need V_long if we have V, but V_long is cleaner than V because it allows negative values (noise).
    # Let's use [V, V_trans] or [V_long, V_trans].
    refine_features = anomaly_config.variables_names

    data_array, scaler, _, _ = transform_and_scale_features(
        df_input=dic_df,
        variables_names=refine_features,
        feature_weights=config.anomaly_detection.feature_weights,
        transform_velocity="power",  # We might want to compresses the low end and stretches the high end velocities.
        transform_params={
            "power": {"exponent": 5}
        },  # This will make low velocities (noise) more distinguishable and high velocities (potential anomalies) more spread out.
        scaler_type="standard",
        make_plots=True,
        output_dir=output_dir,
        base_name=f"{anomaly_base_name}_feature_distributions",
    )

    # 4. Setup Anomalous/Background Priors
    # Bounds for p(anomaly).
    # The actual p(anomaly) for each point will be assigned based on the chosen method(s) and will lie within these bounds. These values do not have to sum to 1 with p(not anomaly) as the probability of not being an anomaly will be 1 - p(anomaly).
    p_lo, p_hi = anomaly_config.prior_anomaly_probability_limits

    # Prior assignment method: "velocity", "y_coord", "kmeans", or combinations like "velocity+kmeans"
    prior_method = config.anomaly_detection.prior_assignment_method
    methods = [m.strip() for m in str(prior_method).split("+")]

    def _prior_velocity(p_threshold: float = 95.0) -> np.ndarray:
        v = dic_df["V"].to_numpy()
        threshold = np.percentile(v, p_threshold)

        # Hard assignment: above threshold → p_hi, below → p_lo
        result = np.where(v >= threshold, p_hi, p_lo).astype(float)

        # Narrow linear blend in a ±30% std band around the threshold to avoid hard edges
        band = v.std() * 0.3
        in_band = np.abs(v - threshold) < band
        t = np.clip((v[in_band] - (threshold - band)) / (2 * band + 1e-12), 0.0, 1.0)
        result[in_band] = p_lo + t * (p_hi - p_lo)

        return result

    def _prior_y_coord() -> np.ndarray:
        y = dic_df["y"].to_numpy()
        score = (y - y.min()) / (y.max() - y.min() + 1e-12)
        return p_lo + score * (p_hi - p_lo)

    def _prior_kmeans() -> tuple[np.ndarray, np.ndarray, int]:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=2, n_init=10, random_state=config.random_seed)
        labels = km.fit_predict(data_array)
        mean_v = [dic_df["V"].to_numpy()[labels == k].mean() for k in range(2)]
        anom_k = int(np.argmax(mean_v))
        logger.info(
            f"KMeans priors: anomaly cluster={anom_k}, "
            f"sizes={[(labels == k).sum() for k in range(2)]}, "
            f"mean V per cluster={[f'{v:.2f}' for v in mean_v]}"
        )
        return np.where(labels == anom_k, p_hi, p_lo), labels, anom_k

    scores: list[np.ndarray] = []
    km_labels = km_anomaly_cluster = None
    for method in methods:
        if method == "velocity":
            scores.append(_prior_velocity(p_threshold=95))
        elif method == "y_coord":
            scores.append(_prior_y_coord())
        elif method == "kmeans":
            p_km, km_labels, km_anomaly_cluster = _prior_kmeans()
            scores.append(p_km)
        else:
            raise ValueError(
                f"Unknown prior_method component: '{method}'. "
                "Choose from: 'velocity', 'y_coord', 'kmeans', or '+'-separated combinations."
            )

    if len(scores) == 1:
        p_anomaly = scores[0]
    else:
        # Bayesian product: element-wise product then rescale to [p_lo, p_hi]
        combined = np.prod(np.stack(scores), axis=0)
        c_min, c_max = combined.min(), combined.max()
        p_anomaly = p_lo + (combined - c_min) / (c_max - c_min + 1e-12) * (p_hi - p_lo)

    prior_probs = np.column_stack([1.0 - p_anomaly, p_anomaly])
    logger.info(
        f"Priors ({prior_method}): p_anomaly in [{p_anomaly.min():.2f}, {p_anomaly.max():.2f}]"
    )

    # Debug plot: prior probability map (+ KMeans labels if used)
    x_pts = dic_df["x"].to_numpy()
    y_pts = dic_df["y"].to_numpy()
    n_panels = 2 if km_labels is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6), squeeze=False)
    if km_labels is not None:
        colors = np.where(km_labels == km_anomaly_cluster, "red", "steelblue")
        axes[0, 0].scatter(x_pts, y_pts, c=colors, s=4, alpha=0.6)
        axes[0, 0].set_title("KMeans: base (blue) / anomaly (red)")
        axes[0, 0].set_aspect("equal")
        axes[0, 0].invert_yaxis()
    sc = axes[0, -1].scatter(
        x_pts, y_pts, c=p_anomaly, s=4, cmap="OrRd", vmin=p_lo, vmax=p_hi
    )
    fig.colorbar(sc, ax=axes[0, -1], label="p(anomaly)")
    axes[0, -1].set_title(
        f"Prior p(anomaly) [{p_anomaly.min():.2f} – {p_anomaly.max():.2f}]"
    )
    axes[0, -1].set_aspect("equal")
    axes[0, -1].invert_yaxis()
    plt.tight_layout()
    fig.savefig(output_dir / f"{base_name}_priors_debug.jpg", dpi=150)
    plt.close(fig)

    # 5. Run MCMC Clustering
    # We construct a dummy sectors dict for the function signature,
    # though spatially they are all in the same "Sector A" envelope.
    # The GMM will separate them based on Feature (Velocity) + Spatial cohesion (MRF).
    dummy_sectors = {"Base": sector_poly, "Anomaly": sector_poly}

    # Enforce that Class 1 (Anomaly) has higher mean velocity than Class 0 (Base) by setting ordered means in the GMM. This is a critical constraint that guides the clustering to find the anomalous cluster as the one with higher velocity. This is disabled when using multiple features to allow more flexibility in clustering, but can be re-enabled if velocity is the main feature and we want to ensure the anomaly cluster is correctly identified.
    do_enforce_ordered_means = True  # if len(refine_features) == 1 else False
    result = clusterize_gaussian_mixture(
        data_array_scaled=data_array,
        prior_probs=prior_probs,
        sectors=dummy_sectors,
        sample_args=anomaly_config.sample_options,
        mu_params=anomaly_config.model_options.mu_params,
        sigma_params=anomaly_config.model_options.sigma_params,
        enforce_ordered_means=do_enforce_ordered_means,
        apply_mrf_regularization=False,  # Critical for spatial coherence
        x_pos=dic_df["x"].to_numpy(),
        y_pos=dic_df["y"].to_numpy(),
        mrf_kwargs=anomaly_config.mrf_options,
        second_pass_sample_args=anomaly_config.second_pass_sample_args,
        random_seed=config.random_seed,
        force_cpu=anomaly_config.force_cpu,
        output_dir=output_dir,
        base_name=f"{base_name}_mcmc",
        debug=True,
        save_ctx={"df_input": dic_df, "scaler": scaler, "img": img},
    )

    # 7. Post-Processing: Vectorization
    labels_map = ["base", "anomaly"]  # Map classes: 0 -> base, 1 -> anomaly
    dic_df["sector"] = [labels_map[p] for p in result.cluster_pred]

    # Save classified points with their assigned sectors
    dic_df.to_csv(output_dir / f"{anomaly_base_name}_points.csv", index=False)

    # 1. Separate points by class
    # We are specifically interested in the 'Anomaly' cluster
    anomaly_df = dic_df[dic_df["sector"] == "anomaly"]
    base_df = dic_df[dic_df["sector"] == "base"]
    logger.info(f"Refinement Stats: Base={len(base_df)}, Anomaly={len(anomaly_df)}")

    if len(anomaly_df) < 10:
        logger.info("Anomaly cluster too small or non-existent.")
        return gpd.GeoDataFrame(), dic_df

    logger.info(f"Vectorizing anomaly cluster with {len(anomaly_df)} points...")

    # Create a grid of the anomaly points for vectorization.
    x_a = anomaly_df["x"].to_numpy()
    y_a = anomaly_df["y"].to_numpy()
    X_sub, Y_sub, label_grid_sub = create_2d_grid(
        x=x_a, y=y_a, labels=np.ones(len(x_a))
    )

    # Apply one iteration of erosion+dilation to clean up the anomaly mask before vectorization. This helps remove small noise clusters and fill small gaps, resulting in cleaner polygons.
    # from ppcluster.griddata import apply_morphological_operations

    # label_grid_sub = apply_morphological_operations(
    #     cluster_grid=label_grid_sub,
    #     erosion_iterations=1,
    #     dilation_iterations=1,
    #     min_cluster_size=20,
    #     connectivity=8,
    # )

    #  Vectorize points to polygon
    anomaly_gdf = vectorize_gridded_sectors(label_grid_sub, X_sub, Y_sub)

    # Check validity and non-emptiness
    if anomaly_gdf.empty or not anomaly_gdf.is_valid.all():
        logger.warning(
            "Anomaly geometry is empty or invalid after vectorization. Skipping smoothing and refinement."
        )
        anomaly_gdf = gpd.GeoDataFrame(geometry=[])
        return anomaly_gdf, dic_df

    # Remove small anomalies that are likely noise (e.g., smaller than 5 points) and fill small holes (e.g., smaller than 3 points) to clean up the geometry. # TODO: MAKE THIS CONFIGURABLE
    n_points_threshold = 5
    n_points_holes_to_fill = 3
    grid_res = abs(float(X_sub[0, 1] - X_sub[0, 0]) if X_sub.shape[1] > 1 else 1.0)

    # Explode MultiPolygons into individual Polygon rows
    anomaly_gdf = anomaly_gdf.explode(index_parts=False, ignore_index=True)

    # Filter Small Sectors
    anomaly_gdf = filter_small_sectors(
        anomaly_gdf, min_area_px2=grid_res**2 * n_points_threshold
    )

    anomaly_gdf = smoothify(
        anomaly_gdf,
        segment_length=grid_res,
        smooth_iterations=1,
        merge_collection=True,  # Merge adjacent polygons to avoid fragmentation
        merge_multipolygons=False,  # Don't merge separate MultiPolygons to preserve distinct anomalies if they exist
        num_cores=1,
    )

    # Fill small holes in the anomaly geometry
    try:
        anomaly_gdf = fill_polygon_holes(
            anomaly_gdf, threshold=grid_res**2 * n_points_holes_to_fill
        )
    except Exception as e:
        logger.error(f"Error during hole filling: {e}")

    # If multiple anomaly geometries remain, select the one with the strongest
    # velocity signal (largest median-velocity difference from the base sector).
    # Points are spatially joined to each candidate polygon; the one whose point
    # median velocity most exceeds the overall base median is kept.
    if len(anomaly_gdf) > 1:
        from shapely.geometry import Point

        base_v_median = dic_df["V"].median()  # rough base reference before any split
        pts_geoseries = gpd.GeoSeries(
            [Point(x, y) for x, y in zip(dic_df["x"], dic_df["y"], strict=True)],
            crs=anomaly_gdf.crs,
        )

        best_idx, best_delta = None, -np.inf
        for idx, candidate_geom in anomaly_gdf.geometry.items():
            inside = pts_geoseries.within(candidate_geom)
            candidate_v = dic_df.loc[inside.values, "V"]
            if len(candidate_v) < 3:
                continue
            delta = candidate_v.median() - base_v_median
            if delta > best_delta:
                best_delta = delta
                best_idx = idx

        if best_idx is not None:
            logger.info(
                f"Multiple anomaly geometries ({len(anomaly_gdf)}): keeping geometry "
                f"#{best_idx} with largest velocity excess "
                f"(Δv_median = {best_delta:.3f})."
            )
            anomaly_gdf = anomaly_gdf.loc[[best_idx]].reset_index(drop=True)
        else:
            logger.warning(
                "Could not rank anomaly geometries by velocity; keeping all of them."
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

    # Log the velocity contrast between the selected anomaly and base
    if "v_median" in gdf_refined.columns:
        stats_by_sector = gdf_refined.set_index("sector")["v_median"]
        if "anomaly" in stats_by_sector.index and "base" in stats_by_sector.index:
            v_anom = stats_by_sector["anomaly"]
            v_base = stats_by_sector["base"]
            logger.info(
                f"Velocity contrast — anomaly v_median: {v_anom:.3f}, "
                f"base v_median: {v_base:.3f}, "
                f"Δv_median: {v_anom - v_base:.3f}"
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
        plot_anomaly_with_velocity(
            anomaly_gdf=gdf_refined,
            sectors_gdf=sectors_gdf,
            dic_df=dic_df,
            img=img,
            sector_name=target_sector,
            out_dir=output_dir,
            base_name=base_name,
        )

    logger.info(f"Anomaly detection completed. Results in {output_dir}")

    return gdf_refined, anomaly_pots


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

    raise SectorNotFoundError(f"Could not find sectors file for {ref_date}.")


def load_best_dic_map(
    ref_date_dt: datetime,
    file_path: str | None,
    search_dir: Path,
    search_pattern: str,
    dt_min: int,
    dt_max: int,
    base_image_dir: Path | None = None,
    mean_global_mad_threshold: float | None = None,
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
        mean_global_mad_threshold: optional MAD threshold to reject noisy maps (None disables this check).
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
            (mean_global_mad_threshold is not None)
            and (mean_mad is not None)
            and (mean_mad > mean_global_mad_threshold)
        ):
            logger.warning(
                f"Rejecting {name}: MAD {mean_mad:.2f} > {mean_global_mad_threshold}"
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
    mean_global_mad_threshold: float | None = None,
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
        mean_global_mad_threshold: optional MAD threshold to reject noisy maps (None disables this check).
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
            (mean_global_mad_threshold is not None)
            and (mean_mad is not None)
            and (mean_mad > mean_global_mad_threshold)
        ):
            logger.warning(
                f"Rejecting {name}: MAD {mean_mad:.2f} > {mean_global_mad_threshold}"
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


def plot_anomaly_with_velocity(
    anomaly_gdf: gpd.GeoDataFrame,
    sectors_gdf: gpd.GeoDataFrame,
    dic_df: pd.DataFrame,
    img: Any,
    sector_name: str,
    out_dir: Path,
    base_name: str,
) -> None:
    """Plot the velocity field with all domain sectors and the detected anomaly overlaid."""
    result = plot_dic_vectors(
        x=dic_df["x"].to_numpy(),
        y=dic_df["y"].to_numpy(),
        u=dic_df["u"].to_numpy(),
        v=dic_df["v"].to_numpy(),
        magnitudes=dic_df["V"].to_numpy(),
        vmax=np.percentile(dic_df["V"], 99),
        background_image=img,
        cmap_name="OrRd",
        figsize=(10, 10),
        title=f"Sector {sector_name} – Velocity Field & Anomaly",
    )
    if result is None:
        return
    fig, ax, _ = result

    # Lock view to the image/quiver extent before adding GeoDataFrame overlays.
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # GeoDataFrame.plot() internally calls ax.set_aspect(1/cos(y_deg)) when it
    # detects ax.get_aspect() == "equal". For pixel-space coordinates this
    # produces a zero or negative cosine and crashes. Temporarily switch to
    # "auto" so geopandas skips that code path, then restore "equal" after.
    ax.set_aspect("auto")

    # Draw all domain sectors (transparent fill, labelled edges)
    if not sectors_gdf.empty:
        sectors_gdf.plot(
            ax=ax,
            facecolor="none",
            edgecolor="steelblue",
            linewidth=1.5,
            alpha=0.8,
            aspect=None,  # Add this to prevent geographic aspect calculation
        )

    # Highlight the refined anomaly geometry on top
    anom_row = anomaly_gdf[anomaly_gdf["sector"] == "anomaly"]
    if not anom_row.empty:
        anom_row.plot(
            ax=ax,
            facecolor="none",
            edgecolor="red",
            linewidth=2.5,
            aspect=None,
        )
        if not anom_row.geometry.is_empty.all():
            c_x = anom_row.geometry.centroid.x.values[0]
            c_y = anom_row.geometry.centroid.y.values[0]
            ax.text(
                c_x,
                c_y,
                "ANOMALY",
                color="white",
                fontweight="bold",
                ha="center",
                fontsize=8,
            )

    # Restore equal aspect and original limits
    ax.set_aspect("equal")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    fig.savefig(
        Path(out_dir) / f"{base_name}_anomaly_result.jpg",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    args = parse_arguments()

    # Load Base Config
    config_path = args.config if args.config else None
    config = load_config(config_path)

    # CLI Overrides
    if args.date:
        config.data.reference_date = args.date
    if args.output_dir:
        config.data.base_output_dir = args.output_dir
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))

    # If subset_name is not provided, use the year of the reference date as default subset name for outputs. This helps to organize outputs by year when processing multiple dates.
    if not config.data.get("subset_name"):
        try:
            ref_date_dt = datetime.strptime(config.data.reference_date, "%Y-%m-%d")
            config.data.subset_name = str(ref_date_dt.year)
            logger.debug(
                f"subset_name not provided. Using year {config.data.subset_name} as default subset name for outputs."
            )
        except Exception as e:
            config.data.subset_name = "unknown"
            logger.warning(
                f"Could not parse reference date for subset naming: {e}. subset_name will remain unset."
            )

    # Resolve Configuration
    OmegaConf.resolve(config)

    # Force CPU for MCMC if specified in config to avoid potential GPU-related issues with JAX in some environments. This should be set before any JAX imports.
    if config.anomaly_detection.force_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"

    # Run the Anomaly Detection Pipeline
    try:
        run_anomaly_pipeline(sectors_file_path=args.sectors_file, config=config)
    except Exception as e:
        logger.error(f"Anomaly Pipeline Failed: {e}", exc_info=True)
        raise
