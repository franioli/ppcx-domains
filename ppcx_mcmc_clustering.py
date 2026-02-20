import argparse
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import DictConfig, ListConfig, OmegaConf
from smoothify import smoothify

from ppcluster import Timer, load_config, mcmc, setup_logger
from ppcluster.cvat import (
    filter_dataframe_by_polygons,
)
from ppcluster.data import (
    find_ensemble_file,
    load_sectors_and_roi,
    read_data_from_db,
    read_data_from_pylamma_nc,
)
from ppcluster.griddata import create_2d_grid, plot_clustering_grid
from ppcluster.mcmc.clustering import (
    clusterize_gaussian_mixture,
    save_sampling_summary,
)
from ppcluster.mcmc.priors import plot_spatial_priors
from ppcluster.preprocessing import (
    apply_dic_filters,
    preprocess_features,
    spatial_subsample,
)
from ppcluster.sectors import (
    assign_sector_labels,
    classify_points_by_polygons,
    clean_vector_sectors,
    compute_sector_stats,
    get_sector_colors,
    plot_sectors,
    plot_sectors_summary,
    vectorize_gridded_sectors,
)

logger = setup_logger(level=logging.INFO, name="ppcx")

HEADLESS = True  # set to True when running in non-GUI environment

if HEADLESS:
    plt.switch_backend("Agg")


def parse_arguments():
    p = argparse.ArgumentParser(
        description="Run MCMC clustering with optional overrides."
    )
    p.add_argument(
        "--date",
        "-d",
        help="Reference date (i.e., final date) to process. Override data.reference_date (YYYY-MM-DD). If not provided, uses config value.",
        default=None,
    )
    p.add_argument(
        "--output_dir",
        "-o",
        help="Output directory. Override config value. If not provided, uses config value.",
        default=None,
    )
    p.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to an optional custom config.yaml file to load instead of the default.",
    )
    p.add_argument(
        "--keep-failed-output",
        action="store_true",
        help="Keep output directory if processing fails (for debugging). By default, output is deleted on failure.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip processing if the run output directory already exists.",
    )

    # Robust generic override mechanism using dotlist (e.g., data.dt_min=24)
    # This captures any leftover arguments in the format key=value
    p.add_argument(
        "overrides",
        nargs="*",
        help="Configuration overrides in dotlist format (e.g., data.dt_min=24 mcmc.sample_options.draws=500).",
    )
    return p.parse_args()


def preprocess_dic_data(
    out: dict[Any, pd.DataFrame],
    roi: Any,
    preproc_config: DictConfig | ListConfig,
) -> pd.DataFrame:
    """
    Run spatial filtering, DIC filters, stacking, and subsampling on raw DIC data.

    Args:
        out: Dictionary mapping source IDs to raw DIC DataFrames.
        roi: ROI polygon for spatial filtering.
        preproc_config: Dictionary of preprocessing parameters.

    Returns:
        pd.DataFrame: The fully preprocessed and stacked DataFrame.
    """
    processed = []
    for src_id, df_src in out.items():
        try:
            # Filter only points inside the spatial priors sectors
            if roi is not None:
                df_src = filter_dataframe_by_polygons(df_src, polygon=roi)

            # Apply other DIC filters if any
            df_src = apply_dic_filters(df_src, **preproc_config.filter_kwargs)

            # Append processed dataframe to the list
            processed.append(df_src)
        except Exception as exc:
            logger.warning(f"Filtering failed for {src_id}: {exc}")
    if not processed:
        raise RuntimeError("No dataframes left after filtering.")

    # Stack all processed dataframes
    dic_df = pd.concat(processed, ignore_index=True)
    logger.info(f"Data shape after filtering and stacking: {dic_df.shape}")

    # Apply subsampling
    if preproc_config.subsample_factor > 1:
        dic_df = spatial_subsample(
            dic_df,
            n_subsample=preproc_config.subsample_factor,
            method=preproc_config.subsample_method,
        )
        logger.info(f"Data shape after subsampling: {dic_df.shape}")

    return dic_df


def run_mcmc_clustering(
    dic_df: pd.DataFrame,
    prior_probs_array: np.ndarray,
    sectors: Any,
    config: DictConfig | ListConfig,
    img: Any,
    output_dir: Path,
    base_name: str,
) -> tuple[Any, Any]:
    """
    Runs the single-pass MCMC clustering pipeline.
    Preprocesses features, sets up parameters, and runs the GMM.
    """
    logger.info("Running MCMC Clustering...")

    # 1. Preprocess Features
    data_array_scaled, scaler, velocities, transform_info = preprocess_features(
        df_input=dic_df,
        variables_names=config.preprocessing.variables_names,
        transform_velocity=config.preprocessing.velocity_transform,
        transform_params=config.preprocessing.transform_params,
        feature_weights=config.preprocessing.feature_weights,
    )
    joblib.dump(scaler, output_dir / f"{base_name}_mcmc_feature_scaler.joblib")

    # TODO: move here other spatial prior preprocessing steps now in the main function

    # 2. Plot Priors if spatial
    fig, axes = mcmc.plot_spatial_priors(dic_df, prior_probs_array, img=img)
    fig.savefig(
        output_dir / f"{base_name}_mcmc_spatial_priors.jpg",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 3. Run Clustering
    result = clusterize_gaussian_mixture(
        data_array_scaled=data_array_scaled,
        prior_probs=prior_probs_array,
        sectors=sectors,
        sample_args=config.mcmc.sample_options,
        mu_params=config.mcmc.model_options.mu_params,
        sigma_params=config.mcmc.model_options.sigma_params,
        apply_mrf_regularization=config.mcmc.mrf_regularization,
        x_pos=dic_df["x"].to_numpy(),
        y_pos=dic_df["y"].to_numpy(),
        mrf_kwargs=config.mcmc.mrf_kwargs,
        second_pass=config.mcmc.second_pass,
        second_pass_sample_args=config.mcmc.second_pass_sample_args,
        force_cpu=config.mcmc.force_cpu,
        random_seed=config.random_seed,
    )

    # 4. Save sampling summary
    save_sampling_summary(
        convergence_flag=result.convergence_flag,
        idata=result.idata,
        output_dir=output_dir,
        base_name=f"{base_name}_mcmc",
        make_plots=True,
        df_input=dic_df,
        cluster_pred=result.cluster_pred,
        posterior_probs=result.posterior_probs,
        scaler=scaler,
        img=img,
    )

    # Plot spatial priors before and after MRF regularization to visualize the effect of the MRF on the spatial distribution of cluster probabilities.
    # TODO: move this plotting logic inside the save_sampling_summary function
    fig, _ = plot_spatial_priors(
        df=dic_df,
        prior_probs=prior_probs_array,
        img=img,
    )
    fig.savefig(
        output_dir / f"{base_name}_spatial_priors_beforeMRF.jpg",
        dpi=150,
        bbox_inches="tight",
    )
    fig, _ = plot_spatial_priors(
        df=dic_df,
        prior_probs=result.idata.constant_data.prior_w.data,
        img=img,
    )
    fig.savefig(
        output_dir / f"{base_name}_spatial_priors_afterMRF.jpg",
        dpi=150,
        bbox_inches="tight",
    )

    return result, scaler


def process_clustering_results(
    df_points: pd.DataFrame,
    cluster_labels: np.ndarray,
    output_dir: Path,
    base_name: str,
    config: DictConfig | ListConfig,
    img: Any = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Vectorize clusters, clean geometries, assign labels, compute stats, and save results.
    Reusable for both main pipeline and refinement steps.

    Args:
        df_points: Original points DataFrame with x, y, and velocity columns.
        cluster_labels: The predicted cluster labels for each point.
        output_dir: Directory to save outputs.
        base_name: Base name for output files.
        config: The full configuration object for parameters.
        img: The original image for plotting (optional).

    Returns:
        gpd.GeoDataFrame: The final sectors GeoDataFrame with statistics.
        gpd.GeoDataFrame: The classified points GeoDataFrame with sector labels.
    """
    # 1. Create grid from points
    x = df_points["x"].to_numpy()
    y = df_points["y"].to_numpy()
    X, Y, kin_cluster_grid = create_2d_grid(x=x, y=y, labels=cluster_labels)
    raster_res = abs(float(X[0, 1] - X[0, 0]) if X.shape[1] > 1 else 1.0)

    # 2. Vectorize & Smooth
    logger.info(f"Vectorizing grid clusters ({base_name})...")
    sectors = vectorize_gridded_sectors(kin_cluster_grid, X, Y)

    # Save raw sectors for debugging
    sectors.to_file(output_dir / f"{base_name}_sectors_raw.geojson", driver="GeoJSON")

    # 3. Clean Vectors
    vect_config = config.postprocessing.vectorization
    sectors = clean_vector_sectors(
        sectors,
        df_points,
        min_area_px2=vect_config.min_area_px2,
        isolation_buffer=vect_config.isolation_buffer,
        velocity_merge_threshold=vect_config.velocity_merge_threshold,
        target_number_of_sectors=vect_config.target_number_of_sectors,
        fill_holes_area=vect_config.fill_holes_area,
        smooth_geometries=vect_config.smooth_geometries,
        smooth_method=vect_config.smooth_method,
        smooth_iterations=vect_config.smooth_iterations,
        raster_res=2 * raster_res,
    )

    # 4. Assign Labels
    # Use config method, or fallback/override if provided
    # For refinement, we might want simple labels, but reusing the logic is fine
    sectors = assign_sector_labels(
        sectors,
        order_by=config.postprocessing.sector_assignment.method,
        ascending=config.postprocessing.sector_assignment.ascending,
    )

    # Drop non-essential columns
    sectors = (
        sectors[["geometry", "sector"]].sort_values(by="sector").reset_index(drop=True)
    )

    # 5. Classify Original Points & Compute Stats
    pts_by_sector = classify_points_by_polygons(
        sectors, df_points, x_col="x", y_col="y"
    )
    pts_by_sector.to_file(
        output_dir / f"{base_name}_points_by_sector.geojson",
        driver="GeoJSON",
    )

    sectors["area_px2"] = sectors.geometry.area
    sectors = compute_sector_stats(
        sectors, pts_by_sector, value_col="V", group_col="sector"
    )
    sectors.to_file(output_dir / f"{base_name}_sectors_final.geojson", driver="GeoJSON")

    stats_path = output_dir / f"{base_name}_sector_stats.csv"
    sectors.drop(columns=[sectors.geometry.name], errors="ignore").to_csv(
        stats_path, index=False, float_format="%.3f"
    )

    # 6. Plotting
    if img is not None:
        sector_colors = get_sector_colors(
            sectors["sector"].tolist(),
            colormap=config.plotting.default_discrete_cmap,
        )

        # Plot raw vs vectorized comparison
        fig, (ax_raw, ax_vec) = plt.subplots(1, 2, figsize=(14, 7))
        plot_clustering_grid(
            ax=ax_raw,
            img=img,
            cluster_grid=kin_cluster_grid,
            X=X,
            Y=Y,
            title="Raw Clusters",
            alpha=0.6,
        )
        plot_sectors(
            ax=ax_vec,
            sectors=sectors,
            img=img,
            sector_colors=sector_colors,
            title="Vectorized Sectors",
        )
        fig.savefig(output_dir / f"{base_name}_raw_vs_vec.jpg", dpi=150)
        plt.close(fig)

    return sectors, pts_by_sector


def run_anomaly_detection(
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

    Returns:
        gpd.GeoDataFrame: The refined geometries for the target sector with anomaly sub-sectors.
        gpd.GeoDataFrame: The classified points for the target sector with anomaly labels.
    """
    anomaly_config = config.anomaly_detection

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    anomaly_base_name = f"{base_name}_sector{target_sector}_anomaly"

    # 1. Filter data for the target sector
    # We use the already classified points or geometric filtering
    sector_row = sectors_gdf[sectors_gdf["sector"] == target_sector]
    if sector_row.empty:
        logger.warning(f"Sector {target_sector} not found. Skipping refinement.")
        return sectors_gdf, dic_df
    sector_poly = sector_row.geometry.iloc[0]

    # Create a buffer to include frontier points that might have been smoothed out
    # 64px is standard grid spacing, so 100px ensures we capture immediate neighbors
    buffer_dist = 100.0
    logger.info(
        f"Applying {buffer_dist}px buffer to Sector {target_sector} for anomaly search."
    )
    sector_poly_buffered = sector_poly.buffer(buffer_dist)

    df_sub = filter_dataframe_by_polygons(dic_df, polygon=sector_poly_buffered)
    if len(df_sub) < 50:  # Arbitrary minimum points
        logger.warning(
            f"Not enough points in Sector {target_sector} ({len(df_sub)}) for refinement."
        )
        return sectors_gdf, dic_df
    logger.info(f"Refining Sector {target_sector} with {len(df_sub)} points...")

    # 1. Compute Median Flow Direction of the Background (Base)
    # Use the median of u and v to find the dominant flow vector
    u_med = df_sub["u"].median()
    v_med = df_sub["v"].median()
    flow_angle = np.arctan2(v_med, u_med)

    # 2. Rotate velocities to align with flow
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

    # 2. Assign Velocity-Based Priors (Unsupervised initialization)
    # We assume 2 classes: 0=Background, 1=Fast/Anomaly
    v_data = df_sub["V"].to_numpy()
    percentile = anomaly_config.prior_percentile_threshold
    threshold = np.percentile(v_data, percentile)

    n_components = 2
    prior_probs = np.zeros((len(df_sub), n_components))

    # Initialize priors: High V points get high probability for Class 1
    for i, v in enumerate(v_data):
        if v > threshold:
            prior_probs[i] = [0.2, 0.8]
        else:
            prior_probs[i] = [0.8, 0.2]

    # 3. Preprocess Features
    # Recalculate scaler specifically for this subset
    data_array, scaler, _, _ = preprocess_features(
        df_input=df_sub,
        variables_names=refine_features,
        # We might want strict raw velocity here, or re-use transforms
        transform_velocity="none",
    )

    # 4. Run Clustering (constrained to this sector)
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

    # 5. Save Sampling Summary (Trace plots, etc.)
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

    # === MANUAL POST-PROCESSING FOR ANOMALY IDENTIFICATION ===

    # Map back 0/1 to specific labels
    labels = ["base", "anomaly"]
    sub_labels = [labels[p] for p in result.cluster_pred]
    df_sub["sector"] = sub_labels

    # Save Sub-sector GeoJSON (points)
    df_sub.to_csv(output_dir / f"{anomaly_base_name}_points.csv", index=False)

    # 1. Separate points by class
    # We are specifically interested in the 'Anomaly' cluster
    anomaly_df = df_sub[df_sub["sector"] == "anomaly"]
    base_df = df_sub[df_sub["sector"] == "base"]

    logger.info(f"Refinement Stats: Base={len(base_df)}, Anomaly={len(anomaly_df)}")

    # Process Anomaly Cluster
    if len(anomaly_df) < 10:  # Minimum threshold to consider it a real cluster
        logger.warning(
            f"Anomaly cluster too small ({len(anomaly_df)} points). Ignored."
        )
        anomaly_gdf = gpd.GeoDataFrame(geometry=[])

        return anomaly_gdf, df_sub

    logger.info(f"Vectorizing anomaly cluster with {len(anomaly_df)} points...")
    x_a = anomaly_df["x"].to_numpy()
    y_a = anomaly_df["y"].to_numpy()

    # Create a localized grid for just the anomaly points
    # reusing grid creation logic but locally
    X_sub, Y_sub, label_grid_sub = create_2d_grid(
        x=x_a,
        y=y_a,
        labels=np.ones(len(x_a), dtype=int) * 1,  # All are class 1 (Anomaly)
    )

    # Vectorize the anomaly cluster grid.
    anomaly_gdf = vectorize_gridded_sectors(label_grid_sub, X_sub, Y_sub)

    # Check validity and non-emptiness
    if anomaly_gdf.empty and anomaly_gdf.is_valid.any():
        logger.warning(
            "Anomaly geometry is empty or invalid after vectorization. Skipping smoothing and refinement."
        )
        anomaly_gdf = gpd.GeoDataFrame(geometry=[])

        return anomaly_gdf, df_sub

    # Explode MultiPolygons into individual Polygon rows
    anomaly_gdf = anomaly_gdf.explode(index_parts=False, ignore_index=True)

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

    # 4. Plotting
    if img is not None:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, cmap="gray")

        # Plot Base
        base_row = gdf_refined[gdf_refined["sector"] == "base"]
        if not base_row.empty:
            base_row.plot(
                ax=ax, facecolor="blue", alpha=0.3, edgecolor="blue", label="Base"
            )

        # Plot Anomaly (Highlighted)
        anom_row = gdf_refined[gdf_refined["sector"] == "anomaly"]
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
        ax.scatter(anomaly_df["x"], anomaly_df["y"], c="yellow", s=1, alpha=0.5)

        plt.title(f"Sector {target_sector} Refinement: Anomaly Detection")
        plt.legend()
        fig.savefig(output_dir / f"{anomaly_base_name}_refinement_map.jpg", dpi=150)
        plt.close(fig)

    logger.info(f"Refinement complete. Results in {output_dir}")

    return gdf_refined, anomaly_pots


def run_pipeline(config: DictConfig | ListConfig) -> bool:
    """
    Main execution pipeline taking a fully merged configuration object.
    """

    timer = Timer()

    if not isinstance(config, DictConfig | ListConfig):
        raise ValueError("config must be an OmegaConf DictConfig or ListConfig object.")

    reference_date = config.data.reference_date
    if not reference_date:
        raise ValueError("reference_date must be provided via CLI or config.")
    reference_date_dt = datetime.strptime(reference_date, "%Y-%m-%d")

    # Log some key configuration values for better traceability
    logger.info(f"Processing reference date: {reference_date}")
    logger.info(f"\tData source: {config.data.get('source', 'database')}")
    logger.info(f"\tCamera name: {config.data.get('camera_name', 'N/A')}")
    logger.info(
        f"\tDt range DIC data selection: {config.data.get('dt_min')} - {config.data.get('dt_max')} hours"
    )
    logger.info(
        f"\tDate range for data selection: {config.data.get('days_before_to_include', 0)} days before to {config.data.get('days_after_to_include', 0)} days after reference date"
    )
    logger.info(f"\tSpatial priors: {list(config.mcmc.priors.probability.keys())}")
    logger.info(f"Spatial prior file: {config.data.sector_prior_path}")
    logger.info(
        f"\tInput features for clustering: {config.preprocessing.variables_names}"
    )

    # Output base directory (output will be saved in a subfolder with camera name and date)
    output_base_dir = Path(config.data.base_output_dir)
    if config.data.run_output_subdir:
        output_dir = output_base_dir / config.data.run_output_subdir
    else:
        output_dir = output_base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define base names for outputs
    base_name = f"{reference_date}"

    # Save a copy of the used config in the output dir with omegaconfig dump
    config_path = output_dir / f"{base_name}_config.yaml"
    OmegaConf.save(config, config_path)

    # === LOAD DATA ===
    # Read sectors for spatial priors and ROI
    # sectors is now dict[str, shapely.Polygon]
    sector_names = list(config.mcmc.priors.probability.keys())
    sectors, roi = load_sectors_and_roi(
        sector_prior_path=config.data.sector_prior_path,
        sector_names=sector_names,
        roi_path=config.data.roi_path,
    )

    # Load dic data and image
    data_source = config.data.get("source", "database")
    if data_source == "file":
        logger.info("Reading data from local file.")

        # Determine the actual file path
        file_path = config.data.get("file_path")
        if file_path and Path(file_path).is_file():
            nc_path = Path(file_path)
        else:
            # Auto-discovery mode
            search_dir = Path(config.data.get("search_dir", "."))
            if file_path and Path(file_path).is_dir():
                search_dir = Path(file_path)
            pattern = config.data.get("search_pattern")
            nc_path = find_ensemble_file(
                search_dir=search_dir,
                reference_date=reference_date,
                dt_hours_min=config.data.dt_min,
                dt_hours_max=config.data.dt_max,
                filename_pattern=pattern,
            )

        if nc_path is None:
            logger.error(f"No file found for reference date {reference_date}.")
            return False

        try:
            # Pass image_dir from config if available
            base_img_dir = config.data.get("image_dir")
            base_img_dir = Path(base_img_dir) if base_img_dir else None
            out, dic_analyses, img = read_data_from_pylamma_nc(
                nc_path, base_image_dir=base_img_dir
            )

        except Exception as e:
            raise ValueError(
                f"Failed to read or parse selected file {nc_path}: {e}"
            ) from e

    elif data_source == "database":
        logger.info("Reading data from database.")

        # Date range for data selection
        days_before_to_include = config.data.get("days_before_to_include", 0)
        ref_date_start = reference_date_dt - pd.Timedelta(days=days_before_to_include)
        days_after_to_include = config.data.get("days_after_to_include", 0)
        ref_date_end = reference_date_dt + pd.Timedelta(days=days_after_to_include)

        # Load data from database with the specified date range
        out, dic_analyses, img = read_data_from_db(
            config, reference_date, ref_date_start, ref_date_end
        )

    else:
        raise ValueError(f"Unknown data source: {data_source}")

    if out is None or dic_analyses is None:
        logger.error("No valid data loaded. Aborting.")
        return False

    date_start = dic_analyses.iloc[0]["master_timestamp"].strftime("%Y-%m-%d")
    date_end = dic_analyses.iloc[0]["slave_timestamp"].strftime("%Y-%m-%d")
    dic_analyses.to_csv(
        output_dir / f"{base_name}_master{date_start}_slave{date_end}_dic_analyses.csv",
        index=False,
    )

    # Apply filter for each df in the dictionary and then stack them
    preproc_config = config.preprocessing
    dic_df = preprocess_dic_data(
        out=out,
        roi=roi,
        preproc_config=preproc_config,
    )
    dic_df.to_csv(output_dir / f"{base_name}_preprocessed_dic_data.csv", index=False)

    timer.update("data_loading_and_preprocessing")

    # ===  MCMC CLUSTERING   === #
    # --- Assign Priors (Spatial or Velocity-based) ---
    # DETECT REFINEMENT OVERRIDE # TODO: find a better way to do this
    # If "Base" or "Fast" (or any specific custom key) is present,
    # we assume the user wants to exclusively use these and ignore A, B, C, D.
    custom_keys = [k for k in config.mcmc.priors.probability if k in ["Base", "Fast"]]
    if custom_keys:
        logger.info(
            f"Custom sector keys detected {custom_keys}. Using these exclusively."
        )
        # Create a new clean dictionary with ONLY the custom keys
        filtered_probs = {}
        for k in custom_keys:
            filtered_probs[k] = config.mcmc.priors.probability[k]

        # Overwrite the config object's dictionary with the filtered one
        config.mcmc.priors.probability = filtered_probs

    # TODO: implement also the possibility to use other types of priors (e.g. velocity-based) without spatial sectors, but for now we require spatial priors if any priors are specified.
    use_spatial_priors = True
    if use_spatial_priors:
        try:
            logger.info("Using SPATIAL priors from polygons.")
            prior_probs_array = mcmc.assign_spatial_priors(
                x=dic_df["x"].to_numpy(),
                y=dic_df["y"].to_numpy(),
                polygons=sectors,
                prior_probs=config.mcmc.priors.probability,
                fade_method=config.mcmc.priors.fade_method,
                fade_options=config.mcmc.priors.fade_options,
            )
        except Exception as exc:
            raise RuntimeError(
                "Error in spatial priors assignment. Check the sector geometries and prior probabilities configuration."
            ) from exc
    else:
        # Default: uniform priors across sectors (not used. fail if spatial priors requested)
        # if not config.mcmc.priors.probability:
        #     n_sectors = len(sectors)
        #     uniform_prob = 1.0 / n_sectors
        #     config.mcmc.priors.probability = {
        #         name: [uniform_prob] * n_sectors for name in sectors
        #     }
        raise NotImplementedError(
            "Velocity-based priors without spatial sectors is not implemented."
        )

    # Perform MCMC clustering
    result, scaler = run_mcmc_clustering(
        dic_df=dic_df,
        prior_probs_array=prior_probs_array,
        sectors=sectors,
        config=config,
        img=img,
        output_dir=output_dir,
        base_name=base_name,
    )
    timer.update("mcmc_clustering")

    # ===  POST-PROCESSING AND CLEANING OF FINAL CLUSTERING  === #

    # Use the shared function for the main data
    logger.info("Processing main clustering results...")
    sectors, pts_by_sector = process_clustering_results(
        df_points=dic_df,  # The original full dataframe
        cluster_labels=result.cluster_pred,
        output_dir=output_dir,
        base_name=base_name,
        config=config,
        img=img,
    )

    # Make a summary plot of the final sectors with points colored by velocity
    sector_colors = get_sector_colors(
        sectors["sector"].tolist(),
        colormap=config.plotting.default_discrete_cmap,
    )
    plot_sectors_summary(
        sectors=sectors,
        points_by_sector=pts_by_sector,
        img=img,
        colors=sector_colors,
        output_dir=output_dir,
        base_name=base_name,
        unit="px",
        quiver_kwargs=config.plotting.quiver,
        figsize=(20, 10),
        dpi=150,
        save_svg=True,
    )

    # Save Pythonized bundle with all dataframes and arrays
    logger.info("Saving sector results...")
    bundle = {
        "reference_date": reference_date,
        "date_start": date_start,
        "date_end": date_end,
        "dic_dataframe": dic_df,
        "posterior_probs": result.posterior_probs,
        "cluster_pred": result.cluster_pred,
        "uncertainty": result.entropy,
        "sectors": sectors,
        "pts_by_sector": pts_by_sector,
    }
    joblib.dump(bundle, output_dir / f"{base_name}_results.joblib")

    # Save geojson with sector geometries and stats in a common base folder
    sector_vector_dir = output_base_dir / "kinematic_sectors_geojson"
    sector_vector_dir.mkdir(exist_ok=True)
    sectors.to_file(
        sector_vector_dir / f"{base_name}_sectors_polygon.geojson",
        driver="GeoJSON",
    )
    pts_by_sector.to_file(
        sector_vector_dir / f"{base_name}_sectors_points.geojson",
        driver="GeoJSON",
    )

    timer.update("post-processing")

    # === OPTIONAL: SECTOR REFINEMENT (e.g. Sector A sub-clustering) ===
    anomaly_gdf = None  # Initialize anomaly_gdf to None
    if config.anomaly_detection.run_anomaly_detection:
        logger.info("Running Anomaly Detection...")
        target_sector = config.anomaly_detection.target_sector
        anomaly_dir = output_dir / f"anomaly_{target_sector}"

        try:
            anomaly_gdf, anomaly_pts = run_anomaly_detection(
                dic_df=dic_df,  # Pass full data, we filter inside
                sectors_gdf=sectors,  # Pass the generated sectors from step 1
                target_sector=target_sector,
                config=config,
                output_dir=anomaly_dir,
                img=img,
                base_name=base_name,
            )

            # Save anomaly geometry and points to geojson in a common folder for all days
            anomaly_vector_dir = output_base_dir / "anomaly_A_geojson"
            anomaly_vector_dir.mkdir(exist_ok=True)
            anomaly_gdf.to_file(
                anomaly_vector_dir / f"{base_name}_anomaly_A_polygon.geojson",
                driver="GeoJSON",
            )
            anomaly_pts.to_file(
                anomaly_vector_dir / f"{base_name}_anomaly_A_points.geojson",
                driver="GeoJSON",
            )
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}", exc_info=True)

    timer.update("anomaly_detection")

    # Make a final summary figure comparing all results ("Bollettino style")
    # TODO: make a unified summary figure and remove hardcoded copying
    kinematic_sectors_dir = output_base_dir / "kinematic_sectors"
    kinematic_sectors_dir.mkdir(exist_ok=True)

    sector_figure_path = output_dir / f"{base_name}_kinematic_sectors_summary.png"
    summary_fig_path = kinematic_sectors_dir / f"{base_name}_bollettino.png"

    # - Make the final bollettino figure with or without anomaly depending on the results of the anomaly detection step. If anomaly detection failed, we fallback to copying the summary figure without anomaly to the base folder for easier access.
    try:
        fig, axes = plot_sectors_summary(
            sectors=sectors,
            points_by_sector=pts_by_sector,
            img=img,
            colors=sector_colors,
            output_dir=None,  # We will save manually after
            unit="px",
            quiver_kwargs=config.plotting.quiver,
            figsize=(20, 10),
            dpi=150,
            save_svg=False,
        )

        # If anomaly detection was successful and we have a valid anomaly geometry, we add it to the plot
        if (
            anomaly_gdf is not None
            and not anomaly_gdf.empty
            and anomaly_gdf.is_valid.any()
        ):
            # get the map axis from the previous plot and plot the anomaly geometry on top with a distinct color and label
            map_ax = axes["map"]
            anom_geoms = anomaly_gdf[anomaly_gdf["sector"] == "anomaly"].union_all()
            anom_geoms = gpd.GeoSeries(anom_geoms)

            # Add the anomaly geometry to the map with a distinct color and label. Keep the facecolor transparent to only show the edge
            anom_geoms.plot(
                ax=map_ax,
                facecolor="none",
                edgecolor="#d3ff0d",
                linewidth=2,
                label="Anomaly",
            )

        fig.savefig(summary_fig_path, dpi=150)
        # plt.close(fig)

    except Exception as e:
        logger.error(
            f"Failed to plot summary with anomaly: {e}. Fallback to copy figure without anomaly.",
            exc_info=True,
        )
        if sector_figure_path.is_file():
            shutil.copy(sector_figure_path, summary_fig_path)
        else:
            logger.error(
                f"Summary figure not found at {sector_figure_path}. Skipping copy."
            )

        # anomaly_dir = (
        #     output_base_dir / f"anomaly_{config.anomaly_detection.target_sector}"
        # )
        # anomaly_dir.mkdir(exist_ok=True)
        # anomaly_fig_path = (
        #     output_dir
        #     / f"anomaly_{config.anomaly_detection.target_sector}"
        #     / f"{base_name}_sector{config.anomaly_detection.target_sector}_anomaly_mcmc_results.jpg"
        # )
        # if anomaly_fig_path.is_file():
        #     shutil.copy(
        #         anomaly_fig_path,
        #         anomaly_dir
        #         / f"{base_name}_sector{config.anomaly_detection.target_sector}_anomaly_mcmc_results.jpg",
        #     )
        # else:
        #     logger.warning(
        #         f"Anomaly detection figure not found at {anomaly_fig_path}. Skipping copy."
        #     )

        # # Concatenate the anomaly vector to the sector geodataframe for a final summary plot
        # try:
        #     file = (
        #         output_dir
        #         / f"anomaly_{config.anomaly_detection.target_sector}"
        #         / f"{base_name}_sector{config.anomaly_detection.target_sector}_anomaly_vector.geojson"
        #     )
        #     anomaly_gdf = gpd.GeoDataFrame.from_file(file)
        #     anomaly_gdf = anomaly_gdf.loc[
        #         anomaly_gdf["sector"]
        #         == f"{config.anomaly_detection.target_sector}_anomaly"
        #     ]
        #     # rename the anomaly with Z to make it visually distinct in the plot
        #     anomaly_gdf["sector"] = anomaly_gdf["sector"].str.replace(
        #         f"{config.anomaly_detection.target_sector}_anomaly",
        #         "Z",
        #     )
        #     sector_with_anomaly = gpd.GeoDataFrame(
        #         pd.concat(
        #             [sectors, anomaly_gdf],
        #             ignore_index=True,
        #         ),
        #         crs=sectors.crs,
        #     )

        #     # Add a specific color for the anomaly sector (Z)
        #     sector_anomaly_colors = sector_colors.copy()
        #     sector_anomaly_colors["Z"] = "#d3ff0d"
        #     plot_sectors_summary(
        #         sectors=sector_with_anomaly,
        #         points_by_sector=pts_by_sector,
        #         img=img,
        #         colors=sector_anomaly_colors,
        #         output_dir=anomaly_dir,
        #         base_name=f"{base_name}_sector{config.anomaly_detection.target_sector}_anomaly",
        #         unit="px",
        #         quiver_kwargs=config.plotting.quiver,
        #         figsize=(20, 10),
        #         dpi=150,
        #         save_svg=False,
        #     )
        # except Exception as e:
        #     logger.error(f"Failed to plot summary with anomaly: {e}", exc_info=True)

    logger.info("Processing complete.")
    timer.print()

    return True


def main():
    args = parse_arguments()

    # 1. Load Base Config
    config_path = args.config if args.config else None
    config = load_config(config_path)

    # 2. Apply Specific CLI Flags (Highest Priority for these shortcuts)
    if args.date:
        config.data.reference_date = args.date
    if args.output_dir:
        config.data.base_output_dir = args.output_dir

    # 3. Apply Generic Dotlist Overrides
    if args.overrides:
        # OmegaConf can natively merge list of dot-string arguments
        # e.g. ["data.dt_min=10", "mcmc.regularization=False"]
        cli_conf = OmegaConf.from_dotlist(args.overrides)
        config = OmegaConf.merge(config, cli_conf)

    # 4. Dynamic update 'year' based on 'reference_date'
    # This must happen before resolution so that paths using ${data.year} are correct
    if not config.data.reference_date:
        raise ValueError("reference_date must be provided via CLI or config.")
    try:
        ref_dt = datetime.strptime(config.data.reference_date, "%Y-%m-%d")
        config.data.year = str(ref_dt.year)
        logger.debug(
            f"CLI: Updating 'data.year' to {config.data.year} based on reference date."
        )
    except ValueError:
        logger.error(
            f"Could not parse year from reference_date: {config.data.reference_date}"
        )
        config.data.year = "unknown"

    # 5. Resolve Configuration
    # This computes all interpolations (e.g. ${data.output_dir}) now.
    OmegaConf.resolve(config)

    # Run the main pipeline with error handling to ensure that if something goes wrong, we log it and optionally clean up any partial outputs.
    base_output_dir = Path(config.data.base_output_dir)
    run_output_subdir = config.data.get("run_output_subdir", None)
    if run_output_subdir:
        output_dir = base_output_dir / run_output_subdir
    else:
        output_dir = base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and output_dir.exists():
        logger.info(f"Skipping run: output directory {output_dir} already exists.")
        return

    # Force CPU for MCMC if specified in config to avoid potential GPU-related issues with JAX in some environments. This should be set before any JAX imports.
    if config.mcmc.force_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"

    try:
        result = run_pipeline(config)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if not args.keep_failed_output:
            logger.info(f"Cleaning up output directory {output_dir} due to failure.")
            shutil.rmtree(output_dir, ignore_errors=True)
        raise

    if result:
        logger.info(f"Pipeline completed successfully. Outputs saved to {output_dir}")
    else:
        logger.error("Pipeline did not complete successfully. Check logs for details.")
        if not args.keep_failed_output:
            logger.info(
                f"Cleaning up output directory {output_dir} due to incomplete results."
            )
            shutil.rmtree(output_dir, ignore_errors=True)


def run_multiscale_mcmc_deprecated(
    dic_df: pd.DataFrame,
    prior_probs_array: np.ndarray,
    sectors: Any,
    config: DictConfig | ListConfig,
):
    """
    DEPRECATED: Multiscale approach iterating over sigma values.
    Kept for reference but not used in the active pipeline.
    """

    raise NotImplementedError(
        "Multiscale MCMC clustering is currently deprecated and not maintained. This function is kept for reference but should not be used in the active pipeline. The old result dict was replaced with a ClusteringResult object that does not contain 'sigma' key anymore."
    )

    # results = []
    # if config.multiscale.sigma_values is None:
    #     return None

    #     # Loop through smoothing scales
    #     results = []
    #     for sigma in config.multiscale.sigma_values:
    #         logger.info(f"Processing with Gaussian smoothing sigma={sigma}...")

    #         # Create scale-specific base name
    #         scale_base_name = f"{mcmc_base_name}_sigma{sigma}"

    #         # Apply Gaussian smoothing if needed (skipped for sigma=0)
    #         df_run = apply_2d_gaussian_filter(dic_df, sigma=sigma)

    #         # Preprocess features for clustering
    #         data_array_scaled, scaler, velocities, transform_info = preprocess_features(
    #             df_input=df_run,
    #             variables_names=config.data.variables_names,
    #             transform_velocity=config.mcmc.velocity_transform,
    #             transform_params=config.mcmc.transform_params,
    #         )
    #         joblib.dump(scaler, output_dir / f"{scale_base_name}_scaler.joblib")

    #         # Run MCMC clustering with the a gaussian mixture model
    #         if sigma > 2:  # For larger sigma, tighten priors
    #             mu_params = {"mu": 0, "sigma": 0.5}
    #             sigma_params = {"sigma": 0.5}
    #         else:
    #             mu_params = config.mcmc.model_options.mu_params
    #             sigma_params = config.mcmc.model_options.sigma_params
    #         result = clusterize_gaussian_mixture(
    #             data_array_scaled=data_array_scaled,
    #             prior_probs=prior_probs_array,
    #             sectors=sectors,
    #             sample_args=config.mcmc.sample_options,
    #             mu_params=mu_params,
    #             sigma_params=sigma_params,
    #             feature_weights=config.mcmc.feature_weights,
    #             apply_mrf_regularization=config.mcmc.mrf_regularization,
    #             mrf_kwargs=config.mcmc.mrf_kwargs,
    #             second_pass=config.mcmc.second_pass,
    #             second_pass_sample_args=config.mcmc.second_pass_sample_args,
    #             random_seed=config.random_seed,
    #         )

    #         # --- Save sampling summary ---
    #         save_sampling_summary(
    #             convergence_flag=result["convergence_flag"],
    #             idata=result["idata"],
    #             output_dir=output_dir,
    #             base_name=scale_base_name,
    #         )

    #         # Add scale information to result
    #         # result["sigma"] = sigma

    #         # Append to results list
    #         results.append(result)

    # # --- Save final clustering results ---
    # cluster_aggregation_outs = {
    #     "cluster_pred": cluster_pred,
    #     "posterior_probs": posterior_probs,
    #     "entropy": entropy,
    #     "similarity_matrix": similarity_matrix,
    #     "stability_score": stability_score,
    #     "valid_scales": valid_scales,
    # }
    # joblib.dump(
    #     cluster_aggregation_outs,
    #     output_dir / f"{mcmc_base_name}_clustering_results_raw.joblib",
    # )

    # return results


if __name__ == "__main__":
    main()
