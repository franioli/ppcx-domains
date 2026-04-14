"""Run MCMC-based kinematic domain identification on DIC displacement data.

This script runs MCMC Gaussian-mixture clustering to identify kinematic domains 
from DIC displacement maps and saves vectorized sector polygons with 
velocity statistics.

Configuration is loaded from ``config.yaml`` by default. Any value can be
overridden at the command line using OmegaConf dot-list syntax
(e.g. ``data.dt_min=24``).

------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------

    python ppcx_identify_domains.py [OPTIONS] [OVERRIDES ...]

OPTIONS
    -d, --date DATE          Reference (end) date to process (YYYY-MM-DD).
    -o, --output_dir DIR     Override the output directory from config.
    -c, --config PATH        Path to a custom config.yaml file.
        --skip-existing      Skip if the output directory already exists.
        --keep-failed-output Keep partial output on failure (for debugging).

OVERRIDES
    Any number of dot-list key=value pairs forwarded to OmegaConf, e.g.:
        data.dt_min=24
        mcmc.sample_options.draws=500
        data.subset_name="2024_24mp"
        mcmc.force_cpu=true

------------------------------------------------------------------------
EXAMPLES
------------------------------------------------------------------------

1. Process a single date with defaults from config.yaml:
    python ppcx_identify_domains.py --date 2024-06-06

2. Process a date with config overrides:
    python ppcx_identify_domains.py --date 2024-06-06 \\
        data.subset_name="2024_18mp" mcmc.force_cpu=true

3. Use a custom config file:
    python ppcx_identify_domains.py --date 2024-06-06 --config my_config.yaml

4. Skip dates that have already been processed:
    python ppcx_identify_domains.py --date 2024-06-06 --skip-existing

5. Generate and run a batch job file for a date range (see ppcx_prepare_job_file.py):
    python ppcx_prepare_job_file.py ppcx_identify_domains.py \\
        --date-range 2024-06-01 2024-10-30 --output jobs.txt
    parallel -j 4 --bar --joblog run.log --resume < jobs.txt
"""

import argparse
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image

from ppcluster import Timer, load_config, mcmc, setup_logger
from ppcluster.cvat import (
    filter_dataframe_by_polygons,
)
from ppcluster.data import (
    find_ensemble_files,
    load_sectors_and_roi,
    read_data_from_db,
    read_data_from_pylamma_nc,
)
from ppcluster.griddata import (
    apply_morphological_operations,
    close_small_holes,
    create_2d_grid,
    plot_clustering_grid,
    remove_small_grid_components,
)
from ppcluster.mcmc.clustering import (
    clusterize_gaussian_mixture,
)
from ppcluster.preprocessing import (
    apply_dic_filters,
    spatial_subsample,
    transform_and_scale_features,
)
from ppcluster.sectors import (
    assign_sector_labels,
    classify_points_by_polygons,
    clean_vector_sectors,
    compute_sector_stats,
    vectorize_gridded_sectors,
)
from ppcluster.visualization import (
    get_sector_colors,
    plot_dic_vectors,
    plot_sectors,
    plot_sectors_summary,
)

logger = setup_logger(level=logging.INFO, name="ppcx")

CONFIG_PATH = "config.yaml"  # Path to the config file. Can be overwritten by --config argument in CLI.

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
        default=CONFIG_PATH,
        help=f"Path to anomaly config file. Default: {CONFIG_PATH}",
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
        pd.DataFrame: The fully preprocessed and stacked DataFramdPl
    """
    processed = []
    for src_id, df_src in out.items():
        # Filter only points inside the spatial priors sectors
        if roi is not None:
            df_src = filter_dataframe_by_polygons(df_src, polygon=roi)

        num_points = len(df_src)

        # Apply MAD filtering if max_point_mad is specified
        max_point_mad = preproc_config.max_point_mad
        if max_point_mad is not None and "mad" in df_src.columns:
            df_src = df_src[df_src["mad"] <= max_point_mad]
            logger.info(
                f"Source {src_id}: Applied point MAD filtering with threshold {max_point_mad}. Points before: {num_points}, after: {len(df_src)}."
            )

        # Apply other DIC filters if any
        df_src = apply_dic_filters(df_src, **preproc_config.filter_kwargs)
        logger.info(
            f"Source {src_id}: Applied DIC filters. Points before: {num_points}, after: {len(df_src)}."
        )

        # Append processed dataframe to the list
        processed.append(df_src)

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


def vectorize_clustering_results(
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
    X, Y, cluster_grid = create_2d_grid(x=x, y=y, labels=cluster_labels)
    raster_res = abs(float(X[0, 1] - X[0, 0]) if X.shape[1] > 1 else 1.0)

    # TODO: hard-coded parameters for now, can be made configurable
    # Remove very small components and merge to nearest neighbor
    cluster_grid_raw = cluster_grid.copy()
    cluster_grid = remove_small_grid_components(
        label_grid=cluster_grid,
        min_size=20,
        connectivity=8,
        merge_strategy="merge",  # merge small components to nearest neighbor
    )
    # Apply a few iterations of erosion + dilation to the cluster grid to remove small noisy clusters before vectorization (
    cluster_grid = apply_morphological_operations(
        cluster_grid=cluster_grid,
        erosion_iterations=1,
        dilation_iterations=2,
        min_cluster_size=20,
        connectivity=8,
    )
    cluster_grid = close_small_holes(cluster_grid, max_hole_size=20)

    # 2. Vectorize & Smooth
    logger.info(f"Vectorizing grid clusters ({base_name})...")
    sectors = vectorize_gridded_sectors(cluster_grid, X, Y)

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

    # 6. Plot raw vs vectorized clusters
    if img is not None:
        sector_colors = get_sector_colors(
            sectors["sector"].tolist(),
            colormap=config.plotting.default_discrete_cmap,
        )

        fig, axes = plt.subplots(1, 3, figsize=(14, 7))
        plot_clustering_grid(
            ax=axes[0],
            img=img,
            cluster_grid=cluster_grid_raw,
            X=X,
            Y=Y,
            title="Raw Clusters",
            alpha=0.6,
        )
        plot_clustering_grid(
            ax=axes[1],
            img=img,
            cluster_grid=cluster_grid,
            X=X,
            Y=Y,
            title="Processed Grid Clusters",
            alpha=0.6,
        )
        plot_sectors(
            ax=axes[2],
            sectors=sectors,
            img=img,
            sector_colors=sector_colors,
            title="Vectorized Sectors",
        )
        fig.savefig(output_dir / f"{base_name}_raw_vs_vec.jpg", dpi=150)
        plt.close(fig)

    return sectors, pts_by_sector


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
    out = {}
    dic_analyses = None
    img = None

    if data_source == "file":
        logger.info("Reading data from local file.")
        file_path = config.data.get("file_path")

        if file_path and Path(file_path).is_file():
            nc_paths = [Path(file_path)]
        else:  # Auto-discover mode
            search_dir = (
                Path(file_path)
                if (file_path and Path(file_path).is_dir())
                else Path(config.data.get("search_dir", "."))
            )
            pattern = config.data.get("search_pattern")
            nc_paths = find_ensemble_files(
                search_dir=search_dir,
                reference_date=reference_date,
                dt_hours_min=config.data.dt_min,
                dt_hours_max=config.data.dt_max,
                filename_pattern=pattern,
            )

        if not nc_paths:
            logger.error(f"No files found for {reference_date}.")
            return False

        # 1. Read all candidates
        base_img_dir = (
            Path(config.data.image_dir) if config.data.get("image_dir") else None
        )
        candidates = {}
        for p in nc_paths:
            try:
                df, meta = read_data_from_pylamma_nc(p, base_image_dir=base_img_dir)
                if not df.empty:
                    candidates[p.stem] = (df, meta)
            except Exception as e:
                logger.warning(f"Failed to read {p}: {e}")

        # 2. Filter by quality
        mean_mad_threshold = config.preprocessing.mean_global_mad_threshold
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

        out = {best["name"]: best["df"]}
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
            ref_date_str = reference_date_dt.strftime("%Y_%m_%d")
            img_candidates = sorted(base_img_dir.glob(f"*{ref_date_str}*.jpg"))

            # Take middle one if multiple candidates found
            if img_candidates:
                img_path = img_candidates[len(img_candidates) // 2]
                try:
                    img = Image.open(img_path)
                    logger.info(f"Loaded background image from {img_path}")
                except Exception as e:
                    logger.warning(f"Could not load image {img_path}: {e}")

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
        output_dir / f"{base_name}_dic_analyses-master{date_start}_slave{date_end}.csv",
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

    # Plot the preprocessed DIC data for visual inspection
    dic_plot_result = plot_dic_vectors(
        x=dic_df["x"].to_numpy(),
        y=dic_df["y"].to_numpy(),
        u=dic_df["u"].to_numpy(),
        v=dic_df["v"].to_numpy(),
        magnitudes=dic_df["V"].to_numpy(),
        background_image=img,
        cmap_name="OrRd",
        figsize=(10, 8),
        title="Preprocessed DIC Vectors",
    )
    if dic_plot_result:
        fig, _, _ = dic_plot_result
        fig.savefig(output_dir / f"{base_name}_preprocessed_dic_vectors.jpg", dpi=150)
        plt.close(fig)

    timer.update("data_loading_and_preprocessing")

    # ===  MCMC CLUSTERING   === #
    # -- Assign Priors (Spatial-based)
    logger.info("Assigning spatial priors based on sectors...")
    prior_probs_array = mcmc.assign_spatial_priors(
        x=dic_df["x"].to_numpy(),
        y=dic_df["y"].to_numpy(),
        polygons=sectors,
        prior_probs=config.mcmc.priors.probability,
        fade_method=config.mcmc.priors.fade_method,
        fade_options=config.mcmc.priors.fade_options,
    )

    # -- Preprocess features and scale them for MCMC
    logger.info("Transforming and scaling features for MCMC...")
    data_array_scaled, scaler, velocities, transform_info = (
        transform_and_scale_features(
            df_input=dic_df,
            variables_names=config.preprocessing.variables_names,
            transform_velocity=config.preprocessing.velocity_transform,
            transform_params=config.preprocessing.transform_params,
            feature_weights=config.preprocessing.feature_weights,
            make_plots=True,
            output_dir=output_dir,
            base_name=base_name,
        )
    )
    joblib.dump(scaler, output_dir / f"{base_name}_mcmc_feature_scaler.joblib")
    np.savetxt(
        output_dir / f"{base_name}_mcmc_scaled_features.csv",
        data_array_scaled,
        delimiter=",",
        header=",".join(config.preprocessing.variables_names),
        comments="",
    )

    # -- Run MCMC Clustering
    logger.info("Running MCMC Clustering...")
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
        output_dir=output_dir,
        base_name=f"{base_name}_mcmc",
        debug=True,
        save_ctx={"df_input": dic_df, "scaler": scaler, "img": img},
    )

    timer.update("mcmc_clustering")

    # ===  POST-PROCESSING AND CLEANING OF FINAL CLUSTERING  === #

    # Use the shared function for the main data
    logger.info("Processing main clustering results...")
    sectors, pts_by_sector = vectorize_clustering_results(
        df_points=dic_df,  # The original full dataframe
        cluster_labels=result.cluster_pred,
        output_dir=output_dir,
        base_name=base_name,
        config=config,
        img=img,
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

    # Make a summary plot of the final sectors with points colored by velocity
    try:
        sector_colors = get_sector_colors(
            sectors["sector"].tolist(),
            colormap=config.plotting.default_discrete_cmap,
        )
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

        # Save the figure inrun-specific output directory
        fig.savefig(output_dir / f"{base_name}.jpg", dpi=150, bbox_inches="tight")
        fig.savefig(output_dir / f"{base_name}.svg", bbox_inches="tight")

        # Save it also in the common kinematic sectors summary folder
        kinematic_sectors_dir = output_base_dir / "kinematic_sectors"
        kinematic_sectors_dir.mkdir(exist_ok=True)
        fig.savefig(
            kinematic_sectors_dir / f"{base_name}.png", dpi=150, bbox_inches="tight"
        )

        plt.close(fig)

    except Exception as e:
        logger.error(f"Failed to plot summary: {e}", exc_info=True)

    logger.info("Processing complete.")
    timer.print()

    return True


def main():
    args = parse_arguments()

    # Load Base Config
    config_path = args.config if args.config else None
    config = load_config(config_path)

    # Apply Specific CLI Flags (Highest Priority for these shortcuts)
    if args.date:
        config.data.reference_date = args.date
    if args.output_dir:
        config.data.base_output_dir = args.output_dir

    # Apply Generic Dotlist Overrides
    if args.overrides:
        # OmegaConf can natively merge list of dot-string arguments
        # e.g. ["data.dt_min=10", "mcmc.regularization=False"]
        cli_conf = OmegaConf.from_dotlist(args.overrides)
        config = OmegaConf.merge(config, cli_conf)

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

    # Resolve Configuration with interpolations (e.g. ${data.output_dir}) now.
    OmegaConf.resolve(config)

    # Run the main pipeline with error handling to ensure that if something goes wrong, we log it and optionally clean up any partial outputs.
    base_output_dir = Path(config.data.base_output_dir)
    run_output_subdir = config.data.get("run_output_subdir", None)
    if run_output_subdir:
        output_dir = base_output_dir / run_output_subdir
    else:
        output_dir = base_output_dir

    # Check if output directory already exists and skip if --skip-existing is set. This prevents overwriting previous runs and allows for safe re-runs without manual cleanup.
    if args.skip_existing and output_dir.exists():
        logger.info(f"Skipping run: output directory {output_dir} already exists.")
        return

    # Create output directory before running the pipeline
    output_dir.mkdir(parents=True, exist_ok=True)

    # Force CPU for MCMC if specified in config to avoid potential GPU-related issues with JAX in some environments. This should be set before any JAX imports.
    if config.mcmc.force_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"

    # Run the pipeline
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


if __name__ == "__main__":
    main()
