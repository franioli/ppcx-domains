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
import shapely
from matplotlib import pyplot as plt
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image

from ppcluster import Timer, load_config, mcmc, setup_logger
from ppcluster.cvat import filter_dataframe_by_polygons
from ppcluster.data import (
    find_ensemble_files,
    load_sectors_and_roi,
    read_data_from_pylamma_nc,
)
from ppcluster.exceptions import DICMapNotFoundError
from ppcluster.griddata import (
    apply_morphological_operations,
    close_small_holes,
    create_2d_grid,
    plot_clustering_grid,
    remove_small_grid_components,
)
from ppcluster.mcmc.clustering import clusterize_gaussian_mixture
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
    smooth_polygons,
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


def run_pipeline(config: DictConfig | ListConfig) -> bool:
    """
    Main execution pipeline.
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
    prior_sectors, roi = load_sectors_and_roi(
        sector_prior_path=config.data.sector_prior_path,
        sector_names=sector_names,
        roi_path=config.data.roi_path,
    )

    base_img_dir = Path(config.data.image_dir) if config.data.get("image_dir") else None

    # Load the best DIC map according to the criteria (temporal proximity, MAD threshold, ensemble size).
    # We load the DIC map of the day before the reference date (t-1, t-dt), since this procedure is meant to run daily with the newest available DIC map (which is usually the day-before one).
    ref_date_to_load = reference_date_dt - pd.Timedelta(days=1)
    logger.info(
        f"Searching for DIC maps for {ref_date_to_load} (dt: {config.data.dt_min}-{config.data.dt_max} hours)."
    )
    try:
        dic_df, dic_analyses, img = load_best_dic_map(
            ref_date=ref_date_to_load,
            file_path=config.data.get("file_path"),
            search_dir=Path(config.data.get("search_dir")),
            search_pattern=config.data.get("search_pattern"),
            dt_min=config.data.dt_min,
            dt_max=config.data.dt_max,
            base_image_dir=base_img_dir,
            mean_mad_threshold=config.preprocessing.mean_global_mad_threshold,
            min_ensemble_size=config.preprocessing.min_ensemble_size,
        )
    except DICMapNotFoundError as e:
        logger.error(str(e))
        return False

    date_start = dic_analyses.iloc[0]["master_timestamp"].strftime("%Y-%m-%d")
    date_end = dic_analyses.iloc[0]["slave_timestamp"].strftime("%Y-%m-%d")
    dic_analyses.to_csv(
        output_dir / f"{base_name}_dic_analyses-master{date_start}_slave{date_end}.csv",
        index=False,
    )

    # Apply filter for each df in the dictionary and then stack them
    dic_df = preprocess_dic_data(
        dic_df=dic_df,
        roi=roi,
        preproc_config=config.preprocessing,
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
    logger.info("Assigning spatial priors based on prior_sectors...")
    prior_probs_array = mcmc.assign_spatial_priors(
        x=dic_df["x"].to_numpy(),
        y=dic_df["y"].to_numpy(),
        polygons=prior_sectors,
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
        sectors=prior_sectors,
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
        priors_sectors=prior_sectors,  # Dictionary of sector_name: Polygon for spatial priors, used to assign sector labels to points and compute stats
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


def load_best_dic_map(
    ref_date: datetime,
    file_path: str | None,
    search_dir: Path,
    search_pattern: str,
    dt_min: int,
    dt_max: int,
    base_image_dir: Path | None = None,
    mean_mad_threshold: float | None = None,
    min_ensemble_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Image.Image | None]:
    """
    Locate, read and select the best DIC NetCDF file for a given reference date.

    Args:
        ref_date: reference date as datetime (used to search day-before maps).
        file_path: explicit file path to a single NetCDF file (if provided, search is skipped).
        search_dir: directory to search for candidate NetCDF files.
        search_pattern: filename glob/pattern used by find_ensemble_files.
        dt_min, dt_max: accepted temporal difference limits passed to find_ensemble_files.
        base_image_dir: optional directory containing background images (used to load an image if metadata lacks one).
        mean_mad_threshold: optional MAD threshold to reject noisy maps (None disables this check).
        min_ensemble_size: optional minimum ensemble size to accept a map (None disables this check).

    Returns:
        tuple of (dic_df, dic_analyses_meta, image_or_none)

    Raises:
        FileNotFoundError: if no candidate files are found.
        RuntimeError: if no candidate passes the optional quality filters.
    """
    # 1. Locate candidate files
    if file_path and Path(file_path).is_file():
        logger.info(f"Using explicitly provided DIC file: {file_path}")
        nc_paths = [Path(file_path)]
    else:
        # discover candidate files
        nc_paths = find_ensemble_files(
            search_dir, ref_date.strftime("%Y-%m-%d"), dt_min, dt_max, search_pattern
        )
    if not nc_paths:
        raise DICMapNotFoundError(
            "No DIC files found for the specified date and criteria."
        )

    # 2. Read candidates
    candidates: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for p in nc_paths:
        try:
            df, meta = read_data_from_pylamma_nc(p, base_image_dir=base_image_dir)
            if not df.empty:
                candidates[p.stem] = (df, meta)
        except Exception as exc:
            logger.warning(f"Failed to read {p}: {exc}")

    if not candidates:
        raise DICMapNotFoundError("No readable DIC candidates found.")

    # 3. Evaluate and filter candidates (skip checks when thresholds are None)
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
            (mean_mad_threshold is not None)
            and (mean_mad is not None)
            and (mean_mad > mean_mad_threshold)
        ):
            logger.warning(
                f"Rejecting {name}: MAD {mean_mad:.2f} > {mean_mad_threshold}"
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
        raise DICMapNotFoundError("No DIC candidates passed quality filters.")

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
            ref_date_str = ref_date.strftime("%Y_%m_%d")
            img_candidates = sorted(Path(base_image_dir).glob(f"*{ref_date_str}*.jpg"))
            if img_candidates:
                img = Image.open(img_candidates[len(img_candidates) // 2])
                logger.info(
                    f"Loaded background image from {img_candidates[len(img_candidates) // 2]}"
                )
        except Exception as exc:
            logger.warning(f"Could not locate/load fallback image: {exc}")

    return dic_df, dic_meta, img


def preprocess_dic_data(
    dic_df: pd.DataFrame,
    roi: Any,
    preproc_config: DictConfig | ListConfig,
) -> pd.DataFrame:
    """
    Run spatial filtering, DIC filters, stacking, and subsampling on raw DIC data.

    Args:
        dic_data: Dictionary mapping source IDs to raw DIC DataFrames.
        roi: ROI polygon for spatial filtering.
        preproc_config: Dictionary of preprocessing parameters.

    Returns:
        pd.DataFrame: The fully preprocessed and stacked DataFrame.
    """
    dic_df = dic_df.copy()  # avoid modifying original dataframe

    # Filter only points inside the spatial priors sectors
    if roi is not None:
        dic_df = filter_dataframe_by_polygons(dic_df, polygon=roi)

    num_points = len(dic_df)

    # Apply MAD filtering if max_point_mad is specified
    max_point_mad = preproc_config.max_point_mad
    if max_point_mad is not None and "mad" in dic_df.columns:
        dic_df = dic_df[dic_df["mad"] <= max_point_mad]
        logger.info(
            f"Applied point MAD filtering with threshold {max_point_mad}. Points before: {num_points}, after: {len(dic_df)}."
        )

        # Apply other DIC filters if any
        dic_df = apply_dic_filters(dic_df, **preproc_config.filter_kwargs)
        logger.info(
            f"Applied DIC filters. Points before: {num_points}, after: {len(dic_df)}."
        )

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
    priors_sectors: gpd.GeoDataFrame
    | dict[str, gpd.GeoSeries | shapely.geometry.Polygon]
    | None = None,
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
        merge_strategy="merge",  # merge small components to nearest neighbor
    )
    # Apply a bit of morphological smoothing (opening→closing per round) to smooth the clusters and remove noise (this is optional but can help with vectorization)
    cluster_grid = apply_morphological_operations(
        cluster_grid=cluster_grid,
        smoothing_rounds=1,
        connectivity=4,
    )
    n_valid_cells = int(np.sum(~np.isnan(cluster_grid) & (cluster_grid >= 0)))
    if n_valid_cells == 0:
        raise RuntimeError(
            f"[{base_name}] Cluster grid is empty after morphological operations. "
            "All cells were erased — clusters may be too thin. "
            "Reduce smoothing_rounds or check the clustering quality."
        )

    cluster_grid = close_small_holes(cluster_grid, max_hole_size=20)

    # 2. Vectorize & Smooth
    logger.info(f"Vectorizing grid clusters ({base_name})...")
    sectors = vectorize_gridded_sectors(cluster_grid, X, Y)
    if sectors.empty:
        raise RuntimeError(
            f"[{base_name}] Vectorization produced no sectors. "
            "The cluster grid may contain no valid polygons — check the grid output."
        )

    # Save raw sectors for debugging
    sectors.to_file(output_dir / f"{base_name}_sectors_raw.geojson", driver="GeoJSON")

    # 3. Clean Vectors
    vect_config = config.postprocessing.vectorization
    sectors = clean_vector_sectors(
        sectors,
        df_points,
        split_disconnected=True,  # split disconnected sectors in multiple polygons
        min_area_px2=vect_config.min_area_px2,
        isolation_buffer=vect_config.isolation_buffer,
        velocity_merge_threshold=vect_config.velocity_merge_threshold,
        target_number_of_sectors=vect_config.target_number_of_sectors,
        max_number_of_sectors=8,  # Hardcoded max number of sectors (if more polygons than this number are found, than smallest will be removed)
        fill_holes_area=vect_config.fill_holes_area,
        smooth_geometries=vect_config.smooth_geometries,
        smooth_method=vect_config.smooth_method,
        smooth_iterations=vect_config.smooth_iterations,
        raster_res=2 * raster_res,
    )
    if sectors is None or sectors.empty:
        raise RuntimeError("No sectors found after vectorization and cleaning.")

    # 4. Assign Labels

    # Assign lables by overlapping with spatial priors, otherwise fallback to sorting by y-centroid
    #  assign each of the vectorized sector to a label (A, B, C, D...) based on the overlap with the priors (each prior is already assigned with a label)
    # multiple sectors can be assigned to the same label (this allows for not filtering the vectorized sectors to 4 polygons, but multiple polygon can belong to the same sector so eventually we will always have only 4 sectoes)

    # Convert priors_sectors dict[str, Polygon] → GeoDataFrame with 'sector_name' column
    priors_gdf: gpd.GeoDataFrame | None = None
    if priors_sectors is not None:
        if isinstance(priors_sectors, dict):
            geoms = [
                v.union_all() if isinstance(v, gpd.GeoSeries) else v
                for v in priors_sectors.values()
            ]
            priors_gdf = gpd.GeoDataFrame(
                {"sector_name": list(priors_sectors.keys())},
                geometry=geoms,
            )
        elif isinstance(priors_sectors, gpd.GeoDataFrame):
            priors_gdf = priors_sectors

    sectors = assign_sector_labels(
        sectors,
        method=config.postprocessing.sector_assignment.method,
        ascending=config.postprocessing.sector_assignment.ascending,
        priors_gdf=priors_gdf,
        df_points=df_points,
        ambiguity_threshold=config.postprocessing.sector_assignment.get(
            "ambiguity_threshold", 0.7
        ),
    )

    # Dissolve sectors by assigned label to ensure one polygon per sector (this allows to have multiple disconnected polygons for the same sector if the vectorization produced that, but we will not have more than 4 sectors in total)
    logger.info(f"Dissolving sectors by assigned labels ({base_name})...")
    sectors = sectors.dissolve(by="sector", as_index=False)

    # Run a final iteration of smoothing of the dissolved sectors to smooth the final geometries after label assignment and dissolving (this is optional but can help with the final output)
    logger.info(f"Smoothing final sector geometries ({base_name})...")
    sectors["geometry"] = sectors.geometry.buffer(0)  # fix potential invalid geometries
    sectors = smooth_polygons(
        sectors,
        smooth_method=vect_config.smooth_method,
        smooth_iterations=1,
        merge_collection=False,
        area_tolerance=0.01,  # % of original area allowed as error
        num_cores=1,  # Keep it single-threaded to avoid overhead on small datasets
        raster_res=2 * raster_res,
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
    if sectors is None:
        raise RuntimeError("No sectors found after computing stats.")

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


if __name__ == "__main__":
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
        exit(0)

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
