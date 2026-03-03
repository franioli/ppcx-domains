import argparse
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
from PIL import Image

matplotlib.use("Agg")

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import DictConfig, ListConfig, OmegaConf

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
    vectorize_gridded_sectors,
)
from ppcluster.visualization import (
    get_sector_colors,
    plot_dic_vectors,
    plot_sectors,
    plot_sectors_summary,
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
    max_mad: float
    | None = None,  # *TODO: include this in the config and apply MAD filtering if specified
) -> pd.DataFrame:
    """
    Run spatial filtering, DIC filters, stacking, and subsampling on raw DIC data.

    Args:
        out: Dictionary mapping source IDs to raw DIC DataFrames.
        roi: ROI polygon for spatial filtering.
        preproc_config: Dictionary of preprocessing parameters.
        max_mad: Optional maximum MAD threshold for filtering. If None, no MAD filtering is applied.

    Returns:
        pd.DataFrame: The fully preprocessed and stacked DataFramdPl
    """
    processed = []
    for src_id, df_src in out.items():
        # Filter only points inside the spatial priors sectors
        if roi is not None:
            df_src = filter_dataframe_by_polygons(df_src, polygon=roi)

        num_points = len(df_src)

        # Apply MAD filtering if max_mad is specified
        if max_mad is not None and "mad" in df_src.columns:
            df_src = df_src[df_src["mad"] <= max_mad]
            logger.info(
                f"Source {src_id}: Applied MAD filtering with threshold {max_mad}. Points before: {num_points}, after: {len(df_src)}."
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

    # Save the scaled array as text file for debugging and inspection
    np.savetxt(
        output_dir / f"{base_name}_mcmc_scaled_features.csv",
        data_array_scaled,
        delimiter=",",
        header=",".join(config.preprocessing.variables_names),
        comments="",
    )

    # --- Debugging Plot: Scaled vs Original Distributions ---
    n_feats = data_array_scaled.shape[1]
    fig, axes = plt.subplots(n_feats, 1, figsize=(10, 4 * n_feats), squeeze=False)

    for i, var_name in enumerate(config.preprocessing.variables_names):
        ax = axes[i, 0]
        scaled_data = data_array_scaled[:, i]
        orig_data = scaler.inverse_transform(data_array_scaled)[:, i]

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
        mu_params=OmegaConf.to_container(
            config.mcmc.model_options.mu_params, resolve=True
        ),
        sigma_params=OmegaConf.to_container(
            config.mcmc.model_options.sigma_params, resolve=True
        ),
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
        mean_mad_threshold = 2.0  # *TODO: hardcoded value!
        min_ensemble_size = 2  # *TODO: hardcoded value!
        valid_results = []

        for name, (df, meta) in candidates.items():
            mean_mad = float(df["mad"].mean())
            min_ens = (
                int(df["ensemble_size"].min()) if "ensemble_size" in df.columns else 99
            )

            if mean_mad > mean_mad_threshold:
                logger.warning(
                    f"Rejecting {name}: MAD {mean_mad:.2f} > {mean_mad_threshold}"
                )
                continue
            if min_ens < min_ensemble_size:
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

        # 3. Select best candidate (lowest MAD, then largest ensemble, then largest dt)
        valid_results.sort(key=lambda x: (x["mad"], -x["ens"], -x["dt"]))
        best = valid_results[0]
        logger.info(
            f"Selected best DIC map: {best['name']} (MAD: {best['mad']:.2f}, DT: {best['dt']:.1f}h)"
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
        output_dir / f"{base_name}_master{date_start}_slave{date_end}_dic_analyses.csv",
        index=False,
    )

    # Apply filter for each df in the dictionary and then stack them
    preproc_config = config.preprocessing
    dic_df = preprocess_dic_data(
        out=out,
        roi=roi,
        preproc_config=preproc_config,
        max_mad=50,  # *TODO: hardcoded value, consider moving to config
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
        fig.savefig(output_dir / f"{base_name}.png", dpi=150, bbox_inches="tight")
        fig.savefig(output_dir / f"{base_name}.svg", dpi=150, bbox_inches="tight")

        # Save it also in the common kinematic sectors summary folder
        kinematic_sectors_dir = output_base_dir / "kinematic_sectors"
        kinematic_sectors_dir.mkdir(exist_ok=True)
        fig.savefig(
            kinematic_sectors_dir / f"{base_name}.png", dpi=150, bbox_inches="tight"
        )

        plt.close(fig)

    except Exception as e:
        logger.error(
            f"Failed to plot summary with anomaly: {e}. Fallback to copy figure without anomaly.",
            exc_info=True,
        )

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

    # Check if output directory already exists and skip if --skip-existing is set. This prevents overwriting previous runs and allows for safe re-runs without manual cleanup.
    if args.skip_existing and output_dir.exists():
        logger.info(f"Skipping run: output directory {output_dir} already exists.")
        return

    # Create output directory before running the pipeline
    output_dir.mkdir(parents=True, exist_ok=True)

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


if __name__ == "__main__":
    main()
