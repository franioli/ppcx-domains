import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

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
    find_ensemble_file,
    load_sectors_and_roi,
    read_data_from_db,
    read_data_from_pylamma_nc,
)
from ppcluster.griddata import create_2d_grid, plot_clustering_grid
from ppcluster.mcmc.clustering import (
    aggregate_multiscale_clustering,
    clusterize_gaussian_mixture,
    save_sampling_summary,
)
from ppcluster.preprocessing import (
    apply_2d_gaussian_filter,
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


def run_pipeline(config: DictConfig | ListConfig):
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

    # Output base directory (output will be saved in a subfolder with camera name and date)
    # config.data.output_dir is already resolved in __main__
    output_base_dir = Path(config.data.output_dir)
    output_dir = output_base_dir / f"{config.data.camera_name}_{reference_date}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define base names for outputs
    base_name = f"{reference_date}"
    mcmc_base_name = f"{reference_date}_mcmc"

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

        try:
            # Pass image_dir from config if available
            base_img_dir = config.data.get("image_dir")
            out, dic_analyses, img = read_data_from_pylamma_nc(
                nc_path, base_image_dir=Path(base_img_dir) if base_img_dir else None
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

    date_start = dic_analyses.iloc[0]["master_timestamp"].strftime("%Y-%m-%d")
    date_end = dic_analyses.iloc[0]["slave_timestamp"].strftime("%Y-%m-%d")
    dic_analyses.to_csv(
        output_dir
        / f"{base_name}_master_{date_start}_slave{date_end}_dic_analyses.csv",
        index=False,
    )
    logger.info("Selected DIC analyses:")
    logger.info(dic_analyses.head())

    # Apply filter for each df in the dictionary and then stack them
    preproc_config = config.preprocessing
    dic_df = preprocess_dic_data(
        out=out,
        roi=roi,
        preproc_config=preproc_config,
    )
    # Save preprocessed DIC data
    dic_df.to_csv(output_dir / f"{base_name}_preprocessed_dic_data.csv", index=False)
    logger.info("Sample of preprocessed DIC data:")
    logger.info(dic_df.head())

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

    # Check if we are in "Sector A Refinement" mode
    # If the user asks for sectors that don't exist in the loaded polygon file,
    # assume we want velocity-based unsupervised clustering.
    # Example config for refinement:
    # priors:
    #   probability:
    #     Base: [0.5, 0.5]
    #     Fast: [0.5, 0.5]
    use_spatial_priors = True
    for name in config.mcmc.priors.probability:
        if name not in sectors:
            use_spatial_priors = False
            break
    use_velocity_priors_for_refinement = True  # TODO: hardcoded for now

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
            logger.error(f"Failed to assign spatial priors: {exc}")
            raise
    else:
        if use_velocity_priors_for_refinement:
            logger.info(
                "Target sectors not found in polygons. Switching to VELOCITY-BASED priors (Unsupervised)."
            )
            # Ensure we have exactly 2 components for this specific task
            n_components = len(config.mcmc.priors.probability)
            if n_components != 2:
                logger.warning(
                    "Velocity refinement is optimized for 2 components. Results may vary."
                )

            # Extract V column
            v_data = dic_df["V"].to_numpy()

            # Custom logic: Split by percentile
            # Cluster 0 = Slow/Base, Cluster 1 = Fast
            percentile_threshold = 90  # Top 10% guess - TODO: hardcoded for now
            threshold = np.percentile(v_data, percentile_threshold)
            prior_probs_array = np.zeros((len(v_data), n_components))
            for i, v_val in enumerate(v_data):
                if v_val > threshold:
                    prior_probs_array[i, :] = [0.2, 0.8]  # 80% provof being Fast
                else:
                    prior_probs_array[i, :] = [0.8, 0.2]  # 80% chance of being Base

            # Update sectors dictionary to match new fictitious names for the output handling
            # We create a dummy geometry (envelope of all points) just to satisfy the pipeline
            sectors = {
                list(config.mcmc.priors.probability.keys())[0]: roi,  # Base
                list(config.mcmc.priors.probability.keys())[1]: roi,  # Fast
            }
        else:
            raise NotImplementedError(
                "Velocity-based priors without spatial sectors is not implemented."
            )
            # Default: uniform priors across sectors (not used. fail if spatial priors requested)
            # if not config.mcmc.priors.probability:
            #     n_sectors = len(sectors)
            #     uniform_prob = 1.0 / n_sectors
            #     config.mcmc.priors.probability = {
            #         name: [uniform_prob] * n_sectors for name in sectors
            #     }

    fig, axes = mcmc.plot_spatial_priors(dic_df, prior_probs_array, img=img)
    fig.savefig(
        output_dir / f"{mcmc_base_name}_spatial_priors.jpg",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Perform MCMC clustering without any multiscale smoothing
    if (
        config.multiscale.sigma_values is None
        or len(config.multiscale.sigma_values) == 1
    ):
        df_run = dic_df.copy()

        # Preprocess features for clustering
        data_array_scaled, scaler, velocities, transform_info = preprocess_features(
            df_input=df_run,
            variables_names=config.preprocessing.variables_names,
            transform_velocity=config.preprocessing.velocity_transform,
            transform_params=config.preprocessing.transform_params,
            feature_weights=config.preprocessing.feature_weights,
        )
        joblib.dump(scaler, output_dir / f"{base_name}_feature_scaler.joblib")

        # Run MCMC clustering with the a gaussian mixture model
        result = clusterize_gaussian_mixture(
            data_array_scaled=data_array_scaled,
            prior_probs=prior_probs_array,
            sectors=sectors,
            sample_args=config.mcmc.sample_options,
            mu_params=config.mcmc.model_options.mu_params,
            sigma_params=config.mcmc.model_options.sigma_params,
            apply_mrf_regularization=config.mcmc.mrf_regularization,
            x_pos=df_run["x"].to_numpy(),
            y_pos=df_run["y"].to_numpy(),
            mrf_kwargs=config.mcmc.mrf_kwargs,
            second_pass=config.mcmc.second_pass,
            second_pass_sample_args=config.mcmc.second_pass_sample_args,
            force_cpu=config.mcmc.force_cpu,
            random_seed=config.random_seed,
        )

        # Save sampling summary
        save_sampling_summary(
            convergence_flag=result.convergence_flag,
            idata=result.idata,
            output_dir=output_dir,
            base_name=f"{base_name}_mcmc",
            make_plots=True,
            df_input=df_run,
            cluster_pred=result.cluster_pred,
            posterior_probs=result.posterior_probs,
            scaler=scaler,
            img=img,
        )

        # Extract results
        cluster_pred = result.cluster_pred
        posterior_probs = result.posterior_probs
        entropy = -np.sum(posterior_probs * np.log(posterior_probs + 1e-10), axis=1)
        similarity_matrix = None
        stability_score = None
        valid_scales = None
    else:
        raise NotImplementedError(
            "Multiscale aggregation is broken, as the old result dict was replaced with a ClusteringResult object that does not contain 'sigma' key anymore."
        )

        # Loop through smoothing scales
        results = []
        for sigma in config.multiscale.sigma_values:
            logger.info(f"Processing with Gaussian smoothing sigma={sigma}...")

            # Create scale-specific base name
            scale_base_name = f"{mcmc_base_name}_sigma{sigma}"

            # Apply Gaussian smoothing if needed (skipped for sigma=0)
            df_run = apply_2d_gaussian_filter(dic_df, sigma=sigma)

            # Preprocess features for clustering
            data_array_scaled, scaler, velocities, transform_info = preprocess_features(
                df_input=df_run,
                variables_names=config.data.variables_names,
                transform_velocity=config.mcmc.velocity_transform,
                transform_params=config.mcmc.transform_params,
            )
            joblib.dump(scaler, output_dir / f"{scale_base_name}_scaler.joblib")

            # Run MCMC clustering with the a gaussian mixture model
            if sigma > 2:  # For larger sigma, tighten priors
                mu_params = {"mu": 0, "sigma": 0.5}
                sigma_params = {"sigma": 0.5}
            else:
                mu_params = config.mcmc.model_options.mu_params
                sigma_params = config.mcmc.model_options.sigma_params
            result = clusterize_gaussian_mixture(
                data_array_scaled=data_array_scaled,
                prior_probs=prior_probs_array,
                sectors=sectors,
                sample_args=config.mcmc.sample_options,
                mu_params=mu_params,
                sigma_params=sigma_params,
                feature_weights=config.mcmc.feature_weights,
                apply_mrf_regularization=config.mcmc.mrf_regularization,
                mrf_kwargs=config.mcmc.mrf_kwargs,
                second_pass=config.mcmc.second_pass,
                second_pass_sample_args=config.mcmc.second_pass_sample_args,
                random_seed=config.random_seed,
            )

            # --- Save sampling summary ---
            save_sampling_summary(
                convergence_flag=result["convergence_flag"],
                idata=result["idata"],
                output_dir=output_dir,
                base_name=scale_base_name,
            )

            # Add scale information to result
            # result["sigma"] = sigma

            # Append to results list
            results.append(result)

        # aggregate multi-scale results
        aggregated_results = aggregate_multiscale_clustering(
            results,
            similarity_threshold=config.multiscale.aggregation.similarity_threshold,
            overall_threshold=config.multiscale.aggregation.overall_threshold,
            fig_path=output_dir / f"{mcmc_base_name}_similarity_heatmap.jpg",
        )
        cluster_pred = aggregated_results["combined_cluster_pred"]
        posterior_probs = aggregated_results["avg_posterior_probs"]
        entropy = aggregated_results["avg_entropy"]
        similarity_matrix = aggregated_results["similarity_matrix"]
        stability_score = aggregated_results["stability_score"]
        valid_scales = aggregated_results["valid_scales"]

    # --- Save final clustering results ---
    cluster_aggregation_outs = {
        "cluster_pred": cluster_pred,
        "posterior_probs": posterior_probs,
        "entropy": entropy,
        "similarity_matrix": similarity_matrix,
        "stability_score": stability_score,
        "valid_scales": valid_scales,
    }
    joblib.dump(
        cluster_aggregation_outs,
        output_dir / f"{mcmc_base_name}_clustering_results_raw.joblib",
    )
    timer.update("mcmc_clustering")

    # ===  POST-PROCESSING AND CLEANING OF FINAL CLUSTERING  === #

    # Retrieve data
    x = dic_df["x"].to_numpy()
    y = dic_df["y"].to_numpy()
    kin_cluster = np.asarray(cluster_pred.copy())

    # Create 2D grid of clustering results
    X, Y, kin_cluster_grid = create_2d_grid(x=x, y=y, labels=kin_cluster)
    raster_res = abs(float(X[0, 1] - X[0, 0]) if X.shape[1] > 1 else 1.0)

    # Apply morphological cleaning to the cluster grid
    # NOTE: Replaced by better vector-based cleaning below
    # kin_cluster_grid = apply_cluster_grid_cleaning(
    #     X=X,
    #     Y=Y,
    #     clusters=kin_cluster_grid,
    #     config=postproc_config,
    #     output_dir=output_dir,
    #     base_name=base_name,
    #     img=img,
    # )

    # ===  KINEMATIC SECTORS COMPUTATION ===
    # 1. Vectorize & Smooth
    logger.info("Vectorizing grid clusters to polygons...")
    sectors = vectorize_gridded_sectors(kin_cluster_grid, X, Y)
    sectors.to_file(output_dir / f"{base_name}_sectors_raw.geojson", driver="GeoJSON")
    vect_config = config.postprocessing.vectorization
    sectors = clean_vector_sectors(
        sectors,
        dic_df,
        min_area_px2=vect_config.min_area_px2,
        isolation_buffer=vect_config.isolation_buffer,
        velocity_merge_threshold=vect_config.velocity_merge_threshold,
        force_minimum_sectors=vect_config.force_minimum_sectors,
        target_number_of_sectors=vect_config.target_number_of_sectors,
        fill_holes_area=vect_config.fill_holes_area,
        smooth_geometries=vect_config.smooth_geometries,
        smooth_method=vect_config.smooth_method,
        smooth_iterations=vect_config.smooth_iterations,
        raster_res=2 * raster_res,
    )

    # 2. Assign Labels (A, B, C...)
    # We use the centroid Y position to order sectors from bottom to top (A=lowest Y). The y axis is inverted in image coordinates (0 at top), hence ascending=False
    sectors = assign_sector_labels(
        sectors,
        order_by=config.postprocessing.sector_assignment.method,
        ascending=config.postprocessing.sector_assignment.ascending,
    )

    # Drop all but essential columns and sort by sector label
    sectors = (
        sectors[["geometry", "sector"]].sort_values(by="sector").reset_index(drop=True)
    )

    # Plot raw clusters vs vectorized sectors
    sector_colors = get_sector_colors(
        sectors["sector"].tolist(),
        colormap=config.plotting.default_discrete_cmap,
    )
    fig, (ax_raw, ax_vectorized) = plt.subplots(1, 2, figsize=(14, 7))
    plot_clustering_grid(
        ax=ax_raw,
        img=img,
        cluster_grid=kin_cluster_grid,
        X=X,
        Y=Y,
        title="Raw clustering assignments",
        show_legend=True,
        show_stats=True,
        alpha=0.6,
    )
    plot_sectors(
        ax=ax_vectorized,
        sectors=sectors,
        img=img,
        velocity_df=None,
        sector_colors=sector_colors,
        label_column="sector",
        add_sector_labels=False,
        title=f"Cleaned vectorized sectors (n={len(sectors)})",
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"{base_name}_clustering_raw_vs_vectorized.jpg", dpi=150)
    plt.close(fig)

    # 3. Classify the original points dataframe and compute statistics
    pts_by_sector = classify_points_by_polygons(sectors, dic_df, x_col="x", y_col="y")
    pts_by_sector.to_file(
        output_dir / f"{base_name}_dic_points_by_sector.geojson",
        driver="GeoJSON",
    )

    # Compute sector statistics
    sectors["area_px2"] = sectors.geometry.area
    sectors = compute_sector_stats(
        sectors, pts_by_sector, value_col="V", group_col="sector"
    )
    sectors.to_file(output_dir / f"{base_name}_sectors_final.geojson", driver="GeoJSON")
    stats = sectors.drop(columns=[sectors.geometry.name], errors="ignore")
    stats.to_csv(
        output_dir / f"{base_name}_sector_stats.csv",
        index=False,
        float_format="%.3f",
    )
    logger.info(f"Saved final sectors GeoJSON with stats to {output_dir}")

    logger.info("Creating summary figure...")
    sector_figure_path = plot_sectors_summary(
        sectors=sectors,
        points_by_sector=pts_by_sector,
        img=img,
        colors=sector_colors,
        output_dir=output_dir,
        base_name=base_name,
        unit="px",
        quiver_kwargs=config.plotting.quiver,
        figsize=(20, 10),
        dpi=300,
        save_svg=True,
    )
    if sector_figure_path is not None:
        sector_figures_dir = output_base_dir / "kinematic_sectors"
        sector_figures_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(sector_figure_path, sector_figures_dir / f"{base_name}_sectors.png")
        logger.info(f"Copied summary figure to {sector_figures_dir}")

    # ==== SAVE FINAL RESULTS ==== #
    logger.info("Saving final sector results...")

    # 1) Pythonized bundle with all dataframes and arrays
    bundle = {
        "reference_date": reference_date,
        "date_start": date_start,
        "date_end": date_end,
        "dic_dataframe": dic_df,
        "posterior_probs": posterior_probs,
        "cluster_pred": cluster_pred,
        "uncertainty": entropy,
        "sectors": sectors,
        "pts_by_sector": pts_by_sector,
    }
    joblib.dump(bundle, output_dir / f"{base_name}_results.joblib")
    timer.update("post-processing")

    logger.info("Processing complete.")
    timer.print()


if __name__ == "__main__":
    args = parse_arguments()

    # 1. Load Base Config
    config_path = args.config if args.config else None
    config = load_config(config_path)

    # 2. Apply Specific CLI Flags (Highest Priority for these shortcuts)
    if args.date:
        config.data.reference_date = args.date
    if args.output_dir:
        config.data.output_dir = args.output_dir

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

    run_pipeline(config)
