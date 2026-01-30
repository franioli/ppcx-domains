import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import DictConfig, ListConfig, OmegaConf
from sqlalchemy import create_engine

from ppcluster import load_config, mcmc, setup_logger
from ppcluster.cvat import (
    filter_dataframe_by_polygons,
    read_polygons_from_cvat,
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
from ppcluster.utils.database import (
    fetch_dic_analysis_ids,
    get_dic_analysis_by_ids,
    get_image,
    get_multi_dic_data,
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


def read_sectors_from_file(sector_prior_path: Path, sector_names: list[str]):
    """
    Load sector polygons from a CVAT XML or geospatial file.
    Returns: sectors_dict.
    """
    sectors = {}
    if sector_prior_path.suffix.lower() in (".xml", ".zip"):
        logger.info(f"Loading sectors from CVAT XML: {sector_prior_path}")
        sectors = read_polygons_from_cvat(
            sector_prior_path,
            image_ids=[0],
            include_labels=sector_names,
        )
    elif sector_prior_path.suffix.lower() in (".geojson", ".shp", ".gpkg"):
        logger.info(f"Loading sectors from geospatial file: {sector_prior_path}")
        gdf_priors = gpd.read_file(sector_prior_path)
        label_col = None
        for candidate in ["sector", "label", "name", "class", "id"]:
            if candidate in gdf_priors.columns:
                label_col = candidate
                break
        if not label_col:
            raise ValueError(
                f"Could not find a classification label column in {sector_prior_path}."
            )
        for _, row in gdf_priors.iterrows():
            lbl = str(row[label_col])
            geom = row.geometry
            if lbl in sector_names:
                if lbl in sectors:
                    sectors[lbl] = sectors[lbl].union(geom)
                else:
                    sectors[lbl] = geom
    else:
        raise ValueError(
            f"Unsupported sector prior file format: {sector_prior_path.suffix}"
        )
    return sectors


def read_roi_from_file(path: Path):
    """
    Load ROI polygon from a CVAT XML or geospatial file.
    Returns: roi_polygon or None.
    """
    roi = None
    if not path or not Path(path).exists():
        return None

    path = Path(path)
    if path.suffix.lower() in (".xml", ".zip"):
        try:
            logger.info(f"Loading ROI from CVAT XML: {path}")
            roi_poly = read_polygons_from_cvat(
                path, image_ids=[0], include_labels=["ROI", "roi"]
            )
            roi = roi_poly.get("ROI") or roi_poly.get("roi")
        except Exception as e:
            logger.warning(f"Failed to read ROI from XML {path}: {e}")
    elif path.suffix.lower() in (".geojson", ".shp", ".gpkg"):
        try:
            logger.info(f"Loading ROI from geospatial file: {path}")
            gdf_roi = gpd.read_file(path)
            roi = gdf_roi.geometry.union_all()
        except Exception as e:
            logger.warning(f"Failed to read ROI from geospatial file {path}: {e}")
    else:
        logger.warning(
            f"Unsupported ROI file format: {path.suffix}. Skipping ROI loading."
        )
    return roi


def load_sectors_and_roi(
    sector_prior_path: Path | str,
    sector_names: list[str],
    roi_path: Path | str | None = None,
):
    """
    Load sector polygons and ROI polygon from a CVAT XML or geospatial file.
    Returns: (sectors_dict, roi_polygon or None).
    """

    # Find matching sector prior file (supports glob patterns)
    prior_file_pattern = Path(sector_prior_path)
    sector_prior_paths = list(
        prior_file_pattern.parent.glob(Path(prior_file_pattern).name)
    )
    if len(sector_prior_paths) == 0:
        raise FileNotFoundError(
            f"No sector prior file found matching: {sector_prior_path}"
        )
    if len(sector_prior_paths) > 1:
        logger.warning(
            f"Multiple sector prior files matched. Using the first one found: {list(sector_prior_paths)}"
        )
    sector_prior_path = sector_prior_paths[0]

    # Load sectors
    sectors = read_sectors_from_file(sector_prior_path, sector_names)
    # if not sectors or any(name not in sectors for name in sector_names):
    #     missing = [name for name in sector_names if name not in sectors]
    #     raise ValueError(
    #         f"Sectors missing in prior file {sector_prior_path}: {missing}. Expected: {sector_names}"
    #     )

    # Try to read ROI from separate file if provided, else from sector_prior_path
    roi = None
    if roi_path is not None:
        roi_path = Path(roi_path)
        roi = read_roi_from_file(roi_path)
    if roi is None:
        roi = read_roi_from_file(sector_prior_path)

    if roi is None:
        logger.warning("No ROI polygon provided. Skipping spatial filtering.")

    return sectors, roi


def run_pipeline(config: DictConfig | ListConfig):
    """
    Main execution pipeline taking a fully merged configuration object.
    """
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

    # Read sectors for spatial priors and ROI
    # sectors is now dict[str, shapely.Polygon]
    sector_names = list(config.mcmc.priors.probability.keys())
    sectors, roi = load_sectors_and_roi(
        sector_prior_path=config.data.sector_prior_path,
        sector_names=sector_names,
        roi_path=config.data.roi_path,
    )

    # Date range for data selection
    reference_start_date = None
    reference_end_date = None
    if config.data.days_before_to_include > 0:
        reference_start_date = reference_date_dt - pd.Timedelta(
            days=config.data.days_before_to_include
        )
    if config.data.days_after_to_include > 0:
        reference_end_date = reference_date_dt + pd.Timedelta(
            days=config.data.days_after_to_include
        )

    # Fetch DIC ids
    db_engine = create_engine(config.db_url)
    dic_ids = fetch_dic_analysis_ids(
        db_engine,
        camera_name=config.data.camera_name,
        reference_date=reference_date,
        reference_date_start=reference_start_date,
        reference_date_end=reference_end_date,
        dt_hours_min=config.data.dt_min,
        dt_hours_max=config.data.dt_max,
    )
    if len(dic_ids) < 1:
        raise ValueError("No DIC analyses found for the given criteria")

    # Get DIC analysis metadata and save to CSV
    dic_analyses = get_dic_analysis_by_ids(db_engine=db_engine, dic_ids=dic_ids)
    logger.info("Fetched DIC analysis:")
    for _, row in dic_analyses.iterrows():
        logger.info(
            f"DIC ID: {row['dic_id']}, date: {row['reference_date']}, dt (hrs): {row['dt_hours']}, Master: {row['master_timestamp']}, Slave: {row['slave_timestamp']}"
        )
    date_start = dic_analyses.iloc[0]["master_timestamp"].strftime("%Y-%m-%d")
    date_end = dic_analyses.iloc[0]["slave_timestamp"].strftime("%Y-%m-%d")
    dic_analyses.to_csv(
        output_dir / f"{date_start}_{date_end}_selected_dic_analyses.csv", index=False
    )
    logger.info("Selected DIC analyses:")
    logger.info(dic_analyses.head())

    # Get master image
    master_image_id = dic_analyses["master_image_id"].iloc[0]
    img = get_image(image_id=master_image_id, config=config.api)

    # Fetch DIC data
    out = get_multi_dic_data(dic_ids, stack_results=False, config=config.api)
    logger.info(f"Found stack of {len(out)} DIC dataframes.")

    # Apply filter for each df in the dictionary and then stack them
    preproc_config = config.preprocessing
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

    # Save preprocessed DIC data
    dic_df.to_csv(output_dir / f"{base_name}_preprocessed_dic_data.csv", index=False)
    logger.info("Sample of preprocessed DIC data:")
    logger.info(dic_df.head())

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
            random_seed=config.random_seed,
        )

        # Save sampling summary
        save_sampling_summary(
            convergence_flag=result.convergence_flag,
            idata=result.idata,
            output_dir=output_dir,
            base_name=base_name,
        )

        # Otherwise extract the single result
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

    logger.info("Processing complete.")


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
