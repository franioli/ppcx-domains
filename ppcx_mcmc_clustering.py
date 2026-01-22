import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path

import arviz as az
import joblib
import numpy as np
import pandas as pd
import pymc as pm
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

from ppcluster import load_config, mcmc, setup_logger
from ppcluster.cvat import (
    filter_dataframe_by_polygons,
    read_polygons_from_cvat,
)
from ppcluster.griddata import create_2d_grid
from ppcluster.postproc import aggregate_multiscale_clustering, plot_clustering_grid
from ppcluster.preprocessing import (
    apply_2d_gaussian_filter,
    apply_dic_filters,
    preprocess_velocity_features,
    spatial_subsample,
)
from ppcluster.sectors import (
    assign_sector_labels,
    classify_points_by_polygons,
    clean_vector_sectors,
    compute_sector_stats,
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


def run_mcmc_clustering(
    df_input,
    prior_probs,
    sectors,
    output_dir,
    base_name,
    img=None,
    variables_names=None,
    transform_velocity="none",
    transform_params=None,
    mu_params=None,
    sigma_params=None,
    feature_weights=None,
    sample_args=None,
    mrf_regularization: bool = False,
    mrf_kwargs: dict | None = None,
    second_pass: str = "full",  # "skip" | "short" | "full"
    second_pass_sample_args: dict | None = None,
    random_seed=8927,
):
    """
    Run MCMC-based clustering on velocity data with flexible velocity transformations.

    Parameters:
    -----------
    df_input : pandas.DataFrame
        Input dataframe with 'x', 'y', 'V' columns
    transform_velocity : str, default="none"
        Type of velocity transformation: "power", "exponential", "threshold", "sigmoid", or "none"
    transform_params : dict, optional
        Parameters for velocity transformation (see preprocess_velocity_features for details)
    """

    # --- helper: build initvals from idata posterior means (warm-start) ---
    def _initvals_from_idata(idata_in, n_chains):
        mu_mean = idata_in.posterior["mu"].mean(dim=["chain", "draw"]).values
        sigma_mean = idata_in.posterior["sigma"].mean(dim=["chain", "draw"]).values
        # Ensure shapes match the model dims; return a list of per-chain dicts
        init = {"mu": mu_mean, "sigma": sigma_mean}
        return [init for _ in range(n_chains)]

    logger.info(f"Running MCMC clustering for {base_name}...")

    # Default parameters if not provided
    if mu_params is None:
        mu_params = {"mu": 0, "sigma": 1}
    if sigma_params is None:
        sigma_params = {"sigma": 1}
    if sample_args is None:
        sample_args = dict(
            target_accept=0.95,
            draws=2000,
            tune=1000,
            chains=4,
            cores=4,
            random_seed=random_seed,
        )
    if variables_names is None:
        variables_names = ["V"]

    if "V" not in df_input.columns:
        raise ValueError("Input dataframe must contain 'V' column for velocities.")

    # Preprocess velocity features to enhance high velocities
    velocities, transform_info = preprocess_velocity_features(
        velocities=df_input["V"].to_numpy(),
        velocity_transform=transform_velocity,
        velocity_params=transform_params,
    )

    # Extract data array for clustering
    if len(variables_names) > 1:
        # Concatenate other features to velocities
        additional_vars = variables_names.copy()
        if "V" in additional_vars:
            additional_vars.remove("V")
        additional_data = df_input[additional_vars].to_numpy()
        data_array = np.column_stack((velocities, additional_data))
    else:
        # Use only velocities
        data_array = velocities.reshape(-1, 1)

    # Scale data for model input
    scaler = StandardScaler()
    scaler.fit(data_array)
    joblib.dump(scaler, output_dir / f"{base_name}_scaler.joblib")
    data_array_scaled = scaler.transform(data_array)

    # Build model
    logger.info(f"Running MCMC clustering for {base_name}...")
    model = mcmc.build_marginalized_mixture_model(
        data_array_scaled,
        prior_probs,
        sectors,
        mu_params=mu_params,
        sigma_params=sigma_params,
        feature_weights=feature_weights,
    )

    # Sample model (1st pass)
    idata, convergence_flag = mcmc.sample_model(
        model, output_dir, base_name, **sample_args
    )
    if not convergence_flag:
        idata_summary = az.summary(idata, var_names=["mu", "sigma"])
        logger.info(f"MCMC did not converge. Summary:\n{idata_summary}")

    # --- MRF regularization of priors and optional re-sample ---
    prior_used = prior_probs
    if mrf_regularization:
        x_pos = df_input["x"].to_numpy()
        y_pos = df_input["y"].to_numpy()
        # mrf_kwargs = {"n_neighbors": 24, "length_scale": 200, "beta": 5, "n_iter": 5} # hard-coded values for debug
        prior_mrf, q_mrf = mcmc.mrf_regularization(
            data_array_scaled, idata, prior_probs, x_pos, y_pos, **mrf_kwargs
        )
        prior_used = prior_mrf
        try:
            fig, _ = mcmc.plot_spatial_priors(df_input, prior_mrf, img=img)
            fig.savefig(
                output_dir
                / f"{base_name}_mrf_priors_neig{mrf_kwargs['n_neighbors']}_ls{mrf_kwargs['length_scale']}_beta{mrf_kwargs['beta']}.jpg",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
        except Exception as exc:
            logger.warning(f"Could not plot MRF priors: {exc}")

    # Decide second pass strategy
    if mrf_regularization and second_pass.lower() == "skip":
        # Fastest: don't re-sample. Use q_mrf as final posterior_probs and argmax as labels.
        posterior_probs = q_mrf
        cluster_pred = np.argmax(posterior_probs, axis=1)
        uncertainty = 1.0 - posterior_probs.max(axis=1)
        # keep idata from 1st pass for plots/params
    else:
        # Re-sample with refined priors (short or full)
        if mrf_regularization:
            with model:
                pm.set_data({"prior_w": prior_used})

        # Allow short second pass and warm start
        sp2_args = dict(**sample_args)
        if second_pass.lower() == "short":
            # much fewer draws/tune; fewer chains can also help
            sp2_args.update(dict(draws=600, tune=400, chains=2, cores=2))
            if second_pass_sample_args:
                sp2_args.update(second_pass_sample_args)
        elif second_pass_sample_args:
            sp2_args.update(second_pass_sample_args)

        # Warm-start from previous posterior means
        initvals = _initvals_from_idata(idata, sp2_args.get("chains", 2))

        with model:
            # pass initvals through sample_model if it supports, else call pm.sample directly
            try:
                idata, convergence_flag = mcmc.sample_model(
                    model,
                    output_dir,
                    base_name + ("_mrf" if mrf_regularization else ""),
                    initvals=initvals,
                    **sp2_args,
                )
            except TypeError:
                # fallback if your wrapper doesn't accept initvals
                idata = pm.sample(**sp2_args)
                convergence_flag = True

        # Compute posterior-based assignments
        posterior_probs, cluster_pred, uncertainty = mcmc.compute_posterior_assignments(
            idata, n_posterior_samples=200
        )

    # Generate plots
    fig = mcmc.plot_velocity_clustering(
        df_features=df_input,
        img=img,
        idata=idata,
        cluster_pred=cluster_pred,
        posterior_probs=posterior_probs,
        scaler=scaler,
    )
    fig.savefig(
        output_dir / f"{base_name}_results.jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Trace plots
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    az.plot_trace(
        idata, var_names=["mu", "sigma"], axes=axes, compact=True, legend=True
    )
    fig.savefig(output_dir / f"{base_name}_trace_plots.jpg", dpi=150)
    plt.close(fig)

    # Forest plots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    az.plot_forest(idata, var_names=["mu", "sigma"], combined=True, ess=True, ax=axes)
    fig.savefig(output_dir / f"{base_name}_forest_plot.jpg", dpi=150)
    plt.close(fig)

    # Collect and save metadata
    metadata = mcmc.collect_run_metadata(
        idata=idata,
        convergence_flag=convergence_flag,
        data_array_scaled=data_array_scaled,
        variables_names=variables_names,
        sectors=sectors,
        prior_probs=prior_probs,
        sample_args=sample_args,
        frame=locals(),
    )
    mcmc.save_run_metadata(output_dir, base_name, metadata)

    # Return results dictionary
    result = {
        "metadata": metadata,
        "idata": idata,
        "scaler": scaler,
        "convergence_flag": convergence_flag,
        "posterior_probs": posterior_probs,
        "cluster_pred": cluster_pred,
        "uncertainty": uncertainty,
    }

    plt.close("all")
    return result


def main(reference_date: str | None = None, output_dir: str | Path | None = None):
    """
    Run the pipeline. When called interactively you can pass:
      - reference_date: "YYYY-MM-DD" to override config.data.reference_date
      - overrides: dict of dot.notation keys -> values to override config entries
    """

    # ===  DATA LOADING AND PREPROCESSING  === #
    config = load_config()

    # Retrieve parameters from CLI or config
    if reference_date:
        config.data.reference_date = reference_date

    reference_date = config.data.reference_date
    if not reference_date:
        raise ValueError("reference_date must be specified either via CLI or config.")

    # Automatically update 'year' in config so that interpolated paths (output_dir, priors) match the date
    try:
        current_year = str(datetime.strptime(reference_date, "%Y-%m-%d").year)
        if config.data.year != current_year:
            logger.info(
                f"Updating config.data.year: {config.data.year} -> {current_year}"
            )
            config.data.year = current_year
    except ValueError:
        logger.warning(f"Could not parse year from reference_date: {reference_date}")

    if output_dir:
        config.data.output_dir = str(output_dir)

    # Retrieve other config parameters
    data_config = config.data
    random_seed = config.random_seed
    camera_name = data_config.camera_name
    variables_names = data_config.variables_names

    # Output base directory (output will be saved in a subfolder with camera name and date)
    output_base_dir = Path(data_config.output_dir)
    output_dir = output_base_dir / f"{camera_name}_{reference_date}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define base names for outputs
    base_name = f"{reference_date}"
    mcmc_base_name = f"{reference_date}_mcmc"

    # Save a copy of the used config in the output dir with omegaconfig dump
    config_path = output_dir / f"{base_name}_config.yaml"
    OmegaConf.save(config, config_path)

    # Date range for data selection
    days_before_to_include = data_config.days_before_to_include
    days_after_to_include = data_config.days_after_to_include
    dt_min = data_config.dt_min
    dt_max = data_config.dt_max
    if days_before_to_include > 0 or days_after_to_include > 0:
        reference_start_date = datetime.strptime(
            reference_date, "%Y-%m-%d"
        ) - pd.Timedelta(days=days_before_to_include)
        reference_end_date = datetime.strptime(
            reference_date, "%Y-%m-%d"
        ) + pd.Timedelta(days=days_after_to_include)
    else:
        reference_start_date = None
        reference_end_date = None

    # Fetch DIC ids
    db_engine = create_engine(config.db_url)
    dic_ids = fetch_dic_analysis_ids(
        db_engine,
        camera_name=camera_name,
        reference_date=reference_date,
        reference_date_start=reference_start_date,
        reference_date_end=reference_end_date,
        dt_hours_min=dt_min,
        dt_hours_max=dt_max,
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

    # Read roi for data filtering
    roi = read_polygons_from_cvat(data_config.roi_path, image_name=None)

    # Read sectors for spatial priors
    prior_file_pattern = data_config.sector_prior_file
    sector_prior_files = list(
        Path(prior_file_pattern).parent.glob(Path(prior_file_pattern).name)
    )
    if len(sector_prior_files) == 0:
        raise FileNotFoundError(
            f"No sector prior file found matching: {data_config.sector_prior_file}"
        )
    if len(sector_prior_files) > 1:
        logger.warning(
            f"Multiple sector prior files matched. Using the first one found: {list(sector_prior_files)}"
        )
    sector_prior_file = sector_prior_files[0]
    sectors = read_polygons_from_cvat(sector_prior_file, image_name=None)

    # Fetch DIC data
    out = get_multi_dic_data(dic_ids, stack_results=False, config=config.api)
    logger.info(f"Found stack of {len(out)} DIC dataframes.")

    # Apply filter for each df in the dictionary and then stack them
    preproc_config = config.preprocessing
    processed = []
    for src_id, df_src in out.items():
        try:
            # Filter only points inside the spatial priors sectors
            df_src = filter_dataframe_by_polygons(df_src, polygons=roi)

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

    # --- Gather parameters ---

    # MCMC parameters
    sample_args = {
        "draws": config.mcmc.sample_options.draws,
        "tune": config.mcmc.sample_options.tune,
        "chains": config.mcmc.sample_options.chains,
        "cores": config.mcmc.sample_options.cores,
        "target_accept": config.mcmc.sample_options.target_accept,
        "random_seed": random_seed,
    }
    model_options = config.mcmc.model_options
    mu_params = model_options.mu_params
    sigma_params = model_options.sigma_params

    # Velocity transformation parameters
    velocity_transform = config.mcmc.velocity_transform
    transform_params = config.mcmc.transform_params

    # MRF regularization parameters
    mrf_regularization = config.mcmc.mrf_regularization
    mrf_kwargs = config.mcmc.mrf_kwargs
    second_pass = config.mcmc.second_pass
    second_pass_sample_args = config.mcmc.second_pass_sample_args

    # Multiscale parameters
    multiscale_config = config.multiscale
    sigma_values = multiscale_config.sigma_values
    aggregation_config = multiscale_config.aggregation

    # Post-processing parameters
    postproc_config = config.postprocessing

    # --- Assign spatial priors ---
    prior_config = config.priors
    if not prior_config.probability:
        # Default: uniform priors across sectors
        n_sectors = len(sectors)
        uniform_prob = 1.0 / n_sectors
        prior_config.probability = {
            name: [uniform_prob] * n_sectors for name in sectors
        }
    prior_probs_array = mcmc.assign_spatial_priors(
        x=dic_df["x"].to_numpy(),
        y=dic_df["y"].to_numpy(),
        polygons=sectors,
        prior_probs=prior_config.probability,
        fade_method=prior_config.fade_method,
        fade_options=prior_config.fade_options.get(prior_config.fade_method, {}),
    )

    fig, axes = mcmc.plot_spatial_priors(dic_df, prior_probs_array, img=img)
    fig.savefig(
        output_dir / f"{mcmc_base_name}_spatial_priors.jpg",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # --- Loop through smoothing scales ---
    results = []
    for sigma in sigma_values:
        logger.info(f"Processing with Gaussian smoothing sigma={sigma}...")

        # Create scale-specific base name
        scale_base_name = f"{mcmc_base_name}_sigma{sigma}"

        # Apply Gaussian smoothing if needed (skipped for sigma=0)
        df_run = apply_2d_gaussian_filter(dic_df, sigma=sigma)

        # For larger sigma, tighten priors
        if sigma > 2:
            mu_params = {"mu": 0, "sigma": 0.5}
            sigma_params = {"sigma": 0.5}

        # Run MCMC clustering with the smoothed data
        result = run_mcmc_clustering(
            df_input=df_run,
            prior_probs=prior_probs_array,
            sectors=sectors,
            output_dir=output_dir,
            base_name=scale_base_name,
            img=img,
            variables_names=variables_names,
            sample_args=sample_args,
            transform_velocity=velocity_transform,
            transform_params=transform_params,
            mu_params=mu_params,
            sigma_params=sigma_params,
            random_seed=random_seed,
            mrf_regularization=mrf_regularization,
            mrf_kwargs=mrf_kwargs,
            second_pass=second_pass,
            second_pass_sample_args=second_pass_sample_args,
        )

        # Add scale information to result
        result["sigma"] = sigma

        # Append to results list
        results.append(result)

    # --- Aggregate multi-scale results (if multiscale approach) ---

    # Multiscale parameters (grouped)
    if len(sigma_values) > 1:
        aggregated_results = aggregate_multiscale_clustering(
            results,
            similarity_threshold=aggregation_config.similarity_threshold,
            overall_threshold=aggregation_config.overall_threshold,
            fig_path=output_dir / f"{mcmc_base_name}_similarity_heatmap.jpg",
        )
        cluster_pred = aggregated_results["combined_cluster_pred"]
        posterior_probs = aggregated_results["avg_posterior_probs"]
        entropy = aggregated_results["avg_entropy"]
        similarity_matrix = aggregated_results["similarity_matrix"]
        stability_score = aggregated_results["stability_score"]
        valid_scales = aggregated_results["valid_scales"]

    else:
        # Otherwise extract the single result
        cluster_pred = results[0]["cluster_pred"]
        posterior_probs = results[0]["posterior_probs"]
        entropy = -np.sum(posterior_probs * np.log(posterior_probs + 1e-10), axis=1)
        similarity_matrix = None
        stability_score = None
        valid_scales = None

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
    # TODO: move parameters to config file!!

    # 1. Vectorize & Smooth
    logger.info("Vectorizing grid clusters to polygons...")
    sectors = vectorize_gridded_sectors(kin_cluster_grid, X, Y)
    sectors.to_file(
        output_dir / f"{base_name}_kinematic_sectors_raw.geojson", driver="GeoJSON"
    )
    sectors = clean_vector_sectors(
        sectors,
        dic_df,
        min_area_px2=100000.0,
        isolation_buffer=30.0,
        velocity_merge_threshold=1,
        target_number_of_sectors=4,
        fill_holes_area=80000.0,
        smooth_geometries=True,
        smooth_method="smoothify",
        smooth_iterations=1,
        raster_res=2 * raster_res,
    )

    # Plot raw clusters vs vectorized sectors
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
        sector_colors=postproc_config.sector_assignment.sector_colors,
        label_column="cluster_id",
        add_sector_labels=False,
        title=f"Cleaned vectorized sectors (n={len(sectors)})",
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"{base_name}_clustering_raw_vs_vectorized.jpg", dpi=150)
    plt.close(fig)

    # 2. Assign Labels (A, B, C...)
    # We use the centroid Y position to order sectors from bottom to top (A=lowest Y)
    # The y axis is inverted in image coordinates (0 at top), hence ascending=False
    sectors = assign_sector_labels(
        sectors,
        order_by=postproc_config.sector_assignment.method,
        ascending=postproc_config.sector_assignment.ascending,
    )

    # Drop all but essential columns
    sectors = sectors[["geometry", "sector"]].copy()

    # 3. Classify the original points dataframe and compute statistics
    pts_by_sector = classify_points_by_polygons(sectors, dic_df, x_col="x", y_col="y")

    # Compute sector statistics
    sectors["area_px2"] = sectors.geometry.area
    sectors = compute_sector_stats(
        sectors, pts_by_sector, value_col="V", group_col="sector"
    )
    sectors.to_file(
        output_dir / f"{base_name}_kinematic_sectors_final.geojson", driver="GeoJSON"
    )
    stats = sectors.drop(columns=[sectors.geometry.name], errors="ignore")
    stats.to_csv(
        output_dir / f"{base_name}_kinematic_sector_stats.csv",
        index=False,
        float_format="%.3f",
    )
    logger.info(f"Saved final kinematic sectors GeoJSON with stats to {output_dir}")

    logger.info("Creating summary figure...")
    sector_colors = postproc_config.sector_assignment.sector_colors
    sector_figure_path = plot_sectors_summary(
        sectors=sectors,
        points_by_sector=pts_by_sector,
        img=img,
        colors=sector_colors,
        output_dir=output_dir,
        base_name=base_name,
        unit="px",
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
    try:
        bundle = {
            "reference_date": reference_date,
            "date_start": date_start,
            "date_end": date_end,
            "dic_dataframe": dic_df,
            "posterior_probs": posterior_probs,
            "cluster_pred": cluster_pred,
            "uncertainty": entropy,
            "sectors": sectors,
            "sector_stats": sector_stats,
            "pts_by_sector": pts_by_sector,
        }
        py_path = output_dir / f"{base_name}_results.joblib"
        joblib.dump(bundle, py_path)
    except Exception as exc:
        logger.warning(f"Failed saving joblib bundle: {exc}")

    logger.info("Processing complete.")


if __name__ == "__main__":
    import argparse

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
    args = p.parse_args()

    main(reference_date=args.date, output_dir=args.output_dir)
