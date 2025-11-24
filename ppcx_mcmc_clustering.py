# %% # ===  IMPORTS  === #
import argparse
import shutil
from datetime import datetime
from pathlib import Path

import arviz as az
import joblib
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pymc as pm
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

from ppcluster import logger, mcmc
from ppcluster.config import ConfigManager
from ppcluster.cvat import (
    filter_dataframe_by_polygons,
    read_polygons_from_cvat,
)
from ppcluster.griddata import create_2d_grid
from ppcluster.mksectors import (
    SectorPolygons,
    assign_sector_labels,
    classify_points_by_sectors,
    compute_sector_stats,
    draw_polygon,
    remove_polygon_overlaps,
    validate_no_overlaps,
    vectorize_clusters,
)
from ppcluster.postproc import (
    aggregate_multiscale_clustering,
    apply_morphological_operations,
    keep_only_largest_clusters,
    plot_clustering_grid,
    remove_small_grid_components,
    split_disconnected_components,
)
from ppcluster.preprocessing import (
    apply_2d_gaussian_filter,
    apply_dic_filters,
    preprocess_velocity_features,
    spatial_subsample,
)
from ppcluster.utils.database import (
    fetch_dic_analysis_ids,
    get_dic_analysis_by_ids,
    get_image,
    get_multi_dic_data,
)

INTERACTIVE = False  # set to True when running in an interactive environment

if not INTERACTIVE:
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
        mkw = dict(n_neighbors=8, length_scale=50, beta=2.0, n_iter=5)
        if mrf_kwargs:
            mkw.update(mrf_kwargs)
        prior_mrf, q_mrf = mcmc.mrf_regularization(
            data_array_scaled, idata, prior_probs, x_pos, y_pos, **mkw
        )
        prior_used = prior_mrf

        # visualize refined priors
        try:
            fig, _ = mcmc.plot_spatial_priors(df_input, prior_mrf, img=img)
            fig.savefig(
                output_dir / f"{base_name}_mrf_priors.png", dpi=150, bbox_inches="tight"
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
        output_dir / f"{base_name}_results.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Trace plots
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    az.plot_trace(
        idata, var_names=["mu", "sigma"], axes=axes, compact=True, legend=True
    )
    fig.savefig(output_dir / f"{base_name}_trace_plots.png", dpi=150)
    plt.close(fig)

    # Forest plots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    az.plot_forest(idata, var_names=["mu", "sigma"], combined=True, ess=True, ax=axes)
    fig.savefig(output_dir / f"{base_name}_forest_plot.png", dpi=150)
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


def main(reference_date: str | None = None):
    """
    Run the pipeline. When called interactively you can pass:
      - reference_date: "YYYY-MM-DD" to override config.data.reference_date
      - overrides: dict of dot.notation keys -> values to override config entries
    """  # %% # ===  LOAD CONFIGURATION  === #

    # %% # ===  CONFIG MANAGER  === #
    config = ConfigManager()

    # Apply CLI overrides to
    if reference_date:
        config.set("data.reference_date", reference_date)

    # Retrieve reference date
    reference_date = config.get("data.reference_date", None)
    if reference_date is None:
        raise ValueError("reference_date must be specified either via CLI or config.")

    db_engine = create_engine(config.db_url)
    random_seed = config.get("random_seed", 8927)

    # %% # ===  DATA LOADING AND PREPROCESSING  === #
    data_config = config.get("data", {})

    # Output base directory (output will be saved in a subfolder with camera name and date range)
    output_base_dir = Path(data_config.get("output_dir", "output"))

    # Data selection parameters
    camera_name = data_config.get("camera_name", "PPCX_Tele")
    days_before_to_include = data_config.get("days_before_to_include", 0)
    days_after_to_include = data_config.get("days_after_to_include", 0)
    dt_min = data_config.get("dt_min", 72)
    dt_max = data_config.get("dt_max", 96)
    reference_start_date = datetime.strptime(reference_date, "%Y-%m-%d") - pd.Timedelta(
        days=days_before_to_include
    )
    reference_end_date = datetime.strptime(reference_date, "%Y-%m-%d") + pd.Timedelta(
        days=days_after_to_include
    )
    variables_names = data_config.get("variables_names", ["V"])

    # Read roi and spatial priors
    roi_path = Path(data_config.get("roi_path", "data/roi.xml"))
    sector_prior_file = Path(
        data_config.get("sector_prior_file", "data/priors_4_sectors.xml")
    )
    roi = read_polygons_from_cvat(roi_path, image_name=None)
    sectors = read_polygons_from_cvat(sector_prior_file, image_name=None)

    # Check that at least the reference date or an interval of dates is provided
    if not (reference_date or (reference_start_date and reference_end_date)):
        raise ValueError(
            "Either reference_date or both reference_start_date and reference_end_date must be provided."
        )

    # Fetch DIC ids
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

    # Get DIC analysis metadata
    dic_analyses = get_dic_analysis_by_ids(db_engine=db_engine, dic_ids=dic_ids)
    logger.info("Fetched DIC analysis:")
    for _, row in dic_analyses.iterrows():
        print(
            f"DIC ID: {row['dic_id']}, date: {row['reference_date']}, dt (hrs): {row['dt_hours']}, Master: {row['master_timestamp']}, Slave: {row['slave_timestamp']}"
        )
    print("Summary of selected DIC analyses:")
    print(dic_analyses.describe())

    # Output paths
    date_start = dic_analyses.iloc[0]["master_timestamp"].strftime("%Y-%m-%d")
    date_end = dic_analyses.iloc[0]["slave_timestamp"].strftime("%Y-%m-%d")
    output_dir = output_base_dir / f"{camera_name}_{date_end}_mcmc"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define base name for outputs
    base_name = f"{date_end}"

    # Get master image
    master_image_id = dic_analyses["master_image_id"].iloc[0]
    img = get_image(image_id=master_image_id, config=config)

    # Fetch DIC data
    out = get_multi_dic_data(dic_ids, stack_results=False, config=config)
    logger.info(f"Found stack of {len(out)} DIC dataframes.")

    # Apply filter for each df in the dictionary and then stack them
    preprocessing_config = config.get("preprocessing", {})
    subsample_factor = preprocessing_config.get("subsample_factor", 1)
    subsample_method = preprocessing_config.get("subsample_method", "random")
    filter_kwargs = preprocessing_config.get("filter_kwargs", {})

    processed = []
    for src_id, df_src in out.items():
        try:
            # Filter only points inside the spatial priors sectors
            df_src = filter_dataframe_by_polygons(df_src, polygons=roi)

            # Apply other DIC filters if any
            df_src = apply_dic_filters(df_src, **filter_kwargs)

            # Append processed dataframe to the list
            processed.append(df_src)
        except Exception as exc:
            logger.warning("Filtering failed for %s: %s", src_id, exc)
    if not processed:
        raise RuntimeError("No dataframes left after filtering.")

    # Stack all processed dataframes
    df = pd.concat(processed, ignore_index=True)
    logger.info("Data shape after filtering and stacking: %s", df.shape)

    # Apply subsampling
    if subsample_factor > 1:
        df_subsampled = spatial_subsample(
            df, n_subsample=subsample_factor, method=subsample_method
        )
        df = df_subsampled
        logger.info(f"Data shape after subsampling: {df.shape}")

    # %% # ===  SPATIAL PRIORS AND INITIAL VISUALIZATIONS  === #

    # Assign spatial priors
    prior_config = config.get("priors", {})
    prior_probability = prior_config.get("probability", None)
    if not prior_probability:
        # Default: uniform priors across sectors
        n_sectors = len(sectors)
        uniform_prob = 1.0 / n_sectors
        prior_probability = {name: [uniform_prob] * n_sectors for name in sectors}
    fade_method = prior_config.get("fade_method", "constant")
    fade_method_options = prior_config.get("fade_options", {}).get(fade_method, {})
    prior_probs_array = mcmc.assign_spatial_priors(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        polygons=sectors,
        prior_probs=prior_probability,
        fade_method=fade_method,
        fade_options=fade_method_options,
    )

    fig, axes = mcmc.plot_spatial_priors(df, prior_probs_array, img=img)
    fig.savefig(
        output_dir / f"{base_name}_mcmc_spatial_priors.jpg",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Plot velocity field
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_title("Velocity Field", fontsize=14, pad=10)
    ax.imshow(img, alpha=0.5, cmap="gray")
    magnitudes = df["V"].to_numpy()
    vmin = 0.0
    vmax = np.max(magnitudes)
    norm = Normalize(vmin=vmin, vmax=vmax)
    q = ax.quiver(
        df["x"].to_numpy(),
        df["y"].to_numpy(),
        df["u"].to_numpy(),
        df["v"].to_numpy(),
        magnitudes,
        scale=None,
        scale_units="xy",
        angles="xy",
        cmap="viridis",
        norm=norm,
        width=0.008,
        headwidth=2.5,
        alpha=1.0,
    )
    cbar = fig.colorbar(q, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    cbar.set_label("Velocity Magnitude", rotation=270, labelpad=15)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    fig.savefig(
        output_dir / f"{base_name}_velocity_field.jpg",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # %% # ===  RUN MCMC CLUSTERING  === #

    # MCMC parameters
    mcmc_config = config.get("mcmc", {})
    sample_options = mcmc_config.get("sample_options", {})
    sample_args = {
        "draws": sample_options.get("draws", 2000),
        "tune": sample_options.get("tune", 1000),
        "chains": sample_options.get("chains", 4),
        "cores": sample_options.get("cores", 4),
        "target_accept": sample_options.get("target_accept", 0.9),
        "random_seed": random_seed,
    }
    model_options = mcmc_config.get("model_options", {})
    mu_params = model_options.get("mu_params", {"mu": 0, "sigma": 1})
    sigma_params = model_options.get("sigma_params", {"sigma": 1})

    # Velocity transformation parameters
    velocity_transform = mcmc_config.get(
        "velocity_transform", None
    )  # also: "power", "exponential", "sigmoid"
    transform_params = mcmc_config.get(
        "transform_params", {}
    )  # also {"midpoint_percentile": 70, "steepness": 2.0},)

    # MRF regularization parameters
    mrf_regularization = mcmc_config.get("mrf_regularization", True)
    mrf_kwargs = mcmc_config.get("mrf_kwargs", {})
    second_pass = mcmc_config.get("second_pass", "short")
    second_pass_sample_args = mcmc_config.get("second_pass_sample_args", {})

    # Multiscale parameters
    multiscale_config = config.get("multiscale", {})
    sigma_values = multiscale_config.get("sigma_values", [2])

    # Aggregation parameters
    aggregation_config = multiscale_config.get("aggregation", {})
    similarity_threshold = aggregation_config.get("similarity_threshold", 0.7)
    overall_threshold = aggregation_config.get("overall_threshold", 0.8)

    # Loop through smoothing scales
    results = []
    for sigma in sigma_values:
        logger.info(f"Processing with Gaussian smoothing sigma={sigma}...")

        # Create scale-specific base name
        scale_base_name = f"{date_start}_{date_end}_mcmc_sigma{sigma}"

        # Apply Gaussian smoothing if needed (skipped for sigma=0)
        df_run = apply_2d_gaussian_filter(df, sigma=sigma)

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

    # %% == =  AGGREGATE MULTI-SCALE RESULTS  (if multiscale approach)=== #

    # Multiscale parameters (grouped)
    multiscale_config = config.get("multiscale", {})
    aggregation_config = multiscale_config.get("aggregation", {})
    similarity_threshold = aggregation_config.get("similarity_threshold", 0.7)
    overall_threshold = aggregation_config.get("overall_threshold", 0.8)

    if len(sigma_values) > 1:
        aggregated_results = aggregate_multiscale_clustering(
            results,
            similarity_threshold=similarity_threshold,
            overall_threshold=overall_threshold,
            fig_path=output_dir
            / f"{reference_start_date}_{reference_end_date}_similarity_heatmap.jpg",
        )

        # Unpack aggregated results
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

    # ===  Save final clustering results
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
        output_dir / f"{base_name}_kinematic_clustering_results.joblib",
    )

    # %% # ===  POST-PROCESSING AND CLEANING OF FINAL CLUSTERING  === #

    # Retrieve data
    df_smooth = apply_2d_gaussian_filter(df, sigma=1)
    x = df_smooth["x"].to_numpy()
    y = df_smooth["y"].to_numpy()
    v = df_smooth["V"].to_numpy()
    kin_cluster = np.asarray(cluster_pred.copy())

    # Save pre-postprocessing clustering for comparison
    cluster_before_postproc = cluster_pred.copy()
    X_before, Y_before, cluster_grid_before_postproc = create_2d_grid(
        x=x, y=y, labels=cluster_before_postproc
    )

    # Create 2D grid of clustering results
    X, Y, kin_cluster_grid = create_2d_grid(x=x, y=y, labels=kin_cluster)

    # Post-processing parameters
    config.reload()
    postproc_config = config.get("postprocessing", {})
    erosion_iterations = postproc_config.get("erosion_iterations", 0)
    dilation_iterations = postproc_config.get("dilation_iterations", 0)
    min_cluster_size = postproc_config.get("min_cluster_size", 50)
    connectivity = postproc_config.get("connectivity", 4)
    keep_only_largest_n = postproc_config.get("keep_only_largest_n", -1)
    logger.info(
        f"Post-processing params: erosion={erosion_iterations}, "
        f"dilation={dilation_iterations}, min_size={min_cluster_size}"
    )

    # === STEP 1: Grid-level post-processing ===

    # 1.1. Split disconnected components first
    if postproc_config.get("split_disconnected_components", True):
        kin_cluster_grid, split_mapping = split_disconnected_components(
            kin_cluster_grid, connectivity=connectivity, start_label=0
        )

    # 1.4. Remove small components at grid level
    if min_cluster_size > 0:
        kin_cluster_grid = remove_small_grid_components(
            label_grid=kin_cluster_grid,
            min_size=min_cluster_size,
            connectivity=connectivity,
            merge_strategy="merge",  # or "merge" to assign to nearest neighbor
        )

    # 1.2. Apply morphological operations (erosion + dilation)
    if erosion_iterations > 0 or dilation_iterations > 0:
        kin_cluster_grid = apply_morphological_operations(
            cluster_grid=kin_cluster_grid,
            erosion_iterations=erosion_iterations,
            dilation_iterations=dilation_iterations,
            min_cluster_size=min_cluster_size,
            connectivity=connectivity,
        )

    # 1.5. Keep only N largest clusters (on grid)
    if keep_only_largest_n > 0:
        kin_cluster_grid = keep_only_largest_clusters(
            label_grid=kin_cluster_grid,
            n_largest=keep_only_largest_n,
            connectivity=connectivity,
        )

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(12, 6))
    plot_clustering_grid(
        ax=ax_before,
        img=img,
        cluster_grid=cluster_grid_before_postproc,
        X=X_before,
        Y=Y_before,
        title="Before Post-Processing",
        show_legend=True,
        show_stats=True,
        alpha=0.5,
    )
    plot_clustering_grid(
        ax=ax_after,
        img=img,
        cluster_grid=kin_cluster_grid,
        X=X,
        Y=Y,
        title="After Post-Processing",
        show_legend=True,
        show_stats=True,
        alpha=0.5,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{base_name}_kinematic_clustering_postproc.jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # === STEP 2: Vectorize clusters to polygons ===
    logger.info("Vectorizing grid clusters to polygons...")

    # Fastest (convex shapes only)
    polygons_from_clusters = vectorize_clusters(
        kin_cluster_grid, X, Y, method="convex_hull"
    )
    # polygons_from_clusters = vectorize_clusters(
    #     kin_cluster_grid,
    #     X,
    #     Y,
    #     method="cell_union",
    #     buffer_distance=3.0,
    #     simplify_tolerance=2.0,
    # )

    # === STEP 3: Assign sector letters (A, B, C, D...) ===
    logger.info("Assigning sector letters to clusters...")
    assignment_result = assign_sector_labels(
        cluster_grid=kin_cluster_grid,
        Y=Y,
        polygons=polygons_from_clusters,
        order_by="y_position",  # Options: 'y_position', 'area', 'cluster_id'
        reverse_order=True,  # True = bottom-to-top (largest Y first)
        label_prefix="",  # Use '' for A,B,C... or 'S' for S0,S1,S2...
    )
    cluster_to_letter = assignment_result["cluster_to_letter"]
    ordered_clusters_ids = assignment_result["ordered_cluster_ids"]
    sector_polygons = assignment_result["sector_polygons"]

    # Remove overlaps between sector polygons  (A has priority, then B, then C, etc.)
    logger.info("Removing overlaps between sector polygons...")
    ordered_sector_letters = [cluster_to_letter[cid] for cid in ordered_clusters_ids]
    sector_polygons = remove_polygon_overlaps(
        polygons=sector_polygons,
        ordered_labels=ordered_sector_letters,
        buffer_after_difference=1.0,  # Small buffer to smooth jagged edges
    )
    if validate_no_overlaps(sector_polygons, tolerance=1):
        logger.info("✓ All sector polygons are non-overlapping")
    else:
        logger.warning("⚠ Some overlaps may remain (check validation)")

    # Define colors for sectors
    morphokinematic_config = config.get("morphokinematic", {})
    sector_colors = morphokinematic_config.get("sector_colors", None)
    default_cmap = plt.get_cmap("tab10")
    colors = {}
    for idx, letter in enumerate(sorted(cluster_to_letter.values())):
        if sector_colors is not None and letter in sector_colors:
            colors[letter] = sector_colors[letter]
        else:
            colors[letter] = mcolors.to_hex(default_cmap(idx % default_cmap.N))

    # Plot morpho-kinematic sectors
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, alpha=0.5, cmap="gray")
    legend_patches = {}
    for letter in sorted(cluster_to_letter.values()):
        poly = sector_polygons.get(letter)
        if poly is not None:
            draw_polygon(ax, poly, colors[letter], fill_alpha=0.15, zorder=1)
            legend_patches[letter] = mpatches.Patch(
                color=colors[letter], label=f"Sector {letter}", alpha=0.5
            )
    if legend_patches:
        ax.legend(
            handles=list(legend_patches.values()),
            labels=list(legend_patches.keys()),
            loc="upper right",
            fontsize=10,
            framealpha=0.9,
        )
    ax.set_title("Morpho-Kinematic Sectors", fontsize=12)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(
        output_dir / f"{base_name}_morphokinematic_sectors.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Compute sector statistics
    velocity_df = df.copy()
    x = velocity_df["x"].to_numpy()
    y = velocity_df["y"].to_numpy()
    v = velocity_df["V"].to_numpy()
    point_sector_labels = classify_points_by_sectors(
        polygons=sector_polygons,
        x=x,
        y=y,
    )
    assigned_mask = point_sector_labels != ""
    x_assigned = x[assigned_mask]
    y_assigned = y[assigned_mask]
    v_assigned = v[assigned_mask]
    labels_assigned = point_sector_labels[assigned_mask]
    logger.info(
        f"Using {len(x_assigned)} / {len(x)} assigned points for statistics "
        f"({100 * len(x_assigned) / len(x):.1f}%)"
    )
    mk_stats = compute_sector_stats(
        polygons=sector_polygons,
        point_labels=labels_assigned,
        x=x_assigned,
        y=y_assigned,
        v=v_assigned,
    )
    mk_stats.to_csv(
        output_dir / f"{base_name}_morphokinematic_sector_stats.csv",
        index=False,
    )

    logger.info(f"Saved sector statistics: {len(mk_stats)} sectors")

    # %% # ===FINAL  OUTPUTS === #
    def create_summary_figure(
        img: np.ndarray,
        df: pd.DataFrame,
        mk_stats: pd.DataFrame,
        polygons: SectorPolygons,
        colors: dict[str, str],
        base_name: str,
        output_dir: Path,
        *,
        cluster_to_letter: dict[int, str],
        figsize: tuple[int, int] = (18, 7),
        dpi: int = 200,
        min_cbar_percentile: float = 5.0,
        max_cbar_percentile: float = 95.0,
        stat_cols: list[str] | None = None,
        max_labels_in_table: int = 12,
    ) -> Path | None:
        """
        Create summary figure with:
          1) Velocity field + vectors
          2) Sector polygons
          3) Compact statistics table

        Args:
            img: background image
            df: dataframe with columns x,y,u,v,V
            mk_stats: stats dataframe from compute_sector_stats
            polygons: SectorPolygons (letter -> coords)
            colors: map {letter: hex_color}
            cluster_to_letter: cluster id -> letter mapping
            stat_cols: optional column subset for table
        """
        try:
            if stat_cols is None:
                stat_cols = [
                    "label",
                    "n_points",
                    "area_px2",
                    "compactness",
                    "v_mean",
                    "v_std",
                    "v_median",
                ]

            available = [c for c in stat_cols if c in mk_stats.columns]
            if "label" not in available:
                logger.warning("mk_stats has no 'label' column; skipping figure.")
                return None

            summary_fig = plt.figure(figsize=figsize)
            gs = summary_fig.add_gridspec(
                1,
                3,
                width_ratios=[1.05, 1.05, 0.9],
                left=0.02,
                right=0.98,
                top=0.93,
                bottom=0.07,
                wspace=0.15,
            )

            ax_vf = summary_fig.add_subplot(gs[0, 0])
            ax_sectors = summary_fig.add_subplot(gs[0, 1])
            ax_table = summary_fig.add_subplot(gs[0, 2])

            # 1) Velocity field
            ax_vf.imshow(img, alpha=0.55, cmap="gray")
            mags = df["V"].to_numpy()
            vmin = np.percentile(mags, min_cbar_percentile)
            vmax = np.percentile(mags, max_cbar_percentile)
            norm = Normalize(vmin=vmin, vmax=vmax)
            q = ax_vf.quiver(
                df["x"].to_numpy(),
                df["y"].to_numpy(),
                df["u"].to_numpy(),
                df["v"].to_numpy(),
                mags,
                norm=norm,
                scale=None,
                scale_units="xy",
                angles="xy",
                cmap="viridis",
                width=0.006,
                headwidth=2.0,
            )
            ax_vf.set_aspect("equal")
            ax_vf.set_xticks([])
            ax_vf.set_yticks([])
            cbar = plt.colorbar(q, ax=ax_vf, fraction=0.046, pad=0.03)
            cbar.set_label("Velocity [px/day]", rotation=270, labelpad=12, fontsize=8)
            cbar.ax.tick_params(labelsize=7)
            ax_vf.set_title("Velocity Field", fontsize=11)

            # 2) Sector polygons
            ax_sectors.imshow(img, alpha=0.5, cmap="gray")
            legend_patches = {}
            for letter in sorted(colors.keys()):
                coords = polygons.get(letter)
                if coords is None:
                    continue
                draw_polygon(
                    ax_sectors, coords, colors[letter], fill_alpha=0.13, zorder=1
                )
                legend_patches[letter] = mpatches.Patch(
                    color=colors[letter], label=f"{letter}", alpha=0.55
                )
            if legend_patches:
                ax_sectors.legend(
                    handles=list(legend_patches.values()),
                    labels=list(legend_patches.keys()),
                    loc="upper right",
                    fontsize=8,
                    framealpha=0.9,
                )
            ax_sectors.set_aspect("equal")
            ax_sectors.set_xticks([])
            ax_sectors.set_yticks([])
            ax_sectors.set_title("Morpho-Kinematic Sectors", fontsize=11)

            # 3) Statistics table
            ax_table.axis("off")
            display_df = mk_stats[available].copy()
            # formatting
            for c in display_df.columns:
                if c == "label":
                    continue
                if c in {"n_points", "area_px2"}:
                    display_df[c] = display_df[c].round(0).astype(int)
                else:
                    display_df[c] = display_df[c].round(2)

            # limit columns (sector labels) if too many
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
            for (i, j), cell in table.get_celld().items():
                if i == 0 or j == -1:
                    cell.set_facecolor("#E8E8E8")
                    cell.set_text_props(weight="bold", size=7)
                else:
                    cell.set_facecolor("white")
            ax_table.set_title("Sector Statistics", fontsize=11, pad=6)

            summary_fig.suptitle(base_name, fontsize=13, weight="bold", y=0.985)

            out_path = output_dir / f"{base_name}_sectors_summary.png"
            summary_fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close(summary_fig)
            logger.info("Saved summary figure to %s", out_path)
            return out_path

        except Exception as exc:
            logger.warning("Failed summary figure: %s", exc, exc_info=True)
            return None

    def save_sector_results(
        output_dir: Path,
        base_name: str,
        *,
        sector_polygons: SectorPolygons,
        cluster_to_letter: dict[int, str],
        mk_stats: pd.DataFrame,
        posterior_probs: np.ndarray | None = None,
        cluster_pred: np.ndarray | None = None,
        uncertainty: np.ndarray | None = None,
        img_shape: tuple[int, int] | None = None,
        export_geojson: bool = True,
        export_shapefile: bool = False,
    ) -> dict[str, Path]:
        """
        Persist final sector results in reusable formats.

        Saves:
        - Python bundle (.joblib): mappings, arrays, stats (as dict)
        - NumPy bundle (.npz): arrays only
        - GeoJSON (+ optional Shapefile) with Y inverted for GIS (origin bottom-left)

        Args:
            img_shape: (H, W) used to invert Y for GIS export (y_qgis = H-1 - y)

        Returns:
            Dict of artifact name -> Path (existing exports only).
        """
        artifacts: dict[str, Path] = {}
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1) Pythonized bundle
        try:
            bundle = {
                "cluster_to_letter": cluster_to_letter,
                "letter_to_cluster": {v: k for k, v in cluster_to_letter.items()},
                "sector_vertices": {
                    lab: coords for lab, coords in sector_polygons.items()
                },
                "mk_stats": mk_stats.to_dict(orient="list"),
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

        # 3) Geo export (GeoJSON / Shapefile)
        if export_geojson or export_shapefile:
            if not sector_polygons.geometries:
                logger.warning("No geometries to export for GIS.")
                return artifacts
            try:
                import geopandas as gpd
                from shapely.geometry import Polygon

                H = img_shape[0] if (img_shape and len(img_shape) >= 1) else None

                records = []
                for lab, geom in sector_polygons.geometries.items():
                    if geom is None or geom.is_empty:
                        continue
                    # Invert Y for GIS if possible
                    if H is not None:
                        inv_exterior = [
                            (x, H - 1 - y) for x, y in geom.exterior.coords[:-1]
                        ]
                        geom_qgis = Polygon(inv_exterior)
                    else:
                        geom_qgis = geom
                    records.append(
                        {
                            "sector": lab,
                            "n_points": int(
                                mk_stats.loc[mk_stats["label"] == lab, "n_points"].iloc[
                                    0
                                ]
                                if lab in mk_stats["label"].values
                                else 0
                            ),
                            "area_px2": float(geom.area),
                            "compactness": float(
                                (4 * np.pi * geom.area) / (geom.length**2 + 1e-12)
                            ),
                            "geometry": geom_qgis,
                        }
                    )

                gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=None)

                if export_geojson:
                    geojson_path = output_dir / f"{base_name}_sectors.geojson"
                    gdf.to_file(geojson_path, driver="GeoJSON")
                    artifacts["geojson"] = geojson_path

                if export_shapefile:
                    shp_dir = output_dir / f"{base_name}_shp"
                    shp_dir.mkdir(exist_ok=True)
                    shp_path = shp_dir / f"{base_name}_sectors.shp"
                    gdf.to_file(shp_path)
                    artifacts["shapefile"] = shp_path

            except Exception as exc:
                logger.warning(f"Failed GIS export: {exc}")

        return artifacts

    logger.info("Creating summary figure...")
    summary_path = create_summary_figure(
        img=img,
        df=velocity_df,
        mk_stats=mk_stats,
        polygons=sector_polygons,
        colors=colors,
        cluster_to_letter=cluster_to_letter,
        base_name=base_name,
        output_dir=output_dir,
        figsize=(18, 7),
        dpi=200,
    )
    if summary_path is not None:
        quick_dir = output_base_dir / "summary"
        quick_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(summary_path, quick_dir / f"{base_name}_sectors_summary.png")
        logger.info("Copied summary figure to %s", quick_dir)

    # After mk_stats creation (right after saving CSV)
    artifacts = save_sector_results(
        output_dir=output_dir,
        base_name=base_name,
        sector_polygons=sector_polygons,
        cluster_to_letter=cluster_to_letter,
        mk_stats=mk_stats,
        posterior_probs=posterior_probs,
        cluster_pred=cluster_pred,
        uncertainty=entropy,  # or use 'uncertainty' if available
        img_shape=img.shape if img is not None else None,
        export_geojson=True,
        export_shapefile=False,
    )
    for k, pth in artifacts.items():
        logger.info(f"Saved {k}: {pth}")


# %%
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Run MCMC clustering with optional overrides."
    )
    p.add_argument(
        "--reference-date", "-d", help="Override data.reference_date (YYYY-MM-DD)."
    )
    args = p.parse_args()

    main(reference_date=args.reference_date)
