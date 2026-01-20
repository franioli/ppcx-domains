import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path

import arviz as az
import geopandas as gpd
import joblib
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pymc as pm
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from shapely.geometry import Polygon
from sklearn.preprocessing import StandardScaler
from smoothify import smoothify
from sqlalchemy import create_engine

from ppcluster import load_config, mcmc, setup_logger
from ppcluster.cvat import (
    filter_dataframe_by_polygons,
    read_polygons_from_cvat,
)
from ppcluster.griddata import create_2d_grid
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
from ppcluster.sectors import (
    assign_sector_labels,
    classify_points_by_polygons,
    clean_morphokinematic_sectors,
    compute_sector_stats,
    vectorize_grid_to_gdf,
)
from ppcluster.utils.database import (
    fetch_dic_analysis_ids,
    get_dic_analysis_by_ids,
    get_image,
    get_multi_dic_data,
)

logger = setup_logger(level=logging.INFO, name="ppcx", force=True)

HEADLESS = True  # set to True when running in non-GUI environment

if HEADLESS:
    plt.switch_backend("Agg")
else:
    plt.switch_backend("QtAgg")


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
                output_dir / f"{base_name}_mrf_priors.jpg", dpi=150, bbox_inches="tight"
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


def apply_cluster_grid_cleaning(
    X: np.ndarray,
    Y: np.ndarray,
    clusters: np.ndarray,
    config: dict,
    output_dir: Path,
    base_name: str,
    img: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply morphological operations and cleaning to the cluster grid.
    Refines clusters, removing noise and small components.

    Args:
        X, Y: 2D meshgrid arrays of point coordinates
        clusters: 2D array of cluster labels on the grid
        config: post-processing configuration parameters
        output_dir: directory to save outputs
        base_name: base name for output files
        img: optional background image for plotting

    Returns:
        Refined cluster grid as 2D numpy array.
    """

    # Retrieve post-processing parameters
    do_split = config.get("split_disconnected_components", True)
    erosion_iters = config.get("erosion_iterations", 0)
    dilation_iters = config.get("dilation_iterations", 0)
    connectivity = config.get("connectivity", 8)
    min_cluster_size = config.get("min_cluster_size", 0)
    keep_only_largest_n = config.get("keep_only_largest_n", 0)

    logger.info(
        f"Post-proc params: erosion={erosion_iters}, "
        f"dilation={dilation_iters}, min_size={min_cluster_size}"
    )

    # Store pre-postprocessing grid for comparison
    cluster_before = clusters.copy()

    # Split disconnected components first
    if do_split:
        clusters, _ = split_disconnected_components(
            clusters,
            connectivity=connectivity,
            start_label=0,
        )

    # Remove very small components and merge to nearest neighbor
    clusters = remove_small_grid_components(
        label_grid=clusters,
        min_size=20,  # initial removal threshold to clean noise (hard-coded)
        connectivity=connectivity,
        merge_strategy="merge",  # merge small components to nearest neighbor
    )

    # Apply morphological operations (erosion + dilation)
    if erosion_iters > 0 or dilation_iters > 0:
        clusters = apply_morphological_operations(
            cluster_grid=clusters,
            erosion_iterations=erosion_iters,
            dilation_iterations=dilation_iters,
            min_cluster_size=min_cluster_size,
            connectivity=connectivity,
        )
    # Remove small components again after morph operations (do not merge)
    if min_cluster_size > 0:
        clusters = remove_small_grid_components(
            label_grid=clusters,
            min_size=min_cluster_size,
            connectivity=connectivity,
            merge_strategy="remove",  # or "merge" to assign to nearest neighbor
        )

    # Keep only N largest clusters (on grid)
    if keep_only_largest_n > 0:
        clusters = keep_only_largest_clusters(
            label_grid=clusters,
            n_largest=keep_only_largest_n,
            connectivity=connectivity,
        )

    # Plot comparison before/after post-processing
    if img is not None:
        fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(12, 6))
        plot_clustering_grid(
            ax=ax_before,
            img=img,
            cluster_grid=cluster_before,
            X=X,
            Y=Y,
            title="Before Post-Processing",
            show_legend=True,
            show_stats=True,
            alpha=0.5,
        )
        plot_clustering_grid(
            ax=ax_after,
            img=img,
            cluster_grid=clusters,
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

    # Save cleaned grid result to file for inspection
    dump_data = {"X": X, "Y": Y, "kin_cluster_grid": clusters}
    joblib.dump(
        dump_data, output_dir / f"{base_name}_kinematic_clustering_cleaned.joblib"
    )

    return clusters


def plot_kinematic_sectors(
    velocity_df: pd.DataFrame,
    sector_gdf: gpd.GeoDataFrame,
    sector_stats: pd.DataFrame,
    img: np.ndarray,
    sector_colors: dict | None,
    output_dir: Path,
    base_name: str,
    figsize: tuple = (18, 7),
    dpi: int = 300,
) -> Path:
    """
    Plot morpho-kinematic sectors summary with velocity field and statistics table.
    """

    from matplotlib import colors as mcolors

    # Configuration
    label_column = "label"
    min_cbar_percentile = 5.0
    max_cbar_percentile = 95.0
    stat_cols = [
        "label",
        "v_mean",
        "v_std",
        "v_median",
        "v_mad",
        "n_points",
        "area_px2",
        "compactness",
    ]
    max_labels_in_table = 12

    if sector_colors is None:
        sector_colors = {}

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.3, 1.0], "wspace": 0.25},
    )
    ax_sectors, ax_table = axes

    # 1) Velocity field with sectors overlay
    mags = velocity_df["V"].to_numpy()
    vmin = np.percentile(mags, min_cbar_percentile)
    vmax = np.percentile(mags, max_cbar_percentile)

    ax_sectors.imshow(img, cmap="gray")
    norm = Normalize(vmin=vmin, vmax=vmax)
    q = ax_sectors.quiver(
        velocity_df["x"].to_numpy(),
        velocity_df["y"].to_numpy(),
        velocity_df["u"].to_numpy(),
        velocity_df["v"].to_numpy(),
        mags,
        norm=norm,
        scale=None,
        scale_units="xy",
        angles="xy",
        cmap="viridis",
        width=0.006,
        headwidth=2.0,
    )
    ax_sectors.set_aspect("equal")
    ax_sectors.set_xticks([])
    ax_sectors.set_yticks([])
    cbar = plt.colorbar(q, ax=ax_sectors, fraction=0.046, pad=0.03)
    cbar.set_label("Velocity [px/day]", rotation=270, labelpad=12, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax_sectors.set_title("Velocity Field", fontsize=11)

    # Sectors colors mapping
    present_labels = sorted(sector_gdf[label_column].unique())
    colors = {}
    fallback_cmap = plt.get_cmap("tab10")
    for i, label in enumerate(present_labels):
        if label in sector_colors:
            colors[label] = sector_colors[label]
        else:
            # Fallback color based on index
            colors[label] = mcolors.to_hex(fallback_cmap(i % 10))

    # Plot sectors
    plot_gdf = sector_gdf.copy()
    plot_gdf["color"] = plot_gdf["label"].map(colors)

    if not plot_gdf.empty:
        # 1. Plot the fill with high transparency to show velocity vectors
        plot_gdf.plot(
            ax=ax_sectors,
            color=plot_gdf["color"],
            alpha=0.1,
            linewidth=0,
        )
        # 2. Plot the edges with full opacity to define limits
        plot_gdf.plot(
            ax=ax_sectors,
            facecolor="none",
            edgecolor=plot_gdf["color"],
            linewidth=2.5,
            alpha=1.0,
        )

    # Add legend manually to match style
    legend_patches = [
        mpatches.Patch(color=colors[label], label=label, alpha=0.8)
        for label in present_labels
        if label in colors
    ]
    if legend_patches:
        ax_sectors.legend(
            handles=legend_patches,
            loc="upper right",
            fontsize=8,
            framealpha=0.9,
        )

    # Add letters to sectors
    for _, row in plot_gdf.iterrows():
        cent = row.geometry.centroid
        ax_sectors.text(
            cent.x,
            cent.y,
            row["label"],
            fontsize=12,
            weight="bold",
            color="white",
            ha="center",
            va="center",
        )

    ax_sectors.set_aspect("equal")
    ax_sectors.set_xticks([])
    ax_sectors.set_yticks([])
    ax_sectors.set_title("Morpho-Kinematic Sectors", fontsize=11)

    # 3) Statistics table
    available = [c for c in stat_cols if c in sector_stats.columns]
    if "label" not in available:
        logger.warning("sector_stats has no 'label' column; skipping table.")
        display_df = pd.DataFrame()
    else:
        display_df = sector_stats[available].copy()

    ax_table.axis("off")

    if not display_df.empty:
        # Formatting
        for c in display_df.columns:
            if c == "label":
                continue
            if c in {"n_points", "area_px2"}:
                display_df[c] = display_df[c].round(0).astype(int)
            else:
                display_df[c] = display_df[c].round(2)

        # Limit rows
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

        # Style table headers
        for (i, j), cell in table.get_celld().items():
            if i == 0 or j == -1:
                cell.set_facecolor("#E8E8E8")
                cell.set_text_props(weight="bold", size=7)
            else:
                cell.set_facecolor("white")

    ax_table.set_title("Sector Statistics", fontsize=11, pad=6)

    fig.suptitle(base_name, fontsize=13, weight="bold", y=0.985)

    out_path = output_dir / f"{base_name}_sectors_summary.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved summary figure to {out_path}")
    return out_path


def save_sector_results(
    output_dir: Path,
    base_name: str,
    *,
    gdf_sectors: gpd.GeoDataFrame,
    mk_stats: pd.DataFrame,
    posterior_probs: np.ndarray | None = None,
    cluster_pred: np.ndarray | None = None,
    uncertainty: np.ndarray | None = None,
    img_shape: tuple[int, int] | None = None,
    export_geojson: bool = True,
    export_shapefile: bool = False,
) -> dict[str, Path]:
    """
    Persist final sector results.

    Saves:
    - Bundle (.joblib): python dict with dataframes
    - Arrays (.npz): raw numpy arrays
    - GIS (.geojson/.shp): Merged geometry + stats, optionally with Y-inversion.
    """
    artifacts: dict[str, Path] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Pythonized bundle (Simplified: just dict of DF/GDF/Arrays)
    try:
        # Convert GDF to list-dict for plain pickling if needed, or just keep GDF
        bundle = {
            "gdf_sectors": gdf_sectors,
            "mk_stats": mk_stats,
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

    # 3) GIS export
    if export_geojson or export_shapefile:
        if gdf_sectors.empty:
            logger.warning("No geometries to export for GIS.")
            return artifacts

        try:
            # Merge stats into GDF for a rich export file
            export_gdf = gdf_sectors.merge(mk_stats, on="label", how="left")

            # Handle Y-Inversion for GIS compatibility (Image Origin vs Map Origin)
            H = img_shape[0] if (img_shape and len(img_shape) >= 1) else None

            if H is not None:
                # Invert Y coordinate: y_new = H - 1 - y_old
                def invert_y(geom):
                    if geom.is_empty:
                        return geom
                    # shapely scalar transform
                    return scale(
                        translate(geom, yoff=-(H - 1)), yfact=-1, origin=(0, 0)
                    )
                    # Or manual rebuild to be safe without imports:
                    # return Polygon([(x, H - 1 - y) for x, y in geom.exterior.coords])

                # Manual reliable inversion loop for polygons
                new_geoms = []
                for geom in export_gdf.geometry:
                    if geom.geom_type == "Polygon":
                        new_g = Polygon(
                            [(x, H - 1 - y) for x, y in geom.exterior.coords]
                        )
                        new_geoms.append(new_g)
                    elif geom.geom_type == "MultiPolygon":
                        # Handle multi-polygons if present
                        parts = [
                            Polygon([(x, H - 1 - y) for x, y in p.exterior.coords])
                            for p in geom.geoms
                        ]
                        from shapely.ops import unary_union

                        new_geoms.append(unary_union(parts))
                    else:
                        new_geoms.append(geom)  # Fallback

                export_gdf.geometry = new_geoms

            if export_geojson:
                geojson_path = output_dir / f"{base_name}_sectors.geojson"
                export_gdf.to_file(geojson_path, driver="GeoJSON")
                artifacts["geojson"] = geojson_path

            if export_shapefile:
                shp_dir = output_dir / f"{base_name}_shp"
                shp_dir.mkdir(exist_ok=True)
                shp_path = shp_dir / f"{base_name}_sectors.shp"
                export_gdf.to_file(shp_path)
                artifacts["shapefile"] = shp_path

        except Exception as exc:
            logger.warning(f"Failed GIS export: {exc}")

    return artifacts


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

    if output_dir:
        config.data.output_dir = str(output_dir)

    # Retrieve other config parameters
    data_config = config.data
    random_seed = config.random_seed
    camera_name = data_config.camera_name
    variables_names = data_config.variables_names

    # Output base directory (output will be saved in a subfolder with camera name and date)
    output_base_dir = Path(data_config.output_dir)
    output_dir = output_base_dir / f"{camera_name}_{reference_date}_mcmc"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define base name for outputs
    base_name = f"{reference_date}"

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

    # Get DIC analysis metadata
    dic_analyses = get_dic_analysis_by_ids(db_engine=db_engine, dic_ids=dic_ids)
    logger.info("Fetched DIC analysis:")
    for _, row in dic_analyses.iterrows():
        logger.info(
            f"DIC ID: {row['dic_id']}, date: {row['reference_date']}, dt (hrs): {row['dt_hours']}, Master: {row['master_timestamp']}, Slave: {row['slave_timestamp']}"
        )
    date_start = dic_analyses.iloc[0]["master_timestamp"].strftime("%Y-%m-%d")
    date_end = dic_analyses.iloc[0]["slave_timestamp"].strftime("%Y-%m-%d")
    logger.info("Selected DIC analyses:")
    logger.info(dic_analyses.head())

    # Get master image
    master_image_id = dic_analyses["master_image_id"].iloc[0]
    img = get_image(image_id=master_image_id, config=config.api)

    # Read roi and spatial priors
    roi = read_polygons_from_cvat(data_config.roi_path, image_name=None)
    sectors = read_polygons_from_cvat(data_config.sector_prior_file, image_name=None)

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

    # print sample of data
    logger.info("Sample of preprocessed DIC data:")
    logger.info(dic_df.head())

    # ===  SPATIAL PRIORS AND INITIAL VISUALIZATIONS  === #

    # Assign spatial priors
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
        output_dir / f"{base_name}_mcmc_spatial_priors.jpg",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ===  RUN MCMC CLUSTERING  === #

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

    # Loop through smoothing scales
    results = []
    for sigma in sigma_values:
        logger.info(f"Processing with Gaussian smoothing sigma={sigma}...")

        # Create scale-specific base name
        scale_base_name = f"{date_start}_{date_end}_mcmc_sigma{sigma}"

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

    # ===  AGGREGATE MULTI-SCALE RESULTS  (if multiscale approach)=== #

    # Multiscale parameters (grouped)
    if len(sigma_values) > 1:
        aggregated_results = aggregate_multiscale_clustering(
            results,
            similarity_threshold=aggregation_config.similarity_threshold,
            overall_threshold=aggregation_config.overall_threshold,
            fig_path=output_dir
            / f"{reference_start_date}_{reference_end_date}_similarity_heatmap.jpg",
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
        output_dir / f"{base_name}_kinematic_clustering_results_raw.joblib",
    )

    # ===  POST-PROCESSING AND CLEANING OF FINAL CLUSTERING  === #

    # Retrieve data
    x = dic_df["x"].to_numpy()
    y = dic_df["y"].to_numpy()
    kin_cluster = np.asarray(cluster_pred.copy())

    # Create 2D grid of clustering results
    X, Y, kin_cluster_grid = create_2d_grid(x=x, y=y, labels=kin_cluster)
    raster_res = abs(float(X[0, 1] - X[0, 0]) if X.shape[1] > 1 else 1.0)

    # Apply morphological cleaning to the cluster grid # NOTE: Replaced by better vector-based cleaning below
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

    def _plot_sct(gdf, path):
        fig, ax = plt.subplots()
        gdf.plot(
            ax=ax, column="cluster_id", legend=True, edgecolor="black", cmap="tab10"
        )
        ax.set_title(f"Sectors (n={len(gdf)})")
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.axis("off")
        fig.savefig(path)
        plt.close(fig)

    sectors = vectorize_grid_to_gdf(kin_cluster_grid, X, Y)
    sectors_cleaned = clean_morphokinematic_sectors(
        sectors,
        dic_df,
        min_area_px2=100000.0,
        isolation_buffer=30.0,
        velocity_merge_threshold=1.6,
        target_number_of_sectors=4,
        force_minimum_sectors=True,
    )

    # Classify the original points dataframe
    sectors_cleaned = smoothify(
        sectors_cleaned,
        segment_length=raster_res,
        smooth_iterations=4,
        merge_collection=True,
        merge_field="cluster_id",
        num_cores=4,
        area_tolerance=0.5,
    )

    # 2. Assign Labels (A, B, C...)
    # We use the centroid Y position to order sectors from bottom to top (A=lowest Y)
    # The y axis is inverted in image coordinates (0 at top), hence ascending=False
    sectors_cleaned = assign_sector_labels(
        sectors_cleaned,
        order_by=postproc_config.sector_assignment.method,
        ascending=postproc_config.sector_assignment.ascending,
    )

    # === Compute sector statistics ===
    # Classify the original points dataframe
    pts_by_sector = classify_points_by_polygons(
        sectors_cleaned, dic_df, x_col="x", y_col="y"
    )

    # rename 'label' to 'sector'
    if "label" in pts_by_sector.columns and "sector" not in pts_by_sector.columns:
        pts_by_sector = pts_by_sector.rename(columns={"label": "sector"})

    # Compute table
    mk_stats = compute_sector_stats(sectors_cleaned, pts_by_sector, value_col="V")
    mk_stats.to_csv(output_dir / f"{base_name}_kinematic_sector_stats.csv", index=False)

    logger.info(f"Saved sector statistics: {len(mk_stats)} sectors")

    # --- Plot Kinematic Sectors ---
    logger.info("Creating summary figure...")
    sector_colors = postproc_config.sector_assignment.sector_colors
    sector_figure_path = plot_kinematic_sectors(
        velocity_df=pts_by_sector,
        sector_gdf=sectors_cleaned,
        sector_stats=mk_stats,
        img=img,
        sector_colors=sector_colors,
        output_dir=output_dir,
        base_name=base_name,
    )
    if sector_figure_path is not None:
        sector_figures_dir = output_base_dir / "kinematic_sectors"
        sector_figures_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(sector_figure_path, sector_figures_dir / f"{base_name}_sectors.png")
        logger.info(f"Copied summary figure to {sector_figures_dir}")

    # Save artifacts
    artifacts = save_sector_results(
        output_dir=output_dir,
        base_name=base_name,
        gdf_sectors=sectors_cleaned,
        mk_stats=mk_stats,
        posterior_probs=posterior_probs,
        cluster_pred=cluster_pred,
        uncertainty=entropy,
        img_shape=img.size if img is not None else None,
        export_geojson=True,
        export_shapefile=False,
    )
    for k, pth in artifacts.items():
        logger.info(f"Saved {k}: {pth}")


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
