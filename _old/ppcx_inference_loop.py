import glob
from pathlib import Path

import arviz as az
import cloudpickle
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from ppcluster import logger
from ppcluster.config import ConfigManager
from ppcluster.griddata import (
    create_2d_grid,
    map_grid_to_points,
)
from ppcluster.mcmc import (
    assign_spatial_priors,
    compute_posterior_assignments,
)
from ppcluster.postproc import (
    close_small_holes,
    compute_cluster_statistics_simple,
    plot_1d_velocity_clustering_simple,
    remove_small_grid_components,
    split_disconnected_components,
)
from ppcluster.preprocessing import (
    apply_dic_filters,
)
from ppcluster.utils.database import (
    fetch_dic_analysis_ids,
    get_dic_analysis_by_ids,
    get_dic_data,
    get_image,
)
from ppcluster.utils.roi import PolygonROISelector

# Use non-interactive backend for matplotlib
plt.switch_backend("Agg")

# Load configuration
config = ConfigManager()
db_engine = create_engine(config.db_url)

# Define a range of dates to process
# dates = [
#     "2024-09-01",
#     "2024-09-02",
#     "2024-09-03",
#     "2024-09-04",
#     "2024-09-05",
#     "2024-09-06",
#     "2024-09-07",
# ]
date_in = "2024-08-23"
date_end = "2024-08-28"
# date_in = "2024-09-01"
# date_end = "2024-09-07"
dates = pd.date_range(start=date_in, end=date_end).strftime("%Y-%m-%d").tolist()

camera_name = "PPCX_Tele"
output_dir = Path("output") / f"{camera_name}_PyMC"

# Load posterior and scaler
PRIOR_STRENGTH = 0.4
reference_start_date = "2024-08-23"
reference_end_date = "2024-08-28"
posterior_base_name = (
    f"PPCX_mcmc_{camera_name}_pooled_{reference_start_date}_{reference_end_date}"
)
idata = az.from_netcdf(output_dir / f"{posterior_base_name}_posterior.idata.nc")
scaler = joblib.load(output_dir / f"{posterior_base_name}_scaler.joblib")

SPLIT_DISCONNECTED_CLUSTERS = False


def process_date(date):
    # Get DIC data
    dic_ids = fetch_dic_analysis_ids(
        db_engine, camera_name=camera_name, reference_date=date
    )
    if len(dic_ids) == 0:
        raise ValueError("No DIC analyses found for the given criteria")
    elif len(dic_ids) > 1:
        logger.warning(
            "Multiple DIC analyses found for the given criteria. Using the first one."
        )
    dic_id = dic_ids[0]

    dic_analyses = get_dic_analysis_by_ids(db_engine=db_engine, dic_ids=[dic_id])
    master_image_id = dic_analyses["master_image_id"].iloc[0]
    img = get_image(master_image_id, camera_name=camera_name, config=config)
    df = get_dic_data(dic_id, config=config)
    df = apply_dic_filters(
        df,
        filter_outliers=config.get("dic.filter_outliers"),
        tails_percentile=config.get("dic.tails_percentile"),
    )

    selector = PolygonROISelector.from_file(config.get("data.roi_path"))
    df = selector.filter_dataframe(df, x_col="x", y_col="y")
    logger.info(f"Data shape after filtering: {df.shape}")

    # Prepare new data
    variables_names = config.get("clustering.variables_names")
    df_features = preproc_features(df)
    X = df_features[variables_names].values
    X_scaled = scaler.transform(X)

    # Assign spatial priors
    sector_files = sorted(glob.glob(config.get("data.sector_prior_pattern")))
    sector_selectors = [PolygonROISelector.from_file(f) for f in sector_files]
    prior_probs = assign_spatial_priors(
        df, sector_selectors, prior_strength=PRIOR_STRENGTH
    )

    # compute assignments for new single-day data (fast option using posterior mean)
    posterior_probs, cluster_pred, entropy = compute_posterior_assignments(
        idata, X_scaled, prior_probs, use_posterior_mean=True
    )

    # Save results to pickle
    basename = f"PPCX_mcmc_{camera_name}_{date}_dic-{dic_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{basename}_clustering_results.pkl", "wb") as f:
        cloudpickle.dump(
            {
                "data": df,
                "df_features": df_features,
                "var_names": variables_names,
                "features_scaled": X_scaled,
                "posterior_probs": posterior_probs,
                "cluster_pred": cluster_pred,
                "entropy": entropy,
                "scaler": scaler,
                "idata": idata,
            },
            f,
        )
        logger.info(
            f"Results saved to {output_dir / f'{basename}_clustering_results.pkl'}"
        )

    # Plot results
    fig, uncertainty, stats = plot_1d_velocity_clustering(
        df_features,
        img,
        idata=idata,
        cluster_pred=cluster_pred,
        posterior_probs=posterior_probs,
        scaler=scaler,
    )
    fig.savefig(
        output_dir
        / f"PPCX_mcmc_{camera_name}_{date}_dic-{dic_id}_clustering_results.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Post-processing of cluster labels
    X, Y, label_grid = create_2d_grid(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        labels=cluster_pred,
        grid_spacing=None,
    )
    logger.info(
        f"Initial grid shape: {label_grid.shape}, NaN cells: {np.isnan(label_grid).sum()}"
    )

    # Remove small cluster components
    remove_small_min_size = [20, 5]
    connectivity = 4
    cleaned_grid = label_grid.copy()
    for min_size in remove_small_min_size:
        cleaned_grid = remove_small_grid_components(
            cleaned_grid, min_size=min_size, connectivity=connectivity
        )
    logger.debug(
        f"After removing small components: shape={cleaned_grid.shape} nan={int(np.isnan(cleaned_grid).sum())} unique_labels={np.unique(cleaned_grid[~np.isnan(cleaned_grid)]) if np.any(~np.isnan(cleaned_grid)) else []}",
    )

    # Close small holes
    cleaned_grid = close_small_holes(
        cleaned_grid,
        max_hole_size=50,
        connectivity=4,
        require_single_neighbor=True,
    )
    nan_after = int(np.isnan(cleaned_grid).sum())
    unique_after = (
        np.unique(cleaned_grid[~np.isnan(cleaned_grid)])
        if np.any(~np.isnan(cleaned_grid))
        else []
    )
    logger.info(
        f"After close_small_holes: NaN cells={nan_after} unique_labels={unique_after}",
    )

    # split disconnected pieces so every connected patch has its own label
    if SPLIT_DISCONNECTED_CLUSTERS:
        cleaned_grid, split_mapping = split_disconnected_components(
            cleaned_grid, connectivity=8, start_label=0
        )

    # Map label grid back to points
    point_labels_cleaned = map_grid_to_points(
        X, Y, cleaned_grid, df, x_col="x", y_col="y", nan_fill=-1
    )
    cluster_pred_cleaned = point_labels_cleaned.astype(int)

    #
    fig, uncertainty, stats = plot_1d_velocity_clustering_simple(
        df_features,
        img,
        cluster_pred=cluster_pred_cleaned,
        posterior_probs=posterior_probs,
    )
    fig.savefig(
        output_dir / f"{basename}_clustering_cleaned.png",
        dpi=300,
        bbox_inches="tight",
    )

    cluster_stats = compute_cluster_statistics_simple(
        df_features,
        cluster_pred=cluster_pred_cleaned,
        posterior_probs=posterior_probs,
    )
    cluster_stats = pd.DataFrame(cluster_stats).T
    cluster_stats.to_csv(output_dir / f"{basename}_clustering_cleaned_stats.csv")

    # Close all figures
    plt.close("all")
    logger.info(f"Results saved to {output_dir}")


def main():
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    for date in dates:
        logger.info(f"Processing date: {date}")

        # process_date(date)
        try:
            process_date(date)
        except Exception as e:
            logger.error(f"Error processing date {date}: {e}")
            continue


if __name__ == "__main__":
    main()
