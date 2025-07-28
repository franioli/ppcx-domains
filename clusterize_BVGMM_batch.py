import logging
from pathlib import Path

import joblib
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

from src.clustering import (
    plot_gmm_clusters,
    preproc_features,
)
from src.config import ConfigManager
from src.database import (
    apply_dic_filters,
    get_dic_analysis_ids,
    get_dic_data,
    get_image,
)
from src.roi import PolygonROISelector, filter_dataframe
from src.visualization import plot_dic_vectors

# use agg backend for matplotlib to avoid display issues
plt.switch_backend("agg")

# Initialize configuration
config_manager = ConfigManager()
config = config_manager.config

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.processing.log_level.upper()),
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def create_db_engine():
    """Create a database engine for each worker process."""
    return create_engine(config_manager.get_db_url())


def process_single_day(camera_name, target_date):
    """Process a single day for a given camera."""

    db_engine = create_db_engine()
    logger = logging.getLogger(f"worker_{camera_name}_{target_date}")

    try:
        logger.info(f"Processing Camera: {camera_name}, Date: {target_date}")

        base_name = f"{camera_name}_{target_date}_GMM"
        output_dir = Path(config.output.base_dir) / camera_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get DIC analysis metadata (filtered by date/camera)
        dic_analyses = get_dic_analysis_ids(
            db_engine, reference_date=target_date, camera_name=camera_name
        )
        if dic_analyses.empty:
            logger.warning(
                f"No DIC analyses found for {camera_name} on {target_date}. Skipping."
            )
            return {"status": "skipped", "reason": "No DIC analyses found"}

        # Get the master image for the DIC analysis via the API
        master_image_id = dic_analyses["master_image_id"].iloc[0]
        img = get_image(master_image_id, camera_name=camera_name)
        if img is None:
            logger.warning(
                f"Image not found for {camera_name} on {target_date}. Skipping."
            )
            return {"status": "skipped", "reason": "Image not found"}

        # Fetch the displacement data for that DIC analysis via the API
        dic_id = dic_analyses["dic_id"].iloc[0]
        df_raw = get_dic_data(
            dic_id,
            app_host=config.api.host,
            app_port=config.api.port,
        )
        # Apply all DIC filters
        df = apply_dic_filters(
            df_raw,
            filter_outliers=config.dic.filter_outliers,
            tails_percentile=config.dic.tails_percentile,
            min_velocity=config.dic.min_velocity,
            apply_2d_median=config.dic.apply_2d_median,
            median_window_size=config.dic.median_window_size,
            median_threshold_factor=config.dic.median_threshold_factor,
        )

        logger.info(f"Loaded {len(df)} DIC data points")

        # Apply ROI filtering
        selector = PolygonROISelector.from_file(config.data.polygon_roi_file)
        df_original_size = len(df)
        df = filter_dataframe(df, selector.polygon_path, x_col="x", y_col="y")
        logger.info(
            f"Filtered data: {df_original_size} -> {len(df)} points "
            f"({len(df) / df_original_size:.1%} kept)"
        )

        # Plot DIC vectors
        if config.output.save_plots:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_dic_vectors(
                x=df["x"].to_numpy(),
                y=df["y"].to_numpy(),
                u=df["u"].to_numpy(),
                v=df["v"].to_numpy(),
                magnitudes=df["V"].to_numpy(),
                background_image=img,
                cmap_name="viridis",
                fig=fig,
                ax=ax,
            )
            dic_plot_path = output_dir / f"{base_name}_dic.png"
            fig.savefig(dic_plot_path, dpi=config.output.plot_dpi, bbox_inches="tight")
            plt.close(fig)
            logger.debug(f"Saved DIC plot to {dic_plot_path}")

        # --- Run Variational Bayesian Gaussian Mixture clustering ---
        logger.info("Starting GMM clustering")
        df_features = preproc_features(df)
        features = df_features[config.clustering.variables_names].values

        # Apply feature scaling with weights
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Fit GMM
        gmm = BayesianGaussianMixture(
            n_components=config.clustering.n_components,
            weight_concentration_prior=config.clustering.weight_concentration_prior,
            covariance_type=config.clustering.covariance_type,
            max_iter=config.clustering.max_iter,
            random_state=config.clustering.random_state,
        )
        gmm.fit(features_scaled)
        labels = gmm.predict(features_scaled)
        n_clusters_found = len(set(labels))
        logger.info(f"GMM clustering completed: Found {n_clusters_found} clusters")

        # Plot clustering results
        if config.output.save_plots:
            fig, ax, stats_df = plot_gmm_clusters(
                df_features,
                labels,
                var_names=["V", "angle_rad"],
                img=img,
                figsize=(8, 6),
            )
            cluster_plot_path = output_dir / f"{base_name}_clusters.png"
            fig.savefig(
                cluster_plot_path, dpi=config.output.plot_dpi, bbox_inches="tight"
            )
            plt.close(fig)

        # Save the GMM model and results
        scaler_path = output_dir / f"{base_name}_scaler.joblib"
        joblib.dump(scaler, scaler_path)
        gmm_fname = (
            f"{base_name}_model_comp{config.clustering.n_components}_"
            f"cov{config.clustering.covariance_type}.joblib"
        )
        joblib.dump(gmm, output_dir / gmm_fname)
        features_path = output_dir / f"{base_name}_features_with_labels.csv"
        df_features["cluster"] = labels
        df_features.to_csv(features_path, index=False)

        logger.info(
            f"Successfully processed {camera_name} on {target_date}: "
            f"Generated {n_clusters_found} clusters from {len(df)} data points"
        )

        return {
            "status": "success",
            "camera_name": camera_name,
            "target_date": target_date,
            "n_clusters": n_clusters_found,
            "n_points": len(df),
            "n_filtered_points": df_original_size - len(df),
        }

    except Exception as e:
        logger.error(f"Error processing {camera_name} on {target_date}: {str(e)}")
        return {
            "status": "error",
            "camera_name": camera_name,
            "target_date": target_date,
            "error": str(e),
        }
    finally:
        # Clean up database connection
        if "db_engine" in locals():
            db_engine.dispose()


def main():
    """Main function to orchestrate the parallel processing."""

    logging.info(f"Processing cameras: {config.data.camera_names}")
    logging.info(f"DIC filter parameters: {config.dic}")
    logging.info(f"GMM parameters: {config.clustering}")
    logging.info(f"Parallel processing with {config.processing.n_jobs} jobs")

    # Create the connection to the database for fetching dates
    logging.info(
        f"Connecting to database at {config.database.host}:{config.database.port}"
    )
    db_engine = create_db_engine()

    # Get all unique dates for all cameras
    query = """SELECT DIC.reference_date, CAM.camera_name
    FROM ppcx_app_dic DIC
    JOIN ppcx_app_image IM ON DIC.master_image_id = IM.id
    JOIN ppcx_app_camera CAM ON IM.camera_id = CAM.id
    GROUP BY DIC.reference_date, CAM.camera_name
    ORDER BY DIC.reference_date ASC, CAM.camera_name ASC
    """
    available_days_by_cam = pd.read_sql(query, db_engine).groupby("camera_name")

    # Build list of tasks to process
    tasks = []
    for camera_name in config.data.camera_names:
        if camera_name in available_days_by_cam.groups:
            days_unique = available_days_by_cam.get_group(camera_name)[
                "reference_date"
            ].unique()
            logging.info(
                f"Camera {camera_name}: Found {len(days_unique)} analysis dates"
            )
            for target_date in days_unique:
                tasks.append((camera_name, target_date))
    tasks.sort(key=lambda x: (x[0], x[1]))  # Sort tasks by camera and date
    logging.info(f"Total tasks to process: {len(tasks)}")

    # Process tasks in parallel
    results = Parallel(n_jobs=config.processing.n_jobs, verbose=1)(
        delayed(process_single_day)(camera_name, target_date)
        for camera_name, target_date in tasks
    )

    # Summarize results
    successful = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    logging.info(
        f"Batch processing completed: {successful} successful, {skipped} skipped, {errors} errors"
    )
    for result in results:
        if result["status"] == "error":
            logging.error(
                f"Error in {result['camera_name']} on {result['target_date']}: {result['error']}"
            )

    # Clean up main database connection
    db_engine.dispose()


if __name__ == "__main__":
    main()
