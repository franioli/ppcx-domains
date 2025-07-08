import logging
import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

from src.clustering import (
    plot_gmm_clusters,
    preproc_features,
)
from src.database import get_dic_analysis_ids, get_dic_data, get_image
from src.roi import PolygonROISelector, filter_dataframe
from src.visualization import plot_dic_vectors

# use agg backend for matplotlib to avoid display issues
plt.switch_backend("agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Load environment variables from .env file
load_dotenv()
DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
APP_HOST = os.environ.get("APP_HOST")
APP_PORT = os.environ.get("APP_PORT")
GET_IMAGE_VIEW = os.environ.get("GET_IMAGE_VIEW")

# Parameters for DIC data processing
# camera_names = ["PPCX_Tele", "PPCX_Wide"]
camera_names = ["PPCX_Tele"]  # Use only one camera for testing
min_velocity = 1  # Minimum velocity threshold in pixels, use -1 to disable
filter_outliers = True  # Whether to filter out low velocity vectors
tails_percentile = 0.001  # Percentile for tail filtering

# Parameters for GMM clustering
variables_names = ["x", "y", "V", "angle_rad"]
n_components = 6
max_iter = 100
random_state = 42
covariance_type = "full"

# Output directory for results
base_output_dir = "output"

# Parallel processing parameters
n_jobs = 4  # Number of parallel jobs, adjust based on your system


def create_db_engine():
    """Create a database engine for each worker process."""
    return create_engine(
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )


def process_single_day(
    camera_name,
    target_date,
    base_output_dir,
    min_velocity,
    filter_outliers,
    tails_percentile,
    variables_names,
    n_components,
    max_iter,
    random_state,
    covariance_type,
):
    """Process a single day for a given camera."""

    # Create database engine for this worker
    db_engine = create_db_engine()

    # Set up logging for this worker
    logger = logging.getLogger(f"worker_{camera_name}_{target_date}")

    try:
        logger.info(f"Processing Camera: {camera_name}, Date: {target_date}")

        # Build the output folder and base_name
        base_name = f"{camera_name}_{target_date}_GMM"
        output_dir = Path(base_output_dir) / camera_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get DIC analysis metadata (filtered by date/camera)
        logger.debug(f"Fetching DIC analysis IDs for {camera_name} on {target_date}")

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
        logger.debug(f"Fetching image ID {master_image_id} for {camera_name}")
        if img is None:
            logger.warning(
                f"Image not found for {camera_name} on {target_date}. Skipping."
            )
            return {"status": "skipped", "reason": "Image not found"}

        # Fetch the displacement data for that DIC analysis via the API
        dic_id = dic_analyses["dic_id"].iloc[0]
        logger.debug(f"Fetching DIC data for analysis ID {dic_id}")
        df = get_dic_data(
            dic_id,
            filter_outliers=filter_outliers,
            tails_percentile=tails_percentile,
            min_velocity=min_velocity,
        )
        logger.info(f"Loaded {len(df)} DIC data points")

        # Load the selector from a saved polygon
        selector = PolygonROISelector.from_file(
            "data/PPCX_Tele_glacier_ROI.json",
        )
        df_original_size = len(df)
        df = filter_dataframe(
            df,
            selector.polygon_path,
            x_col="x",
            y_col="y",
        )
        logger.info(
            f"Filtered data: {df_original_size} -> {len(df)} points ({len(df) / df_original_size:.1%} kept)"
        )

        # Plot DIC vectors
        logger.debug("Plotting DIC vectors")
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
        fig.savefig(dic_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"Saved DIC plot to {dic_plot_path}")

        # --- Run Variational Bayesian Gaussian Mixture clustering and plot ---
        logger.info("Starting GMM clustering")
        df_features = preproc_features(df)
        features = df_features[variables_names].values
        logger.debug(f"Features shape: {features.shape}")

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        logger.debug("Features scaled with StandardScaler")

        gmm = BayesianGaussianMixture(
            n_components=10,
            weight_concentration_prior=1e-3,
            covariance_type=covariance_type,
            max_iter=max_iter,
            random_state=random_state,
        )
        logger.debug(f"Fitting GMM with {10} components")
        gmm.fit(features_scaled)
        labels = gmm.predict(features_scaled)
        n_clusters_found = len(set(labels))
        logger.info(f"GMM clustering completed: Found {n_clusters_found} clusters")

        fig, ax, stats_df = plot_gmm_clusters(
            df_features,
            labels,
            var_names=["V", "angle_rad"],
            img=img,
            figsize=(8, 6),
        )
        cluster_plot_path = output_dir / f"{base_name}_clusters.png"
        fig.savefig(cluster_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"Saved cluster plot to {cluster_plot_path}")

        # Save the GMM model
        scaler_path = output_dir / f"{base_name}_scaler.joblib"
        joblib.dump(scaler, scaler_path)
        logger.debug(f"Saved scaler to {scaler_path}")

        gmm_fname = f"{base_name}_model_comp{n_components}_cov{covariance_type}.joblib"
        gmm_path = output_dir / gmm_fname
        joblib.dump(gmm, gmm_path)
        logger.debug(f"Saved GMM model to {gmm_path}")

        # Save the features DataFrame with labels
        features_path = output_dir / f"{base_name}_features_with_labels.csv"
        df_features.to_csv(features_path, index=False)
        logger.debug(f"Saved features with labels to {features_path}")
        
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
            "n_filtered_points": len(df) - df_original_size,
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

    logging.info(f"Processing cameras: {camera_names}")
    logging.info(
        f"DIC filter parameters: min_velocity={min_velocity}, filter_outliers={filter_outliers}, tails_percentile={tails_percentile}"
    )
    logging.info(
        f"GMM parameters: n_components={n_components}, covariance_type={covariance_type}"
    )
    logging.info(f"Parallel processing with {n_jobs} jobs")

    # Create the connection to the database for fetching dates
    logging.info(f"Connecting to database at {DB_HOST}:{DB_PORT}/{DB_NAME}")
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
    logging.info(f"Found {len(available_days_by_cam)} cameras with available DIC data.")

    # Build list of tasks to process
    tasks = []
    for camera_name in camera_names:
        days_unique = available_days_by_cam.get_group(camera_name)[
            "reference_date"
        ].unique()
        logging.info(f"Camera {camera_name}: Found {len(days_unique)} analysis dates")

        for target_date in days_unique:
            tasks.append((camera_name, target_date))

    # Sort tasks by camera and date
    tasks.sort(key=lambda x: (x[0], x[1]))
    logging.info(f"Total tasks to process: {len(tasks)}")

    # Process tasks in parallel
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(process_single_day)(
            camera_name,
            target_date,
            base_output_dir,
            min_velocity,
            filter_outliers,
            tails_percentile,
            variables_names,
            n_components,
            max_iter,
            random_state,
            covariance_type,
        )
        for camera_name, target_date in tasks
    )

    # Summarize results and Log any errors
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
