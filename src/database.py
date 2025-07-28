import io
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from src.config import ConfigManager

logger = logging.getLogger(__name__)

config = ConfigManager()


def get_dic_analysis_ids(
    db_engine,
    *,
    reference_date: str | datetime | None = None,
    master_timestamp: str | datetime | None = None,
    camera_id: int | None = None,
    camera_name: str | None = None,
) -> pd.DataFrame:
    """
    Get DIC analysis metadata (including IDs) for a specific date/timestamp and optional camera filter.
    """
    query = """
    SELECT 
        DIC.id as dic_id,
        CAM.camera_name,
        DIC.master_timestamp,
        DIC.slave_timestamp,
        DIC.master_image_id,
        DIC.slave_image_id,
        DIC.time_difference_hours
    FROM ppcx_app_dic DIC
    JOIN ppcx_app_image IMG ON DIC.master_image_id = IMG.id
    JOIN ppcx_app_camera CAM ON IMG.camera_id = CAM.id
    WHERE 1=1
    """
    params = []
    if reference_date is not None:
        query += " AND DATE(DIC.reference_date) = %s"
        params.append(str(reference_date))
    if master_timestamp is not None:
        query += " AND DATE(DIC.master_timestamp) = %s"
        params.append(str(master_timestamp))
    if camera_id is not None:
        query += " AND CAM.id = %s"
        params.append(camera_id)
    if camera_name is not None:
        query += " AND CAM.camera_name = %s"
        params.append(camera_name)
    query += " ORDER BY DIC.master_timestamp"
    return pd.read_sql(query, db_engine, params=tuple(params))


def get_dic_data(
    dic_id: int,
    app_host: str = None,
    app_port: str = None,
) -> pd.DataFrame:
    """
    Fetch DIC displacement data from the Django API endpoint as a DataFrame.
    """
    # Use config defaults if not provided
    if app_host is None:
        app_host = config.get("api.host")
    if app_port is None:
        app_port = config.get("api.port")

    url = f"http://{app_host}:{app_port}/API/dic/{dic_id}/"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Could not fetch DIC data for id {dic_id}: {response.text}")

    data = response.json()
    # If the data is empty, raise an error
    if not data or "points" not in data:
        raise ValueError(f"No valid DIC data found for id {dic_id}")

    points = data["points"]
    vectors = data["vectors"]
    magnitudes = data["magnitudes"]

    # Convert to DataFrame
    df = pd.DataFrame(points, columns=["x", "y"])
    df["u"] = [v[0] for v in vectors]
    df["v"] = [v[1] for v in vectors]
    df["V"] = magnitudes

    return df


def filter_outliers_by_percentile(
    df: pd.DataFrame, tails_percentile: float = 0.01, velocity_column: str = "V"
) -> pd.DataFrame:
    """
    Filter out extreme tails based on the specified percentile.

    Args:
        df: DataFrame with DIC data
        tails_percentile: Percentile for tail filtering (e.g., 0.01 removes bottom 1% and top 1%)
        velocity_column: Column name for velocity magnitude

    Returns:
        Filtered DataFrame
    """
    prob_threshold = (tails_percentile, 1 - tails_percentile)
    velocity_percentiles = df[velocity_column].quantile(prob_threshold).values

    df_filtered = df[
        (df[velocity_column] >= velocity_percentiles[0])
        & (df[velocity_column] <= velocity_percentiles[1])
    ].reset_index(drop=True)

    logger.info(
        f"Percentile filtering: {len(df)} -> {len(df_filtered)} points "
        f"(removed {len(df) - len(df_filtered)} outliers)"
    )

    return df_filtered


def filter_by_min_velocity(
    df: pd.DataFrame, min_velocity: float, velocity_column: str = "V"
) -> pd.DataFrame:
    """
    Filter out low velocity vectors if specified.

    Args:
        df: DataFrame with DIC data
        min_velocity: Minimum velocity threshold
        velocity_column: Column name for velocity magnitude

    Returns:
        Filtered DataFrame
    """
    if min_velocity < 0:
        logger.info("Minimum velocity filtering disabled")
        return df

    df_filtered = df[df[velocity_column] >= min_velocity].reset_index(drop=True)

    logger.info(
        f"Min velocity filtering: {len(df)} -> {len(df_filtered)} points "
        f"(removed {len(df) - len(df_filtered)} points below {min_velocity})"
    )

    return df_filtered


def create_2d_grid(df: pd.DataFrame, grid_spacing: float = None) -> tuple:
    """
    Create a 2D grid from scattered DIC points.

    Args:
        df: DataFrame with columns 'x', 'y', 'u', 'v', 'V'
        grid_spacing: Spacing between grid points. If None, estimated from data

    Returns:
        tuple: (x_grid, y_grid, u_grid, v_grid, v_mag_grid, valid_mask)
    """
    if grid_spacing is None:
        # Estimate grid spacing from minimum distances
        points = df[["x", "y"]].values
        tree = cKDTree(points)
        distances, _ = tree.query(
            points, k=2
        )  # k=2 to get distance to nearest neighbor
        grid_spacing = np.median(
            distances[:, 1]
        )  # distances[:, 1] is distance to nearest neighbor
        logger.info(f"Estimated grid spacing: {grid_spacing:.2f}")

    # Create regular grid
    x_min, x_max = df["x"].min(), df["x"].max()
    y_min, y_max = df["y"].min(), df["y"].max()

    x_grid = np.arange(x_min, x_max + grid_spacing, grid_spacing)
    y_grid = np.arange(y_min, y_max + grid_spacing, grid_spacing)

    # Create meshgrid
    X, Y = np.meshgrid(x_grid, y_grid)

    # Initialize grids
    u_grid = np.full_like(X, np.nan)
    v_grid = np.full_like(Y, np.nan)
    v_mag_grid = np.full_like(X, np.nan)

    # Map points to grid
    for _, row in df.iterrows():
        i = np.argmin(np.abs(y_grid - row["y"]))
        j = np.argmin(np.abs(x_grid - row["x"]))

        u_grid[i, j] = row["u"]
        v_grid[i, j] = row["v"]
        v_mag_grid[i, j] = row["V"]

    # Create valid mask
    valid_mask = ~np.isnan(v_mag_grid)

    logger.info(f"Created 2D grid: {X.shape}, {np.sum(valid_mask)} valid points")

    return X, Y, u_grid, v_grid, v_mag_grid, valid_mask


def apply_2d_median_filter(
    df: pd.DataFrame,
    window_size: int = None,
    threshold_factor: float = None,
    velocity_column: str = "V",
) -> pd.DataFrame:
    """
    Apply 2D median filter to remove outliers based on local neighborhood.

    Args:
        df: DataFrame with DIC data containing 'x', 'y', 'u', 'v', 'V' columns
        window_size: Size of the median filter window (odd number recommended)
        threshold_factor: Factor for outlier detection (n * median_deviation)
        velocity_column: Column name for velocity magnitude

    Returns:
        Filtered DataFrame
    """
    # Use config defaults if not provided
    if window_size is None:
        window_size = config.get("dic.median_window_size")
    if threshold_factor is None:
        threshold_factor = config.get("dic.median_threshold_factor")

    logger.info(
        f"Applying 2D median filter: window_size={window_size}, threshold_factor={threshold_factor}"
    )

    # Create 2D grid from scattered points
    X, Y, u_grid, v_grid, v_mag_grid, valid_mask = create_2d_grid(df)

    # Apply median filter only to valid points
    v_mag_filtered = np.full_like(v_mag_grid, np.nan)

    # Create a copy for filtering
    v_mag_work = v_mag_grid.copy()
    v_mag_work[~valid_mask] = np.nan

    # Apply median filter
    v_mag_median = ndimage.median_filter(v_mag_work, size=window_size)

    # Calculate local median absolute deviation (MAD)
    # MAD = median(|x_i - median(x)|)
    mad_grid = np.abs(v_mag_work - v_mag_median)
    mad_median = ndimage.median_filter(mad_grid, size=window_size)

    # Create outlier mask
    outlier_threshold = threshold_factor * mad_median
    outlier_mask = np.abs(v_mag_work - v_mag_median) > outlier_threshold

    # Count outliers
    n_outliers = np.sum(outlier_mask & valid_mask)
    logger.info(f"Detected {n_outliers} outliers in 2D median filter")

    # Create list of valid point indices to keep
    keep_indices = []
    for idx, row in df.iterrows():
        # Find grid position
        i = np.argmin(np.abs(Y[:, 0] - row["y"]))
        j = np.argmin(np.abs(X[0, :] - row["x"]))

        # Check if this point should be kept
        if not outlier_mask[i, j]:
            keep_indices.append(idx)

    # Filter DataFrame
    df_filtered = df.iloc[keep_indices].reset_index(drop=True)
    logger.info(
        f"2D median filtering: {len(df)} -> {len(df_filtered)} points "
        f"(removed {len(df) - len(df_filtered)} outliers)"
    )

    return df_filtered


def apply_dic_filters(
    df: pd.DataFrame,
    filter_outliers: bool = None,
    tails_percentile: float = None,
    min_velocity: float = None,
    apply_2d_median: bool = None,
    median_window_size: int = None,
    median_threshold_factor: float = None,
) -> pd.DataFrame:
    """
    Apply all DIC data filters in sequence.

    Args:
        df: Raw DIC DataFrame
        filter_outliers: Whether to apply percentile-based outlier filtering
        tails_percentile: Percentile for tail filtering
        min_velocity: Minimum velocity threshold
        apply_2d_median: Whether to apply 2D median filter
        median_window_size: Window size for median filter
        median_threshold_factor: Threshold factor for median filter

    Returns:
        Filtered DataFrame
    """
    # Use config defaults if not provided
    if filter_outliers is None:
        filter_outliers = config.get("dic.filter_outliers")
    if tails_percentile is None:
        tails_percentile = config.get("dic.tails_percentile")
    if min_velocity is None:
        min_velocity = config.get("dic.min_velocity")
    if apply_2d_median is None:
        apply_2d_median = config.get("dic.apply_2d_median")
    if median_window_size is None:
        median_window_size = config.get("dic.median_window_size")
    if median_threshold_factor is None:
        median_threshold_factor = config.get("dic.median_threshold_factor")

    logger.info(f"Starting DIC filtering pipeline with {len(df)} points")
    df_filtered = df.copy()

    # 1. Apply percentile-based outlier filtering
    if filter_outliers:
        df_filtered = filter_outliers_by_percentile(
            df_filtered, tails_percentile=tails_percentile
        )

    # 2. Apply minimum velocity filtering
    if min_velocity >= 0:
        df_filtered = filter_by_min_velocity(df_filtered, min_velocity=min_velocity)

    # 3. Apply 2D median filter
    if apply_2d_median:
        df_filtered = apply_2d_median_filter(
            df_filtered,
            window_size=median_window_size,
            threshold_factor=median_threshold_factor,
        )
    logger.info(
        f"DIC filtering pipeline completed: {len(df)} -> {len(df_filtered)} points "
        f"(removed {len(df) - len(df_filtered)} total)"
    )

    return df_filtered


def get_image(
    image_id: int,
    app_host: str = None,
    app_port: str = None,
    camera_name: str | None = None,
) -> Image.Image:
    """Get an image from the database by its ID and rotate if from Tele camera."""
    # Use config defaults if not provided
    if app_host is None:
        app_host = config.get("api.host")
    if app_port is None:
        app_port = config.get("api.port")

    url = f"http://{app_host}:{app_port}/API/images/{image_id}/"
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(io.BytesIO(response.content))
        # Rotate if camera_name is Tele (portrait mode)
        if camera_name is not None and "tele" in camera_name.lower():
            img = img.rotate(90, expand=True)  # 90° clockwise
        return img
    else:
        raise ValueError(f"Image with ID {image_id} not found.")
