import logging
from typing import Literal

import numpy as np
import pandas as pd
from scipy import ndimage

from ppcluster.griddata import create_2d_grid_from_df
from ppcluster.lamma_filter import vector_field_filter

logger = logging.getLogger("ppcx")
RANDOM_SEED = 8927


# === GRID data filtering ===


def apply_dic_filters(
    df: pd.DataFrame,
    min_velocity: float | None = None,
    filter_outliers: bool = True,
    tails_percentile: float = 0.01,
    apply_2d_median: bool = False,
    median_window_size: int = 5,
    median_threshold_factor: float = 3.0,
    apply_2d_gaussian: bool = False,
    gaussian_sigma: float = 1.0,
    apply_lamma_filter: bool = False,
    lamma_method: str = "Neighbours",
    lamma_k: int = 4,
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

    logger.info(f"Starting DIC filtering pipeline with {len(df)} points")
    df_filtered = df.copy()

    # Apply percentile-based outlier filtering
    if filter_outliers and tails_percentile is not None and tails_percentile > 0:
        df_filtered = filter_outliers_by_percentile(
            df_filtered, tails_percentile=tails_percentile
        )

    # Apply minimum velocity filtering
    if min_velocity is not None and min_velocity >= 0:
        df_filtered = filter_by_min_velocity(df_filtered, min_velocity=min_velocity)

    # Apply LAMMA vector field filter
    if apply_lamma_filter:
        df_filtered = apply_2d_lamma_filter(
            df_filtered,
            method=lamma_method,
            k=lamma_k,
        )

    # Apply 2D median filter
    if apply_2d_median:
        df_filtered = apply_2d_median_filter(
            df_filtered,
            window_size=median_window_size,
            threshold_factor=median_threshold_factor,
        )

    # Apply 2D Gaussian smoothing to velocity magnitude
    if apply_2d_gaussian:
        df_filtered = apply_2d_gaussian_filter(df_filtered, sigma=gaussian_sigma)

    logger.info(
        f"DIC filtering pipeline completed: {len(df)} -> {len(df_filtered)} points "
        f"(removed {len(df) - len(df_filtered)} total)"
    )

    return df_filtered


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


def apply_2d_gaussian_filter(df: pd.DataFrame, sigma: float = 1.0) -> pd.DataFrame:
    """
    Smooth u, v and velocity magnitude with a 2D Gaussian filter.

    - Uses mask-normalized gaussian filtering to respect missing cells.
    - Maps smoothed grid values back to original points.
    """

    if sigma <= 0:
        logger.info(
            "Gaussian smoothing disabled (sigma <= 0). Returning original DataFrame."
        )
        return df

    logger.info(f"Applying 2D Gaussian filter (u,v,V): sigma={sigma}")

    # Build grid from points
    X, Y, u_grid, v_grid, v_mag_grid, valid_mask = create_2d_grid_from_df(df)

    if not np.any(valid_mask):
        logger.warning(
            "apply_2d_gaussian_filter: no valid grid cells found, returning original df"
        )
        return df

    # Prepare arrays for convolution: replace NaN with 0 and use validity mask
    u_work = np.where(valid_mask, u_grid, 0.0)
    v_work = np.where(valid_mask, v_grid, 0.0)
    V_work = np.where(valid_mask, v_mag_grid, 0.0)
    mask_work = valid_mask.astype(float)

    # Smooth numerator and denominator for each component
    num_u = ndimage.gaussian_filter(u_work, sigma=sigma, mode="constant", cval=0.0)
    num_v = ndimage.gaussian_filter(v_work, sigma=sigma, mode="constant", cval=0.0)
    num_V = ndimage.gaussian_filter(V_work, sigma=sigma, mode="constant", cval=0.0)
    den = ndimage.gaussian_filter(mask_work, sigma=sigma, mode="constant", cval=0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        u_smoothed = num_u / den
        v_smoothed = num_v / den
        V_smoothed = num_V / den

    u_smoothed[den == 0] = np.nan
    v_smoothed[den == 0] = np.nan
    V_smoothed[den == 0] = np.nan

    # Map smoothed grid back to points
    df_out = df.copy()
    sm_u = []
    sm_v = []
    sm_V = []
    x_grid = X[0, :]
    y_grid = Y[:, 0]
    for _, row in df_out.iterrows():
        i = np.argmin(np.abs(y_grid - row["y"]))
        j = np.argmin(np.abs(x_grid - row["x"]))
        sm_u.append(u_smoothed[i, j])
        sm_v.append(v_smoothed[i, j])
        sm_V.append(V_smoothed[i, j])

    df_out["u"] = sm_u
    df_out["v"] = sm_v
    df_out["V"] = sm_V

    n_none = int(
        np.isnan(df_out["V"]).sum()
        + np.isnan(df_out["u"]).sum()
        + np.isnan(df_out["v"]).sum()
    )
    logger.info(
        f"apply_2d_gaussian_filter: smoothed values assigned, NaN mapped points (total components)={n_none}"
    )
    return df_out


def apply_2d_median_filter(
    df: pd.DataFrame,
    window_size: int | None = 5,
    threshold_factor: float | None = 3.0,
) -> pd.DataFrame:
    """
    Apply 2D median filter to remove outliers based on local neighborhood.
    Now evaluates u, v and V components. A point is removed if any component
    is an outlier relative to its local median +/- threshold_factor * MAD.
    """
    logger.info(
        f"Applying 2D median filter (u,v,V): window_size={window_size}, threshold_factor={threshold_factor}"
    )

    # Create 2D grid from scattered points
    X, Y, u_grid, v_grid, v_mag_grid, valid_mask = create_2d_grid_from_df(df)

    # Prepare work arrays with NaNs where invalid
    u_work = u_grid.copy().astype(float)
    v_work = v_grid.copy().astype(float)
    V_work = v_mag_grid.copy().astype(float)

    u_work[~valid_mask] = np.nan
    v_work[~valid_mask] = np.nan
    V_work[~valid_mask] = np.nan

    size = int(window_size) if window_size is not None else 5

    # Helper to compute nanmedian with generic_filter
    def nanmedian(arr):
        return float(np.nanmedian(arr))

    u_median = ndimage.generic_filter(
        u_work, function=nanmedian, size=size, mode="constant", cval=np.nan
    )
    v_median = ndimage.generic_filter(
        v_work, function=nanmedian, size=size, mode="constant", cval=np.nan
    )
    V_median = ndimage.generic_filter(
        V_work, function=nanmedian, size=size, mode="constant", cval=np.nan
    )

    # compute local MAD (median of absolute deviations)
    u_abs_dev = np.abs(u_work - u_median)
    v_abs_dev = np.abs(v_work - v_median)
    V_abs_dev = np.abs(V_work - V_median)

    u_mad = ndimage.generic_filter(
        u_abs_dev, function=nanmedian, size=size, mode="constant", cval=np.nan
    )
    v_mad = ndimage.generic_filter(
        v_abs_dev, function=nanmedian, size=size, mode="constant", cval=np.nan
    )
    V_mad = ndimage.generic_filter(
        V_abs_dev, function=nanmedian, size=size, mode="constant", cval=np.nan
    )

    # thresholding (avoid zero MAD by adding tiny eps)
    eps = 1e-12
    outlier_u = (
        np.abs(u_work - u_median) > (threshold_factor * (u_mad + eps))
    ) & valid_mask
    outlier_v = (
        np.abs(v_work - v_median) > (threshold_factor * (v_mad + eps))
    ) & valid_mask
    outlier_V = (
        np.abs(V_work - V_median) > (threshold_factor * (V_mad + eps))
    ) & valid_mask

    # combined outlier: any component flagged
    outlier_mask = outlier_u | outlier_v | outlier_V

    n_outliers = int(np.sum(outlier_mask))
    logger.info(f"Detected {n_outliers} outliers (u/v/V combined) in 2D median filter")

    # Build list of valid indices to keep (not an outlier)
    keep_indices = []
    for idx, row in df.iterrows():
        i = np.argmin(np.abs(Y[:, 0] - row["y"]))
        j = np.argmin(np.abs(X[0, :] - row["x"]))
        if not outlier_mask[i, j]:
            keep_indices.append(idx)

    df_filtered = df.iloc[keep_indices].reset_index(drop=True)
    logger.info(
        f"2D median filtering (u,v,V): {len(df)} -> {len(df_filtered)} points (removed {len(df) - len(df_filtered)})"
    )
    return df_filtered


def apply_2d_lamma_filter(
    df: pd.DataFrame,
    method: str = "Neighbours",
    k: int | None = 4,
) -> pd.DataFrame:
    """
    Apply the LAMMA vector field filter to the DataFrame.

    Adapts the specific I/O of lamma.vector_field_filter to the pandas DataFrame.

    Args:
        df: Input DataFrame containing 'x', 'y' coordinates and 'u', 'v' velocity components.
        method: Filtering method ('Delaunay', 'Neighbours', 'Radius').
        k: Number of neighbors or radius (depending on method).

    Returns:
        Filtered DataFrame with updated 'u', 'v' and 'V'.
    """
    if df.empty:
        return df

    logger.info(f"Applying LAMMA filter: method={method}, k={k}")

    # Prepare inputs for LAMMA
    # Nodes: coordinates [x, y]
    nodes = df[["x", "y"]].to_numpy()

    # Values: list of arrays [u, v]
    # Ensure they are flat arrays as expected by some paths in lamma,
    # though vector_field_filter handles shape checks.
    u = df["u"].to_numpy()
    v = df["v"].to_numpy()
    values = [u, v]

    try:
        # Call LAMMA function
        # Interface: vector_field_filter(values, nodes, method="Delaunay", k=None)
        # Returns: out[0]->DX, out[1]->DY, out[2]->DZ, out[3]->nodes
        out = vector_field_filter(values, nodes, method=method, k=k)

        u_filtered = out[0]
        v_filtered = out[1]

        # Create output DataFrame
        df_out = df.copy()
        df_out["u"] = u_filtered
        df_out["v"] = v_filtered

        # Recompute magnitude V
        df_out["V"] = np.sqrt(df_out["u"] ** 2 + df_out["v"] ** 2)

        logger.info("LAMMA filtering completed.")
        return df_out

    except Exception as e:
        logger.error(f"Failed to apply LAMMA filter: {e}")
        # Return original on failure or re-raise
        return df


# === SPATIAL SUBSAMPLING ===
def spatial_subsample(df, n_subsample=5, method="regular", random_state=RANDOM_SEED):
    """
    Subsample DIC data spatially to reduce computational load.

    Parameters:
    - n_subsample: Take every nth point (for regular) or fraction (for random)
    - method: 'regular', 'random''
    """
    available_methods = ["regular", "random"]
    if method == "regular":
        # Take every nth point in spatial order
        df_sorted = df.sort_values(["x", "y"]).reset_index(drop=True)
        subsample_idx = np.arange(0, len(df_sorted), n_subsample)
        df_sub = df_sorted.iloc[subsample_idx].copy()

    elif method == "random":
        # Random sampling (n_subsample as fraction)
        n_samples = (
            int(len(df) * n_subsample)
            if n_subsample < 1
            else int(len(df) / n_subsample)
        )
        df_sub = df.sample(n=min(n_samples, len(df)), random_state=random_state).copy()

    else:
        raise ValueError(
            f"Unknown subsampling method: {method}. Available: {available_methods}"
        )

    print(
        f"Subsampled from {len(df)} to {len(df_sub)} points ({len(df_sub) / len(df) * 100:.1f}%)"
    )
    return df_sub


# === Feature preprocessing ===
def preprocess_velocity_features(
    velocities: np.ndarray,
    velocity_transform: str = "power",
    velocity_params: dict | None = None,
):
    """
    Preprocess velocity features with various transformation options.

    Parameters:
    -----------
    velocities : numpy.ndarray
        1D array of velocity magnitudes
    velocity_transform : str, default="power"
        Type of velocity transformation: "power", "exponential", "threshold", "sigmoid", or "none"
    velocity_params : dict, optional
        Parameters for velocity transformation. Defaults depend on transform type:
        - "power": {"exponent": 2}
        - "exponential": {"scale": 0.2}
        - "threshold": {"threshold_percentile": 50, "enhancement": 3.0}
        - "sigmoid": {"midpoint_percentile": 70, "steepness": 2.0}
        - "none": {}

            Returns:
    --------
    tuple : (velocities_transformed, transform_info)
        - velocities_transformed: numpy.ndarray of transformed velocities
        - transform_info: Dictionary with transformation metadata
    """
    # Ensure input is numpy array
    velocities = np.asarray(velocities)

    # Set default velocity transformation parameters
    if velocity_params is None:
        velocity_params = {
            "power": {"exponent": 2},
            "exponential": {"scale": 0.2},
            "threshold": {"threshold_percentile": 50, "enhancement": 3.0},
            "sigmoid": {"midpoint_percentile": 70, "steepness": 2.0},
            "none": {},
        }.get(velocity_transform, {})

    # Apply velocity transformation
    if velocity_transform == "power":
        exponent = velocity_params.get("exponent", 2)
        velocities_transformed = velocities**exponent
        transform_info = {"type": "power", "exponent": exponent}

    elif velocity_transform == "exponential":
        scale = velocity_params.get("scale", 0.2)
        velocities_transformed = np.exp(scale * velocities) - 1
        transform_info = {"type": "exponential", "scale": scale}

    elif velocity_transform == "threshold":
        threshold_percentile = velocity_params.get("threshold_percentile", 50)
        enhancement = velocity_params.get("enhancement", 3.0)
        threshold = np.percentile(velocities, threshold_percentile)
        velocities_transformed = np.where(
            velocities > threshold,
            velocities * enhancement,
            velocities,
        )
        transform_info = {
            "type": "threshold",
            "threshold": threshold,
            "enhancement": enhancement,
        }

    elif velocity_transform == "sigmoid":
        midpoint_percentile = velocity_params.get("midpoint_percentile", 70)
        steepness = velocity_params.get("steepness", 2.0)
        midpoint = np.percentile(velocities, midpoint_percentile)
        velocities_transformed = 1 / (1 + np.exp(-steepness * (velocities - midpoint)))
        transform_info = {
            "type": "sigmoid",
            "midpoint": midpoint,
            "steepness": steepness,
        }

    else:  # "none" or any other value
        logger.info("No velocity transformation applied.")
        return velocities, {"type": "none"}

    logger.info(
        f"Applied {velocity_transform} transformation with params: {velocity_params}"
    )
    logger.info(f"Feature shape: {velocities_transformed.shape}")

    return velocities_transformed, transform_info


def compute_second_derivative_feature(
    grid_values: np.ndarray,
    *,
    x_coords: np.ndarray | None = None,
    y_coords: np.ndarray | None = None,
    spacing: float | tuple[float, float] | None = None,
    direction: Literal["x", "y", "gradient"] = "gradient",
    order: Literal[1, 2] = 2,
    edge_order: int = 2,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Compute directional derivatives for a 2D gridded scalar field.

    Parameters
    ----------
    grid_values : numpy.ndarray
        Scalar field on a regular grid with shape (ny, nx).
    x_coords, y_coords : array-like, optional
        Coordinate axes for the grid. If omitted, evenly spaced coordinates are generated
        from ``spacing`` (or unit spacing if not provided).
    spacing : float or tuple(float, float), optional
        Grid spacing (dy, dx). Overrides the spacing inferred from ``x_coords``/``y_coords``.
    direction : {"x", "y", "gradient"}, default "gradient"
        Direction of the derivative. Use "gradient" to obtain derivatives along both axes.
    order : {1, 2}, default 2
        Derivative order. ``order=1`` yields first derivatives, ``order=2`` yields second derivatives.
    edge_order : int, default 2
        Edge handling order passed to ``numpy.gradient``.

    Returns
    -------
    numpy.ndarray or tuple of numpy.ndarray
        Derivative grid(s) with the same shape as ``grid_values``. When ``direction="gradient"``,
        a tuple ``(d/dy, d/dx)`` is returned. For ``order=2`` the tuple contains the second
        directional derivatives.

    Raises
    ------
    ValueError
        If the input grid is not 2D, spacing is invalid, or the direction/order is unsupported.
    """
    field = np.asarray(grid_values, float)
    if field.ndim != 2:
        raise ValueError("grid_values must be a 2D array.")

    def _spacing_from_input(sp_value, idx):
        if sp_value is None:
            return None
        if np.isscalar(sp_value):
            val = float(sp_value)
        else:
            seq = tuple(sp_value)
            if len(seq) != 2:
                raise ValueError(
                    "spacing must be a float or a tuple of two floats (dy, dx)."
                )
            val = float(seq[idx])
        if val <= 0:
            raise ValueError("Spacing values must be positive.")
        return val

    def _build_axis(size, coords, sp_value, idx):
        if coords is not None:
            axis_vals = np.asarray(coords, float)
            if axis_vals.size != size:
                raise ValueError(
                    f"Expected {size} coordinates for axis {idx}, received {axis_vals.size}."
                )
            return axis_vals
        step = sp_value if sp_value is not None else 1.0
        return np.arange(size, dtype=float) * step

    spacing_y = _spacing_from_input(spacing, 0)
    spacing_x = _spacing_from_input(spacing, 1)

    y_axis = _build_axis(field.shape[0], y_coords, spacing_y, 0)
    x_axis = _build_axis(field.shape[1], x_coords, spacing_x, 1)

    def _axis_spacing(axis_vals):
        if axis_vals.size < 2:
            return 1.0
        diffs = np.diff(axis_vals)
        diffs = diffs[np.abs(diffs) > 0]
        if diffs.size == 0:
            return 1.0
        return float(np.mean(diffs))

    dy = spacing_y if spacing_y is not None else _axis_spacing(y_axis)
    dx = spacing_x if spacing_x is not None else _axis_spacing(x_axis)
    if dy <= 0 or dx <= 0:
        raise ValueError("Computed grid spacing must be positive.")

    def _directional_derivative(axis: int) -> np.ndarray:
        step = dy if axis == 0 else dx
        first = np.gradient(field, step, axis=axis, edge_order=edge_order)
        if order == 1:
            return first
        if order == 2:
            return np.gradient(first, step, axis=axis, edge_order=edge_order)
        raise ValueError("order must be either 1 or 2.")

    direction_norm = str(direction).lower()
    if direction_norm in {"y", "vertical"}:
        result = _directional_derivative(0)
    elif direction_norm in {"x", "horizontal"}:
        result = _directional_derivative(1)
    elif direction_norm in {"gradient", "both"}:
        result = (_directional_derivative(0), _directional_derivative(1))
    else:
        raise ValueError("direction must be 'x', 'y', or 'gradient'.")

    logger.debug(
        "Computed %s derivative(s) (order=%d) on grid shape %s",
        direction_norm,
        order,
        field.shape,
    )
    return result
