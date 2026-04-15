import ast
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import DictConfig, ListConfig
from sqlalchemy import create_engine

from ppcluster.cvat import (
    filter_dataframe_by_polygons,
    read_polygons_from_cvat,
)
from ppcluster.preprocessing import (
    apply_dic_filters,
    spatial_subsample,
)
from ppcluster.utils.database import (
    fetch_dic_analysis_ids,
    get_dic_analysis_by_ids,
    get_image,
    get_multi_dic_data,
)

logger = logging.getLogger("ppcx")


class DataLoadingError(Exception):
    """Custom exception raised when data loading from the database fails."""

    pass


DEFAULT_FILENAME_PATTERN = r"day_dic_(?P<slave>\d{4}-\d{2}-\d{2})_(?P<master>\d{4}-\d{2}-\d{2})_dt(?P<dt>\d+)_.*\.nc"


def find_ensemble_files(
    search_dir: Path,
    reference_date: str | datetime,
    dt_hours_min: int,
    dt_hours_max: int,
    filename_pattern: str | None = None,
) -> list[Path]:
    """
    Find ensemble NetCDF files in search_dir matching the reference date and dt range.

    Args:
        search_dir: Directory to search in.
        reference_date: The date string (YYYY-MM-DD) or datetime object corresponding to the slave/final date.
        dt_hours_min: Minimum time delta in hours.
        dt_hours_max: Maximum time delta in hours.
        filename_pattern: Optional regex pattern. If None, uses default: day_dic_{slave}_{master}_dt{dt}_ensemble{ens}.nc
        where {slave} and {master} are dates in YYYY-MM-DD format, and {dt} is the time delta in days."
    Returns:
        List[Path] of selected NetCDF files matching the criteria, sorted by dt and filename. If no files match, returns an empty list.
    """
    if not search_dir.exists():
        raise DataLoadingError(f"Search directory does not exist: {search_dir}")

    # Normalize reference_date to YYYY-MM-DD string
    if isinstance(reference_date, datetime):
        reference_date = reference_date.strftime("%Y-%m-%d")

    # Convert dt range from hours to days (integer)
    dt_days_min = round(dt_hours_min / 24)
    dt_days_max = round(dt_hours_max / 24)

    # Use default pattern if not provided
    # Default matches: day_dic_2025-01-02_2025-01-01_dt1_ensemble10.nc
    if not filename_pattern:
        regex = re.compile(
            r"day_dic_(?P<slave>\d{4}-\d{2}-\d{2})_(?P<master>\d{4}-\d{2}-\d{2})_dt(?P<dt>\d+)_.*\.nc"
        )
    else:
        try:
            regex = re.compile(filename_pattern)
        except re.error as e:
            raise DataLoadingError(
                f"Invalid regex pattern provided: {filename_pattern}"
            ) from e

    logger.info(
        f"Searching in {search_dir} for date={reference_date}, dt=[{dt_days_min}-{dt_days_max}] days"
    )

    candidates = []
    for file_path in search_dir.glob("*.nc"):
        match = regex.match(file_path.name)
        if not match:
            continue

        groups = match.groupdict()

        # Check date (slave_date == reference_date)
        # If user supplied custom regex without 'slave' group, we skip date check or rely on partial match?
        # Enforce that custom regex must catch at least 'slave' date or be specific enough.
        slave_date = groups.get("slave")
        if slave_date and slave_date != reference_date:
            continue

        # Check dt
        dt_str = groups.get("dt")
        if dt_str:
            try:
                dt_val = int(dt_str)
                if not (dt_days_min <= dt_val <= dt_days_max):
                    continue
            except ValueError:
                continue  # dt not an integer, skip

        candidates.append(file_path)

    if not candidates:
        logger.warning(
            f"No ensemble files found in {search_dir} matching reference date {reference_date} "
            f"and dt range {dt_days_min}-{dt_days_max} days."
        )
        return []

    # Sort candidates by extracted dt (if present)
    file_dt_pairs = []
    for p in candidates:
        m = regex.match(p.name)
        dt_val = None
        if m:
            dt_str = m.groupdict().get("dt")
            try:
                dt_val = int(dt_str)
            except Exception:
                dt_val = None
        file_dt_pairs.append((p, dt_val if dt_val is not None else -1))

    # sort by dt then filename for deterministic order
    file_dt_pairs.sort(key=lambda t: (t[1], t[0].name))

    selected_files = [p for p, _ in file_dt_pairs]
    logger.info(f"Auto-selected files: {[p.name for p in selected_files]}")

    return selected_files


def read_data_from_pylamma_nc(
    nc_path: Path, base_image_dir: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read pylamma ensemble NetCDF and return data structures matching the DB reader.

    Args:
        nc_path (Path): Path to the .nc file.
        base_image_dir (Path | None): User-specified directory to look for images.

    Returns:
        tuple: (DIC dataframe, metadata dataframe).
    """

    def _parse_datetime_with_midday_or_none(val: str | None) -> pd.Timestamp:
        """Parse a date or datetime string, using 12:00 if only date is present."""
        if val is None or pd.isna(val):
            return pd.Timestamp("NaT")
        dt = pd.to_datetime(val, errors="coerce")
        if pd.isna(dt):
            return pd.Timestamp("NaT")
        # If time is missing, set to 12:00
        if dt.hour == 0 and dt.minute == 0 and dt.second == 0 and len(str(val)) <= 10:
            return dt.replace(hour=12, minute=0, second=0)
        return dt

    with xr.open_dataset(nc_path) as ds:
        attrs = ds.attrs

        # --- 1. Date Parsing ---
        # Try 'date1'/'date2' first, then 'master_day'/'slave_day'
        master_ts = pd.NaT
        for key in ["date1", "master_day"]:
            master_ts = _parse_datetime_with_midday_or_none(attrs.get(key))
            if not pd.isna(master_ts):
                break
        slave_ts = pd.NaT
        for key in ["date2", "slave_day"]:
            slave_ts = _parse_datetime_with_midday_or_none(attrs.get(key))
            if not pd.isna(slave_ts):
                break

        # Fallback: Parse filename if attributes are missing or NaT
        if pd.isna(master_ts) or pd.isna(slave_ts):
            try:
                # Regex matches: day_dic_{slave}_{master}_...
                match = re.search(
                    r"day_dic_(?P<slave>\d{4}-\d{2}-\d{2})_(?P<master>\d{4}-\d{2}-\d{2})",
                    nc_path.name,
                )
                if match:
                    if pd.isna(slave_ts):
                        slave_ts = _parse_datetime_with_midday_or_none(
                            match.group("slave")
                        )
                    if pd.isna(master_ts):
                        master_ts = _parse_datetime_with_midday_or_none(
                            match.group("master")
                        )
            except Exception as e:
                logger.warning(
                    f"Could not infer dates from filename {nc_path.name}: {e}"
                )

        # Safe string formatting for reference_date
        if not pd.isna(slave_ts):
            ref_date_str = slave_ts.strftime("%Y-%m-%d")
        else:
            ref_date_str = "unknown"

        # --- 2. Metadata Extraction ---
        camera_name = attrs.get("camera_name", "unknown")
        master_list_str = attrs.get("master_list", "[]")
        slave_list_str = attrs.get("slave_list", "[]")
        nc_image_dir_str = attrs.get("image_dir", "")
        dt_hours = float(attrs.get("dt_days", 0)) * 24
        dic_analyses = pd.DataFrame(
            [
                {
                    "dic_id": np.nan,
                    "reference_date": ref_date_str,
                    "camera_name": camera_name,
                    "master_timestamp": master_ts,
                    "slave_timestamp": slave_ts,
                    "master_image_ids": np.nan,
                    "slave_image_ids": np.nan,
                    "image_dir": nc_image_dir_str,
                    "master_list": master_list_str,
                    "slave_list": slave_list_str,
                    "dt_hours": dt_hours,
                }
            ]
        )
        # --- 3. Image path validation (do not load image) ---
        image_path_str = None
        try:
            master_list = ast.literal_eval(master_list_str)

            mid_idx = len(master_list) // 2
            mid_filename = master_list[mid_idx]

            # Option A: User specified base_image_dir (Overrides NetCDF attribute)
            if base_image_dir:
                candidate = Path(base_image_dir) / mid_filename
                if candidate.exists():
                    image_path_str = str(candidate.resolve())
                else:
                    logger.debug(
                        f"Image {mid_filename} not found in user-provided dir {base_image_dir}"
                    )

            # Option B: Fallback to 'image_dir' attribute from NetCDF
            if image_path_str is None and nc_image_dir_str:
                nc_image_dir = Path(nc_image_dir_str)

                # B1. Try as absolute or relative path
                candidate = nc_image_dir / mid_filename
                if candidate.exists():
                    image_path_str = str(candidate.resolve())

                # B2. Try relative to the NetCDF file location
                if image_path_str is None:
                    candidate = (nc_path.parent / nc_image_dir / mid_filename).resolve()
                    if candidate.exists():
                        image_path_str = str(candidate)

            if image_path_str is None:
                logger.warning(
                    f"Image {mid_filename} could not be located for NetCDF {nc_path.name}."
                )

        except Exception as e:
            logger.warning(
                f"Error while trying to locate image for NetCDF {nc_path.name}: {e}"
            )
            image_path_str = None

        if image_path_str is not None:
            logger.info(f"Associated image found for {nc_path.name}: {image_path_str}")
            dic_analyses["image_path"] = image_path_str
        else:
            logger.warning(f"No associated image found for {nc_path.name}.")
            dic_analyses["image_path"] = np.nan

        # --- 4. Data Processing ---
        # Process DIC data into DataFrame with structure: x, y, u, v, V, MAD
        df = ds.to_dataframe().reset_index()

        # Check that all the required columns are present
        required_cols = ["x", "y", "vx", "vy", "mad", "ensemble_size"]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.warning(
                f"NetCDF {nc_path.name} is missing required columns: {missing}. Found columns: {df.columns.tolist()}"
            )
            return pd.DataFrame(), dic_analyses

        # --- FIX: x and y need swapping based on coordinate system conventions
        # Pylamma/TICOI NetCDF saves typically with dims ('y', 'x').
        # If the visual result is rotated, it means 'x' in the NC file corresponds to the 'y' axis in the plot, or vice versa.
        df = df.rename(columns={"x": "y_temp", "y": "x"}).rename(
            columns={"y_temp": "y"}
        )

        # Rename vx, vy to u, v and calculate V
        df = df.rename(columns={"vx": "u", "vy": "v"})

        if "mid_date" in df.columns:
            df = df.drop(columns=["mid_date"])

        df["V"] = np.sqrt(df["u"] ** 2 + df["v"] ** 2)

        # Keep only required columns and drop NaNs
        column_to_keep = ["x", "y", "u", "v", "V", "mad", "ensemble_size"]
        df = df[column_to_keep].dropna(subset=["V"])

    return df, dic_analyses


def read_data_from_db(
    config: DictConfig | ListConfig,
    reference_date: str | datetime,
    reference_start_date: str | datetime | None = None,
    reference_end_date: str | datetime | None = None,
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame, Any]:
    """
    Fetch DIC data and metadata from the database.

    Args:
        config: Configuration object containing database and data specs.
        reference_date: Target reference date.
        reference_start_date: Start of the temporal search window.
        reference_end_date: End of the temporal search window.

    Returns:
        tuple: (dict of DIC dataframes, metadata dataframe, background image).
    """
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
        raise DataLoadingError("No DIC analyses found for the given criteria")

    dic_analyses = get_dic_analysis_by_ids(db_engine=db_engine, dic_ids=dic_ids)
    logger.info("Fetched DIC analysis:")
    for _, row in dic_analyses.iterrows():
        logger.info(
            f"DIC ID: {row['dic_id']}, date: {row['reference_date']}, dt (hrs): {row['dt_hours']}, Master: {row['master_timestamp']}, Slave: {row['slave_timestamp']}"
        )

    master_image_id = dic_analyses["master_image_id"].iloc[0]

    img = get_image(image_id=master_image_id, config=config.api)

    out = get_multi_dic_data(dic_ids, stack_results=False, config=config.api)
    logger.info(f"Found stack of {len(out)} DIC dataframes.")

    # Rename columns to match the standard u,v format if necessary
    for key in out:
        out[key] = out[key].rename(columns={"dx": "u", "dy": "v"}, errors="ignore")

    return out, dic_analyses, img


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
            raise DataLoadingError(
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
        raise DataLoadingError(
            f"Unsupported sector prior file format: {sector_prior_path.suffix}"
        )
    return sectors


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


def read_roi_from_file(path: Path | str | None) -> Any | None:
    """
    Load ROI polygon from CVAT XML or geospatial files.

    Args:
        path: Path to the ROI definition file.

    Returns:
        Optional[Any]: Shapely polygon if found, else None.
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
) -> tuple[dict[str, Any], Any]:
    """
    Load sector polygons and ROI polygon from a CVAT XML or geospatial file.

    Args:
        sector_prior_path: Path to the file containing sector definitions.
        sector_names: List of sector names to extract.
        roi_path: Optional separate path for the ROI definition.

    Returns:
        tuple: (dictionary of sector polygons, ROI polygon).
    """

    # Find matching sector prior file (supports glob patterns)
    prior_file_pattern = Path(sector_prior_path)
    sector_prior_paths = list(
        prior_file_pattern.parent.glob(Path(prior_file_pattern).name)
    )
    if len(sector_prior_paths) == 0:
        raise DataLoadingError(
            f"No sector prior file found matching: {sector_prior_path}"
        )
    if len(sector_prior_paths) > 1:
        logger.warning(
            f"Multiple sector prior files matched. Using the first one found: {list(sector_prior_paths)}"
        )
    sector_prior_path = sector_prior_paths[0]

    # Load sectors
    sectors = read_sectors_from_file(sector_prior_path, sector_names)

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
