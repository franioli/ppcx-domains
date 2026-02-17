from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from cvatkit import CvatReader
from shapely.geometry import Point
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

logger = logging.getLogger("ppcx")


def read_polygons_from_cvat(
    xml_source: str | Path,
    image_ids: int | Sequence[int] | None = 0,
    include_labels: Sequence[str] | None = None,
    exclude_labels: Sequence[str] | None = None,
) -> dict[str, ShapelyPolygon]:
    """Parse polygons from a CVAT export using cvatkit.

    Args:
        xml_source: Path to the CVAT XML or ZIP annotation.
        image_ids: ID(s) of the images to extract. 0 (default) for first image,
            None for all images.
        include_labels: Whitelist of labels.
        exclude_labels: Blacklist of labels.

    Returns:
        dict[str, ShapelyPolygon]: Map of label names to Shapely Polygon objects.
    """
    reader = CvatReader(xml_source)
    polygons: dict[str, list[ShapelyPolygon]] = {}

    # Normalize image_ids to a list or None
    target_ids = [image_ids] if isinstance(image_ids, int) else image_ids

    # If target_ids is None, CvatReader handles fetching from all images by not passing image_id
    if target_ids is None:
        cvat_polygons = reader.get_polygons(
            labels=include_labels, exclude_labels=exclude_labels
        )
    else:
        cvat_polygons = []
        for iid in target_ids:
            cvat_polygons.extend(
                reader.get_polygons(
                    image_id=iid, labels=include_labels, exclude_labels=exclude_labels
                )
            )

    # Convert CvatPolygon to ShapelyPolygon
    for cvat_poly in cvat_polygons:
        label = cvat_poly.label or "unnamed"
        poly = cvat_poly.to_shapely()
        if poly:
            polygons.setdefault(label, []).append(poly)

    # Merge polygons with the same label into a MultiPolygon or single Polygon
    merged_polygons: dict[str, ShapelyPolygon] = {}
    for label, polys in polygons.items():
        if len(polys) == 1:
            merged_polygons[label] = polys[0]
        else:
            merged = unary_union(polys)
            merged_polygons[label] = merged

    return merged_polygons


def filter_dataframe_by_polygons(
    df: pd.DataFrame | pd.Series,
    polygon: ShapelyPolygon,
    x_col: str = "x",
    y_col: str = "y",
    invert: bool = False,
    return_mask: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    """
    Filter a DIC dataframe keeping only points inside (or outside if invert=True) a Shapely Polygon.

    Parameters
    ----------
    df : pandas.DataFrame | pandas.Series
        Input dataframe containing point coordinates.
    polygon : ShapelyPolygon
        Polygon object defining the area(s) to filter by.
    x_col, y_col : str
        Column names in df with x and y coordinates.
    invert : bool
        If True return points outside the selected polygons.
    return_mask : bool
        If True return tuple (filtered_df, boolean_mask) where mask is aligned with df.

    Returns
    -------
    pandas.DataFrame
        Filtered dataframe (and mask if return_mask=True).
    """
    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(
            f"Coordinates columns '{x_col}' and/or '{y_col}' not found in dataframe"
        )

    pts = np.column_stack((df[x_col].to_numpy(), df[y_col].to_numpy()))
    mask = np.array([polygon.contains(Point(x, y)) for x, y in pts])
    if invert:
        mask = ~mask
    filtered_df = df[mask]

    if return_mask:
        return filtered_df, mask

    return filtered_df
