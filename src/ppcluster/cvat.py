from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cvatkit import CvatReader
from matplotlib import pyplot as plt
from matplotlib.path import Path as MplPath

logger = logging.getLogger("ppcx")


@dataclass
class Polygon:
    """Light wrapper exposing contains_points(x, y)"""

    name: str
    path: MplPath

    def contains_points(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pts = np.column_stack((x, y))
        return self.path.contains_points(pts)

    def bounds(self) -> tuple[float, float, float, float]:
        verts = np.asarray(self.path.vertices)
        xmin, ymin = verts.min(axis=0)
        xmax, ymax = verts.max(axis=0)
        return float(xmin), float(ymin), float(xmax), float(ymax)

    def plot(self, ax=None, close_polygon: bool = True, **plot_kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        verts = np.asarray(self.path.vertices)
        if close_polygon and not np.allclose(verts[0], verts[-1]):
            verts = np.vstack([verts, verts[0]])
        ax.plot(verts[:, 0], verts[:, 1], **plot_kwargs)
        return ax

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "vertices": np.array(self.path.vertices).tolist()}


def read_polygons_from_cvat(
    xml_source: str | Path,
    image_ids: int | Sequence[int] | None = 0,
    include_labels: Sequence[str] | None = None,
    exclude_labels: Sequence[str] | None = None,
) -> dict[str, Polygon]:
    """Parse polygons from a CVAT export using cvatkit.

    Args:
        xml_source: Path to the CVAT XML or ZIP annotation.
        image_ids: ID(s) of the images to extract. 0 (default) for first image,
            None for all images.
        include_labels: Whitelist of labels.
        exclude_labels: Blacklist of labels.

    Returns:
        dict[str, Polygon]: Map of label names to Polygon objects.
    """
    reader = CvatReader(xml_source)
    polygons: dict[str, Polygon] = {}

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

    # Convert CvatPolygon to Polygon (using MplPath)
    for cvat_poly in cvat_polygons:
        label = cvat_poly.label or "unnamed"
        path = cvat_poly.to_mpl_path()
        if path:
            polygons[label] = Polygon(name=label, path=path)

    return polygons


def filter_dataframe_by_polygons(
    df: pd.DataFrame,
    polygons: dict[str, Polygon] | Polygon | None,
    x_col: str = "x",
    y_col: str = "y",
    polygon_names: Iterable[str] | None = None,
    invert: bool = False,
    return_mask: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    """
    Filter a DIC dataframe keeping only points inside (or outside if invert=True)
    one or more Polygon objects produced by read_spatial_priors_from_cvat().

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing point coordinates.
    polygons : dict[str, Polygon] | Polygon | None
        Polygons dictionary (name -> Polygon) or a single Polygon instance.
        If None the dataframe is returned unchanged (or inverted if invert=True).
    x_col, y_col : str
        Column names in df with x and y coordinates.
    polygon_names : iterable of str, optional
        If `polygons` is a dict, restrict to the given polygon keys (union).
        If None use all polygons in the dict.
    invert : bool
        If True return points outside the selected polygons.
    return_mask : bool
        If True return tuple (filtered_df, boolean_mask) where mask is aligned with df.

    Returns
    -------
    pandas.DataFrame
        Filtered dataframe (and mask if return_mask=True).
    """
    if polygons is None:
        mask = np.ones(len(df), dtype=bool)
        if invert:
            mask = ~mask
        out_df = df[mask]
        return (out_df, mask) if return_mask else out_df

    # extract coordinate arrays (handle possible missing columns)
    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(
            f"Coordinates columns '{x_col}' and/or '{y_col}' not found in dataframe"
        )

    x_arr = df[x_col].to_numpy()
    y_arr = df[y_col].to_numpy()

    # build combined mask
    combined_mask = np.zeros_like(x_arr, dtype=bool)

    if isinstance(polygons, Polygon):
        combined_mask = polygons.contains_points(x_arr, y_arr)
    elif isinstance(polygons, dict):
        keys = list(polygons.keys())
        if polygon_names is not None:
            # ensure provided names exist
            sel = [k for k in polygon_names]
            missing = [k for k in sel if k not in polygons]
            if missing:
                raise KeyError(f"Requested polygon_names not found: {missing}")
            keys = sel
        for k in keys:
            poly = polygons[k]
            combined_mask |= poly.contains_points(x_arr, y_arr)
    else:
        raise TypeError("polygons must be a dict[str, Polygon], a Polygon, or None")

    if invert:
        combined_mask = ~combined_mask

    filtered_df = df[combined_mask]

    return (filtered_df, combined_mask) if return_mask else filtered_df


## -- Mask -- ##


def read_mask_element_from_cvat(
    xml_source: str | Path,
    image_name: str | None = None,
    exclude_labels: Sequence[str] | None = None,
) -> list[dict]:
    """
    Parse <mask> annotations from a CVAT export and return a list of mask info dicts.

    - If image_name is provided, only masks for that image are returned.
    - exclude_labels: optional sequence of label names to ignore.
    - masks are sorted by z_order to preserve annotation stacking order.

    Each returned dict contains: 'label', 'z', 'rle', 'left', 'top', 'width', 'height', 'attrs'.
    """
    exclude = set(exclude_labels or ())
    reader = CvatReader(xml_source)

    # If no specific image requested, use first image
    if image_name is None:
        images = reader.get_images()
        if not images:
            return []
        image_name = images[0].name

    # Get all masks for the specified image
    cvat_masks = reader.get_masks(image_name=image_name)

    # Sort by z_order
    cvat_masks.sort(key=lambda m: m.z_order)

    masks: list[dict] = []
    for cvat_mask in cvat_masks:
        if cvat_mask.label in exclude:
            logger.debug("Skipping excluded mask label: %s", cvat_mask.label)
            continue

        mask_info = {
            "label": cvat_mask.label,
            "z": cvat_mask.z_order,
            "rle": cvat_mask.rle,
            "left": cvat_mask.left,
            "top": cvat_mask.top,
            "width": cvat_mask.width,
            "height": cvat_mask.height,
            "attrs": cvat_mask.attributes,
            "occluded": cvat_mask.occluded,
        }
        masks.append(mask_info)

    return masks
