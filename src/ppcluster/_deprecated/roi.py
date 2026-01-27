import logging
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.widgets import PolygonSelector

logger = logging.getLogger("ppcx")


class PolygonROISelector:
    """Interactive polygon selector for defining regions of interest on images.

    This class no longer launches an interactive selector at construction.
    Use start_interactive(...) to run the GUI, or set/load a polygon programmatically
    with set_polygon(...) / load_from_file(...). Use create_mask(...) or
    contains_points(...) to obtain boolean masks compatible with DataFrame filtering.

    Attributes:
        polygon_points: List of (x,y) tuples defining the polygon vertices.
        polygon_path: Matplotlib Path object for the polygon.
        file_path: Optional path to save/load polygon points as JSON.
        image: Optional background image for interactive selection.
    """

    def __init__(
        self,
        polygon_points: list[tuple[float, float]] | None = None,
        file_path: str = "polygon_roi.json",
    ):
        """
        Initialize the PolygonROISelector.
        Args:
            polygon_points: Optional initial list of (x,y) tuples for the polygon.
            file_path: Optional path to save/load polygon points as JSON.
        """
        self.polygon_points = list(polygon_points or [])
        self.polygon_path = None
        self.file_path = file_path
        self.fig = None
        self.ax = None
        self.polygon_selector = None

        if self.polygon_points and len(self.polygon_points) >= 3:
            self.polygon_path = MplPath(self.polygon_points)

    @classmethod
    def from_file(cls, file_path):
        """Construct selector from file without launching interactive UI."""
        selector = cls(polygon_points=None, file_path=file_path)
        return selector.load_from_file(file_path)

    # --- Interactive helpers (call explicitly) ---
    def draw_interactive(self, image=None, useblit: bool = True):
        """Start interactive polygon selection. Blocks until closed.

        Args:
            image: Optional background image (2D or 3D array) to display.
            useblit: Whether to use blitting for faster drawing (default True).
        """
        if image is not None:
            self.image = image

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        if self.image is not None:
            self.ax.imshow(self.image, alpha=0.8)
        self.ax.set_title(
            f"Select ROI polygon\nClick to add points, press Enter to finish. Output path: {self.file_path}.",
            fontsize=10,
        )

        # Initialize polygon selector using internal callback
        self.polygon_selector = PolygonSelector(
            self.ax, self._on_polygon_select, useblit=useblit
        )
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        plt.show()

    def _on_polygon_select(self, verts):
        """Internal callback when polygon selector finishes or updates."""
        self.polygon_points = list(verts)
        if len(self.polygon_points) >= 3:
            self.polygon_path = MplPath(self.polygon_points)
        logger.info(f"Polygon updated: {len(self.polygon_points)} vertices")

    def _on_key_press(self, event):
        """Handle key press events for the interactive selector."""
        # finish selection
        import contextlib

        if event.key == "enter":
            if len(self.polygon_points) >= 3:
                logger.info("Polygon selection completed!")
                if self.file_path:
                    self.to_file(self.file_path)
                    logger.info(f"Polygon points saved to {self.file_path}")
                # Close the figure to end interaction
                with contextlib.suppress(Exception):
                    plt.close(self.fig)
            else:
                logger.info("Need at least 3 points to form a polygon")
        elif event.key == "escape":
            logger.info("Polygon selection cancelled")
            with contextlib.suppress(Exception):
                plt.close(self.fig)

    def filter_dataframe(self, df, x_col="x", y_col="y"):
        """Filter the dataframe to keep only points inside the polygon."""
        if self.polygon_path is None:
            logger.info("No polygon defined, returning original dataframe")
            return df
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"DataFrame must contain columns '{x_col}' and '{y_col}'")
        mask = self.create_mask(df=df, x_col=x_col, y_col=y_col)
        filtered_df = df[mask].reset_index(drop=True)
        logger.info(
            f"Filtered {len(df)} points to {len(filtered_df)} points inside polygon"
        )
        return filtered_df

    def visualize(self, df, img=None, figsize=(12, 5)):
        """Visualize polygon overlaid on data (non-interactive)."""
        visualize_polygon_filter(df, self, img, figsize=figsize)

    # --- Programmatic polygon management ---
    def set_polygon(self, points):
        """Set polygon points programmatically (list of (x,y) pairs)."""
        self.polygon_points = list(points)
        if len(self.polygon_points) >= 3:
            self.polygon_path = MplPath(self.polygon_points)
        else:
            self.polygon_path = None

    # --- Mask / filtering utilities ---
    def contains_points(self, x, y):
        """Return a boolean mask for points (x, y) inside the polygon."""
        if self.polygon_path is None:
            return np.zeros_like(x, dtype=bool)
        points = np.column_stack([x, y])
        return self.polygon_path.contains_points(points)

    def to_file(self, path):
        """Save the selector's polygon points to a JSON file."""
        import json

        if not self.polygon_points:
            logger.info("No polygon points selected to save")
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.suffix:
            path = path.with_suffix(".json")
        data = {"polygon_points": self.polygon_points}
        with open(path, "w") as f:
            json.dump(data, f)
        self.file_path = str(path)
        logger.info(f"Polygon selector saved to {path}")

    def load_from_file(self, path):
        """Load polygon points from a JSON file and set them on this selector."""
        import json

        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        polygon_points = data.get("polygon_points", [])
        self.set_polygon(polygon_points)
        self.file_path = str(path)
        return self

    def create_mask(
        self,
        points: np.ndarray | None = None,
        x_col: str | None = None,
        y_col: str | None = None,
        df=None,
    ):
        """
        Create boolean mask for a set of points.

        - points: Nx2 array of (x,y) coordinates
        - or provide a pandas DataFrame via df with columns x_col and y_col
        """
        if self.polygon_path is None:
            # If no polygon, return mask of all False (no points selected)
            if df is not None:
                return np.zeros(len(df), dtype=bool)
            if points is not None:
                return np.zeros(len(points), dtype=bool)
            return np.array([], dtype=bool)

        if df is not None:
            if x_col is None or y_col is None:
                raise ValueError(
                    "x_col and y_col must be provided when passing a DataFrame"
                )
            pts = df[[x_col, y_col]].values
            return self.polygon_path.contains_points(pts)

        if points is not None:
            pts = np.asarray(points)
            if pts.ndim != 2 or pts.shape[1] != 2:
                raise ValueError("points must be an Nx2 array")
            return self.polygon_path.contains_points(pts)

        raise ValueError("Either points or df must be provided to create_mask")


def points_inside_polygon(polygon_path, points) -> np.ndarray:
    """Check if points are inside the selected polygon."""
    if polygon_path is None:
        return np.ones(len(points), dtype=bool)  # If no polygon, all points are inside
    return polygon_path.contains_points(points)


def filter_dataframe(df, polygon_path, x_col="x", y_col="y"):
    """Filter dataframe to keep only points inside the polygon."""
    if polygon_path is None:
        logger.info("No polygon defined, returning original dataframe")
        return df
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"DataFrame must contain columns '{x_col}' and '{y_col}'")
    mask = points_inside_polygon(polygon_path, df[[x_col, y_col]].values)
    filtered_df = df[mask].reset_index(drop=True)
    logger.info(
        f"Filtered {len(df)} points to {len(filtered_df)} points inside polygon"
    )
    return filtered_df


def visualize_polygon_filter(df, polygon_selector, img=None, figsize=(12, 5)):
    """Visualize the effect of polygon filtering on DIC data."""

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot 1: Original data with polygon overlay
    if img is not None:
        ax.imshow(img, alpha=0.7)

    # Plot all vectors
    q1 = ax.quiver(
        df["x"],
        df["y"],
        df["u"],
        df["v"],
        df["V"],
        scale=None,
        scale_units="xy",
        angles="xy",
        cmap="viridis",
        width=0.003,
        alpha=0.6,
    )

    # Overlay polygon if defined
    if polygon_selector.polygon_path is not None:
        polygon_points = np.array(polygon_selector.polygon_points)
        # Close the polygon
        polygon_closed = np.vstack([polygon_points, polygon_points[0]])
        ax.plot(
            polygon_closed[:, 0],
            polygon_closed[:, 1],
            "r-",
            linewidth=2,
            label="ROI boundary",
        )
        ax.fill(polygon_closed[:, 0], polygon_closed[:, 1], "red", alpha=0.1)

    ax.set_title(f"Original Data (n={len(df)})")
    ax.set_aspect("equal")
    ax.grid(False)

    plt.tight_layout()
    plt.show()
