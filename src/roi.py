from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.widgets import PolygonSelector


class PolygonROISelector:
    """Interactive polygon selector for defining regions of interest on images."""

    def __init__(
        self,
        image=None,
        title="Select ROI polygon",
        polygon_points=None,
        file_path=None,
    ):
        self.image = image
        self.polygon_points = polygon_points or []
        self.polygon_path = None
        self.file_path = file_path

        if self.polygon_points and len(self.polygon_points) >= 3:
            self.polygon_path = MplPath(self.polygon_points)

        if image is not None and not self.polygon_points:
            # Create figure and axis
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            self.ax.imshow(image, alpha=0.8)
            self.ax.set_title(
                f"{title}\nClick to add points, press Enter to finish. Click 's' to save polygon points to a json file."
            )

            # Initialize polygon selector
            self.polygon_selector = PolygonSelector(
                self.ax, self.on_polygon_select, useblit=True
            )

            # Connect events
            self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
            plt.show()

    def on_polygon_select(self, verts, selector):
        """Callback when polygon is selected."""
        self.polygon_points = list(verts)
        if len(self.polygon_points) >= 3:
            self.polygon_path = MplPath(self.polygon_points)
            print(f"Polygon selected with {len(self.polygon_points)} vertices")

    def on_key_press(self, event):
        """Handle key press events."""
        if event.key == "enter":
            if len(self.polygon_points) >= 3:
                print("Polygon selection completed!")
                plt.close(self.fig)
            else:
                print("Need at least 3 points to form a polygon")
        elif event.key == "escape":
            print("Polygon selection cancelled")
            plt.close(self.fig)

        if event.key == "s":
            """Save the polygon points to a file."""
            if self.file_path:
                self.to_file(self.file_path)
            else:
                print("No file path specified for saving polygon points")

    def to_file(self, path):
        """Save the selector's polygon points to a JSON file."""
        import json

        if not self.polygon_points:
            print("No polygon points selected to save")
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.suffix:
            path = path.with_suffix(".json")
        data = {
            "polygon_points": self.polygon_points,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Polygon selector saved to {path}")

    @classmethod
    def from_file(cls, path, image=None, title="Select ROI polygon"):
        """Load a selector from a JSON file. Optionally attach an image for visualization."""
        import json

        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        polygon_points = data.get("polygon_points", [])
        return cls(image=image, title=title, polygon_points=polygon_points)


def points_inside_polygon(polygon_path, points) -> np.ndarray:
    """Check if points are inside the selected polygon."""
    if polygon_path is None:
        return np.ones(len(points), dtype=bool)  # If no polygon, all points are inside
    return polygon_path.contains_points(points)


def filter_dataframe(df, polygon_path, x_col="x", y_col="y"):
    """Filter dataframe to keep only points inside the polygon."""
    if polygon_path is None:
        print("No polygon defined, returning original dataframe")
        return df
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"DataFrame must contain columns '{x_col}' and '{y_col}'")
    mask = points_inside_polygon(polygon_path, df[[x_col, y_col]].values)
    filtered_df = df[mask].reset_index(drop=True)
    print(f"Filtered {len(df)} points to {len(filtered_df)} points inside polygon")
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
