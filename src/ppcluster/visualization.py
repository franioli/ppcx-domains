import logging
from pathlib import Path
from typing import Literal

import cmcrameri.cm as cmc
import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PIL import Image as PILImage
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("ppcx")

# === DIC === #


def plot_dic_vectors(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    magnitudes: np.ndarray,
    background_image: np.ndarray | PILImage.Image | None = None,
    vmin: float = 0.0,
    vmax: float | None = None,
    scale: float | None = None,
    scale_units: str = "xy",
    width: float = 0.003,
    headwidth: float = 2.5,
    quiver_alpha: float = 1,
    image_alpha: float = 0.7,
    cmap_name: str = "batlow",
    figsize: tuple[int, int] = (12, 10),
    dpi: int = 300,
    ax: Axes | None = None,
    fig: Figure | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes, object] | None:
    """Base function to plot DIC displacement vectors using numpy arrays.

    Args:
        x: X coordinates (seed points)
        y: Y coordinates (seed points)
        u: X displacement components
        v: Y displacement components
        magnitudes: Displacement magnitudes
        background_image: Optional background image array
        vmin: Minimum value for color normalization
        vmax: Maximum value for color normalization
        scale: Quiver scale parameter
        scale_units: Units for quiver scaling
        width: Width of the quiver arrows
        headwidth: Headwidth for quiver arrows
        alpha: Alpha transparency for quiver arrows
        cmap_name: Name of colormap
        figsize: Figure size as (width, height)
        dpi: Dots per inch for the figure
        ax: Optional matplotlib Axes to plot on
        fig: Optional matplotlib Figure to use
        title: Optional plot title

    Returns:
        Tuple of (figure, axes, quiver_object)

    Raises:
        ValueError: If input arrays have different lengths or are empty
    """
    # Input validation
    arrays = [x, y, u, v, magnitudes]
    if not all(len(arr) == len(arrays[0]) for arr in arrays):
        raise ValueError("All input arrays must have the same length")

    if len(x) == 0:
        raise ValueError("Input arrays cannot be empty")

    # Handle colormap selection
    cmap = None
    if hasattr(cmc, cmap_name):
        cmap = getattr(cmc, cmap_name)
    elif cmap_name in plt.colormaps():
        cmap = plt.colormaps.get_cmap(cmap_name)
    else:
        print(f"Colormap '{cmap_name}' not found. Falling back to 'viridis'.")
        cmap = plt.colormaps.get_cmap("viridis")

    # Set up color normalization
    max_magnitude = vmax if vmax is not None else np.max(magnitudes)
    norm = Normalize(vmin=vmin, vmax=max_magnitude)

    # Set up figure and axes
    if ax is not None:
        target_ax: Axes = ax
        target_fig: Figure = fig if fig is not None else ax.figure  # type: ignore
    else:
        new_fig, new_ax = plt.subplots(figsize=figsize, dpi=dpi)
        target_fig, target_ax = new_fig, new_ax

    # Display background image if provided
    if background_image is not None:
        target_ax.imshow(background_image, alpha=image_alpha)
    else:
        # If no background image reverse the y axis to match image coordinates
        target_ax.invert_yaxis()

    # Create quiver plot
    q = target_ax.quiver(
        x,
        y,
        u,
        v,
        magnitudes,
        scale=scale,
        scale_units=scale_units,
        angles="xy",
        cmap=cmap,
        norm=norm,
        width=width,
        headwidth=headwidth,
        alpha=quiver_alpha,
    )

    # Add colorbar
    cbar = target_fig.colorbar(q, ax=target_ax)
    cbar.set_label("Displacement Magnitude (pixels)")

    # Set title and labels
    if title:
        target_ax.set_title(title)

    # Disable axis grid and labels
    target_ax.grid(False)
    target_ax.set_xlabel("")
    target_ax.set_ylabel("")
    target_ax.set_xticks([])
    target_ax.set_yticks([])
    target_ax.set_aspect("equal")

    return target_fig, target_ax, q


def plot_dic_scatter(
    x: np.ndarray,
    y: np.ndarray,
    magnitudes: np.ndarray,
    background_image: np.ndarray | None = None,
    vmin: float = 0.0,
    vmax: float | None = None,
    cmap_name: str = "batlow",
    s: float = 20,
    alpha: float = 0.8,
    figsize: tuple[int, int] = (12, 10),
    dpi: int = 300,
    ax: Axes | None = None,
    fig: Figure | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes, object]:
    """Plot DIC displacement data as a scatter plot colored by magnitude.

    Args:
        x: X coordinates (seed points)
        y: Y coordinates (seed points)
        magnitudes: Displacement magnitudes
        background_image: Optional background image array
        vmin: Minimum value for color normalization
        vmax: Maximum value for color normalization
        cmap_name: Name of colormap
        s: Size of scatter points
        alpha: Alpha transparency
        figsize: Figure size as (width, height)
        dpi: Dots per inch for the figure
        ax: Optional matplotlib Axes to plot on
        fig: Optional matplotlib Figure to use
        title: Optional plot title

    Returns:
        Tuple of (figure, axes, scatter_object)
    """
    # Input validation
    if len(x) != len(y) or len(x) != len(magnitudes):
        raise ValueError("All input arrays must have the same length")

    if len(x) == 0:
        raise ValueError("Input arrays cannot be empty")

    # Handle colormap selection
    cmap = None
    if hasattr(cmc, cmap_name):
        cmap = getattr(cmc, cmap_name)
    elif cmap_name in plt.colormaps():
        cmap = plt.colormaps.get_cmap(cmap_name)
    else:
        cmap = plt.colormaps.get_cmap("viridis")

    # Set up figure and axes
    if ax is not None:
        ax = ax
        fig = fig if fig is not None else ax.figure
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Display background image if provided
    if background_image is not None:
        ax.imshow(background_image, alpha=0.7)

    # Create scatter plot
    scatter = ax.scatter(
        x,
        y,
        c=magnitudes,
        cmap=cmap,
        s=s,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax or np.max(magnitudes),
    )

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Displacement Magnitude (pixels)")

    # Set title and labels
    if title:
        ax.set_title(title)
    else:
        ax.set_title("DIC Displacement Magnitude")

    # Disable axis grid and labels
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    return fig, ax, scatter


def visualize_dic_dataframe(
    df: pd.DataFrame,
    plot_type: str = "quiver",
    background_image: PILImage.Image | np.ndarray | None = None,
    output_dir: str | Path | None = None,
    filename: str | None = None,
    vmin: float = 0.0,
    vmax: float | None = None,
    scale: float | None = None,
    scale_units: str = "xy",
    width: float = 0.003,
    headwidth: float = 2.5,
    alpha: float = 0.8,
    cmap_name: str = "batlow",
    show: bool = False,
    figsize: tuple[int, int] = (12, 10),
    dpi: int = 300,
    ax: Axes | None = None,
    fig: Figure | None = None,
    **kwargs,
) -> tuple[Figure, Axes, object] | None:
    """Visualize DIC displacement data from a pandas DataFrame.

    Args:
        df: DataFrame containing DIC displacement data with original column names
        plot_type: Type of plot ("quiver" or "scatter")
        background_image: Optional background image
        output_dir: Directory to save plots
        filename: Custom filename for saved plot
        vmin: Minimum value for color normalization
        vmax: Maximum value for color normalization
        scale: Quiver scale parameter (quiver plots only)
        scale_units: Units for quiver scaling (quiver plots only)
        width: Width of quiver arrows (quiver plots only)
        headwidth: Headwidth for quiver arrows (quiver plots only)
        alpha: Alpha transparency
        cmap_name: Name of colormap
        show: If True, show the plot interactively
        figsize: Figure size as (width, height)
        dpi: Dots per inch for the figure
        ax: Optional matplotlib Axes to plot on
        fig: Optional matplotlib Figure to use
        **kwargs: Additional keyword arguments for plot functions

    Returns:
        If ax is provided, returns (fig, ax, plot_obj). Otherwise returns None.

    Raises:
        ValueError: If DataFrame is empty or missing required columns
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    # Define column mapping
    columns_to_extract = {
        "seed_x_px": "x",
        "seed_y_px": "y",
        "displacement_x_px": "u",
        "displacement_y_px": "v",
        "displacement_magnitude_px": "V",
    }

    # Check for required columns
    required_columns = list(columns_to_extract.keys())
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")

    # Extract and rename columns
    plot_df = df[required_columns].rename(columns=columns_to_extract)

    # Convert to numpy arrays
    x = plot_df["x"].values
    y = plot_df["y"].values
    u = plot_df["u"].values
    v = plot_df["v"].values
    magnitudes = plot_df["V"].values

    # Handle background image
    bg_array = None
    if background_image is not None:
        if isinstance(background_image, PILImage.Image):
            bg_array = np.array(background_image)
        else:
            bg_array = background_image

    # Generate title from timestamp if available
    title = None
    if "master_timestamp" in df.columns and not df["master_timestamp"].isnull().all():
        timestamp = df["master_timestamp"].iloc[0]
        if isinstance(timestamp, str):
            title = f"DIC Displacement - {timestamp}"
        elif hasattr(timestamp, "strftime"):
            title = f"DIC Displacement - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

    # Create plot based on type
    if plot_type.lower() == "quiver":
        result = plot_dic_vectors(
            x=x,
            y=y,
            u=u,
            v=v,
            magnitudes=magnitudes,
            background_image=bg_array,
            vmin=vmin,
            vmax=vmax,
            scale=scale,
            scale_units=scale_units,
            width=width,
            headwidth=headwidth,
            alpha=alpha,
            cmap_name=cmap_name,
            figsize=figsize,
            dpi=dpi,
            ax=ax,
            fig=fig,
            title=title,
            **kwargs,
        )
    elif plot_type.lower() == "scatter":
        result = plot_dic_scatter(
            x,
            y,
            magnitudes,
            background_image=bg_array,
            vmin=vmin,
            vmax=vmax,
            cmap_name=cmap_name,
            alpha=alpha,
            figsize=figsize,
            dpi=dpi,
            ax=ax,
            fig=fig,
            title=title,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Use 'quiver' or 'scatter'")

    # Handle output if no axes were provided
    if ax is None:
        fig, ax, plot_obj = result

        # Generate filename if not provided
        if output_dir and not filename:
            if (
                "master_timestamp" in df.columns
                and not df["master_timestamp"].isnull().all()
            ):
                timestamp = df["master_timestamp"].iloc[0]
                if isinstance(timestamp, str):
                    safe_time_str = timestamp.replace(":", "-").replace(" ", "_")
                elif hasattr(timestamp, "strftime"):
                    safe_time_str = timestamp.strftime("%Y%m%d_%H%M%S")
                else:
                    safe_time_str = str(timestamp)
                filename = f"dic_{plot_type}_{safe_time_str}"
            else:
                filename = f"dic_{plot_type}"

        # Save or show plot
        _save_or_show_plot(fig, output_dir, filename, show, dpi)
        return None
    else:
        return result


def visualize_uv_plt(df, ax=None, **kwargs):
    """
    Visualize the u-v scatter plot with optional background image.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    V = df["V"].values if "V" in df else np.sqrt(df["u"] ** 2 + df["v"] ** 2)
    scatter = ax.scatter(
        df["u"], df["v"], s=1, c=V, alpha=0.6, cmap="viridis", **kwargs
    )
    ax.set_xlabel("u (displacement in x direction)")
    ax.set_ylabel("v (displacement in y direction)")
    ax.set_title("Displacement Vectors (u-v Scatter Plot)")
    plt.colorbar(scatter, ax=ax)
    ax.set_aspect("equal", adjustable="box")


def visualize_pca(df, columns_to_extract=None, normalize=False):
    "visualize the enhanced features using PCA for dimensionality reduction"
    from sklearn.decomposition import PCA

    # Prepare data for PCA
    if columns_to_extract is None:
        columns_to_extract = ["x", "y", "u", "v", "V", "angle_deg"]

    # Ensure all required columns are present in the DataFrame
    missing_columns = set(columns_to_extract) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in DataFrame: {missing_columns}")

    if normalize:
        scaler = StandardScaler()
        data = scaler.fit_transform(df[columns_to_extract])
    else:
        data = df[columns_to_extract].values

    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    df_reduced = pd.DataFrame(reduced_data, columns=["PC1", "PC2"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df_reduced["PC1"], df_reduced["PC2"], s=1, alpha=0.6)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA Reduced Features Scatter Plot")
    ax.set_aspect("equal", adjustable="box")


# === Sectors === #


def get_sector_colors(
    sector_labels: list[str], colormap: str = "tab10"
) -> dict[str, str]:
    """
    Generate a dictionary mapping sector labels to colors.

    Args:
        sector_labels (list[str]): List of sector label strings to assign colors to.
        default_colormap (str, optional): Name of the matplotlib colormap to use if no custom colors are provided. Defaults to "tab10".

    Returns:
        dict[str, str]: Dictionary mapping each sector label to a hex color string.
    """
    cmap = plt.get_cmap(colormap)
    unique_labels = sorted(set(sector_labels))

    return {
        label: mcolors.to_hex(cmap(i % cmap.N)) for i, label in enumerate(unique_labels)
    }


def plot_sectors(
    sectors: gpd.GeoDataFrame,
    img: np.ndarray | None = None,
    velocity_df: pd.DataFrame | gpd.GeoDataFrame | None = None,
    velocity_mode: Literal["quiver", "scatter"] = "quiver",
    sector_colors: dict | None = None,
    velocity_cmap: str | None = None,
    min_cbar: float | None = None,
    max_cbar: float | None = None,
    min_cbar_percentile: float | None = None,
    max_cbar_percentile: float | None = None,
    label_column: str = "sector",
    add_sector_labels: bool = False,
    title: str = "Kinematic Sectors",
    ax: Axes | None = None,
    img_kwargs: dict | None = None,
    quiver_kwargs: dict | None = None,
    scatter_kwargs: dict | None = None,
    sector_kwargs: dict | None = None,
    sector_fill_kwargs: dict | None = None,
    sector_edge_kwargs: dict | None = None,
    label_kwargs: dict | None = None,
) -> Axes | None:
    """
    Plot velocity field and overlay sector geometries on a given axis.

    Args:
        sectors: GeoDataFrame with sector geometries and labels.
        img: 2D array for background image.
        velocity_df: DataFrame/Geodataframe with velocity vectors (columns: x, y, u, v, V).
        velocity_mode: 'quiver' or 'scatter' for velocity plotting.
        sector_colors: Dict mapping sector labels to colors.
        velocity_cmap: Colormap for velocity field, ignored if velocity_df is None (default 'Blues').
        min_cbar_percentile: Minimum percentile for velocity colorbar scaling.
        max_cbar_percentile: Maximum percentile for velocity colorbar scaling.
        label_column: Column in sectors GeoDataFrame for labeling.
        add_sector_labels: Whether to add text labels at sector centroids.
        title: Title of the plot.
        ax: Matplotlib Axes to plot on. If None, a new figure and axes are created.
        img_kwargs: Keyword arguments for ax.imshow()
        quiver_kwargs: Keyword arguments for ax.quiver()
        scatter_kwargs: Keyword arguments for ax.scatter()
        sector_kwargs: General keyword arguments for sector plotting (applied to both fill and edge if specialized kwargs not provided)
        sector_fill_kwargs: Specific keyword arguments for sector fill
        sector_edge_kwargs: Specific keyword arguments for sector edges
        label_kwargs: Keyword arguments for ax.text() labels

    Returns:
        The matplotlib Axes object with the plot.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if sector_colors is None:
        sector_colors = {}

    if sectors.empty:
        logger.warning("No sectors to plot.")
        return ax
    else:
        plt_gdf = sectors.copy()

    # Plot background image if provided
    if img is not None:
        # Default defaults
        img_defaults = {"cmap": "gray"}
        if img_kwargs:
            img_defaults.update(img_kwargs)
        ax.imshow(img, **img_defaults)

    # Plot Velocity field if provided
    if velocity_df is not None and not velocity_df.empty:
        X = velocity_df["x"].to_numpy()
        Y = velocity_df["y"].to_numpy()
        mags = velocity_df["V"].to_numpy()

        # Define default color normalization
        vmin = float(np.min(mags))
        vmax = float(np.max(mags))
        if vmin == vmax:
            vmin = 0.0  # avoid zero-range

        # Try to get vmin/vmax from quiver_kwargs if present
        if quiver_kwargs is not None:
            vmin = quiver_kwargs.pop("vmin", None)
            vmax = quiver_kwargs.pop("vmax", None)

        # If min/max_cbar or min/max_cbar_percentile are provided, they take precedence
        if min_cbar is not None and max_cbar is not None:
            vmin, vmax = float(min_cbar), float(max_cbar)
        elif min_cbar_percentile is not None and max_cbar_percentile is not None:
            vmin = float(np.percentile(mags, min_cbar_percentile))
            vmax = float(np.percentile(mags, max_cbar_percentile))

        norm = Normalize(vmin=vmin, vmax=vmax)

        if velocity_mode == "quiver":
            # Override defaults args with user kwargs
            q_defaults = {
                "scale": None,
                "scale_units": "xy",
                "angles": "xy",
                "width": 0.006,
                "headwidth": 2.0,
                "norm": norm,
                "cmap": velocity_cmap,
            }
            if quiver_kwargs:
                q_defaults.update(quiver_kwargs)

            # Note: X, Y, U, V, C are positional for quiver usually, but we pass C (mags) as 5th arg.
            q = ax.quiver(
                X,
                Y,
                velocity_df["u"].to_numpy(),
                velocity_df["v"].to_numpy(),
                mags,
                **q_defaults,
            )

        elif velocity_mode == "scatter":
            # Override defaults args with user kwargs
            s_defaults = {
                "c": mags,
                "cmap": velocity_cmap,
                "norm": norm,
                "s": 10,
                "edgecolors": "none",
            }
            if scatter_kwargs:
                s_defaults.update(scatter_kwargs)

            q = ax.scatter(X, Y, **s_defaults)
        else:
            logger.warning(
                f"Unknown velocity_mode '{velocity_mode}'. Skipping velocity plot."
            )
            q = None
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        # Colorbar
        if q is not None:
            cbar = plt.colorbar(q, ax=ax, fraction=0.046, pad=0.03)
            cbar.set_label("Velocity [px/day]", rotation=270, labelpad=12, fontsize=8)
            cbar.ax.tick_params(labelsize=7)
            ax.set_title("Velocity Field", fontsize=11)

    # Plot Sectors Overlay
    if label_column not in plt_gdf.columns:
        logger.warning(
            f"Label column '{label_column}' not found in sectors GeoDataFrame."
        )
        return ax

    present_labels = sorted(plt_gdf[label_column].unique())
    colors = {}
    fallback_cmap = plt.get_cmap("tab10")
    for i, label in enumerate(present_labels):
        if label in sector_colors:
            colors[label] = sector_colors[label]
        else:
            colors[label] = mcolors.to_hex(fallback_cmap(i % 10))
    # Assign colors to sectors
    plt_gdf["color"] = plt_gdf[label_column].map(colors)

    # --- Plot Fill (transparent) ---
    # Merge default styles < sector_kwargs < sector_fill_kwargs
    fill_styles = {
        "color": plt_gdf["color"],
        "alpha": 0.1,
        "linewidth": 0,
        "aspect": None,
    }
    if sector_kwargs:
        fill_styles.update(sector_kwargs)
    if sector_fill_kwargs:
        fill_styles.update(sector_fill_kwargs)
    plt_gdf.plot(ax=ax, **fill_styles)

    # --- Plot Edges (opaque) ---
    # Merge default styles < sector_kwargs < sector_edge_kwargs
    edge_styles = {
        "facecolor": "none",
        "edgecolor": plt_gdf["color"],
        "linewidth": 2.5,
        "alpha": 1.0,
        "aspect": None,
    }
    if sector_kwargs:
        edge_styles.update(sector_kwargs)
    if sector_edge_kwargs:
        edge_styles.update(sector_edge_kwargs)

    plt_gdf.plot(ax=ax, **edge_styles)

    # Manual Legend
    legend_patches = [
        mpatches.Patch(color=colors[label], label=label, alpha=0.8)
        for label in present_labels
        if label in colors
    ]
    if legend_patches:
        ax.legend(
            handles=legend_patches,
            loc="upper right",
            fontsize=8,
            framealpha=0.9,
        )

    # Labels on centroids
    if add_sector_labels:
        lbl_defaults = {
            "fontsize": 12,
            "weight": "bold",
            "color": "white",
            "ha": "center",
            "va": "center",
        }
        if label_kwargs:
            lbl_defaults.update(label_kwargs)

        for _, row in plt_gdf.iterrows():
            cent = row.geometry.centroid
            ax.text(
                cent.x,
                cent.y,
                row[label_column],
                **lbl_defaults,
            )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")

    return ax


def render_sector_stats_table(
    ax: Axes,
    sector_stats: pd.DataFrame,
    max_rows: int = 12,
) -> Axes:
    """
    Render a formatted statistics table on a given axis.

    Args:
        ax: Matplotlib Axes to render the table on.
        sector_stats: DataFrame containing sector statistics.
        max_rows: Maximum number of rows to display.

    Returns:
        The matplotlib Axes object with the table.
    """
    stat_cols = [
        "sector",
        "v_mean",
        "v_std",
        "v_median",
        "v_mad",
        "n_points",
        "area_px2",
        "compactness",
    ]

    available = [c for c in stat_cols if c in sector_stats.columns]
    if "sector" not in available:
        logger.warning("sector_stats has no 'sector' column; skipping table.")
        display_df = pd.DataFrame()
    else:
        display_df = sector_stats[available].copy()

    ax.axis("off")
    ax.set_title("Sector Statistics", fontsize=11, pad=6)

    if display_df.empty:
        return ax

    # Formatting
    for c in display_df.columns:
        if c == "sector":
            continue
        if c in {"n_points", "area_px2"}:
            display_df[c] = display_df[c].round(0).astype(int)
        else:
            display_df[c] = display_df[c].round(2)

    # Limit rows
    if display_df.shape[0] > max_rows:
        display_df = display_df.iloc[:max_rows, :]

    table_df = display_df.set_index("sector").T
    table = ax.table(
        cellText=table_df.values,
        colLabels=list(table_df.columns),
        rowLabels=list(table_df.index),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.05, 1.6)

    # Style table headers
    for (i, j), cell in table.get_celld().items():
        if i == 0 or j == -1:
            cell.set_facecolor("#E8E8E8")
            cell.set_text_props(weight="bold", size=7)
        else:
            cell.set_facecolor("white")

    return ax


def plot_sectors_summary(
    sectors: gpd.GeoDataFrame,
    points_by_sector: pd.DataFrame | gpd.GeoDataFrame,
    img: np.ndarray,
    colors: dict[str, str],
    output_dir: Path | str | None,
    base_name: str = "sectors_summary",
    unit: str = "px",
    figsize: tuple = (20, 10),
    dpi: int = 300,
    save_svg: bool = False,
    img_kwargs: dict | None = None,
    quiver_kwargs: dict | None = None,
    scatter_kwargs: dict | None = None,
    sector_kwargs: dict | None = None,
    sector_fill_kwargs: dict | None = None,
    sector_edge_kwargs: dict | None = None,
    label_kwargs: dict | None = None,
) -> tuple[Figure, dict[str, Axes]]:
    """
    Plot kinematic sectors summary with velocity field and statistics table.

    Args:
        sectors: GeoDataFrame with sector geometries and labels.
        points_by_sector: DataFrame or GeoDataFrame with velocity data and sector labels.
        img: Background image as a numpy array.
        colors: Dictionary mapping sector labels to colors.
        output_dir: Output directory for saving the figure.
        base_name: Base name for the output file.
        unit: Unit for velocity and area.
        figsize: Figure size.
        dpi: Dots per inch for saved figure.
        save_svg: Whether to also save as SVG.
        img_kwargs: Keyword arguments for image plotting.
        quiver_kwargs: Keyword arguments for quiver plotting.
        scatter_kwargs: Keyword arguments for scatter plotting.
        sector_kwargs: General sector plotting kwargs.
        sector_fill_kwargs: Fill kwargs for sectors.
        sector_edge_kwargs: Edge kwargs for sectors.
        label_kwargs: Label kwargs for sector labels.

    Returns:
        Tuple of (Figure, dict[str, Axes]) for the summary figure.
    """

    # Ensure colors is a standard dictionary (Seaborn can fail with OmegaConf DictConfig)
    if hasattr(colors, "to_container"):
        colors = colors.to_container()
    colors = dict(colors)

    # Handle missing keys for sectors present in the data
    unique_sectors = sorted(points_by_sector["sector"].unique())
    cmap = plt.get_cmap("tab10")
    for i, sec in enumerate(unique_sectors):
        if sec not in colors:
            colors[sec] = plt.colors.to_hex(cmap(i % 10))

    # Sort sectors and points_by_sector by sector for consistent plotting
    sectors = sectors.sort_values(by="sector").copy()
    pts = points_by_sector.sort_values(by="sector").copy()

    # --- Figure Layout ---
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        3,
        2,
        width_ratios=[1.3, 1],
        height_ratios=[1, 1, 0.6],
        figure=fig,
        hspace=0.35,
        wspace=0.15,
    )

    # 1. Left Panel: Map (spans all rows)
    ax_map = fig.add_subplot(gs[:, 0])
    plot_sectors(
        sectors=sectors,
        img=img,
        velocity_df=pts,
        sector_colors=colors,
        add_sector_labels=True,
        title="Kinematic Sectors",
        ax=ax_map,
        img_kwargs=img_kwargs,
        quiver_kwargs=quiver_kwargs,
        scatter_kwargs=scatter_kwargs,
        sector_kwargs=sector_kwargs,
        sector_fill_kwargs=sector_fill_kwargs,
        sector_edge_kwargs=sector_edge_kwargs,
        label_kwargs=label_kwargs,
    )
    # If no image is passed, invert the y axis to keep the correct orientation of the sectors
    if img is None:
        ax_map.invert_yaxis()

    # 2. Right Panel Top: Boxplot of Velocities
    ax_box = fig.add_subplot(gs[0, 1])
    sns.boxplot(
        data=pts,
        x="sector",
        y="V",
        palette=colors,
        ax=ax_box,
        hue="sector",
        showfliers=True,  # Show outliers
        fliersize=2,  # Make outlier markers smaller
    )
    ax_box.set_title("Velocity Distribution per Sector", weight="bold")
    ax_box.set_ylabel(f"Velocity [{unit}/day]")
    ax_box.set_xlabel("")
    ax_box.legend([], [], frameon=False)  # Hide legend if hue creates one

    # 3. Right Panel Middle: Histogram of Velocities
    ax_hist = fig.add_subplot(gs[1, 1])
    sns.histplot(
        data=pts,
        x="V",
        hue="sector",
        palette=colors,
        element="step",
        stat="density",
        common_norm=False,
        ax=ax_hist,
        alpha=0.3,
    )
    # Dynamically limit Y-axis to avoid squashing the boxes
    # We show up to the 98th percentile + 20% padding
    all_velocities = pts["V"].dropna()
    if not all_velocities.empty:
        y_limit = np.percentile(all_velocities, 98) * 1.2
        ax_box.set_ylim(0, y_limit)

    ax_hist.set_title("Velocity Histogram (Density)", weight="bold")
    ax_hist.set_xlabel(f"Velocity [{unit}/day]")
    if ax_box.get_legend() is not None:
        ax_box.get_legend().remove()

    # 4. Right Panel Bottom: Text Statistics Table
    ax_stats = fig.add_subplot(gs[2, 1])
    ax_stats.axis("off")
    ax_stats.set_title("Statistics Summary", weight="bold", pad=10)

    # Prepare table data
    header = ["Sector", "Median V", "NMAD", f"Area [{unit}²]", "N points"]
    relevant_sectors = sorted(pts["sector"].unique())
    stats_data = []
    for sec in relevant_sectors:
        row = sectors[sectors["sector"] == sec]
        if row.empty:
            continue
        row = row.iloc[0]
        nmad = row.get("v_nmad", np.nan)
        stats_data.append(
            [
                sec,
                f"{row['v_median']:.2f}",
                f"{nmad:.2f}",
                f"{int(row['area_px2']):,}",
                f"{int(row['n_points'])}",
            ]
        )

    # Render table
    if stats_data:
        table = ax_stats.table(
            cellText=stats_data,
            colLabels=header,
            loc="center",
            cellLoc="center",
            colWidths=[0.15, 0.2, 0.2, 0.25, 0.2],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.8)

        # Style headers and cells
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor("#e0e0e0")
                cell.set_text_props(weight="bold")
            else:
                # Color the sector letter cell with the sector color
                if j == 0:
                    sec_label = stats_data[i - 1][0]
                    c = colors.get(sec_label, "#ffffff")
                    cell.set_facecolor(c)
                    cell.set_text_props(weight="bold", color="black")
                    cell.set_alpha(0.6)

    if output_dir is not None:
        output_dir = Path(output_dir)
        out_path = output_dir / f"{base_name}_kinematic_sectors_summary.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        if save_svg:
            fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")

        plt.close(fig)
        logger.info(f"Saved summary figure to {out_path}")

    # Return the figure and the all the axes in case the caller wants to further customize or save it.
    axes = {
        "map": ax_map,
        "box": ax_box,
        "hist": ax_hist,
        "stats": ax_stats,
    }

    return fig, axes


# === Helper functions for specific use cases ===#


def _save_or_show_plot(
    fig: Figure,
    output_dir: str | Path | None = None,
    filename: str | None = None,
    show: bool = False,
    dpi: int = 300,
    close_after_save: bool = True,
) -> None:
    """Save plot to file or show it interactively.

    Args:
        fig: Matplotlib figure to save/show
        output_dir: Directory to save plots
        filename: Filename (without extension)
        show: If True, show the plot interactively
        dpi: Dots per inch for saved figure
        close_after_save: If True, close figure after saving
    """
    if output_dir and filename:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / f"{filename}.png"
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
        if close_after_save:
            plt.close(fig)
    elif show:
        plt.show()
    else:
        if close_after_save:
            plt.close(fig)


# Convenience functions for specific use cases
def plot_dic_from_arrays(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    magnitudes: np.ndarray,
    background_image: np.ndarray | None = None,
    plot_type: str = "quiver",
    **kwargs,
) -> tuple[Figure, Axes, object]:
    """Convenience function to plot DIC data from numpy arrays."""
    if plot_type.lower() == "quiver":
        return plot_dic_vectors(x, y, u, v, magnitudes, background_image, **kwargs)
    elif plot_type.lower() == "scatter":
        return plot_dic_scatter(x, y, magnitudes, background_image, **kwargs)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")


def plot_dic_from_dict(
    data_dict: dict,
    background_image: np.ndarray | None = None,
    plot_type: str = "quiver",
    **kwargs,
) -> tuple[Figure, Axes, object]:
    """Convenience function to plot DIC data from a dictionary with keys x, y, u, v, V."""
    required_keys = (
        ["x", "y", "u", "v", "V"] if plot_type == "quiver" else ["x", "y", "V"]
    )
    missing_keys = set(required_keys) - set(data_dict.keys())
    if missing_keys:
        raise ValueError(f"Dictionary missing required keys: {missing_keys}")

    if plot_type.lower() == "quiver":
        return plot_dic_vectors(
            data_dict["x"],
            data_dict["y"],
            data_dict["u"],
            data_dict["v"],
            data_dict["V"],
            background_image,
            **kwargs,
        )
    elif plot_type.lower() == "scatter":
        return plot_dic_scatter(
            data_dict["x"], data_dict["y"], data_dict["V"], background_image, **kwargs
        )
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")
