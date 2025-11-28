"""
Analyze time series of clustering results across multiple dates.

Reads clustering outputs from multiple date folders and creates:
1. Time series plots of velocity statistics per sector
2. Multi-panel visualization of sector evolution over time
"""

import argparse
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps as cm
from matplotlib import patches as mpatches
from matplotlib.colors import Normalize

from ppcluster import logger
from ppcluster.config import ConfigManager
from ppcluster.mksectors import draw_polygon


def find_result_folders(base_dir: Path, pattern: str = r".*_\d{4}-\d{2}-\d{2}_mcmc$"):
    """
    Find all result folders matching the pattern.

    Args:
        base_dir: Base directory to search
        pattern: Regex pattern for folder names (default matches CAMERA_YYYY-MM-DD_mcmc)

    Returns:
        List of (date, folder_path) tuples sorted by date
    """
    folders = []
    date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})")

    for folder in base_dir.iterdir():
        if not folder.is_dir():
            continue
        if re.match(pattern, folder.name):
            # Extract date from folder name
            match = date_pattern.search(folder.name)
            if match:
                date_str = match.group(1)
                folders.append((date_str, folder))

    # Sort by date
    folders.sort(key=lambda x: x[0])
    return folders


def load_sector_results(folder: Path, base_name: str | None = None):
    """
    Load sector results from a single date folder.

    Args:
        folder: Path to result folder
        base_name: Optional base name pattern (auto-detected if None)

    Returns:
        Dictionary with sector data or None if loading fails
    """
    try:
        # Auto-detect base_name from folder if not provided
        if base_name is None:
            # Look for the bundle file
            bundle_files = list(folder.glob("*_sectors_bundle.joblib"))
            if not bundle_files:
                logger.warning(f"No bundle file found in {folder}")
                return None
            base_name = bundle_files[0].stem.replace("_sectors_bundle", "")

        # Load main results bundle
        bundle_path = folder / f"{base_name}_sectors_bundle.joblib"
        if not bundle_path.exists():
            logger.warning(f"Bundle not found: {bundle_path}")
            return None

        bundle = joblib.load(bundle_path)

        # Load stats CSV
        stats_path = folder / f"{base_name}_morphokinematic_sector_stats.csv"
        if stats_path.exists():
            stats_df = pd.read_csv(stats_path)
        else:
            logger.warning(f"Stats CSV not found: {stats_path}")
            # Reconstruct from bundle
            stats_df = pd.DataFrame(bundle.get("mk_stats", {}))

        return {
            "date": base_name,
            "folder": folder,
            "bundle": bundle,
            "stats": stats_df,
            "sector_polygons": bundle.get("sector_vertices", {}),
            "cluster_to_letter": bundle.get("cluster_to_letter", {}),
        }

    except Exception as exc:
        logger.error(f"Failed to load results from {folder}: {exc}")
        return None


def create_velocity_time_series(
    results_list: list,
    config: ConfigManager,
    output_path: Path,
):
    """
    Create time series plots of velocity statistics per sector.

    Args:
        results_list: List of result dictionaries from load_sector_results
        output_path: Path to save the figure
    """
    # Collect data for each sector
    dates = []
    sector_data = {}

    for result in results_list:
        if result is None:
            continue

        date = result["date"]
        dates.append(date)
        stats = result["stats"]

        for _, row in stats.iterrows():
            sector = row["label"]
            if sector not in sector_data:
                sector_data[sector] = {
                    "v_median": [],
                    "v_mad": [],
                    "v_mean": [],
                    "v_std": [],
                    "n_points": [],
                    "area_px2": [],
                }

            sector_data[sector]["v_median"].append(row.get("v_median", np.nan))
            sector_data[sector]["v_mad"].append(row.get("v_mad", np.nan))
            sector_data[sector]["v_mean"].append(row.get("v_mean", np.nan))
            sector_data[sector]["v_std"].append(row.get("v_std", np.nan))
            sector_data[sector]["n_points"].append(row.get("n_points", 0))
            sector_data[sector]["area_px2"].append(row.get("area_px2", 0))

    if not dates:
        logger.warning("No valid dates found for time series")
        return

    dates_dt = pd.to_datetime(dates)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Get colormap
    colors = config.get("morphokinematic").get("sector_colors", {})
    if not colors:
        # Use default colormap if no colors are specified
        cmap = plt.get_cmap("tab10")
        colors = {
            sector: cmap(i % cmap.N)
            for i, sector in enumerate(sorted(sector_data.keys()))
        }

    # Plot 1: Median velocity with MAD error bands
    ax1 = axes[0]
    for sector in sorted(sector_data.keys()):
        data = sector_data[sector]
        v_median = np.array(data["v_median"])
        v_mad = np.array(data["v_mad"])
        ax1.fill_between(
            dates_dt,
            v_median - v_mad,
            v_median + v_mad,
            color=colors[sector],
            alpha=0.2,
            label=f"Sector {sector} ±MAD",
        )
        ax1.plot(
            dates_dt,
            v_median,
            marker="o",
            markersize=4,
            linestyle="-",
            linewidth=1.5,
            label=f"Sector {sector}",
            color=colors[sector],
        )
    ax1.set_ylabel("Velocity [px/day]", fontsize=11)
    ax1.set_title("Median Velocity per Sector (±MAD)", fontsize=12, weight="bold")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Area of each sector over time
    ax2 = axes[1]
    for sector in sorted(sector_data.keys()):
        data = sector_data[sector]
        area_px2 = np.array(data["area_px2"])
        ax2.plot(
            dates_dt,
            area_px2,
            marker="s",
            markersize=4,
            linestyle="-",
            linewidth=1.5,
            label=f"Sector {sector}",
            color=colors[sector],
        )

    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylabel("Area [px²]", fontsize=11)
    ax2.set_title("Area per Sector", fontsize=12, weight="bold")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()

    # plt.show()  # For interactive debugging; remove in production

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved velocity time series to {output_path}")

    # Combined plot velocity-area for each sector separately
    fig, axes = plt.subplots(4, 1, figsize=(14, 8))
    for ax, sector in zip(axes, sorted(sector_data.keys()), strict=False):
        data = sector_data[sector]
        v_median = np.array(data["v_median"])
        v_mad = np.array(data["v_mad"])
        area = np.array(data["area_px2"])
        norm = Normalize(vmin=np.min(area), vmax=np.max(area))
        color = cm.get_cmap("Reds")
        colors_area = color(norm(area))
        ax.errorbar(
            dates_dt,
            v_median,
            yerr=v_mad,
            markersize=0,
            linestyle="-",
            linewidth=0.5,
            label=f"Sector {sector}",
            color="k",
            alpha=0.7,
        )
        ax.scatter(
            dates_dt,
            v_median,
            s=70,
            c=colors_area,
            label="Area (px²)",
            edgecolors="k",
            zorder=3,
        )
        ax.set_ylabel("Velocity [px/day]", fontsize=11)
        ax.set_title(
            f"Sector {sector}: Median Velocity with Area Indication",
            fontsize=12,
            weight="bold",
        )
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

        # force y-limits for better comparison
        ax.set_ylim([0, 20])

    fig.tight_layout()
    fig.savefig(
        output_path.parent / "velocity_area_time_series.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def create_sectors_evolution_figure(
    results_list: list,
    output_path: Path,
    config: ConfigManager,
    max_dates: int = 12,
    ncols: int = 4,
):
    """
    Create multi-panel figure showing sector evolution over time.

    Args:
        results_list: List of result dictionaries
        output_path: Path to save the figure
        config: ConfigManager instance for loading images
        max_dates: Maximum number of dates to show
        ncols: Number of columns in the subplot grid
    """
    # Limit to max_dates
    if len(results_list) > max_dates:
        step = len(results_list) // max_dates
        results_list = results_list[::step][:max_dates]

    n_dates = len(results_list)
    nrows = int(np.ceil(n_dates / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if n_dates == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Get consistent colors for sectors
    all_sectors = set()
    for result in results_list:
        if result and result["stats"] is not None:
            all_sectors.update(result["stats"]["label"].values)

    cmap = plt.get_cmap("tab10")
    colors = {sector: cmap(i % cmap.N) for i, sector in enumerate(sorted(all_sectors))}

    for idx, result in enumerate(results_list):
        ax = axes[idx]

        if result is None:
            ax.axis("off")
            continue

        # Load image
        try:
            if config:
                img = None
            else:
                img = None

            if img is not None:
                ax.imshow(img, alpha=0.5, cmap="gray")
        except Exception:
            pass

        # Draw sector polygons
        sector_polygons = result["sector_polygons"]
        legend_patches = {}

        for sector in sorted(sector_polygons.keys()):
            coords = sector_polygons[sector]
            if coords is None:
                continue

            color = colors.get(sector, "#888888")
            draw_polygon(ax, coords, color, fill_alpha=0.2, zorder=1)
            legend_patches[sector] = mpatches.Patch(
                color=color, label=f"{sector}", alpha=0.5
            )

        # Add legend only to first subplot
        if idx == 0 and legend_patches:
            ax.legend(
                handles=list(legend_patches.values()),
                labels=list(legend_patches.keys()),
                loc="upper right",
                fontsize=7,
                framealpha=0.9,
            )

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(result["date"], fontsize=9)

    # Hide unused subplots
    for idx in range(n_dates, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        "Morpho-Kinematic Sectors Evolution", fontsize=14, weight="bold", y=0.995
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved sectors evolution figure to {output_path}")


def export_statistics_table(results_list: list, output_path: Path):
    """
    Export combined statistics table for all dates.

    Args:
        results_list: List of result dictionaries
        output_path: Path to save CSV file
    """
    all_stats = []

    for result in results_list:
        if result is None or result["stats"] is None:
            continue

        stats_df = result["stats"].copy()
        stats_df["date"] = result["date"]
        all_stats.append(stats_df)

    if not all_stats:
        logger.warning("No statistics to export")
        return

    combined_df = pd.concat(all_stats, ignore_index=True)

    # Reorder columns to have date first
    cols = ["date"] + [c for c in combined_df.columns if c != "date"]
    combined_df = combined_df[cols]

    combined_df.to_csv(output_path, index=False)
    logger.info(f"Saved combined statistics to {output_path}")


def main(args: argparse.Namespace, config_path: Path | None = None):
    config = ConfigManager(config_path) if config_path else ConfigManager()

    input_dir = args.input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = args.output_dir or (input_dir / "time_series")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find result folders
    logger.info(f"Searching for result folders in {input_dir}...")
    folders = find_result_folders(input_dir, pattern=args.folder_pattern)
    logger.info(f"Found {len(folders)} result folders")

    if not folders:
        logger.error("No result folders found!")
        return

    # Load results from all dates
    logger.info("Loading results from all dates...")
    results_list = []
    for date, folder in folders:
        logger.info(f"Loading {date}...")
        result = load_sector_results(folder)
        results_list.append(result)

    valid_results = [r for r in results_list if r is not None]
    logger.info(f"Successfully loaded {len(valid_results)} / {len(folders)} results")

    if not valid_results:
        logger.error("No valid results loaded!")
        return

    # Create time series plots
    logger.info("Creating velocity time series plot...")
    create_velocity_time_series(
        valid_results,
        config=config,
        output_path=output_dir / "velocity_time_series.png",
    )

    # Create sectors evolution figure
    logger.info("Creating sectors evolution figure...")
    create_sectors_evolution_figure(
        valid_results,
        output_dir / "sectors_evolution.png",
        max_dates=args.max_dates,
    )

    # Export combined statistics table
    logger.info("Exporting combined statistics table...")
    export_statistics_table(valid_results, output_dir / "combined_statistics.csv")

    logger.info(f"Time series analysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze time series of clustering results"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Base directory containing result folders (e.g., output)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for time series plots (default: input_dir/time_series)",
    )
    parser.add_argument(
        "--folder-pattern",
        default=r".*_\d{4}-\d{2}-\d{2}_mcmc$",
        help="Regex pattern for result folder names",
    )
    parser.add_argument(
        "--max-dates",
        type=int,
        default=31,
        help="Maximum number of dates to show in evolution figure",
    )
    args = parser.parse_args()
    main(args=args)
