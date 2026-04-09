import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for batch processing
from matplotlib import pyplot as plt

from bollettino_functions import (  # noqa: F401
    collect_points_time_series,
    collect_statistics,
    create_sectors_evolution_mosaic,
    create_time_series_plot,
    find_result_folders,
    load_sector_results,
    plot_area_panel,
    plot_sector_evolution_boxplots,
    plot_swimmer_all_sectors,
    plot_velocity_panel,
    plot_velocity_separation_panel,
)
from ppcluster import load_config, setup_logger

logger = setup_logger(logging.INFO, name="ppcx")


# Temporary dictionary with day to discard per year (e.g. due to bad weather, data issues, etc.)
# TODO: this should be replaced by a more systematic approach (e.g. a metadata file with flags for each date)
DISCARDED_DAYS = {
    "2015": [],
    "2016": [
        "2016-06-02",
        "2016-06-14",
        "2016-07-25",
        "2016-09-14",
        "2016-09-15",
        "2016-09-20",
        "2016-09-21",
        "2016-10-01",
        "2016-10-13",
        "2016-10-14",
        "2016-10-15",
        "2016-10-16",
        "2016-10-17",
        "2016-10-18",
        "2016-10-19",
        "2016-10-20",
    ],
    "2017": [],
    "2018": [
        "2018-10-04",
        "2018-10-05",
        "2018-10-06",
        "2018-10-07",
    ],
    "2019": [
        "2019-08-08",
        "2019-08-09",
        "2019-08-10",
        "2019-08-11",
        "2019-10-02",
        "2019-10-03",
        "2019-10-04",
        "2019-10-05",
    ],
    "2020": [
        "2020-10-17",
        "2020-10-18",
    ],
    "2021": [],
    "2022": [],
    "2023": [],
    "2024_18mp": [
        "2024-07-24",
    ],
    "2024_24mp": [],
    "2025": [],
}


# ==== Default options for CLI parser ====
DEFAULT_OUTPUT_SUBDIR = "kinematic_sectors_time_series"
DEFAULT_DIR_PATTERN = r"\d{4}-\d{2}-\d{2}$"
DEFAULT_RESULTS_PATTERN = "*results.joblib"
DEFAULT_N_JOBS = 1
DEFAULT_SECTORS = ["A", "B", "C"]
DEFAULT_UNIT = "px"
DEFAULT_UNIT_SCALE = 1.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyze sectors time series (batch-friendly)."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        required=True,
        help="Base input directory containing result folders (required)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help=f"Output directory (default: <input-dir>/{DEFAULT_OUTPUT_SUBDIR})",
    )
    parser.add_argument(
        "--dir-pattern",
        type=str,
        default=DEFAULT_DIR_PATTERN,
        help=f"Regex to match result folders (default: {DEFAULT_DIR_PATTERN})",
    )
    parser.add_argument(
        "--results-pattern",
        type=str,
        default=DEFAULT_RESULTS_PATTERN,
        help=f"Glob pattern to find results bundle files (default: {DEFAULT_RESULTS_PATTERN})",
    )
    parser.add_argument(
        "--make-mosaic",
        action="store_true",
        help="Enable creation of mosaic of daily sector plots (disabled by default).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help=f"Number of parallel jobs for mosaic creation (default: {DEFAULT_N_JOBS})",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plots.",
    )
    return parser.parse_args(argv)


def main(args):
    in_dir = Path(args.input_dir)
    out_dir = (
        Path(args.output_dir) if args.output_dir else in_dir / DEFAULT_OUTPUT_SUBDIR
    )
    dir_pattern = args.dir_pattern
    results_pattern = args.results_pattern
    make_mosaic = args.make_mosaic
    n_jobs = args.jobs
    show_plots = args.show

    year = in_dir.name.split("_")[0]

    if not in_dir.exists():
        logger.error(f"Input directory not found: {in_dir}")
        return

    # Load the config file
    config = load_config()
    if not config:
        logger.error("Failed to load configuration; aborting.")
        return

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Searching for result folders in {in_dir}...")
    folders = find_result_folders(in_dir, pattern=dir_pattern)
    logger.info(f"Found {len(folders)} result folders")

    logger.info("Loading results from all dates...")

    day_to_discard = DISCARDED_DAYS.get(year, [])
    results_list = []
    loaded = 0
    skipped = 0
    failed = 0
    for folder in folders:
        # Skip date if the days is in the discarded day dictionary
        date_str = folder.name.split("_")[-1]
        if date_str in day_to_discard:
            logger.info(f"Skipping discarded date {date_str} in year {year}")
            skipped += 1
            continue

        res = load_sector_results(folder, search_pattern=results_pattern)
        if not res:
            failed += 1
            continue
        results_list.append(res)
        loaded += 1

    logger.info(
        f"Successfully loaded {loaded} results out of {len(folders)} folders (skipped: {skipped}, failed: {failed})"
    )

    # Collect and prepare data
    df_sectors = collect_statistics(results_list)
    if df_sectors.empty:
        logger.warning("No sector-level data collected; aborting plotting steps.")
    df_points_ts = collect_points_time_series(results_list)
    df_points_ts = df_points_ts.drop(
        columns=["geometry", "cluster_id", "area"], errors="ignore"
    )

    logger.info(f"Points time series shape: {df_points_ts.shape}")
    logger.debug(
        f"Columns: {df_points_ts.columns.tolist() if not df_points_ts.empty else 'no columns'}"
    )

    # Sector colors
    sector_colors = config.postprocessing.sector_assignment.sector_colors
    if not sector_colors:
        cmap = plt.get_cmap("tab10")
        sectors_list = sorted(df_sectors["sector"].unique())
        sector_colors = {s: cmap(i % cmap.N) for i, s in enumerate(sectors_list)}
    velocity_cmap = config.plotting.default_continuous_cmap or "OrRd"

    # Boxplots evolution
    try:
        _ = plot_sector_evolution_boxplots(
            df_points_ts,
            col="V",
            unit=DEFAULT_UNIT,
            sectors=DEFAULT_SECTORS,
            output_path=out_dir / "sectors_velocity_boxplot_evolution.png",
        )
    except Exception:
        logger.exception("Failed to create sector evolution boxplots")

    # Time series figure (velocity, area, separation)
    try:
        ts_out = out_dir / f"{year}_sectors_time_series.png"
        create_time_series_plot(
            df_sectors,
            df_points_ts,
            ts_out,
            unit=DEFAULT_UNIT,
            sector_colors=sector_colors,
            sector_pairs=[("A", "B")],
            rolling_days_area=3,
            rolling_days_sep=3,
            rolling_function="mean",
            show=show_plots,
        )
    except Exception:
        logger.exception("Failed to create time series plot")

    # Velocity / area panels per sector
    try:
        out_varea = out_dir / f"{year}_sectors_velocity_area_time_series.png"
        plot_sectors_velocity_area_time_series(df_sectors, out_varea)
    except Exception:
        logger.exception("Failed to create sectors velocity/area panels")

    # Swimmer plots (centroid movement)
    try:
        fig_sep = plot_swimmer_all_sectors(
            df_sectors,
            sector_colors,
            plot_sector_extent=False,
            subtract_mean=True,
            rolling_days=5,
        )
        if fig_sep is not None:
            fig_sep.savefig(out_dir / f"{year}_sectors_centroid_location.png", dpi=300)
    except Exception:
        logger.exception("Failed to create swimmer plots")

    # Mosaic plots (optional)
    if make_mosaic:
        try:
            output_path_mosaic = out_dir / f"{year}_sectors_evolution_mosaic"
            create_sectors_evolution_mosaic(
                df_sectors,
                output_path_mosaic,
                image=None,
                df_points=df_points_ts,
                max_dates_per_figure=30,
                ncols=6,
                nrows=None,
                velocity_mode="scatter",
                velocity_cmap=velocity_cmap,
                min_cbar=0.0,
                max_cbar=10.0,
                quiver_kwargs=None,
                scatter_kwargs={"s": 10, "alpha": 0.6},
                sector_kwargs=None,
                sector_fill_kwargs={"alpha": 0},
                sector_edge_kwargs={"linewidth": 5.0},
                save_svg=False,
                n_jobs=n_jobs,
            )
        except Exception:
            logger.exception("Failed to create mosaic plots")

    logger.info(f"All outputs saved to {out_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
