from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger("ppcx")

# Load .env file variables into the system environment so oc.env can find them
load_dotenv(find_dotenv(usecwd=True), override=False)

# -----------------------------------------------------------------------------
# Structured Configuration Definitions (Schemas)
# -----------------------------------------------------------------------------


@dataclass
class DataConfig:
    # Main parameters
    source: str = field(
        default="file",
        metadata={
            "help": "Data source type. Can be 'file' for loading from files or 'database' for querying from a database."
        },
    )
    reference_date: str | None = field(
        default=None,
        metadata={
            "help": "Date to process (e.g., '2017-06-15'). Can be overwritten by the --date argument in the command line."
        },
    )

    # File source parameters
    file_path: str | None = None
    subset_name: str | None = (
        None  # Subset name (e.g., "2025" or a custom string e.g., "2024_18mp") used to construct file paths and output directories.   By default the year extracted from reference_date is used as subset_name, but it can be overwritten by setting this parameter.
    )
    search_dir: str | None = (
        None  # Directory to search for files when source is "file". Should be used in conjunction with search_pattern.
    )
    search_pattern: str | None = (
        None  # Regex pattern to match files when searching in search_dir. Should include named groups for 'slave', 'master', and 'dt'.
    )
    image_dir: str | None = None  # Directory where input images are stored

    # Query parameters for database source
    camera_name: str | None = None
    days_before_to_include: int = 0
    days_after_to_include: int = 0

    # Dt-filtering parameters (applicable to both sources)
    dt_min: float | None = None  # Float to allow 0.5 hours etc
    dt_max: float | None = None

    # Prior and ROI paths (can be used for both file and database sources)
    sector_prior_path: str | None = None
    roi_path: str | None = None

    # Output directory
    base_output_dir: str = "outputs"
    run_output_subdir: str | None = None


@dataclass
class PreprocessingFilterConfig:
    min_velocity: float | None = None
    filter_outliers: bool = False
    tails_percentile: float = 0.01
    apply_2d_median: bool = False
    median_window_size: int = 5
    median_threshold_factor: float = 3.0
    apply_2d_gaussian: bool = False
    gaussian_sigma: float = 1.0
    apply_lamma_filter: bool = False
    lamma_method: str = "Neighbours"
    lamma_k: int = 4


@dataclass
class PreprocessingConfig:
    # Variables to process, weights and transorm options
    variables_names: list[str] = field(default_factory=list)
    feature_weights: list[float] | None = None
    velocity_transform: str | None = None
    transform_params: dict | None = None

    # Subsampling options
    subsample_factor: int = 1
    subsample_method: str = "regular"

    # Filtering options
    filter_kwargs: PreprocessingFilterConfig = field(
        default_factory=PreprocessingFilterConfig
    )

    # Mad filtering parameters (applicable to both sources) #TODO: Check position of these params, maybe in DataConfig?
    mean_global_mad_threshold: float | None = (
        None  # Minimum global MAD (mean of the MAD on the DIC map) threshold. DIC maps with mean MAD below this threshold will be discarded. Set to None to disable global MAD filtering.
    )

    min_ensemble_size: int | None = (
        None  # Minimum number of DIC maps (ensembles) required for a given date to be included in the analysis. Set to None to disable ensemble size filtering.
    )
    max_point_mad: float | None = (
        None  # Maximum MAD threshold for filtering. Points with MAD above this value will be removed. Set to None to disable MAD filtering.
    )


@dataclass
class PriorsConfig:
    # dictionary of probability priors for each sector, e.g. {"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3}. Mandatory option
    probability: dict = field(default_factory=dict)
    fade_method: str = "constant"
    fade_options: dict = field(default_factory=dict)


@dataclass
class GaussianMixtureModelConfig:
    mu_params: dict = field(default_factory=lambda: {"mu": 0, "sigma": 1})
    sigma_params: dict = field(default_factory=lambda: {"sigma": 1})


@dataclass
class McmcSampleOptions:
    target_accept: float = 0.9
    draws: int = 2000
    tune: int = 1000
    chains: int = 4
    cores: int = 4
    random_seed: int | None = None


@dataclass
class MrfOptions:
    n_neighbors: int = 8
    length_scale: float = 50
    beta: float = 2.0
    n_iter: int = 5


@dataclass
class McmcConfig:
    priors: PriorsConfig = field(default_factory=PriorsConfig)
    model_options: GaussianMixtureModelConfig = field(
        default_factory=GaussianMixtureModelConfig
    )
    sample_options: McmcSampleOptions = field(default_factory=McmcSampleOptions)
    mrf_regularization: bool = True
    mrf_kwargs: MrfOptions = field(default_factory=MrfOptions)
    second_pass: str = "short"
    second_pass_sample_args: McmcSampleOptions = field(
        default_factory=lambda: McmcSampleOptions(
            draws=500, tune=300, target_accept=0.9
        )
    )
    force_cpu: bool = False


@dataclass
class MultiscaleConfig:
    sigma_values: list[float] | None = None
    aggregation: dict = field(
        default_factory=lambda: {
            "similarity_threshold": 0.7,
            "overall_threshold": 0.8,
        }
    )


@dataclass
class VectorizationConfig:
    method: str = "smoothify"
    buffer_distance: float = 2.0
    simplify_tolerance: float = 0.0
    min_area_px2: float = 100000.0
    isolation_buffer: float = 30.0
    velocity_merge_threshold: float = 1.0
    force_minimum_sectors: bool = True
    target_number_of_sectors: int = 4
    fill_holes_area: float = 80000.0
    smooth_geometries: bool = True
    smooth_method: str = "smoothify"
    smooth_iterations: int = 1


@dataclass
class SectorAssignmentConfig:
    method: str = "y_position"
    ascending: bool = False
    sector_colors: dict[str, str] | None = None
    # Example of custom colors: sector_colors = field(default_factory=lambda: {
    #         "A": "#b3140b", # Red
    #         "B": "#ee9c21", # Orange
    #         "C": "#f1ee30", # Yellow
    #         "D": "#5fb61c", # Green
    # }


@dataclass
class AnomalyDetectionConfig:
    target_sector: str = "A"
    sector_buffer: float | None = (
        100.0  # Buffer distance in meters to expand sector polygons for data selection. This can help include points near the sector boundaries that may be relevant for anomaly detection.
    )

    prior_assignment_method: str = "velocity+y_coord"  # Method to assign priors for anomaly probability. Options: "velocity", "y_coord", "kmeans", or combinations like "velocity+y_coord". This will influence the initial p(anomaly) for each point in the MCMC sampling.
    prior_anomaly_probability_limits: list[float] = field(
        default_factory=lambda: [0.2, 0.8]
    )

    variables_names: list[str] = field(default_factory=lambda: ["V"])
    feature_weights: list[float] | None = None

    # MCMC options
    model_options: GaussianMixtureModelConfig = field(
        default_factory=lambda: GaussianMixtureModelConfig(
            mu_params={"mu": 0, "sigma": 2},
            sigma_params={"sigma": 1},
        )
    )
    sample_options: McmcSampleOptions = field(
        default_factory=lambda: McmcSampleOptions(
            draws=4000, tune=3000, target_accept=0.99
        )
    )
    mrf_options: MrfOptions = field(
        default_factory=lambda: MrfOptions(
            n_neighbors=8, length_scale=50, beta=3.0, n_iter=5
        )
    )
    second_pass_sample_args: McmcSampleOptions = field(
        default_factory=lambda: McmcSampleOptions(
            draws=1000, tune=500, target_accept=0.9, chains=4
        )
    )

    force_cpu: bool = False


@dataclass
class PostProcessingConfig:
    vectorization: VectorizationConfig = field(default_factory=VectorizationConfig)
    sector_assignment: SectorAssignmentConfig = field(
        default_factory=SectorAssignmentConfig
    )


@dataclass
class PlottingConfig:
    default_discrete_cmap: str = "tab10"
    default_continuous_cmap: str = "OrRd"
    quiver: dict | None = None


@dataclass
class DatabaseConfig:
    # Uses oc.env to pull from environment variables, defaults to localhost if not found
    host: str = "${oc.env:DB_HOST,localhost}"
    port: str = "${oc.env:DB_PORT,5432}"
    name: str = "${oc.env:DB_NAME,planpincieux}"
    user: str = "${oc.env:DB_USER,postgres}"
    password: str = "${oc.env:DB_PASSWORD,password}"


@dataclass
class ApiConfig:
    host: str = "${oc.env:APP_HOST,localhost}"
    port: str = "${oc.env:APP_PORT,8080}"
    image_view: str = "${oc.env:GET_IMAGE_VIEW,images}"


@dataclass
class GlobalConfig:
    """Global configuration for the entire pipeline, including all sections and computed variables."""

    # Currently not used, just a placeholder in case we want to add global-level parameters or computed variables in the future.

    force_cpu: bool = False  # Global flag to force CPU usage for MCMC sampling (specific parameters should be moved here)
    random_seed: int | None = (
        None  # Global random seed for reproducibility (should be moved here and used across all components that require randomness)
    )


@dataclass
class PipelineConfig:
    """Top Level Configuration"""

    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    priors: PriorsConfig = field(default_factory=PriorsConfig)
    mcmc: McmcConfig = field(default_factory=McmcConfig)
    multiscale: MultiscaleConfig = field(default_factory=MultiscaleConfig)
    postprocessing: PostProcessingConfig = field(default_factory=PostProcessingConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    anomaly_detection: AnomalyDetectionConfig = field(
        default_factory=AnomalyDetectionConfig
    )

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: ApiConfig = field(default_factory=ApiConfig)

    random_seed: int | None = None

    # Computed variables (Interpolations)
    db_url: str = "postgresql://${database.user}:${database.password}@${database.host}:${database.port}/${database.name}"
    api_url: str = "http://${api.host}:${api.port}"


# -----------------------------------------------------------------------------
# Configuration Loading Logic
# -----------------------------------------------------------------------------


def _find_config_path() -> Path:
    """Find config.yaml via env var or by searching common locations."""
    env_path = os.getenv("PPCX_CONFIG")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.exists():
            return path

    # Search order: CWD, CWD/config.yaml, Parent dirs
    candidates = ["config.yaml", "config.yml"]
    search_dirs = [Path.cwd()]

    here = Path(__file__).resolve()
    search_dirs.extend([here.parent, here.parent.parent])

    for base in search_dirs:
        for name in candidates:
            p = base / name
            if p.exists():
                return p.resolve()

    # If nothing found, return default location (user might create it)
    return (Path.cwd() / "config.yaml").resolve()


def load_config(config_path: Path | str | None = None) -> ListConfig | DictConfig:
    """
    Load configuration with the following precedence order (Low to High):
    1. Structured Defaults (DataClasses)
    2. Config File (YAML)
    3. Environment Variables (Explicit overrides handled by oc.env in schema)
    """

    # 1. Initialize strictly typed configuration from schema
    # This sets up the default values and expected types
    base_cfg = OmegaConf.structured(PipelineConfig)

    # 2. Merge from file
    path_to_load = Path(config_path) if config_path else _find_config_path()

    if path_to_load.exists():
        file_cfg = OmegaConf.load(path_to_load)

        # Merge file config on top of base schema
        # This will validate types (e.g. error if you pass string to int field)
        base_cfg = OmegaConf.merge(base_cfg, file_cfg)
        logger.debug(f"Loaded config from {path_to_load}")
    else:
        logger.debug(f"Config file not found at {path_to_load}. Using defaults.")

    # If sector colors are not defined, set default colors for the sectors defined in the priors section
    if base_cfg.postprocessing.sector_assignment.sector_colors is None:
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        colormap = base_cfg.plotting.default_discrete_cmap
        logger.debug(
            f"No sector colors defined in config; assigning default colors from colormap '{colormap}'."
        )

        try:
            sector_labels = list(base_cfg.mcmc.priors.probability.keys())
            cmap = plt.get_cmap(colormap)
            colors = {
                label: mcolors.to_hex(cmap(i % cmap.N))
                for i, label in enumerate(sorted(sector_labels))
            }

            base_cfg.postprocessing.sector_assignment.sector_colors = colors
        except Exception as e:
            logger.error(f"Failed to assign default sector colors: {e}")

    # Note: We no longer need the manual env_overrides loop because
    # the DataClasses use "${oc.env:VAR_NAME,default}" which OmegaConf resolves automatically.

    return base_cfg
