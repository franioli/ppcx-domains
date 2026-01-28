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
    output_dir: str = "output"
    reference_date: str | None = None
    year: str | None = None
    camera_name: str | None = "PPCX_Tele"
    days_before_to_include: int = 0
    days_after_to_include: int = 0
    dt_min: float | None = None  # Float to allow 0.5 hours etc
    dt_max: float | None = None
    roi_path: str | None = None
    sector_prior_path: str | None = None
    variables_names: list[str] = field(default_factory=lambda: ["V"])


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
    subsample_factor: int = 1
    subsample_method: str = "regular"
    filter_kwargs: PreprocessingFilterConfig = field(
        default_factory=PreprocessingFilterConfig
    )


@dataclass
class PriorFadeOptions:
    idw: dict = field(default_factory=lambda: {"power": 2})
    linear: dict = field(default_factory=lambda: {"max_distance": 100})
    exponential: dict = field(default_factory=lambda: {"decay_rate": 0.001})


@dataclass
class PriorsConfig:
    probability: dict | None = None
    fade_method: str = "constant"
    fade_options: PriorFadeOptions = field(default_factory=PriorFadeOptions)


@dataclass
class McmcSampleOptions:
    target_accept: float = 0.9
    draws: int = 2000
    tune: int = 1000
    chains: int = 4
    cores: int = 4
    random_seed: int = 8927


@dataclass
class McmcConfig:
    sample_options: McmcSampleOptions = field(default_factory=McmcSampleOptions)
    model_options: dict = field(
        default_factory=lambda: {
            "mu_params": {"mu": 0, "sigma": 1},
            "sigma_params": {"sigma": 1},
        }
    )
    velocity_transform: str | None = None
    transform_params: dict = field(default_factory=dict)
    mrf_regularization: bool = True
    mrf_kwargs: dict = field(
        default_factory=lambda: {
            "n_neighbors": 8,
            "length_scale": 50,
            "beta": 2.0,
            "n_iter": 5,
        }
    )
    second_pass: str = "short"
    second_pass_sample_args: dict = field(
        default_factory=lambda: {
            "draws": 500,
            "tune": 300,
            "chains": 4,
            "cores": 4,
            "target_accept": 0.9,
        }
    )


@dataclass
class MultiscaleConfig:
    sigma_values: list[float] = field(default_factory=lambda: [0.0])
    aggregation: dict = field(
        default_factory=lambda: {"similarity_threshold": 0.7, "overall_threshold": 0.8}
    )


@dataclass
class VectorizationConfig:
    method: str = "smoothify"
    buffer_distance: float = 2.0
    simplify_tolerance: float = 0.0
    min_area_px2: float = 100000.0
    isolation_buffer: float = 30.0
    velocity_merge_threshold: float = 1.0
    target_number_of_sectors: int = 4
    fill_holes_area: float = 80000.0
    smooth_geometries: bool = True
    smooth_method: str = "smoothify"
    smooth_iterations: int = 1


@dataclass
class SectorAssignmentConfig:
    method: str = "y_position"
    ascending: bool = False
    sector_colors: dict = field(
        default_factory=lambda: {
            "A": "#b3140b",
            "B": "#ee9c21",
            "C": "#f1ee30",
            "D": "#5fb61c",
        }
    )


@dataclass
class PostProcessingConfig:
    vectorization: VectorizationConfig = field(default_factory=VectorizationConfig)
    sector_assignment: SectorAssignmentConfig = field(
        default_factory=SectorAssignmentConfig
    )
    # Generic fields can be typed as Any or specific types if known
    split_disconnected_components: bool = True
    min_cluster_size: int = 50


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
class AppConfig:
    """Top Level Configuration"""

    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    priors: PriorsConfig = field(default_factory=PriorsConfig)
    mcmc: McmcConfig = field(default_factory=McmcConfig)
    multiscale: MultiscaleConfig = field(default_factory=MultiscaleConfig)
    postprocessing: PostProcessingConfig = field(default_factory=PostProcessingConfig)

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: ApiConfig = field(default_factory=ApiConfig)

    random_seed: int = 8927

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
    base_cfg = OmegaConf.structured(AppConfig)

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

    # Note: We no longer need the manual env_overrides loop because
    # the DataClasses use "${oc.env:VAR_NAME,default}" which OmegaConf resolves automatically.

    return base_cfg
