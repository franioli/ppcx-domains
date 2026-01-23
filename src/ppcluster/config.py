from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("ppcx")


load_dotenv(find_dotenv(usecwd=True), override=False)

# -----------------------------------------------------------------------------
# Default Configuration Definition
# -----------------------------------------------------------------------------
_DEFAULT_CONFIG = {
    "data": {
        "output_dir": "output",
        "reference_date": None,
        "camera_name": "PPCX_Tele",
        "days_before_to_include": 0,
        "days_after_to_include": 0,
        "dt_min": None,
        "dt_max": None,
        "roi_path": "",
        "sector_prior_file": None,
        "variables_names": ["V"],
    },
    "preprocessing": {
        "subsample_factor": 1,
        "subsample_method": "regular",
        "filter_kwargs": {
            "min_velocity": None,
            "filter_outliers": False,
            "tails_percentile": 0.01,
            "apply_2d_median": False,
            "median_window_size": 5,
            "median_threshold_factor": 3.0,
            "apply_2d_gaussian": False,
            "gaussian_sigma": 1.0,
            "apply_lamma_filter": False,
            "lamma_method": "Neighbours",
            "lamma_k": 4,
        },
    },
    "priors": {
        "probability": None,
        "fade_method": "constant",
        "fade_options": {
            "idw": {"power": 2},
            "linear": {"max_distance": 100},
            "exponential": {"decay_rate": 0.001},
        },
    },
    "mcmc": {
        "sample_options": {
            "target_accept": 0.9,
            "draws": 2000,
            "tune": 1000,
            "chains": 4,
            "cores": 4,
            "random_seed": 8927,
        },
        "model_options": {
            "mu_params": {"mu": 0, "sigma": 1},
            "sigma_params": {"sigma": 1},
        },
        "velocity_transform": None,
        "transform_params": {},
        "mrf_regularization": True,
        "mrf_kwargs": {"n_neighbors": 8, "length_scale": 50, "beta": 2.0, "n_iter": 5},
        "second_pass": "short",
        "second_pass_sample_args": {
            "draws": 500,
            "tune": 300,
            "chains": 4,
            "cores": 4,
            "target_accept": 0.9,
        },
    },
    "multiscale": {
        "sigma_values": [0],
        "aggregation": {
            "similarity_threshold": 0.7,
            "overall_threshold": 0.8,
        },
    },
    "postprocessing": {
        "split_disconnected_components": True,
        "erosion_iterations": 0,
        "dilation_iterations": 0,
        "min_cluster_size": 50,
        "connectivity": 8,
        "keep_only_largest_n": -1,
        "sector_assignment": {
            "method": "y_position",  # We use the centroid Y position to order sectors from bottom to top (A=lowest Y)
            "ascending": False,  # The y axis is inverted in image coordinates (0 at top), hence ascending=False
            "sector_colors": {
                "A": "#b3140b",
                "B": "#ee9c21",
                "C": "#f1ee30",
                "D": "#5fb61c",
            },
        },
        "vectorization": {
            "method": "smoothify",
            "buffer_distance": 2.0,
            "simplify_tolerance": 0.0,
        },
    },
    "random_seed": 8927,
    "database": {
        "host": "localhost",
        "port": 5432,
        "name": "planpincieux",
        "user": "postgres",
        "password": "password",
    },
    "api": {
        "host": "localhost",
        "port": 8080,
        "image_view": "images",
    },
    # Interpolated strings for convenience
    "db_url": "postgresql://${database.user}:${database.password}@${database.host}:${database.port}/${database.name}",
    "api_url": "http://${api.host}:${api.port}",
}


def _find_config_path() -> Path:
    """Find config.yaml via env var or by searching common locations."""
    env_path = os.getenv("PPCX_CONFIG")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.exists():
            return path

    candidates = ["config.yaml", "config.yml"]
    search_dirs = [Path.cwd()]

    here = Path(__file__).resolve()
    search_dirs.extend([here.parent, *here.parents])

    for base in search_dirs:
        for name in candidates:
            p = base / name
            if p.exists():
                return p.resolve()

    # Fallback to CWD even if missing
    return (Path.cwd() / "config.yaml").resolve()


def load_config(config_path: Path | str | None = None) -> DictConfig:
    """
    Load configuration: Defaults -> File -> Env Vars.
    Returns an OmegaConf DictConfig object.
    """
    # 1. Start with defaults
    cfg = OmegaConf.create(_DEFAULT_CONFIG)

    # 2. Merge from file
    config_path = Path(config_path) if config_path else _find_config_path()

    if config_path.exists():
        file_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(cfg, file_cfg)
        logger.debug(f"Loaded config from {config_path}")
    else:
        logger.debug(f"Config file {config_path} not found. Using defaults.")

    # 3. Merge from Environment Variables (Specific Overrides)
    # Map Env Var Name -> Config Key (dot notation)
    env_overrides = {
        "DB_HOST": "database.host",
        "DB_PORT": "database.port",
        "DB_NAME": "database.name",
        "DB_USER": "database.user",
        "DB_PASSWORD": "database.password",
        "APP_HOST": "api.host",
        "APP_PORT": "api.port",
        "GET_IMAGE_VIEW": "api.image_view",
    }

    for env_key, cfg_key in env_overrides.items():
        val = os.getenv(env_key)
        if val is not None:
            OmegaConf.update(cfg, cfg_key, val)

    # Ensure we return a DictConfig (OmegaConf.load can return a ListConfig for top-level lists)
    if not isinstance(cfg, DictConfig):
        raise TypeError(
            f"Loaded configuration must be a DictConfig, got {type(cfg).__name__!r}"
        )

    return cfg
