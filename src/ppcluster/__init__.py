__version__ = "0.1.0"

import logging

# Import modules
from ppcluster import (
    mcmc,  # noqa: F401
    utils,  # noqa: F401
)

# Import specific functions and classes
from ppcluster.config import load_config  # noqa: F401
from ppcluster.utils.logger import get_logger, set_log_level, setup_logger  # noqa: F401

# Create the logger but DO NOT add StreamHandlers/FileHandlers here.
# Only set a default level (INFO), but no output will happen yet.
logger = logging.getLogger("ppcx")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)

__all__ = [
    "mcmc",
    "utils",
    "load_config",
    "get_logger",
    "set_log_level",
    "setup_logger",
]
