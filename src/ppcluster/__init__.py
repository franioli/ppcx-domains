__version__ = "0.1.0"

import logging

# Import modules
from ppcluster import (
    cvat,
    data,
    griddata,
    mcmc,
    sectors,
    utils,
    visualization,
)

# Import specific functions and classes
from ppcluster.config import load_config
from ppcluster.exceptions import DICMapNotFoundError, SectorNotFoundError
from ppcluster.utils.logger import get_logger, set_log_level, setup_logger
from ppcluster.utils.timer import Timer

# Create the logger but DO NOT add StreamHandlers/FileHandlers here.
# Only set a default level (INFO), but no output will happen yet.
logger = logging.getLogger("ppcx")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)

# Define __all__ to specify the namespace of the package and what is available for import when using 'from ppcluster import *'.
__all__ = [
    "cvat",
    "data",
    "griddata",
    "sectors",
    "mcmc",
    "utils",
    "visualization",
    "load_config",
    "get_logger",
    "set_log_level",
    "setup_logger",
    "Timer",
    "DICMapNotFoundError",
    "SectorNotFoundError",
]
