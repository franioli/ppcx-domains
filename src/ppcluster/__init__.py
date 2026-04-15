__version__ = "0.1.0"

import logging

# Import modules
from ppcluster import (
    cvat,  # noqa: F401
    data,  # noqa: F401
    griddata,  # noqa: F401
    mcmc,  # noqa: F401
    sectors,  # noqa: F401
    utils,  # noqa: F401
    visualization,  # noqa: F401
)

# Import specific functions and classes
from ppcluster.config import load_config  # noqa: F401
from ppcluster.exceptions import DICMapNotFoundError, SectorNotFoundError  # noqa: F401
from ppcluster.utils.logger import get_logger, set_log_level, setup_logger  # noqa: F401
from ppcluster.utils.timer import Timer  # noqa: F401

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
