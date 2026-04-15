class SectorNotFoundError(Exception):
    """Custom exception raised when the specified sector is not found in the sectors GeoDataFrame."""

    pass


class DICMapNotFoundError(Exception):
    """Custom exception raised when no suitable DIC map is found for the given reference date."""

    pass
