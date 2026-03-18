"""
Configuration and constants for the ES Data Pipeline.
"""

from enum import IntEnum

class DataStatus(IntEnum):
    """
    Enum for data provenance tracking.
    """
    MEASURED = 0
    INTERPOLATED = 1
    MISSING = 2

# Standardized Column Names
COL_TIME = "time"
COL_HEAT = "heat_kwh"
COL_TEMP_OUT = "temp_out"
COL_WIND_SPEED = "wind_speed"
COL_SOLAR = "solar_irradiance"
COL_STATUS = "data_status"

# Column Mapping (Raw -> Standard)
# Note: This mapping might need to be dynamic for building sensors (e.g. BHB01...)
# but global weather columns can be mapped here.
WEATHER_COL_MAPPING = {
    "Air temperature": COL_TEMP_OUT,
    "Wind speed": COL_WIND_SPEED,
    "Solar radiation": COL_SOLAR,  # Adjust based on exact source name
    "temp_c": COL_TEMP_OUT,        # For ERA5 
    "wind_speed_ms": COL_WIND_SPEED, # For ERA5
    "ssrd_W_per_m2": COL_SOLAR,      # For ERA5
    "tcc": "cloud_cover"             # Total Cloud Cover
}

# Cleaning Constants
SIGMA_CLIP_ITERS = 2
MAX_INTERPOLATE_HOURS = 6
MIN_TEMP_HEATING = 10.0 # deg C, approximate balance point for filtering
