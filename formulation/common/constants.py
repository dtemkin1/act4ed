import os
from pathlib import Path

CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

NETWORK_TYPE = "drive"

MPH_TO_KM_PER_MIN = 37.282
"divide the mph value by 37.282"

METERS_PER_KM = 1000.0
METERS_PER_MILE = 1609.344

# https://malegislature.gov/Laws/GeneralLaws/PartI/TitleXIV/Chapter90/Section17
BUS_SPEED_NOT_HIGHWAY: float = 40.0 / MPH_TO_KM_PER_MIN
BUS_SPEED_SCHOOL_ZONE: float = 20.0 / MPH_TO_KM_PER_MIN

CACHE_DIR = CURRENT_FILE_DIR / ".." / "cache"
