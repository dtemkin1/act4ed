from formulation.common.classes import SchoolType


NETWORK_TYPE = "drive"

TAU = tuple(SchoolType)
"""school types"""

MPH_TO_KM_PER_MIN = 37.282
"divide the mph value by 37.282"

# https://malegislature.gov/Laws/GeneralLaws/PartI/TitleXIV/Chapter90/Section17
BUS_SPEED_NOT_HIGHWAY: float = 40.0 / MPH_TO_KM_PER_MIN
BUS_SPEED_SCHOOL_ZONE: float = 20.0 / MPH_TO_KM_PER_MIN
