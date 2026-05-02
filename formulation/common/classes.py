from dataclasses import dataclass
from enum import IntEnum
from functools import cache, cached_property
import json
import os
from random import random
from typing import NamedTuple

from dotenv import load_dotenv
import re
import censusgeocode as cg
from census import Census

from formulation.common.constants import CACHE_DIR

try:
    from shapely.geometry import Point
except Exception as exc:
    raise ImportError(
        "Shapely not found. Please install it with 'pip install shapely'"
        " and ensure Java is properly configured."
    ) from exc

load_dotenv()  # load environment variables from .env file

type NodeId = int
"""node id in OSM and service graph"""


class SchoolType(IntEnum):
    E = 0
    """elementary"""
    MS = 1
    """middle school"""
    HS = 2
    """high school"""


class BusType(IntEnum):
    C = 0
    """71 passengers, no wheelchair access"""
    BWC = 1
    """31 passengers, 2 wheelchair access"""
    B = 2
    """48 passengers, no wheelchair access"""
    WC = 3
    """4 wheelchair access"""


class Attributes(NamedTuple):
    """attributes for a student"""

    special_ed: bool
    wheelchair_user: bool


class CensusGeoData(NamedTuple):
    """census geocode data for a location"""

    statefp: str
    countyfp: str
    tract: str


class CensusTractInfo(NamedTuple):
    """demographic info for a student"""

    total_population: float

    total_language_at_home: float
    english_only_language_at_home: float

    total_vehicle_households: float
    not_car_owning_households: float


class CensusDemographics(NamedTuple):
    english_at_home: bool
    not_car_owning_household: bool


_CACHE_CENSUS_GEOCODE = CACHE_DIR / "census_geocode_cache.json"
_CACHE_CENSUS_DEMOGRAPHIC = CACHE_DIR / "census_demographics_cache.json"


@cache
def _get_census_geocode(x: float, y: float) -> CensusGeoData:
    """get census geocode for a location"""

    if not _CACHE_CENSUS_GEOCODE.is_file():
        cache_data = {}
    else:
        with open(_CACHE_CENSUS_GEOCODE, "r") as f:
            cache_data = json.load(f)

    if str(x) in cache_data and str(y) in cache_data[str(x)]:
        cached = cache_data[str(x)][str(y)]
        return CensusGeoData(
            statefp=cached["statefp"],
            countyfp=cached["countyfp"],
            tract=cached["tract"],
        )

    try:
        geo: cg.censusgeocode.GeographyResult = cg.coordinates(x, y)
    except ValueError:
        return None

    if not geo:
        return None

    statefp = geo["States"][0]["STATE"]
    countyfp = geo["Counties"][0]["COUNTY"]
    tract = geo["Census Tracts"][0]["TRACT"]

    if str(x) not in cache_data:
        cache_data[str(x)] = {}
    cache_data[str(x)][str(y)] = {
        "statefp": statefp,
        "countyfp": countyfp,
        "tract": tract,
    }

    with open(_CACHE_CENSUS_GEOCODE, "w") as f:
        json.dump(cache_data, f)

    return CensusGeoData(
        statefp=geo["States"][0]["STATE"],
        countyfp=geo["Counties"][0]["COUNTY"],
        tract=geo["Census Tracts"][0]["TRACT"],
    )


@cache
def _get_census_tract_info(state: str, county: str, tract: str) -> CensusTractInfo:
    """get demographic info for a census tract"""

    if not _CACHE_CENSUS_DEMOGRAPHIC.is_file():
        cache_data = {}
    else:
        with open(_CACHE_CENSUS_DEMOGRAPHIC, "r") as f:
            cache_data = json.load(f)

    if (
        state in cache_data
        and county in cache_data[state]
        and tract in cache_data[state][county]
    ):
        cached = cache_data[state][county][tract]
        return CensusTractInfo(
            total_population=cached["total_population"],
            total_language_at_home=cached["total_language_at_home"],
            english_only_language_at_home=cached["english_only_language_at_home"],
            total_vehicle_households=cached["total_vehicle_households"],
            not_car_owning_households=cached["not_car_owning_households"],
        )

    census_api_key = os.getenv("CENSUS_API_KEY")
    if census_api_key is None:
        raise ValueError("CENSUS_API_KEY not found in environment variables")

    c = Census(census_api_key)
    tract_data = c.acs5.state_county_tract(
        fields=[
            "B01001_001E",  # total population
            "C16001_001E",  # total language at home
            "C16001_002E",  # english only language at home
            "B08201_001E",  # total households
            "B08201_002E",  # not car owning households
        ],
        state_fips=state,
        county_fips=county,
        tract=tract,
    )

    if not tract_data:
        return None

    if state not in cache_data:
        cache_data[state] = {}
    if county not in cache_data[state]:
        cache_data[state][county] = {}
    cache_data[state][county][tract] = {
        "total_population": tract_data[0]["B01001_001E"],
        "total_language_at_home": tract_data[0]["C16001_001E"],
        "english_only_language_at_home": tract_data[0]["C16001_002E"],
        "total_vehicle_households": tract_data[0]["B08201_001E"],
        "not_car_owning_households": tract_data[0]["B08201_002E"],
    }

    with open(_CACHE_CENSUS_DEMOGRAPHIC, "w") as f:
        json.dump(cache_data, f)

    return CensusTractInfo(
        total_population=tract_data[0]["B01001_001E"],
        total_language_at_home=tract_data[0]["C16001_001E"],
        english_only_language_at_home=tract_data[0]["C16001_002E"],
        total_vehicle_households=tract_data[0]["B08201_001E"],
        not_car_owning_households=tract_data[0]["B08201_002E"],
    )


@dataclass(frozen=True)
class Base:
    """base class for all entities in the problem, just has a name for now"""

    name: str

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class LocationData(Base):
    """
    base class for entities with geographic location,
    i.e. stops, schools, depots, and students
    """

    geographic_location: Point

    @cached_property
    def census_geo(self) -> CensusGeoData | None:
        """get the census tract info for this location"""
        return _get_census_geocode(
            self.geographic_location.x, self.geographic_location.y
        )

    @cached_property
    def census_data(self) -> CensusTractInfo:
        """get relevant census data for this location"""

        geo = self.census_geo

        if geo is None:
            return None

        return _get_census_tract_info(geo.statefp, geo.countyfp, geo.tract)


@dataclass(frozen=True)
class NodeLocationData(LocationData):
    """
    base class for entities that have both geographic location
    and a corresponding node in the graph,
    i.e. stops, schools, and depots
    """

    node_id: NodeId


@dataclass(frozen=True)
class Stop(NodeLocationData):
    """
    a stop where students can be picked up,
    corresponds to a node in the graph
    """

    ...


@dataclass(frozen=True)
class Depot(NodeLocationData):
    """
    a depot where buses start and end their routes,
    corresponds to a node in the graph
    """

    ...


@dataclass(frozen=True)
class School(NodeLocationData):
    """
    a school where students can be dropped off,
    corresponds to a node in the graph
    """

    id: str
    type: SchoolType
    start_time: int
    """mins from midnight"""

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class Bus(Base):
    """
    a bus that can be used for transportation,
    has a capacity, range (in miles), and may have wheelchair access
    """

    id: str
    capacity: int
    range: float
    has_wheelchair_access: bool
    depot: Depot
    type: BusType | None = None

    @property
    def range_km(self) -> float:
        """range in kilometers"""
        return self.range * 1.60934  # convert miles to km

    @property
    def has_monitor(self) -> bool:
        return re.fullmatch(r"M\d{2}", self.name) is not None

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class Student(LocationData):
    """
    a student that needs to be picked up and dropped off,
    has a school, stop, and specific needs
    """

    id: str
    school: School
    stop: Stop
    attributes: Attributes
    grade: str | None = None

    @cached_property
    def demographics(self) -> CensusDemographics | None:
        """get demographic info for this student based on their census tract"""

        census_data = self.census_data

        if census_data is None:
            return None

        english_at_home = (
            (
                random()
                < (
                    census_data.english_only_language_at_home
                    / census_data.total_language_at_home
                )
            )
            if census_data.total_language_at_home > 0
            else False
        )
        car_owning_household = (
            random()
            < (
                1
                - (
                    census_data.not_car_owning_households
                    / census_data.total_vehicle_households
                )
            )
            if census_data.total_vehicle_households > 0
            else False
        )

        return CensusDemographics(
            english_at_home=english_at_home,
            car_owning_household=car_owning_household,
        )

    @cached_property
    def __str__(self):
        return self.name


Place = School | Depot | Stop
