import json
import os
import random
import re
from dataclasses import dataclass
from enum import IntEnum, auto
from functools import cache, cached_property
from typing import NamedTuple, TypedDict, cast

import censusgeocode as cg
from census import Census
from dotenv import load_dotenv

from formulation.common.constants import CACHE_DIR

try:
    from shapely.geometry import Point
except Exception as exc:
    raise ImportError(
        "Shapely not found. Please install it with 'pip install shapely'"
        " and ensure Java is properly configured."
    ) from exc

random.seed(42)  # set random seed for reproducibility

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


class IncomeLevel(IntEnum):
    LESS_THAN_10K = auto()
    """less than $10,000"""
    BETWEEN_10K_TO_15K = auto()
    """$10,000 to $14,999"""
    BETWEEN_15K_TO_20K = auto()
    """$15,000 to $19,999"""
    BETWEEN_20K_TO_25K = auto()
    """$20,000 to $24,999"""
    BETWEEN_25K_TO_30K = auto()
    """$25,000 to $29,999"""
    BETWEEN_30K_TO_35K = auto()
    """$30,000 to $34,999"""
    BETWEEN_35K_TO_40K = auto()
    """$35,000 to $39,999"""
    BETWEEN_40K_TO_45K = auto()
    """$40,000 to $44,999"""
    BETWEEN_45K_TO_50K = auto()
    """$45,000 to $49,999"""
    BETWEEN_50K_TO_60K = auto()
    """$50,000 to $59,999"""
    BETWEEN_60K_TO_75K = auto()
    """$60,000 to $74,999"""
    BETWEEN_75K_TO_100K = auto()
    """$75,000 to $99,999"""
    BETWEEN_100K_TO_125K = auto()
    """$100,000 to $124,999"""
    BETWEEN_125K_TO_150K = auto()
    """$125,000 to $149,999"""
    BETWEEN_150K_TO_200K = auto()
    """$150,000 to $199,999"""
    MORE_THAN_200K = auto()
    """$200,000 or more"""


class CensusTractInfo(TypedDict):
    """demographic info for a student"""

    total_population: float

    total_language_at_home: float
    english_only_language_at_home: float

    total_vehicle_households: float
    not_car_owning_households: float

    income_brackets: dict[IncomeLevel, float]
    """proportion of households in each income bracket"""


class CensusDemographics(NamedTuple):
    english_at_home: bool
    owns_car: bool
    income_level: IncomeLevel


_CACHE_CENSUS_GEOCODE = CACHE_DIR / "census_geocode_cache.json"
_CACHE_CENSUS_DEMOGRAPHIC = CACHE_DIR / "census_demographics_cache.json"


@cache
def _get_census_geocode(x: float, y: float) -> CensusGeoData | None:
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
        geo = cg.coordinates(x, y)
        geo = cast(cg.censusgeocode.GeographyResult, geo)
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
def _get_census_tract_info(
    state: str, county: str, tract: str
) -> CensusTractInfo | None:
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
        and all(
            keys in cache_data[state][county][tract]
            for keys in CensusTractInfo.__annotations__.keys()
        )
    ):
        cached = cast(CensusTractInfo, cache_data[state][county][tract])
        return CensusTractInfo(
            total_population=cached["total_population"],
            total_language_at_home=cached["total_language_at_home"],
            english_only_language_at_home=cached["english_only_language_at_home"],
            total_vehicle_households=cached["total_vehicle_households"],
            not_car_owning_households=cached["not_car_owning_households"],
            income_brackets={
                IncomeLevel(int(key)): value
                for key, value in (
                    cast(dict[str, float], cached["income_brackets"])
                ).items()
            },
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
            "B19001_001E",  # total income households
            "B19001_002E",  # less than $10,000
            "B19001_003E",  # $10,000 to $14,999
            "B19001_004E",  # $15,000 to $19,999
            "B19001_005E",  # $20,000 to $24,999
            "B19001_006E",  # $25,000 to $29,999
            "B19001_007E",  # $30,000 to $34,999
            "B19001_008E",  # $35,000 to $39,999
            "B19001_009E",  # $40,000 to $44,999
            "B19001_010E",  # $45,000 to $49,999
            "B19001_011E",  # $50,000 to $59,999
            "B19001_012E",  # $60,000 to $74,999
            "B19001_013E",  # $75,000 to $99,999
            "B19001_014E",  # $100,000 to $124,999
            "B19001_015E",  # $125,000 to $149,999
            "B19001_016E",  # $150,000 to $199,999
            "B19001_017E",  # $200,000 or more
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
        "income_brackets": {
            IncomeLevel.LESS_THAN_10K: tract_data[0]["B19001_002E"],
            IncomeLevel.BETWEEN_10K_TO_15K: tract_data[0]["B19001_003E"],
            IncomeLevel.BETWEEN_15K_TO_20K: tract_data[0]["B19001_004E"],
            IncomeLevel.BETWEEN_20K_TO_25K: tract_data[0]["B19001_005E"],
            IncomeLevel.BETWEEN_25K_TO_30K: tract_data[0]["B19001_006E"],
            IncomeLevel.BETWEEN_30K_TO_35K: tract_data[0]["B19001_007E"],
            IncomeLevel.BETWEEN_35K_TO_40K: tract_data[0]["B19001_008E"],
            IncomeLevel.BETWEEN_40K_TO_45K: tract_data[0]["B19001_009E"],
            IncomeLevel.BETWEEN_45K_TO_50K: tract_data[0]["B19001_010E"],
            IncomeLevel.BETWEEN_50K_TO_60K: tract_data[0]["B19001_011E"],
            IncomeLevel.BETWEEN_60K_TO_75K: tract_data[0]["B19001_012E"],
            IncomeLevel.BETWEEN_75K_TO_100K: tract_data[0]["B19001_013E"],
            IncomeLevel.BETWEEN_100K_TO_125K: tract_data[0]["B19001_014E"],
            IncomeLevel.BETWEEN_125K_TO_150K: tract_data[0]["B19001_015E"],
            IncomeLevel.BETWEEN_150K_TO_200K: tract_data[0]["B19001_016E"],
            IncomeLevel.MORE_THAN_200K: tract_data[0]["B19001_017E"],
        },
    }

    with open(_CACHE_CENSUS_DEMOGRAPHIC, "w") as f:
        json.dump(cache_data, f)

    return CensusTractInfo(
        total_population=tract_data[0]["B01001_001E"],
        total_language_at_home=tract_data[0]["C16001_001E"],
        english_only_language_at_home=tract_data[0]["C16001_002E"],
        total_vehicle_households=tract_data[0]["B08201_001E"],
        not_car_owning_households=tract_data[0]["B08201_002E"],
        income_brackets={
            IncomeLevel.LESS_THAN_10K: tract_data[0]["B19001_002E"],
            IncomeLevel.BETWEEN_10K_TO_15K: tract_data[0]["B19001_003E"],
            IncomeLevel.BETWEEN_15K_TO_20K: tract_data[0]["B19001_004E"],
            IncomeLevel.BETWEEN_20K_TO_25K: tract_data[0]["B19001_005E"],
            IncomeLevel.BETWEEN_25K_TO_30K: tract_data[0]["B19001_006E"],
            IncomeLevel.BETWEEN_30K_TO_35K: tract_data[0]["B19001_007E"],
            IncomeLevel.BETWEEN_35K_TO_40K: tract_data[0]["B19001_008E"],
            IncomeLevel.BETWEEN_40K_TO_45K: tract_data[0]["B19001_009E"],
            IncomeLevel.BETWEEN_45K_TO_50K: tract_data[0]["B19001_010E"],
            IncomeLevel.BETWEEN_50K_TO_60K: tract_data[0]["B19001_011E"],
            IncomeLevel.BETWEEN_60K_TO_75K: tract_data[0]["B19001_012E"],
            IncomeLevel.BETWEEN_75K_TO_100K: tract_data[0]["B19001_013E"],
            IncomeLevel.BETWEEN_100K_TO_125K: tract_data[0]["B19001_014E"],
            IncomeLevel.BETWEEN_125K_TO_150K: tract_data[0]["B19001_015E"],
            IncomeLevel.BETWEEN_150K_TO_200K: tract_data[0]["B19001_016E"],
            IncomeLevel.MORE_THAN_200K: tract_data[0]["B19001_017E"],
        },
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
    def census_data(self) -> CensusTractInfo | None:
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


@dataclass(frozen=True)
class Bus(Base):
    """
    a bus that can be used for transportation,
    has a capacity, range (in miles), and may have wheelchair access
    """

    id: str
    capacity: int
    range: float
    wheelchair_capacity: int
    depot: Depot
    type: BusType | None = None

    @property
    def range_km(self) -> float:
        """range in kilometers"""
        return self.range * 1.60934  # convert miles to km

    @property
    def has_monitor(self) -> bool:
        return re.fullmatch(r"M\d{2}", self.name) is not None


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
                random.random()
                < (
                    census_data["english_only_language_at_home"]
                    / census_data["total_language_at_home"]
                )
            )
            if census_data["total_language_at_home"] > 0
            else False
        )
        owns_car = (
            random.random()
            < (
                1
                - census_data["not_car_owning_households"]
                / census_data["total_vehicle_households"]
            )
            if census_data["total_vehicle_households"] > 0
            else False
        )

        # select income level according to percentage likelihood
        income_level = random.choices(
            population=list(census_data["income_brackets"].keys()),
            weights=list(census_data["income_brackets"].values()),
            k=1,
        )[0]

        return CensusDemographics(
            english_at_home=english_at_home,
            owns_car=owns_car,
            income_level=income_level,
        )


Place = School | Depot | Stop
