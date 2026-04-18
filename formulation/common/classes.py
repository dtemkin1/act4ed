from dataclasses import dataclass
from enum import IntEnum

try:
    from shapely.geometry import Point
except Exception as exc:
    raise ImportError(
        "Shapely not found. Please install it with 'pip install shapely'"
        " and ensure Java is properly configured."
    ) from exc

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

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class Student(LocationData):
    """
    a student that needs to be picked up and dropped off,
    has a school, stop, and specific needs
    """

    school: School
    stop: Stop
    requires_monitor: bool
    requires_wheelchair: bool

    def __str__(self):
        return self.name


Place = School | Depot | Stop
