from dataclasses import dataclass, replace
from abc import ABC, abstractmethod
from enum import IntEnum
from functools import cache, cached_property
import os
from pathlib import Path
import warnings

import datetime as dt
import pickle
from collections.abc import Hashable

import geopandas as gpd
from matplotlib import pyplot as plt
import networkx as nx
import osmnx as ox
import pandas as pd

try:
    import r5py
except Exception:
    # cant use r5py but let run
    warnings.warn(
        "Warning: r5py not found. Please install it with 'pip install r5py'"
        " and ensure Java is properly configured.",
    )


try:
    from shapely.geometry import Point
except Exception as exc:
    raise ImportError(
        "Shapely not found. Please install it with 'pip install shapely'"
        " and ensure Java is properly configured."
    ) from exc

CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

type NodeId = int
"""node id in OSM and service graph"""


NETWORK_TYPE = "drive"


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
    geographic_location: Point


@dataclass(frozen=True)
class Stop(NodeLocationData):
    """
    a stop where students can be picked up,
    corresponds to a node in the graph
    """


@dataclass(frozen=True)
class Depot(NodeLocationData):
    """
    a depot where buses start and end their routes,
    corresponds to a node in the graph
    """


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


@dataclass(frozen=True)
class ProblemData(ABC):
    """
    class to hold all problem data and perform necessary preprocessing,
    including graph construction and shortest path calculations
    """

    name: str
    """name of the problem instance, used for saving results and loading data"""

    @property
    @abstractmethod
    def service_graph(self) -> "nx.MultiDiGraph[NodeId]":
        """
        network graph with edge weights corresponding to travel times in minutes and length in km,
        only containing nodes in N and edges corresponding to shortest paths between nodes in N.
        uses integer node_id
        """

    @property
    @abstractmethod
    def stops(self) -> list[Stop]:
        """stops where students can be picked up"""

    @property
    @abstractmethod
    def schools(self) -> list[School]:
        """schools students can be dropped off at"""

    @property
    @abstractmethod
    def depots(self) -> list[Depot]:
        """depots where buses start and end their routes"""

    @property
    @abstractmethod
    def students(self) -> list[Student]:
        """students to be picked up and dropped off"""

    @property
    @abstractmethod
    def buses(self) -> list[Bus]:
        """buses available for transportation"""

    @property
    def all_nodes(self) -> list[Place]:
        """all nodes in the problem, including stops, schools, and depots"""
        return self.stops + self.schools + self.depots


@dataclass(frozen=True)
class ProblemDataToy(ProblemData):
    """
    class to hold (provided) toy problem data and perform necessary preprocessing,
    including graph construction and shortest path calculations
    """

    base_graph: "nx.MultiDiGraph[NodeId]"

    _stops: list[Stop]
    _schools: list[School]
    _depots: list[Depot]
    _students: list[Student]
    _buses: list[Bus]

    @property
    def stops(self) -> list[Stop]:
        return self._stops

    @property
    def schools(self) -> list[School]:
        return self._schools

    @property
    def depots(self) -> list[Depot]:
        return self._depots

    @property
    def students(self) -> list[Student]:
        return self._students

    @property
    def buses(self) -> list[Bus]:
        return self._buses

    def _get_shortest_path_base(
        self, start: NodeId, end: NodeId, weight: str = "length"
    ) -> tuple[float, list[NodeId]]:
        return get_shortest_path(self.base_graph, start, end, weight)

    @cached_property
    def service_graph(self):
        service_graph: "nx.MultiDiGraph[NodeId]" = nx.MultiDiGraph()

        def add_edge_if_path_exists(start: Place, end: Place):
            # check if edge in graph already, if so skip
            start_id = start.node_id
            end_id = end.node_id

            if service_graph.has_edge(start_id, end_id):
                return

            if start_id == end_id:
                service_graph.add_edge(
                    start_id, end_id, length=0, path=[start_id, end_id]
                )
                return

            try:
                length, path = self._get_shortest_path_base(start_id, end_id)
                service_graph.add_edge(start_id, end_id, length=length, path=path)
            except nx.NetworkXNoPath:
                print(f"Warning: no path between {start} and {end} in the graph")

        # Depots -> Stops
        for depot in self.depots:
            for stop in self.stops:
                add_edge_if_path_exists(depot, stop)

        # Stops -> Stops
        # Stops -> Schools
        for stop1 in self.stops:
            for stop2 in self.stops:
                if stop1 != stop2:
                    add_edge_if_path_exists(stop1, stop2)

            for school in self.schools:
                add_edge_if_path_exists(stop1, school)

        # Schools -> Stops
        # Schools -> Schools
        # Schools -> Depot
        for school in self.schools:
            for stop in self.stops:
                add_edge_if_path_exists(school, stop)
            for other_school in self.schools:
                if school != other_school:
                    add_edge_if_path_exists(school, other_school)
            for depot in self.depots:
                add_edge_if_path_exists(school, depot)

        return service_graph


@dataclass(frozen=True)
class ProblemDataReal(ProblemData):
    """
    class to hold all real-world problem data and perform necessary preprocessing,
    including graph construction and shortest path calculations
    """

    # inputs
    schools_path: Path
    """path to schools csv file"""
    stops_path: Path
    """path to stops csv file"""
    depots_path: Path
    """path to depots csv file"""
    students_path: Path
    """path to students csv file"""
    buses_path: Path
    """path to buses csv file"""
    place_name: str
    """place name to geocode for graph construction, e.g. 'Framingham, MA'"""
    boundary_buffer_km: float = 1.0
    """buffer in kilometers to apply to the place boundary when constructing the graph"""
    osm_pbf_path: Path | None = None
    """path to osm pbf file, required if use_r5 is True"""
    use_r5: bool = False
    """flag to use r5py for more accurate travel time estimates, or networkx for faster shortest path calculations"""
    prune: int | None = None
    """flag for whether to prune the service graph based on stop distance (in kilometers)."""

    # post init data
    @cached_property
    def service_graph(self):
        return self._make_service_graph()

    @cached_property
    def osm_graph(self) -> "nx.MultiDiGraph[NodeId]":
        """road network graph without edge weights, used for shortest path calculations"""
        return self._make_osm_graph()

    @cached_property
    def stops(self) -> list[Stop]:
        return self._make_stops()

    @cached_property
    def schools(self) -> list[School]:
        return self._make_schools()

    @cached_property
    def depots(self) -> list[Depot]:
        return self._make_depots()

    @cached_property
    def students(self) -> list[Student]:
        return self._make_students()

    @cached_property
    def buses(self) -> list[Bus]:
        return self._make_buses()

    @cached_property
    def _transportation_network(self) -> "r5py.TransportNetwork":
        assert (
            r5py is not None
        ), "r5py must be installed to use r5 for service graph construction"
        if not self.osm_pbf_path:
            raise ValueError("osm_pbf_path must be provided if use_r5 is True")
        return r5py.TransportNetwork(osm_pbf=self.osm_pbf_path)

    @cached_property
    def gdf(self) -> gpd.GeoDataFrame:
        return ox.geocode_to_gdf(self.place_name)

    def _make_osm_graph(self):
        # get boundary polygon (similar to analysis.ipynb)
        gdf = self.gdf

        # project to utm for meters-based buffering
        projected = gdf.to_crs(gdf.estimate_utm_crs())
        projected["geometry"] = projected.buffer(self.boundary_buffer_km * 1000)

        # project back to original crs for osmnx
        buffered = projected.to_crs(gdf.crs)
        buffered_poly = buffered.geometry.iloc[0]

        # download street network
        graph = ox.graph_from_polygon(buffered_poly, network_type=NETWORK_TYPE)

        # simplify
        graph = ox.truncate.largest_component(graph, strongly=False)
        # G = ox.simplify_graph(G)

        # remove self-loops
        graph.remove_edges_from(list(nx.selfloop_edges(graph)))

        # save as graphml for :sparkles: later :sparkles:
        # os.makedirs(os.path.dirname(GRAPHML_FILE), exist_ok=True)
        # ox.save_graphml(G, GRAPHML_FILE)

        return graph

    def _get_shortest_path_osm(
        self, start: NodeId, end: NodeId, weight: str = "length"
    ) -> tuple[float, list[NodeId]]:
        return get_shortest_path(self.osm_graph, start, end, weight)

    def _make_service_graph(self) -> "nx.MultiDiGraph[NodeId]":
        service_graph: "nx.MultiDiGraph[NodeId]" = nx.MultiDiGraph()

        def add_edge_if_path_exists(start: Place, end: Place):
            # check if edge in graph already, if so skip
            start_id = start.node_id
            end_id = end.node_id

            if service_graph.has_edge(start_id, end_id):
                return

            if start_id == end_id:
                service_graph.add_edge(
                    start_id, end_id, length=0, path=[start_id, end_id]
                )
                return

            try:
                # NOTE: length from osm is in meters, convert to km for service graph
                length, path = self._get_shortest_path_osm(start_id, end_id)

                if self.prune and isinstance(start, Stop) and isinstance(end, Stop):
                    if length > self.prune * 1000:  # convert km to m
                        return

                service_graph.add_edge(
                    start_id, end_id, length=(length / 1000), path=path
                )
            except nx.NetworkXNoPath:
                print(f"Warning: no path between {start} and {end} in the graph")

        if not self.use_r5:
            # Depots -> Stops
            for depot in self.depots:
                for stop in self.stops:
                    add_edge_if_path_exists(depot, stop)

            # Stops -> Stops
            # Stops -> Schools
            for stop1 in self.stops:
                for stop2 in self.stops:
                    if stop1 != stop2:
                        add_edge_if_path_exists(stop1, stop2)

                for school in self.schools:
                    add_edge_if_path_exists(stop1, school)

            # Schools -> Stops
            # Schools -> Schools
            # Schools -> Depot
            for school in self.schools:
                for stop in self.stops:
                    add_edge_if_path_exists(school, stop)
                for other_school in self.schools:
                    if school != other_school:
                        add_edge_if_path_exists(school, other_school)
                for depot in self.depots:
                    add_edge_if_path_exists(school, depot)

        else:
            assert (
                r5py is not None
            ), "r5py must be installed to use r5 for service graph construction"
            if not self.osm_pbf_path:
                raise ValueError("osm_pbf_path must be provided if use_r5 is True")

            nodes_gdf = gpd.GeoDataFrame(
                {
                    "id": [node.name for node in self.all_nodes],
                    "geometry": [node.geographic_location for node in self.all_nodes],
                }
            )

            detailed_itineraries = r5py.DetailedItineraries(
                self._transportation_network,
                origins=nodes_gdf,
                destinations=nodes_gdf,
                transport_modes=[r5py.TransportMode.CAR],
                snap_to_network=True,
                force_all_to_all=True,
            )

            for start_node in self.stops:
                for end_node in self.all_nodes:
                    if start_node != end_node:
                        detailed_itineraries[
                            detailed_itineraries["from_id"] == start_node.node_id
                        ][detailed_itineraries["to_id"] == end_node.node_id]
                        entry = detailed_itineraries[
                            detailed_itineraries["from_id"] == start_node.node_id
                        ][detailed_itineraries["to_id"] == end_node.node_id]
                        service_graph.add_edge(
                            start_node.node_id,
                            end_node.node_id,
                            length=(
                                entry["distance"].iloc[0] / 1000
                            ),  # convert m to km
                            path=entry["geometry"].iloc[0],
                        )

        return service_graph

    def save(self):
        """save problem data to disk for later loading and use in formulation"""
        with open(
            CURRENT_FILE_DIR / "cache" / f"{self.name}_problem_data.pkl", "wb+"
        ) as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, name: str, prune: int | None = None) -> "ProblemDataReal":
        """load problem data from disk"""
        prob_name = f"{name}{'_' + str(prune) if prune else ''}_problem_data"
        return cls.load_path(CURRENT_FILE_DIR / "cache" / f"{prob_name}.pkl")

    @classmethod
    def load_path(cls, path: Path) -> "ProblemDataReal":
        with open(path, "rb") as f:
            return pickle.load(f)

    def _make_schools(self) -> list[School]:
        schools_df = pd.read_csv(
            self.schools_path,
            dtype={
                "id": str,
                "lon": float,
                "lat": float,
                "type": str,
                "start_time": str,
                "name": str,
            },
        )
        return_schools: list[School] = []
        for _, row in schools_df.iterrows():
            geographic_location = Point(row["lon"], row["lat"])
            nearest_node_id = self._get_nearest_node_id(geographic_location)
            start_time = dt.datetime.strptime(row["start_time"], "%H:%M").time()
            school = School(
                id=row["id"],
                name=row["name"],
                node_id=nearest_node_id,
                geographic_location=geographic_location,
                type=SchoolType[row["type"]],
                start_time=start_time.hour * 60 + start_time.minute,
            )
            return_schools.append(school)
        return return_schools

    def _make_depots(self) -> list[Depot]:
        depots_df = pd.read_csv(
            self.depots_path, dtype={"id": str, "lon": float, "lat": float}
        )
        return_depots: list[Depot] = []
        for _, row in depots_df.iterrows():
            geographic_location = Point(row["lon"], row["lat"])
            nearest_node_id = self._get_nearest_node_id(geographic_location)
            depot = Depot(
                name=row["id"],
                node_id=nearest_node_id,
                geographic_location=geographic_location,
            )
            return_depots.append(depot)
        return return_depots

    def _make_stops(self) -> list[Stop]:
        stops_df = pd.read_csv(
            self.stops_path, dtype={"id": str, "lon": float, "lat": float}
        )
        return_stops: list[Stop] = []

        for _, row in stops_df.iterrows():
            geographic_location = Point(row["lon"], row["lat"])
            nearest_node_id = self._get_nearest_node_id(geographic_location)
            stop = Stop(
                name=row["id"],
                node_id=nearest_node_id,
                geographic_location=geographic_location,
            )
            return_stops.append(stop)
        return return_stops

    def _make_students(self) -> list[Student]:
        students_df = pd.read_csv(
            self.students_path,
            dtype={
                "id": str,
                "lon": float,
                "lat": float,
                "school_id": str,
                "is_sp_ed": bool,
                "is_wheelchair_user": bool,
            },
        )
        return_students: list[Student] = []

        for _, row in students_df.iterrows():
            school = next(s for s in self.schools if s.id == row["school_id"])
            geographic_location = Point(row["lon"], row["lat"])

            # find nearest stop to student
            nearest_stop = self._get_nearest_stop(geographic_location)

            this_student = Student(
                name=f"Student {row['id']}",
                geographic_location=geographic_location,
                school=school,
                stop=nearest_stop,
                requires_monitor=bool(row["is_sp_ed"]),
                requires_wheelchair=bool(row["is_wheelchair_user"]),
            )
            return_students.append(this_student)
        return return_students

    def _make_buses(
        self,
    ) -> list[Bus]:
        buses_df = pd.read_csv(
            self.buses_path,
            dtype={
                "id": str,
                "depot_name": str,
                "capacity": int,
                "range": float,
                "has_wheelchair_access": bool,
                "type": str,
            },
        )
        return_buses: list[Bus] = []
        for _, row in buses_df.iterrows():
            depot = next(d for d in self.depots if d.name == row["depot_name"])
            bus = Bus(
                name=row["id"],
                capacity=row["capacity"],
                range=row["range"],
                depot=depot,
                has_wheelchair_access=bool(row["has_wheelchair_access"]),
                type=(
                    BusType[row["type"]] if row["type"] in BusType.__members__ else None
                ),
            )
            return_buses.append(bus)
        return return_buses

    def _get_nearest_node_id(self, geographic_location: Point) -> NodeId:
        """Get the nearest node in the graph to a given point."""
        return ox.distance.nearest_nodes(
            self.osm_graph, geographic_location.x, geographic_location.y
        )

    def _get_nearest_stop(self, geo_location: Point) -> Stop:
        if self.use_r5:
            assert (
                r5py is not None
            ), "r5py must be installed to use r5 for nearest stop calculation"
            nearest_stop_id = (
                r5py.DetailedItineraries(
                    self._transportation_network,
                    origins=gpd.GeoDataFrame(
                        {
                            "id": ["student_location"],
                            "geometry": [geo_location],
                        }
                    ),
                    destinations=gpd.GeoDataFrame(
                        {
                            "id": [stop.node_id for stop in self.stops],
                            "geometry": [
                                stop.geographic_location for stop in self.stops
                            ],
                        }
                    ),
                    snap_to_network=True,
                    transport_modes=[r5py.TransportMode.WALK],
                    force_all_to_all=False,
                )
                .sort(by="distance")
                .iloc[0]["to_id"]
            )
            return next(stop for stop in self.stops if stop.node_id == nearest_stop_id)
        else:
            # Get nearest node in the OSM graph to the student's location
            nearest_node = self._get_nearest_node_id(geo_location)
            distances = nx.single_source_dijkstra_path_length(
                self.osm_graph, source=nearest_node, weight="length"
            )
            all_stop_distances = {
                stop: distances.get(stop.node_id, float("inf")) for stop in self.stops
            }

            # find the stop with the minimum distance to the student
            nearest_stop = min(all_stop_distances.items(), key=lambda item: item[1])[0]
            return nearest_stop

    def sanity_checks(
        self, boundary_gdf: gpd.GeoDataFrame | None = None, save: bool = False
    ):
        """perform sanity checks on the transportation network."""

        # nodes v edges
        print("Number of nodes:", len(self.osm_graph.nodes))
        print("Number of edges:", len(self.osm_graph.edges))

        # degree distribution
        degrees = [deg for _, deg in self.osm_graph.degree()]
        print("Min degree:", min(degrees))
        print("Max degree:", max(degrees))
        print("Mean degree:", sum(degrees) / len(degrees))

        # attributes
        print(
            "Node attributes:",
            [data for n, (_, data) in enumerate(self.osm_graph.nodes.items()) if n < 1][
                0
            ],
        )
        print(
            "Edge attributes:",
            [
                data
                for e, (_, _, data) in enumerate(self.osm_graph.edges(data=True))
                if e < 1
            ][0],
        )

        print("# of Stops:", len(self.stops))
        print("# of Schools:", len(self.schools))
        print("# of Depots:", len(self.depots))
        print("# of Students:", len(self.students))
        print("# of Buses:", len(self.buses))

        # plot map rq with outline of city boundary
        _, ax = ox.plot_graph(
            self.osm_graph,
            node_size=5,
            edge_linewidth=0.5,
            figsize=(8, 8),
            show=False,
            close=False,
        )
        if boundary_gdf is not None:
            boundary_gdf.boundary.plot(ax=ax, color="red", linewidth=2)
        # plt.show()

        if save:
            with open(CURRENT_FILE_DIR / "outputs" / "sanity_checks.png", "wb+") as f:
                plt.savefig(f, dpi=300)

        # service graph checks
        print("Total needed nodes (stops + schools + depots):", len(self.all_nodes))
        print("Number of nodes in service graph:", len(self.service_graph.nodes))
        print("Number of edges in service graph:", len(self.service_graph.edges))

        # note for self: make sure only one node per intersection/dead end,
        # and that there are no duplicate edges or goofy artifacts


TAU = list(SchoolType)
"""school types"""


@cache
def make_place_copy[T: Place](place: T, suffix: str = "copy") -> T:
    return replace(place, name=place.name + f" ({suffix})")


@cache
def make_school_copy(school: School) -> School:
    return make_place_copy(school)


@cache
def make_depot_end_copy(depot: Depot) -> Depot:
    return make_place_copy(depot, "end copy")


@cache
def make_depot_start_copy(depot: Depot) -> Depot:
    return make_place_copy(depot, "start copy")


@cache
def get_shortest_path[T: Hashable](
    graph: "nx.MultiDiGraph[T]", start: T, end: T, weight: str = "length"
) -> tuple[float, list[T]]:
    """returns the length and path of the shortest path between start and end"""
    return nx.bidirectional_dijkstra(graph, source=start, target=end, weight=weight)


def p_m(m: Student):
    """pickup stop of student m"""
    return m.stop


def s_m(m: Student):
    """school of student m"""
    return m.school


def tau_m(m: Student):
    """type of school of student m"""
    return m.school.type


def f_m(m: Student):
    """1 if student m if flagged"""
    return 1 if m.requires_monitor or m.requires_wheelchair else 0


def depot_b(b: Bus):
    """depot of bus b"""
    return b.depot


def C_b(b: Bus):
    """capacity of bus b"""
    return b.capacity


def Wh_b(b: Bus):
    """1 if bus b has wheelchair access"""
    return 1 if b.has_wheelchair_access else 0


def R_b(b: Bus):
    """range of bus b in km"""
    return b.range_km


def h_s(s: School):
    """start time of school s in minutes from midnight"""
    return s.start_time


def slack_s(s: School):
    """required slack time for school s in minutes, same in our case"""
    return 30


def l_s(s: School):
    """latest allowable arrival time at school s in minutes from midnight"""
    return h_s(s) - slack_s(s)
