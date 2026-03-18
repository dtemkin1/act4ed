from dataclasses import dataclass, field
from enum import IntEnum
from functools import cache, cached_property
from pathlib import Path

import datetime as dt
import pickle

import geopandas as gpd
from matplotlib import pyplot as plt
import networkx as nx
import osmnx as ox
import pandas as pd
import r5py


try:
    from shapely.geometry import Point
except ImportError as exc:
    raise ImportError(
        "Shapely not found. Please install it with 'pip install shapely'"
        " and ensure Java is properly configured."
    ) from exc


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

    type: SchoolType
    start_time: int  # in minutes from midnight

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class Bus(Base):
    """
    a bus that can be used for transportation,
    has a capacity, range, and may have wheelchair access
    """

    capacity: int
    range: int
    has_wheelchair_access: bool
    depot: Depot

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


type Node_Type = School | Depot | Stop


@dataclass
class ProblemData:
    """
    class to hold all problem data and perform necessary preprocessing,
    including graph construction and shortest path calculations
    """

    # inputs
    name: str
    """name of the problem instance, used for saving results and loading data"""
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
    boundary_buffer_m: int = 1000
    """buffer in meters to apply to the place boundary when constructing the graph"""
    osm_pbf_path: Path | None = None
    """path to osm pbf file, required if use_r5 is True"""
    use_r5: bool = False
    """flag to use r5py for more accurate travel time estimates, or networkx for faster shortest path calculations"""

    # post init data
    service_graph: nx.MultiDiGraph = field(init=False)
    """
    network graph with edge weights corresponding to travel times in minutes, 
    only containing nodes in N and edges corresponding to shortest paths between nodes in N. 
    uses node_id from osm
    """
    osm_graph: nx.MultiDiGraph = field(init=False)
    """road network graph without edge weights, used for shortest path calculations"""
    stops: list[Stop] = field(default_factory=list, init=False)
    """stops where students can be picked up"""
    schools: list[School] = field(default_factory=list, init=False)
    """schools students can be dropped off at"""
    depots: list[Depot] = field(default_factory=list, init=False)
    """depots where buses start and end their routes"""
    students: list[Student] = field(default_factory=list, init=False)
    """students to be picked up and dropped off"""

    @cached_property
    def all_nodes(self) -> list[Node_Type]:
        """all nodes in the problem, including stops, schools, and depots"""
        return self.stops + self.schools + self.depots

    @cached_property
    def _transportation_network(self) -> r5py.TransportNetwork:
        if not self.osm_pbf_path:
            raise ValueError("osm_pbf_path must be provided if use_r5 is True")
        return r5py.TransportNetwork(osm_pbf=self.osm_pbf_path)

    def __post_init__(self):
        self.osm_graph = self._make_osm_graph()

        self.schools = self._get_schools()
        self.depots = self._get_depots()
        self.buses = self._create_buses()
        self.stops = self._get_stops()

        self.service_graph = self._make_service_graph()

        self.students = self._create_students()

    def _make_osm_graph(self):
        # get boundary polygon (similar to analysis.ipynb)
        framingham_gdf = ox.geocode_to_gdf(self.place_name)

        # project to utm for meters-based buffering
        framingham_projected = framingham_gdf.to_crs(framingham_gdf.estimate_utm_crs())
        framingham_projected["geometry"] = framingham_projected.buffer(
            self.boundary_buffer_m
        )

        # project back to original crs for osmnx
        framingham_buffered = framingham_projected.to_crs(framingham_gdf.crs)
        framingham_buffered_poly = framingham_buffered.geometry.iloc[0]

        # download street network
        graph = ox.graph_from_polygon(
            framingham_buffered_poly, network_type=NETWORK_TYPE
        )

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
        self, start: Node_Type, end: Node_Type, weight: str = "length"
    ) -> tuple[float, list[NodeId]]:
        return nx.bidirectional_dijkstra(
            self.osm_graph,
            source=start.node_id,
            target=end.node_id,
            weight=weight,
        )

    def _make_service_graph(self):
        service_graph = nx.MultiDiGraph()

        def add_edge_if_path_exists(start: Node_Type, end: Node_Type):
            try:
                length, path = self._get_shortest_path_osm(start, end)
                service_graph.add_edge(start, end, length=length, path=path)
            except nx.NetworkXNoPath:
                pass

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
            # Schools -> Depot
            for school in self.schools:
                for stop in self.stops:
                    add_edge_if_path_exists(school, stop)
                for depot in self.depots:
                    add_edge_if_path_exists(school, depot)

        else:
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
                            start_node,
                            end_node,
                            length=entry["distance"].iloc[0],
                            path=entry["geometry"].iloc[0],
                        )

        return service_graph

    def save(self):
        """save problem data to disk for later loading and use in formulation"""
        with open(f"cache/{self.name}_problem_data.pkl", "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(name: str) -> "ProblemData":
        """load problem data from disk"""
        with open(f"cache/{name}_problem_data.pkl", "rb") as f:
            return pickle.load(f)

    def _get_schools(self) -> list[School]:
        schools_df = pd.read_csv(self.schools_path)
        return_schools = []
        for _, row in schools_df.iterrows():
            geographic_location = Point(row["lon"], row["lat"])
            nearest_node_id = self._get_nearest_node_id(geographic_location)
            start_time = dt.datetime.strptime(row["start_time"], "%H:%M").time()
            school = School(
                name=row["id"],
                node_id=nearest_node_id,
                geographic_location=geographic_location,
                type=SchoolType[row["type"].upper()],
                # mins from midnight
                start_time=start_time.hour * 60 + start_time.minute,
            )
            return_schools.append(school)
        return return_schools

    def _get_depots(self) -> list[Depot]:
        depots_df = pd.read_csv(self.depots_path)
        return_depots = []
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

    def _get_stops(self) -> list[Stop]:
        stops_df = pd.read_csv(self.stops_path)
        return_stops = []

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

    def _create_students(self) -> list[Student]:
        students_df = pd.read_csv(self.students_path)
        return_students = []

        for _, row in students_df.iterrows():
            school = next(s for s in self.schools if s.name == row["school_id"])
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

    def _create_buses(
        self,
    ) -> list[Bus]:
        buses_df = pd.read_csv(self.buses_path)
        return_buses = []
        for _, row in buses_df.iterrows():
            depot = next(d for d in self.depots if d.name == row["depot_name"])
            bus = Bus(
                name=row["id"],
                capacity=row["capacity"],
                range=row["range"],
                depot=depot,
                has_wheelchair_access=bool(row["has_wheelchair_access"]),
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
            distances: dict[NodeId, float] = nx.shortest_path_length(
                self.osm_graph, source=nearest_node, target=None, weight="length"
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
            [node for n, node in enumerate(self.osm_graph.nodes(data=True)) if n < 1],
        )
        print(
            "Edge attributes:",
            [edge for e, edge in enumerate(self.osm_graph.edges(data=True)) if e < 1],
        )

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
            with open("outputs/sanity_checks.png", "wb+") as f:
                plt.savefig(f, dpi=300)

        # note for self: make sure only one node per intersection/dead end,
        # and that there are no duplicate edges or goofy artifacts


TAU = list(SchoolType)
"""school types"""


@cache
def make_school_copy(school: School) -> School:
    return School(
        name=school.name + " (copy)",
        geographic_location=school.geographic_location,
        node_id=school.node_id,
        type=school.type,
        start_time=school.start_time,
    )


@cache
def make_depot_copy(depot: Depot, suffix: str) -> Depot:
    return Depot(
        name=depot.name + f" ({suffix})",
        geographic_location=depot.geographic_location,
        node_id=depot.node_id,
    )


@cache
def make_depot_start_copy(depot: Depot) -> Depot:
    return make_depot_copy(depot, "start")


@cache
def make_depot_end_copy(depot: Depot) -> Depot:
    return make_depot_copy(depot, "end")


# @cache
# def get_shortest_path(
#     graph: nx.MultiDiGraph, start: Location, end: Location, weight: str = "length"
# ) -> tuple[float, list[Location]]:
#     """returns the length and path of the shortest path between start and end"""
#     return nx.bidirectional_dijkstra(graph, source=start, target=end, weight=weight)


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
    """range of bus b in miles"""
    return b.range


def h_s(s: School):
    """start time of school s in minutes from midnight"""
    return s.start_time


def slack_s(s: School):
    """required slack time for school s in minutes, same in our case"""
    return 30


def l_s(s: School):
    """latest allowable arrival time at school s in minutes from midnight"""
    return h_s(s) - slack_s(s)
