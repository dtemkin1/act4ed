import datetime as dt
import pickle
import warnings
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Optional, cast

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd

from formulation.common.classes import (Attributes, Bus, BusType, Depot,
                                        NodeId, Place, School, SchoolType,
                                        Stop, Student)
from formulation.common.constants import CACHE_DIR, NETWORK_TYPE
from formulation.common.utils import (ensure_service_graph_kilometers,
                                      get_shortest_path, meters_to_kilometers)

try:
    import r5py
except Exception:
    # cant use r5py but let run
    warnings.warn(
        "Warning: r5py not found. Please install it with 'pip install r5py'"
        " and ensure Java is properly configured.",
    )


try:
    from shapely.geometry import Point, Polygon
except Exception as exc:
    raise ImportError(
        "Shapely not found. Please install it with 'pip install shapely'"
        " and ensure Java is properly configured."
    ) from exc

try:
    from osmnx import settings

    settings.use_cache = True
    settings.cache_folder = CACHE_DIR
except Exception:
    warnings.warn(
        "Warning: osmnx cache settings could not be configured. Please ensure you have the latest version of osmnx installed."
    )


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
    def base_graph(self) -> "nx.MultiDiGraph[NodeId]":
        """
        base network, e.g. road network graph for real data or grid graph for toy data.
        length is assumed to be in meters.
        """
        ...

    @property
    @abstractmethod
    def service_graph(self) -> "nx.MultiDiGraph[NodeId]":
        """
        network graph with edge weights corresponding to travel distances in kilometers,
        only containing nodes in N and edges corresponding to shortest paths between nodes in N.
        uses integer node_id
        """
        ...

    @property
    @abstractmethod
    def stops(self) -> tuple[Stop, ...]:
        """stops where students can be picked up"""
        ...

    @property
    @abstractmethod
    def schools(self) -> tuple[School, ...]:
        """schools students can be dropped off at"""
        ...

    @property
    @abstractmethod
    def depots(self) -> tuple[Depot, ...]:
        """depots where buses start and end their routes"""
        ...

    @property
    @abstractmethod
    def students(self) -> tuple[Student, ...]:
        """students to be picked up and dropped off"""
        ...

    @property
    @abstractmethod
    def buses(self) -> tuple[Bus, ...]:
        """buses available for transportation"""
        ...

    @property
    def all_nodes(self) -> tuple[Place, ...]:
        """all nodes in the problem, including stops, schools, and depots"""
        return self.stops + self.schools + self.depots

    def get_shortest_path_base(
        self, start: NodeId, end: NodeId, weight: str = "length"
    ) -> tuple[float, list[NodeId]]:
        """get shortest path and length in meters between two nodes in the base graph"""
        return get_shortest_path(self.base_graph, start, end, weight)

    def _stop_school_types(self) -> dict[Stop, set[SchoolType]]:
        stop_school_types: dict[Stop, set[SchoolType]] = {}
        for student in self.students:
            stop_school_types.setdefault(student.stop, set()).add(student.school.type)
        return stop_school_types

    def _service_graph_pairs(self) -> tuple[tuple[Place, Place], ...]:
        pairs: list[tuple[Place, Place]] = []

        for depot in self.depots:
            for stop in self.stops:
                pairs.append((depot, stop))

        for stop1 in self.stops:
            for stop2 in self.stops:
                if stop1 != stop2:
                    pairs.append((stop1, stop2))

            for school in self.schools:
                pairs.append((stop1, school))

        for school in self.schools:
            for stop in self.stops:
                pairs.append((school, stop))
            for other_school in self.schools:
                if school != other_school:
                    pairs.append((school, other_school))
            for depot in self.depots:
                pairs.append((school, depot))

        return tuple(pairs)

    def get_shortest_paths_base(
        self, nodes: Sequence[NodeId], weight: str = "length"
    ) -> tuple[float, list[NodeId]]:
        length = 0.0
        all_path: list[NodeId] = []
        for i in range(len(nodes) - 1):
            start = nodes[i]
            end = nodes[i + 1]
            length_m, path = self.get_shortest_path_base(start, end, weight)
            length += length_m

            while all_path and path and path[0] == all_path[-1]:
                path = path[1:]
            all_path.extend(path)

        return length, all_path

    def special_ed_students_in_stop(self, stop: Stop) -> tuple[Student, ...]:
        """Return students with special educational needs who are assigned to a specific stop."""
        return tuple(
            student
            for student in self.students
            if student.attributes.special_ed and student.stop == stop
        )

    def sanity_checks(self):
        """perform sanity checks on the transportation network."""

        # nodes v edges
        print("Number of nodes:", len(self.base_graph.nodes))
        print("Number of edges:", len(self.base_graph.edges))

        # degree distribution
        degrees = [deg for _, deg in self.base_graph.degree()]
        print("Min degree:", min(degrees))
        print("Max degree:", max(degrees))
        print("Mean degree:", sum(degrees) / len(degrees))

        # attributes
        print(
            "Node attributes:",
            [
                data
                for n, (_, data) in enumerate(self.base_graph.nodes.items())
                if n < 1
            ][0],
        )
        print(
            "Edge attributes:",
            [
                data
                for e, (_, _, data) in enumerate(self.base_graph.edges(data=True))
                if e < 1
            ][0],
        )

        print("# of Stops:", len(self.stops))
        print("# of Schools:", len(self.schools))
        print("# of Depots:", len(self.depots))
        print("# of Students:", len(self.students))
        print("# of Buses:", len(self.buses))

        # plot map rq with outline of city boundary
        # _, ax = ox.plot_graph(
        #     self.base_graph,
        #     node_size=5,
        #     edge_linewidth=0.5,
        #     figsize=(8, 8),
        #     show=False,
        #     close=False,
        # )
        # if boundary_gdf is not None:
        #     boundary_gdf.boundary.plot(ax=ax, color="red", linewidth=2)
        # plt.show()

        # if save:
        #     with open(
        #         CURRENT_FILE_DIR / ".." / "outputs" / "sanity_checks.png", "wb+"
        #     ) as f:
        #         plt.savefig(f, dpi=300)

        # service graph checks
        print("Total needed nodes (stops + schools + depots):", len(self.all_nodes))
        print("Number of nodes in service graph:", len(self.service_graph.nodes))
        print("Number of edges in service graph:", len(self.service_graph.edges))

        # note for self: make sure only one node per intersection/dead end,
        # and that there are no duplicate edges or goofy artifacts

    def restricted(
        self,
        *,
        school_ids: Iterable[str | int] | None = None,
        school_types: Iterable[SchoolType | str | int] | None = None,
    ) -> "FilteredProblemData":
        if not school_ids and not school_types:
            raise ValueError("must provide at least one school id or school type")

        allowed_school_ids = None if not school_ids else set(school_ids)
        allowed_school_types = (
            None
            if not school_types
            else {_coerce_school_type(value) for value in school_types}
        )

        selected_schools = [
            school
            for school in self.schools
            if (allowed_school_ids is None or school.id in allowed_school_ids)
            and (allowed_school_types is None or school.type in allowed_school_types)
        ]
        if not selected_schools:
            raise ValueError("no schools matched the requested restriction")

        selected_school_ids = {school.id for school in selected_schools}
        selected_students = tuple(
            student
            for student in self.students
            if student.school.id in selected_school_ids
        )
        if not selected_students:
            raise ValueError("no students matched the requested school restriction")

        selected_school_ids = {student.school.id for student in selected_students}
        selected_schools = tuple(
            school for school in selected_schools if school.id in selected_school_ids
        )
        selected_stops_set = {student.stop for student in selected_students}
        selected_stops = tuple(
            stop for stop in self.stops if stop in selected_stops_set
        )

        return FilteredProblemData(
            name=f"{self.name}_{_restriction_suffix(school_ids, school_types)}",
            base_problem_data=self,
            _stops=selected_stops,
            _schools=selected_schools,
            _depots=self.depots,
            _students=selected_students,
            _buses=self.buses,
        )

    def restrict_to_school(self, school: School | str | int) -> "FilteredProblemData":
        school_id = school.id if isinstance(school, School) else school
        return self.restricted(school_ids=[school_id])

    def restrict_to_school_type(
        self,
        school_type: SchoolType | str | int,
    ) -> "FilteredProblemData":
        return self.restricted(school_types=[school_type])


def _coerce_school_type(value: SchoolType | str | int) -> SchoolType:
    if isinstance(value, SchoolType):
        return value
    if isinstance(value, str):
        return SchoolType[value]
    return SchoolType(value)


def _restriction_suffix(
    school_ids: Iterable[str | int] | None,
    school_types: Iterable[SchoolType | str | int] | None,
) -> str:
    parts: list[str] = []
    if school_ids:
        parts.append("schools_" + "-".join(str(value) for value in school_ids))
    if school_types:
        parts.append(
            "types_"
            + "-".join(_coerce_school_type(value).name for value in school_types)
        )
    return "_".join(parts)


@dataclass(frozen=True)
class FilteredProblemData(ProblemData):
    base_problem_data: ProblemData
    _stops: Optional[tuple[Stop, ...]] = None
    _schools: Optional[tuple[School, ...]] = None
    _depots: Optional[tuple[Depot, ...]] = None
    _students: Optional[tuple[Student, ...]] = None
    _buses: Optional[tuple[Bus, ...]] = None

    @cached_property
    def _service_graph_cached(self) -> "nx.MultiDiGraph[NodeId]":
        if (
            self._stops is None
            and self._schools is None
            and self._depots is None
            and self._students is None
        ):
            return self.base_problem_data.service_graph

        service_graph: "nx.MultiDiGraph[NodeId]" = nx.MultiDiGraph()
        service_graph.graph.update(self.base_problem_data.service_graph.graph)
        service_graph.graph["distance_unit"] = "km"

        stop_school_types: dict[Stop, set[SchoolType]] = {}
        for student in self.students:
            stop_school_types.setdefault(student.stop, set()).add(student.school.type)

        for start, end in self._service_graph_pairs():
            self._add_service_edge(service_graph, start, end, stop_school_types)

        ensure_service_graph_kilometers(service_graph)
        return service_graph

    def _service_graph_pairs(self) -> tuple[tuple[Place, Place], ...]:
        pairs: list[tuple[Place, Place]] = []

        for depot in self.depots:
            for stop in self.stops:
                pairs.append((depot, stop))

        for stop1 in self.stops:
            for stop2 in self.stops:
                if stop1 != stop2:
                    pairs.append((stop1, stop2))

            for school in self.schools:
                pairs.append((stop1, school))

        for school in self.schools:
            for stop in self.stops:
                pairs.append((school, stop))
            for other_school in self.schools:
                if school != other_school:
                    pairs.append((school, other_school))
            for depot in self.depots:
                pairs.append((school, depot))

        return tuple(pairs)

    def _service_edge_allowed(
        self,
        start: Place,
        end: Place,
        stop_school_types: dict[Stop, set[SchoolType]],
        length: float | None = None,
    ) -> bool:
        if isinstance(start, Stop) and isinstance(end, School):
            return end.type in stop_school_types.get(start, set())

        if isinstance(start, Stop) and isinstance(end, Stop):
            if start.node_id == end.node_id:
                return True
            if stop_school_types.get(start, set()).isdisjoint(
                stop_school_types.get(end, set())
            ):
                return False

        if start.node_id == end.node_id:
            return True

        prune = getattr(self.base_problem_data, "prune", None)
        if (
            length is not None
            and prune is not None
            and isinstance(start, Stop)
            and isinstance(end, Stop)
        ):
            return length <= prune

        return True

    def _add_service_edge(
        self,
        service_graph: "nx.MultiDiGraph[NodeId]",
        start: Place,
        end: Place,
        stop_school_types: dict[Stop, set[SchoolType]],
    ) -> None:
        start_id = start.node_id
        end_id = end.node_id

        if service_graph.has_edge(start_id, end_id):
            return

        if not self._service_edge_allowed(start, end, stop_school_types):
            return

        if start_id == end_id:
            service_graph.add_edge(
                start_id,
                end_id,
                length=0.0,
                path=[start_id, end_id],
            )
            return

        base_edge = self.base_problem_data.service_graph.get_edge_data(
            start_id,
            end_id,
            key=0,
        )
        if base_edge is not None:
            length = float(base_edge["length"])
            if not self._service_edge_allowed(
                start,
                end,
                stop_school_types,
                length=length,
            ):
                return
            service_graph.add_edge(start_id, end_id, **dict(base_edge))
            return

        try:
            length_m, path = get_shortest_path(self.base_graph, start_id, end_id)
        except (KeyError, nx.NetworkXNoPath):
            print(f"Warning: no path between {start} and {end} in the graph")
            return

        length_km = meters_to_kilometers(length_m)
        if not self._service_edge_allowed(
            start,
            end,
            stop_school_types,
            length=length_km,
        ):
            return

        service_graph.add_edge(
            start_id,
            end_id,
            length=length_km,
            path=path,
        )

    @property
    def base_graph(self) -> "nx.MultiDiGraph[NodeId]":
        return self.base_problem_data.base_graph

    @property
    def service_graph(self) -> "nx.MultiDiGraph[NodeId]":
        return self._service_graph_cached

    @property
    def stops(self) -> tuple[Stop, ...]:
        return self._stops or self.base_problem_data.stops

    @property
    def schools(self) -> tuple[School, ...]:
        return self._schools or self.base_problem_data.schools

    @property
    def depots(self) -> tuple[Depot, ...]:
        return self._depots or self.base_problem_data.depots

    @property
    def students(self) -> tuple[Student, ...]:
        return self._students or self.base_problem_data.students

    @property
    def buses(self) -> tuple[Bus, ...]:
        return self._buses or self.base_problem_data.buses

    def __getattr__(self, name: str):
        return getattr(self.base_problem_data, name)


@dataclass(frozen=True)
class ProblemDataToy(ProblemData):
    """
    class to hold (provided) toy problem data and perform necessary preprocessing,
    including graph construction and shortest path calculations
    """

    _base_graph: "nx.MultiDiGraph[NodeId]"

    _stops: tuple[Stop, ...]
    _schools: tuple[School, ...]
    _depots: tuple[Depot, ...]
    _students: tuple[Student, ...]
    _buses: tuple[Bus, ...]

    @property
    def base_graph(self):
        return self._base_graph

    @property
    def stops(self) -> tuple[Stop, ...]:
        return self._stops

    @property
    def schools(self) -> tuple[School, ...]:
        return self._schools

    @property
    def depots(self) -> tuple[Depot, ...]:
        return self._depots

    @property
    def students(self) -> tuple[Student, ...]:
        return self._students

    @property
    def buses(self) -> tuple[Bus, ...]:
        return self._buses

    @property
    def service_graph(self) -> "nx.MultiDiGraph[NodeId]":
        service_graph: "nx.MultiDiGraph[NodeId]" = nx.MultiDiGraph()
        service_graph.graph["distance_unit"] = "km"

        def add_edge_if_path_exists(start: Place, end: Place):
            # check if edge in graph already, if so skip
            start_id = start.node_id
            end_id = end.node_id

            if service_graph.has_edge(start_id, end_id):
                return

            if start_id == end_id:
                path = (start_id, end_id)
                service_graph.add_edge(start_id, end_id, length=0.0, path=path)
                return

            try:
                length, path_list = self.get_shortest_path_base(start_id, end_id)
                path = tuple(path_list)
                service_graph.add_edge(
                    start_id,
                    end_id,
                    length=meters_to_kilometers(length),
                    path=path,
                )
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
    def _service_graph_cached(self) -> "nx.MultiDiGraph[NodeId]":
        return self._make_service_graph()

    @property
    def base_graph(self) -> "nx.MultiDiGraph[NodeId]":
        return self.osm_graph

    @property
    def service_graph(self) -> "nx.MultiDiGraph[NodeId]":
        return self._service_graph_cached

    @cached_property
    def osm_graph(self) -> "nx.MultiDiGraph[NodeId]":
        """road network graph without edge weights, used for shortest path calculations"""
        return self._make_osm_graph()

    @cached_property
    def _stops_cached(self) -> tuple[Stop, ...]:
        return self._make_stops()

    @property
    def stops(self) -> tuple[Stop, ...]:
        return self._stops_cached

    @cached_property
    def _schools_cached(self) -> tuple[School, ...]:
        return self._make_schools()

    @property
    def schools(self) -> tuple[School, ...]:
        return self._schools_cached

    @cached_property
    def _depots_cached(self) -> tuple[Depot, ...]:
        return self._make_depots()

    @property
    def depots(self) -> tuple[Depot, ...]:
        return self._depots_cached

    @cached_property
    def _students_cached(self) -> tuple[Student, ...]:
        return self._make_students()

    @property
    def students(self) -> tuple[Student, ...]:
        return self._students_cached

    @cached_property
    def _buses_cached(self) -> tuple[Bus, ...]:
        return self._make_buses()

    @property
    def buses(self) -> tuple[Bus, ...]:
        return self._buses_cached

    @cached_property
    def _transportation_network(self) -> "r5py.TransportNetwork":
        if not self.osm_pbf_path:
            raise ValueError("osm_pbf_path must be provided if use_r5 is True")
        return r5py.TransportNetwork(osm_pbf=self.osm_pbf_path)

    @cached_property
    def gdf(self) -> gpd.GeoDataFrame:
        return ox.geocode_to_gdf(self.place_name)

    def _make_osm_graph(self):
        # get boundary polygon (similar to analysis.ipynb)
        gdf = self.gdf
        crs = gdf.crs
        assert crs is not None

        # project to utm for meters-based buffering
        projected = gdf.to_crs(gdf.estimate_utm_crs())
        projected["geometry"] = projected.buffer(self.boundary_buffer_km * 1000)

        # project back to original crs for osmnx
        buffered = projected.to_crs(crs)
        buffered_poly = cast(Polygon, buffered.geometry.iloc[0])

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

    def get_shortest_path_base(
        self, start: NodeId, end: NodeId, weight: str = "length"
    ) -> tuple[float, list[NodeId]]:

        # check if length and path are already in service_graph
        if "_service_graph_cached" in self.__dict__ and weight == "length":
            service_graph = self._service_graph_cached
            edge = service_graph.get_edge_data(start, end, 0, None)

            if edge is not None:
                # service graph length is in km
                length_km = edge["length"]
                path = edge["path"]

                return length_km * 1000.0, path

        # return super().get_shortest_path_base(start, end, weight)
        return get_shortest_path(self.base_graph, start, end, weight)

    def _service_edge_allowed(
        self,
        start: Place,
        end: Place,
        stop_school_types: dict[Stop, set[SchoolType]],
        length: float | None = None,
    ) -> bool:
        if isinstance(start, Stop) and isinstance(end, School):
            return end.type in stop_school_types.get(start, set())

        if isinstance(start, Stop) and isinstance(end, Stop):
            if start.node_id == end.node_id:
                return True
            if stop_school_types.get(start, set()).isdisjoint(
                stop_school_types.get(end, set())
            ):
                return False

        if start.node_id == end.node_id:
            return True

        if (
            length is not None
            and self.prune is not None
            and isinstance(start, Stop)
            and isinstance(end, Stop)
        ):
            return length <= self.prune

        return True

    @staticmethod
    def _r5_itinerary_lookup(
        detailed_itineraries: pd.DataFrame,
    ) -> dict[tuple[Hashable, Hashable], dict[str, object]]:
        itinerary_lookup: dict[tuple[Hashable, Hashable], dict[str, object]] = {}
        for row in detailed_itineraries.itertuples(index=False):
            key = (row.from_id, row.to_id)
            if key not in itinerary_lookup:
                entry: dict[str, object] = {"distance": row.distance}
                geometry = getattr(row, "geometry", None)
                if geometry is not None:
                    entry["geometry"] = geometry
                itinerary_lookup[key] = entry
        return itinerary_lookup

    def _make_spatio_temporal_graph(self):
        raise NotImplementedError(
            "spatio-temporal graph construction not implemented yet"
        )

    def _make_service_graph(self) -> "nx.MultiDiGraph[NodeId]":
        service_graph: "nx.MultiDiGraph[NodeId]" = nx.MultiDiGraph()
        service_graph.graph["distance_unit"] = "km"
        stop_school_types = self._stop_school_types()
        pairs = self._service_graph_pairs()

        def add_edge(
            start: Place,
            end: Place,
            edge_resolver: Callable[
                [Place, Place], tuple[float, list[NodeId], dict[str, Any]]
            ],
        ):
            start_id = start.node_id
            end_id = end.node_id

            if service_graph.has_edge(start_id, end_id):
                return

            if not self._service_edge_allowed(start, end, stop_school_types):
                return

            if start_id == end_id:
                service_graph.add_edge(
                    start_id, end_id, length=0.0, path=[start_id, end_id]
                )
                return

            try:
                length_m, path, extra_attrs = edge_resolver(start, end)
                length_km = meters_to_kilometers(length_m)
                if not self._service_edge_allowed(
                    start, end, stop_school_types, length=length_km
                ):
                    return
                service_graph.add_edge(
                    start_id,
                    end_id,
                    length=length_km,
                    path=path,
                    **extra_attrs,
                )
            except (KeyError, nx.NetworkXNoPath):
                print(f"Warning: no path between {start} and {end} in the graph")

        if not self.use_r5:

            def edge_resolver_not_r5(start: Place, end: Place):
                length, path = self.get_shortest_path_base(start.node_id, end.node_id)
                return length, path, {}

            for start, end in pairs:
                add_edge(start, end, edge_resolver_not_r5)

        else:
            if not self.osm_pbf_path:
                raise ValueError("osm_pbf_path must be provided if use_r5 is True")
            places = self.all_nodes
            place_ids = {id(place): f"place_{idx}" for idx, place in enumerate(places)}

            nodes_gdf = gpd.GeoDataFrame(
                {
                    "id": [place_ids[id(place)] for place in places],
                    "geometry": [place.geographic_location for place in places],
                }
            )

            itinerary_lookup = self._r5_itinerary_lookup(
                r5py.DetailedItineraries(
                    self._transportation_network,
                    origins=nodes_gdf,
                    destinations=nodes_gdf,
                    transport_modes=[r5py.TransportMode.CAR],
                    snap_to_network=True,
                    force_all_to_all=True,
                )
            )

            def edge_resolver_r5(start: Place, end: Place):
                entry = itinerary_lookup[(place_ids[id(start)], place_ids[id(end)])]
                extra_attrs: dict[str, object] = {}
                geometry = entry.get("geometry")
                if geometry is not None:
                    extra_attrs["geometry"] = geometry
                return float(entry["distance"]), [], extra_attrs

            for start, end in pairs:
                add_edge(start, end, edge_resolver_r5)

        return service_graph

    def save(self, cache_dir: Path | None = None):
        """save problem data to disk for later loading and use in formulation"""
        cache_dir = cache_dir or CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)

        prob_name = (
            f"{self.name}{'_' + str(self.prune) if self.prune else ''}_problem_data"
        )
        with open(cache_dir / f"{prob_name}.pkl", "wb+") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, name: str, prune: int | None = None) -> "ProblemDataReal":
        """load problem data from disk"""
        # Try the prune-specific cache first, then fall back to legacy unpruned naming.
        candidate_names = [f"{name}{'_' + str(prune) if prune else ''}_problem_data"]
        # if prune is None:
        #     candidate_names.append(f"{name}_problem_data")

        for candidate in candidate_names:
            path = CACHE_DIR / f"{candidate}.pkl"
            if path.exists():
                print(f"Loading {path}")
                return cls.load_path(path)

        raise FileNotFoundError(f"No cached problem data found in {CACHE_DIR}")

    @classmethod
    def load_path(cls, path: Path) -> "ProblemDataReal":
        with open(path, "rb") as f:
            problem_data = pickle.load(f)

        cached_service_graph = vars(problem_data).get("_service_graph_cached")
        if cached_service_graph is not None:
            ensure_service_graph_kilometers(cached_service_graph)

        return problem_data

    def _make_schools(self) -> tuple[School, ...]:
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
        return tuple(return_schools)

    def _make_depots(self) -> tuple[Depot, ...]:
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
        return tuple(return_depots)

    def _make_stops(self) -> tuple[Stop, ...]:
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
        return tuple(return_stops)

    def _make_students(self) -> tuple[Student, ...]:
        students_df = pd.read_csv(
            self.students_path,
            dtype={
                "id": str,
                "name": str,
                "lon": float,
                "lat": float,
                "school_id": str,
                "grade": str,
                "is_sp_ed": bool,
                "is_wheelchair_user": bool,
            },
        )
        return_students: list[Student] = []

        outside_boundary = 0
        no_stop = 0
        for _, row in students_df.iterrows():
            school = next(s for s in self.schools if s.id == row["school_id"])
            geographic_location = Point(row["lon"], row["lat"])

            # check if in gdf bounds
            if not (
                cast(Polygon, self.gdf.geometry.iloc[0]).contains(geographic_location)
            ):
                outside_boundary += 1
                continue

            # find nearest stop to student
            nearest_stop = self._get_nearest_stop(geographic_location)

            if nearest_stop is None:
                no_stop += 1
                continue

            this_student = Student(
                id=row["id"],
                name=row["name"],
                geographic_location=geographic_location,
                school=school,
                stop=nearest_stop,
                attributes=Attributes(
                    special_ed=bool(row["is_sp_ed"]),
                    wheelchair_user=bool(row["is_wheelchair_user"]),
                ),
                grade=(
                    str(row["grade"])
                    if "grade" in students_df.columns and not pd.isna(row["grade"])
                    else None
                ),
            )
            return_students.append(this_student)

        if outside_boundary > 0:
            print(
                f"{outside_boundary} student(s) were located outside the boundary and excluded from the problem."
            )

        if no_stop > 0:
            print(f"{no_stop} student(s) could not be assigned stops.")

        return tuple(return_students)

    def _make_buses(
        self,
    ) -> tuple[Bus, ...]:
        buses_df = pd.read_csv(
            self.buses_path,
            dtype={
                "id": str,
                "num": str,
                "depot_name": str,
                "capacity": int,
                "range": float,
                "wheelchair_capacity": int,
                "type": str,
            },
        )
        return_buses: list[Bus] = []
        for _, row in buses_df.iterrows():
            depot = next(d for d in self.depots if d.name == row["depot_name"])
            bus_type = (
                BusType[row["type"]] if row.get("type") in BusType.__members__ else None
            )
            if "wheelchair_capacity" in buses_df.columns and not pd.isna(
                row["wheelchair_capacity"]
            ):
                wheelchair_capacity = int(row["wheelchair_capacity"])
            elif bus_type == BusType.WC:
                wheelchair_capacity = 4
            elif bus_type == BusType.BWC:
                wheelchair_capacity = 2
            else:
                wheelchair_capacity = 0
            bus = Bus(
                id=row["id"],
                name=row["num"],
                capacity=row["capacity"],
                range=row["range"],
                depot=depot,
                wheelchair_capacity=wheelchair_capacity,
                type=bus_type,
            )
            return_buses.append(bus)
        return tuple(return_buses)

    def _get_nearest_node_id(self, geographic_location: Point) -> NodeId:
        """Get the nearest node in the graph to a given point."""
        return ox.distance.nearest_nodes(
            self.osm_graph, geographic_location.x, geographic_location.y
        )

    def _get_nearest_stop(self, geo_location: Point) -> Stop | None:
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
            # https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html#sentinel-node-trick-for-multi-target-queries

            source = self._get_nearest_node_id(geo_location)
            targets = {stop.node_id for stop in self.stops}
            sentinel: NodeId = -1  # sentinel node id that is not used in the graph

            self.osm_graph.add_node(sentinel)
            for target in targets:
                self.osm_graph.add_edge(target, sentinel, length=0.0)

            try:
                path = nx.shortest_path(
                    self.osm_graph, source=source, target=sentinel, weight="length"
                )
            except nx.NetworkXNoPath:
                self.osm_graph.remove_node(sentinel)
                return None

            closest_target = path[-2]
            stop = next(stop for stop in self.stops if stop.node_id == closest_target)
            self.osm_graph.remove_node(sentinel)

            return stop


@dataclass(frozen=True)
class ProblemDataRealSurrogate(ProblemDataReal):
    """
    surrogate version of ProblemDataReal, which divides the geographic area into a grid of hexagons
    and assigns all nodes to the nearest grid cell centroid, to speed up shortest path calculations
    and reduce noise in the graph
    """

    prune = None
    osm_pbf_path = None
    use_r5 = False

    # 40 by 40, making each hexagon ~0.5km in framingham
    NUM_OF_HEXAGONS = 1600

    @property
    def _file_name(self):
        return f"{self.name}_hex_problem_data"

    @classmethod
    def load(cls, name: str, prune: int | None = None) -> "ProblemDataReal":
        """load problem data from disk"""
        prob_name = f"{name}_hex_problem_data"
        return cls.load_path(CACHE_DIR / f"{prob_name}.pkl")

    @property
    def hex_graph(self) -> "nx.MultiDiGraph[tuple[int, int]]":
        """
        hexagonal lattice graph used for surrogate service graph construction,
        with edge weights corresponding to distance in meters
        """

        rows = int(self.NUM_OF_HEXAGONS**0.5)
        cols = int(self.NUM_OF_HEXAGONS**0.5)

        hex_graph = nx.hexagonal_lattice_graph(rows, cols, create_using=nx.MultiDiGraph)
        hex_graph = cast("nx.MultiDiGraph[tuple[int, int]]", hex_graph)

        max_x_osm: float = max(data["x"] for _, data in self.osm_graph.nodes(data=True))
        min_x_osm: float = min(data["x"] for _, data in self.osm_graph.nodes(data=True))
        max_y_osm: float = max(data["y"] for _, data in self.osm_graph.nodes(data=True))
        min_y_osm: float = min(data["y"] for _, data in self.osm_graph.nodes(data=True))

        max_x_pos: float = max(data["pos"][0] for _, data in hex_graph.nodes(data=True))
        min_x_pos: float = min(data["pos"][0] for _, data in hex_graph.nodes(data=True))
        max_y_pos: float = max(data["pos"][1] for _, data in hex_graph.nodes(data=True))
        min_y_pos: float = min(data["pos"][1] for _, data in hex_graph.nodes(data=True))

        for node, data in hex_graph.nodes.items():
            x_pos: float = data["pos"][0]
            y_pos: float = data["pos"][1]

            # scale x and y from hex graph to original graph coordinates
            x_scaled = min_x_osm + (x_pos - min_x_pos) / (max_x_pos - min_x_pos) * (
                max_x_osm - min_x_osm
            )
            y_scaled = min_y_osm + (y_pos - min_y_pos) / (max_y_pos - min_y_pos) * (
                max_y_osm - min_y_osm
            )

            hex_graph.nodes[node]["x"] = x_scaled
            hex_graph.nodes[node]["y"] = y_scaled

        for u, v in hex_graph.edges():
            # distance between centroids in m
            hex_graph.edges[u, v, 0]["length"] = (
                # FIX: this gives distance in lat long, fix...
                (hex_graph.nodes[u]["x"] - hex_graph.nodes[v]["x"]) ** 2
                + (hex_graph.nodes[u]["y"] - hex_graph.nodes[v]["y"]) ** 2
            ) ** 0.5

        return hex_graph

    @cached_property
    def mapping_hex_base(self) -> dict[tuple[int, int], NodeId]:
        """
        get mapping from hex graph node ids to nearest node ids in the base graph,
        for use in shortest path calculations
        """
        mapping: dict[tuple[int, int], NodeId] = {}
        for i, node in enumerate(self.hex_graph.nodes()):
            mapping[node] = i
        return mapping

    @property
    def base_graph(self) -> "nx.MultiDiGraph[NodeId]":
        return self._make_base_graph

    @cached_property
    def _make_base_graph(self) -> "nx.MultiDiGraph[NodeId]":
        hex_graph = self.hex_graph

        base_graph_return: "nx.MultiDiGraph[NodeId]" = nx.MultiDiGraph()
        for node, data in hex_graph.nodes.items():
            base_graph_return.add_node(self.mapping_hex_base[node], **data)
        for u, v, data in hex_graph.edges(data=True):
            base_graph_return.add_edge(
                self.mapping_hex_base[u], self.mapping_hex_base[v], **data
            )

        return base_graph_return

    def _get_nearest_hex_node_id(self, geographic_location: Point) -> tuple[int, int]:
        """Get the nearest node in the hex graph to a given point."""
        x = geographic_location.x
        y = geographic_location.y

        # find nearest hex node by using geographic coordinates
        nearest_node = min(
            self.hex_graph.nodes(),
            key=lambda node: (
                (
                    (self.hex_graph.nodes[node]["x"] - x) ** 2
                    + (self.hex_graph.nodes[node]["y"] - y) ** 2
                )
                ** 0.5
            ),
        )

        return nearest_node

    def _make_stops(self):
        # rather than make multiple stops assigned to the same hex node,
        # we will assign each stop to the nearest hex node and create a new stop there
        # with the same attributes (except for node_id and geographic_location)
        stops: list[Stop] = []
        for stop in super()._make_stops():
            nearest_hex_node_id = self._get_nearest_hex_node_id(
                stop.geographic_location
            )

            geo_point = (
                self.base_graph.nodes[self.mapping_hex_base[nearest_hex_node_id]]["x"],
                self.base_graph.nodes[self.mapping_hex_base[nearest_hex_node_id]]["y"],
            )
            new_stop = Stop(
                name=stop.name,
                node_id=self.mapping_hex_base[nearest_hex_node_id],
                geographic_location=Point(geo_point[0], geo_point[1]),
            )
            stops.append(new_stop)
        return tuple(stops)

    def _get_nearest_stop(self, geo_location):
        stop_locations = [stop.geographic_location for stop in self.stops]
        nearest_stop_location = min(
            stop_locations,
            key=lambda loc: (
                ((loc.x - geo_location.x) ** 2 + (loc.y - geo_location.y) ** 2) ** 0.5
            ),
        )

        # get stop assigned to this hex node
        for stop in self.stops:
            if stop.geographic_location == nearest_stop_location:
                return stop
