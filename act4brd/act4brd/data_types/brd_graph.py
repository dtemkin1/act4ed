import json
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
from uuid import uuid4

import networkx as nx

BUS_OPERATOR_SALARY = 45_028


@dataclass(frozen=True)
class Bus:
    name: str
    capacity: int
    per_mile_emissions: float
    procurement_price: int
    annual_maintenance_cost: int
    discomfort_level: int
    avg_speed: float
    _uuid: str = field(default_factory=lambda: str(uuid4()), init=False)


@dataclass(frozen=True)
class Route:
    name: str
    nodes: list[int]

    def __post_init__(self):
        if len(self.nodes) < 2:
            raise ValueError("Route must contain at least 2 nodes")

    def __len__(self):
        return len(self.nodes) - 1


@dataclass(frozen=True)
class ODFlow:
    origin: int
    destination: int
    flow: int

    def __post_init__(self):
        if self.flow < 0:
            raise ValueError("Flow must be non-negative")

        if self.origin == self.destination:
            raise ValueError("Origin and destination must be different")


@dataclass(frozen=True)
class Operator:
    annual_salary: int = BUS_OPERATOR_SALARY

    def __post_init__(self):
        if self.annual_salary < 0:
            raise ValueError("Operator salary must be non-negative")


@dataclass
class FleetComposition:
    fleet: list[Bus]
    operators: Optional[list[Operator]] = None

    def __post_init__(self):
        if len(self.fleet) < 1:
            raise ValueError("Fleet composition must include at least one bus")

        if self.operators is None:
            self.operators = [Operator()] * len(self.fleet)
        elif len(self.operators) != len(self.fleet):
            raise ValueError("Each bus must have a corresponding operator")

    @property
    def total_capacity(self) -> int:
        return sum(bus.capacity for bus in self.fleet)

    @property
    def total_capital_cost(self) -> int:
        return sum(bus.procurement_price for bus in self.fleet)

    @property
    def total_operational_cost(self) -> int:
        return sum(op.annual_salary for op in self.operators) + sum(bus.annual_maintenance_cost for bus in self.fleet)

    @property
    def num_buses(self) -> int:
        return len(self.fleet)


@dataclass
class FleetAssignment:
    composition: FleetComposition
    assigned_buses: dict[Bus, bool] = None
    assigned_bus_routes: dict[Bus, Optional[Route]] = None

    def __post_init__(self):
        if self.assigned_buses is None:
            self.assigned_buses = {bus: False for bus in self.composition.fleet}

        if self.assigned_bus_routes is None:
            self.assigned_bus_routes = {bus: None for bus in self.composition.fleet}

    def assign_bus_to_route(self, bus: Bus, route: Route) -> None:
        """
        Assign a bus to a specific route.

        Args:
            bus (Bus): The index of the bus in the fleet.
            route (Route): The route to which the bus is assigned.

        Raises:
            ValueError: If the bus is already assigned or the bus index is invalid.
        """
        # if bus_index < 0 or bus_index >= len(self.composition.fleet):
        #     raise ValueError(f"Invalid bus index: {bus_index}")
        if not bus in self.assigned_buses:
            raise ValueError(f"Bus {bus} is not in the fleet.")
        if self.assigned_buses[bus]:
            raise ValueError(f"Bus {bus} is already assigned.")
        self.assigned_buses[bus] = True
        self.assigned_bus_routes[bus] = route

    def capacity_for_route(self, route: Route) -> int:
        """Calculate the total capacity of buses assigned to a specific route."""
        return sum(
            bus.capacity
            for bus, assigned, assigned_route in
            zip(self.composition.fleet, self.assigned_buses, self.assigned_bus_routes)
            if assigned and assigned_route == route
        )

    def capital_cost_for_route(self, route: Route) -> int:
        """Calculate the total procurement cost of buses assigned to a specific route."""
        return sum(
            bus.procurement_price
            for bus, assigned, assigned_route in
            zip(self.composition.fleet, self.assigned_buses, self.assigned_bus_routes)
            if assigned and assigned_route == route
        )

    def operational_cost_for_route(self, route: Route) -> int:
        """Calculate the operational cost for buses assigned to a specific route."""
        gen_1 = zip(self.composition.operators, self.assigned_buses, self.assigned_bus_routes)
        gen_2 = zip(self.composition.fleet, self.assigned_buses, self.assigned_bus_routes)
        return (
                sum(op.annual_salary for op, assigned, assigned_route in gen_1 if
                    assigned and assigned_route == route) +
                sum(bus.annual_maintenance_cost for bus, assigned, assigned_route in gen_2 if
                    assigned and assigned_route == route)
        )

    @property
    def total_emissions(self) -> float:
        """Calculate the total emissions of all assigned buses."""
        return sum(
            bus.per_mile_emissions
            for bus, assigned in zip(self.composition.fleet, self.assigned_buses)
            if assigned
        )

    def emissions_for_route(self, route: Route) -> float:
        """Calculate the emissions for a specific route."""
        gen = list(zip(self.composition.fleet, self.assigned_buses, self.assigned_bus_routes))
        return sum(
            bus.per_mile_emissions
            for bus, assigned, assigned_route in gen
            if assigned and assigned_route == route
        )

    @property
    def num_buses(self) -> int:
        """Get the total number of buses in the fleet."""
        return len(self.composition.fleet)

    @property
    def num_buses_assigned(self) -> int:
        """Get the number of buses that have been assigned to routes."""
        return sum(self.assigned_buses)

    @property
    def buses_assigned(self) -> bool:
        """Check if all buses have been assigned to routes."""
        return all(self.assigned_buses)

    def num_buses_assigned_to_route(self, route: Route) -> int:
        """Get the number of buses assigned to a specific route."""
        gen = list(zip(self.composition.fleet, self.assigned_buses, self.assigned_bus_routes))
        return sum(1 for _, assigned, assigned_route in gen if assigned and assigned_route == route)


@dataclass
class ServiceNetworkDesign:
    routes: list[Route]
    od_flows: list[ODFlow]
    fleet_composition: FleetComposition
    street_network_graph: nx.Graph
    fleet_assignments: list[FleetAssignment] = field(default_factory=list, init=False)
    service_network_graphs: list[nx.MultiDiGraph] = field(default_factory=list, init=False)

    def __post_init__(self):
        if len(self.routes) < 1:
            raise ValueError("At least one route must be provided")

        if len(self.od_flows) < 1:
            raise ValueError("At least one OD flow must be provided")

        if len(self.fleet_composition.fleet) < len(self.routes):
            raise ValueError("At least one bus must be provided for each route")

        # Ensure that all routes have a unique name
        if len(set(route.name for route in self.routes)) != len(self.routes):
            raise ValueError("All routes must have a unique name")

        # Check graph paths for OD flows and route validity
        for od_flow in self.od_flows:
            if not nx.has_path(self.street_network_graph, od_flow.origin, od_flow.destination):
                raise ValueError(f"No path between OD pair {od_flow.origin} and {od_flow.destination}")

        for route in self.routes:
            for u, v in zip(route.nodes[:-1], route.nodes[1:]):
                if not self.street_network_graph.has_edge(u, v):
                    raise ValueError(f"No edge between nodes {u} and {v} in route {route}")

        # Check if all edges in the graph have a weight
        if not all("weight" in edge_data for _, _, edge_data in self.street_network_graph.edges(data=True)):
            raise ValueError("All edges in the graph must have a 'weight' attribute")

    @property
    def demand_profile_name(self) -> str:
        """Get the name of the demand profile. Example: demand_1_8_190__2_6_10"""
        return 'demand_' + "__".join(f"{od.origin}_{od.destination}_{od.flow}" for od in self.od_flows)



    def _get_empty_service_network_graph(self) -> nx.MultiDiGraph:
        return nx.MultiDiGraph(
            (u, v, key, data)
            for u, v, key, data in self.street_network_graph.edges(data=True)
            if data.get('type') != 'distance'
        )

    def _update_service_network_graphs(self) -> None:
        """
        Update the service network graph based on the current fleet assignments.
        """
        # Reset the graph to its original state
        service_network_graph = self._get_empty_service_network_graph()

        # Add buses to the graph
        assignment = self.fleet_assignments[-1]

        for route in self.routes:
            for bus in self.fleet_composition.fleet:
                if assignment.assigned_buses[bus] and assignment.assigned_bus_routes[bus] == route:
                    for u, v in zip(route.nodes[:-1], route.nodes[1:]):
                        travel_time = self.street_network_graph[u][v]['weight'] / bus.avg_speed
                        service_network_graph.add_edge(
                            u, v,
                            weight=travel_time,
                            type="bus",
                            route_id=route.name,
                            discomfort=bus.discomfort_level
                        )

        self.service_network_graphs.append(service_network_graph)

    def analyze_flow_for_assignment(self, assignment: FleetAssignment, od_flow: ODFlow) -> dict[str, Union[float, int]]:
        origin = od_flow.origin
        destination = od_flow.destination
        graph = self.service_network_graphs[self.fleet_assignments.index(assignment)]

        # Compute the shortest path using travel time as weight
        path = nx.shortest_path(graph, origin, destination, weight="weight")
        edges = [
            (path[i], path[i + 1])
            for i in range(len(path) - 1)
        ]

        total_time = 0
        total_edges = len(edges)
        unique_routes = set()
        transfers = 0
        discomfort = 0
        prev_route = None
        segments = 0

        for u, v in edges:
            edge_data = graph[u][v][0]  # Handle the first (or only) edge between nodes
            total_time += edge_data["weight"]
            if edge_data["type"] == "bus":
                unique_routes.add(edge_data["route_id"])
                discomfort += edge_data["discomfort"]
                if prev_route and prev_route != edge_data["route_id"]:
                    transfers += 1
                prev_route = edge_data["route_id"]
                segments += 1

        avg_travel_time = total_time / total_edges if total_edges > 0 else 0
        avg_discomfort = discomfort / segments if segments > 0 else 0

        return {
            "average_travel_time": avg_travel_time,
            "number_of_hops": total_edges,
            "number_of_services": len(unique_routes),
            "transfers": transfers,
            "average_discomfort": avg_discomfort,
        }

    def assign_buses(self, bus_assignment_routine: callable) -> None:
        """
        Assign buses based on a given bus assignment routine, updating fleet assignments.
        The routine has to accept a ServiceNetworkDesign instance and return a FleetAssignment instance.
        """
        new_assignment = bus_assignment_routine(copy(self))
        # Ensure all buses have been assigned in at least one assignment
        if not new_assignment.buses_assigned:
            raise ValueError("Not all buses have been assigned in this fleet assignment")

        self.fleet_assignments.append(new_assignment)
        self._update_service_network_graphs()

    def remove_assignment(self, assignment: FleetAssignment) -> None:
        """
        Remove a specific fleet assignment from the list of assignments.
        """
        self.fleet_assignments.remove(assignment)

    def remove_all_assignments(self) -> None:
        """
        Remove all fleet assignments.
        """
        self.fleet_assignments = []

    @property
    def total_emissions(self) -> float:
        """
        Compute total emissions across all fleet assignments.
        """
        return sum(assignment.total_emissions for assignment in self.fleet_assignments)

    @property
    def total_capital_cost(self) -> int:
        """
        Compute total capital cost across all fleet assignments.
        """
        return sum(assignment.composition.total_capital_cost for assignment in self.fleet_assignments)

    @property
    def total_operational_cost(self) -> int:
        """
        Compute total operational cost across all fleet assignments.
        """
        return sum(assignment.operational_cost_for_route(route)
                   for assignment in self.fleet_assignments
                   for route in self.routes)

    @property
    def satisfied_demand(self) -> dict[ODFlow, int]:
        """
        Compute the satisfied demand for all OD pairs based on the current assignments.
        """
        satisfied_demand = {od_flow: 0 for od_flow in self.od_flows}
        for assignment in self.fleet_assignments:
            for route in self.routes:
                for od_flow in self.od_flows:
                    # Only routes serving the OD pair should contribute
                    if od_flow.origin in route.nodes and od_flow.destination in route.nodes:
                        satisfied_demand[od_flow] += assignment.capacity_for_route(route)
        return satisfied_demand

    def avg_travel_time_for_assignment(self, assignment: FleetAssignment) -> float:
        """
        Compute the average travel time for a specific fleet assignment.
        """
        return sum(
            self.analyze_flow_for_assignment(assignment, od_flow)["average_travel_time"] * od_flow.flow
            for od_flow in self.od_flows
        ) / sum(od_flow.flow for od_flow in self.od_flows)

    @property
    def avg_travel_times(self) -> list[float]:
        """
        Compute the average travel time for all fleet assignments and routes.
        """
        return [self.avg_travel_time_for_assignment(assignment) for assignment in self.fleet_assignments]

    def avg_discomfort_level_for_assignment(self, assignment: FleetAssignment) -> float:
        """
        Compute the average discomfort level for a specific fleet assignment.
        """
        return sum(
            self.analyze_flow_for_assignment(assignment, od_flow)["average_discomfort"] * od_flow.flow
            for od_flow in self.od_flows
        ) / sum(od_flow.flow for od_flow in self.od_flows)

    @property
    def avg_discomfort_levels(self) -> list[float]:
        """
        Compute the average discomfort level for all fleet assignments and routes.
        """
        return [self.avg_discomfort_level_for_assignment(assignment) for assignment in self.fleet_assignments]

    def avg_transfers_for_assignment(self, assignment: FleetAssignment) -> float:
        """
        Compute the average number of transfers for a specific fleet assignment.
        """
        return sum(
            self.analyze_flow_for_assignment(assignment, od_flow)["transfers"] * od_flow.flow
            for od_flow in self.od_flows
        ) / sum(od_flow.flow for od_flow in self.od_flows)

    @property
    def avg_transfers(self) -> list[float]:
        """
        Compute the average number of transfers across all fleet assignments and routes.
        """
        return [self.avg_transfers_for_assignment(assignment) for assignment in self.fleet_assignments]

    def avg_hops_for_assignment(self, assignment: FleetAssignment) -> float:
        """
        Compute the average number of hops for a specific fleet assignment.
        """
        return sum(
            self.analyze_flow_for_assignment(assignment, od_flow)["number_of_hops"] * od_flow.flow
            for od_flow in self.od_flows
        ) / sum(od_flow.flow for od_flow in self.od_flows)

    @property
    def avg_hops(self) -> list[float]:
        """
        Compute the average number of hops across all fleet assignments and routes.
        """
        return [self.avg_hops_for_assignment(assignment) for assignment in self.fleet_assignments]

    def num_buses_assigned_to_route(self, route: Route) -> int:
        """
        Compute the total number of buses assigned to a specific route across all assignments.
        """
        return sum(assignment.num_buses_assigned_to_route(route) for assignment in self.fleet_assignments)


def get_bus_generator(emissions_file: Path, prices_file: Path, capacity_file: Path,
               avg_mph_file: Path, discomfort_level_file: Path) -> callable:
    # Load JSON data
    with open(emissions_file, 'r') as f:
        emissions_data = json.load(f)
    with open(prices_file, 'r') as f:
        prices_data = json.load(f)
    with open(capacity_file, 'r') as f:
        capacity_data = json.load(f)
    with open(avg_mph_file, 'r') as f:
        avg_mph_data = json.load(f)
    with open(discomfort_level_file, 'r') as f:
        discomfort_level_data = json.load(f)

    # Combine data and create Bus instances
    def bus_generator(bus_type):
        return Bus(
            name=bus_type,
            per_mile_emissions=emissions_data[bus_type],
            capacity=capacity_data[bus_type],
            procurement_price=prices_data[bus_type],
            annual_maintenance_cost=prices_data[bus_type],
            avg_speed=avg_mph_data[bus_type],
            discomfort_level=discomfort_level_data[bus_type],
        )
    return bus_generator
