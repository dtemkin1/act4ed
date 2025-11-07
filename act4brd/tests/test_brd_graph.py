import pytest
from act4brd.data_types.brd_graph import (
    Bus,
    Route,
    ODFlow,
    Operator,
    FleetComposition,
    FleetAssignment,
    ServiceNetworkDesign,
    BUS_OPERATOR_SALARY,
)
import networkx as nx


# Test Bus class
def test_bus_creation():
    bus = Bus(
        name="Standard Bus",
        capacity=60,
        per_mile_emissions=0.0148,
        procurement_price=300000,
        annual_maintenance_cost=25000,
        avg_speed=12.0,
        discomfort_level=1,
    )
    assert bus.name == "Standard Bus"
    assert bus.capacity == 60
    assert bus.per_mile_emissions == 0.0148
    assert bus.procurement_price == 300000
    assert bus.annual_maintenance_cost == 25000


# Test Route class
def test_route_creation():
    route = Route(name='route_1', nodes=[0, 1, 2])
    assert len(route) == 2
    assert route.nodes == [0, 1, 2]

    with pytest.raises(ValueError):
        Route(name='route_1', nodes=[0])  # Should raise error for routes with less than 2 nodes


# Test ODFlow class
def test_odflow_creation():
    od_flow = ODFlow(origin=0, destination=1, flow=100)
    assert od_flow.origin == 0
    assert od_flow.destination == 1
    assert od_flow.flow == 100

    with pytest.raises(ValueError):
        ODFlow(origin=0, destination=0, flow=100)  # Origin and destination must differ

    with pytest.raises(ValueError):
        ODFlow(origin=0, destination=1, flow=-10)  # Flow must be non-negative


# Test Operator class
def test_operator_creation():
    operator = Operator(annual_salary=BUS_OPERATOR_SALARY)
    assert operator.annual_salary == BUS_OPERATOR_SALARY

    with pytest.raises(ValueError):
        Operator(annual_salary=-1)  # Negative salary not allowed


# Test FleetComposition class
def test_fleet_composition_creation():
    bus1 = Bus("Bus1", 60, 0.0148, 300000, 25000, 1, 12.0)
    bus2 = Bus("Bus2", 90, 0.022, 750000, 40000, 1, 12.0)

    fleet = FleetComposition(
        fleet=[bus1, bus2]
    )
    assert fleet.num_buses == 2
    assert fleet.total_capacity == 150
    assert fleet.total_capital_cost == 1050000
    assert fleet.total_operational_cost == 25000 + 40000 + 2 * BUS_OPERATOR_SALARY

    with pytest.raises(ValueError):
        FleetComposition(fleet=[])


# Test FleetAssignment class
def test_fleet_assignment_creation():
    bus1 = Bus("Bus1", 60, 0.0148, 300000, 25000, 1, 12.0)
    bus2 = Bus("Bus2", 90, 0.022, 750000, 40000, 1, 12.0)
    fleet = FleetComposition(fleet=[bus1, bus2])

    assignment = FleetAssignment(composition=fleet)
    assert len(assignment.assigned_buses) == 2
    assert len(assignment.assigned_bus_routes) == 2
    assert not any(assignment.assigned_buses)

    route = Route(name='route_1', nodes=[0, 1, 2])
    assignment.assigned_buses[0] = True
    assignment.assigned_bus_routes[0] = route
    assert assignment.num_buses_assigned == 1
    assert assignment.capacity_for_route(route) == 60


# Test ServiceNetworkDesign class
def test_service_network_design_creation():
    bus = Bus("Bus1", 60, 0.0148, 300000, 25000, 1, 12.0)
    fleet = FleetComposition(fleet=[bus])
    route = Route(name='route_1', nodes=[0, 1, 2])
    assignment = FleetAssignment(
        composition=fleet,
        assigned_buses=[True],
        assigned_bus_routes=[route]
    )
    od_flow = ODFlow(origin=0, destination=1, flow=100)
    street_network_graph = nx.Graph()
    street_network_graph.add_edges_from([(0, 1), (1, 2)], weight=1, distance=1)

    service_network = ServiceNetworkDesign(
        routes=[route],
        od_flows=[od_flow],
        fleet_composition=fleet,
        street_network_graph=street_network_graph
    )

    def mock_assignment(network):
        return FleetAssignment(
            composition=network.fleet_composition,
            assigned_buses=[True],
            assigned_bus_routes=[route]
        )

    service_network.assign_buses(mock_assignment)

    assert service_network.routes == [route]
    assert service_network.od_flows == [od_flow]
    assert service_network.fleet_assignments == [assignment]
    assert service_network.street_network_graph == street_network_graph

    with pytest.raises(ValueError):
        ServiceNetworkDesign(
            routes=[],
            od_flows=[od_flow],
            fleet_composition=fleet,
            street_network_graph=street_network_graph
        )  # At least one route required

    with pytest.raises(ValueError):
        ServiceNetworkDesign(
            routes=[route],
            od_flows=[],
            fleet_composition=fleet,
            street_network_graph=street_network_graph
        )  # At least one OD flow required


# Test assignment routine
def test_assign_buses():
    bus1 = Bus("Bus1", 60, 0.0148, 300000, 25000, 1, 12.0)
    bus2 = Bus("Bus2", 90, 0.022, 750000, 40000, 1, 12.0)
    fleet = FleetComposition(fleet=[bus1, bus2])
    route = Route(name='route_1', nodes=[0, 1, 2])
    od_flow = ODFlow(origin=0, destination=1, flow=100)
    street_network_graph = nx.Graph()
    street_network_graph.add_edges_from([(0, 1), (1, 2)])

    service_network = ServiceNetworkDesign(
        routes=[route],
        od_flows=[od_flow],
        fleet_composition=fleet,
        street_network_graph=street_network_graph
    )

    def mock_assignment_routine(network: ServiceNetworkDesign):
        assignment = FleetAssignment(
            composition=network.fleet_composition,
            assigned_buses=[True, False],
            assigned_bus_routes=[route, None]
        )
        return assignment

    with pytest.raises(ValueError):
        service_network.assign_buses(mock_assignment_routine)

    def mock_assign_all_buses_routine(network):
        assignment = FleetAssignment(
            composition=network.fleet_composition,
            assigned_buses=[True, True],
            assigned_bus_routes=[route, route]
        )
        return assignment

    service_network.assign_buses(mock_assign_all_buses_routine)

    assert service_network.fleet_assignments[0].num_buses_assigned == 2
    assert service_network.fleet_assignments[0].assigned_bus_routes[0] == route
    assert service_network.fleet_assignments[0].assigned_bus_routes[1] == route


def test_fleet_assignment_assign_bus():
    # Setup
    bus1 = Bus("Bus1", 60, 0.0148, 300000, 25000, 1, 12.0)
    bus2 = Bus("Bus2", 90, 0.022, 750000, 40000, 1, 12.0)
    fleet = FleetComposition(fleet=[bus1, bus2])
    assignment = FleetAssignment(composition=fleet)
    route1 = Route(name='route_1', nodes=[0, 1, 2])

    # Test initial state
    assert assignment.num_buses_assigned == 0
    assert not assignment.buses_assigned

    # Assign a bus
    assignment.assign_bus_to_route(0, route1)
    assert assignment.num_buses_assigned == 1
    assert assignment.assigned_buses[0] is True
    assert assignment.assigned_bus_routes[0] == route1

    # Assign the second bus
    assignment.assign_bus_to_route(1, route1)
    assert assignment.num_buses_assigned == 2
    assert assignment.assigned_buses[1] is True
    assert assignment.assigned_bus_routes[1] == route1
    assert assignment.buses_assigned

    # Test invalid bus index
    with pytest.raises(ValueError):
        assignment.assign_bus_to_route(2, route1)  # Index out of range

    # Test assigning an already assigned bus
    with pytest.raises(ValueError):
        assignment.assign_bus_to_route(0, route1)
