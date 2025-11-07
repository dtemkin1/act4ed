# Import necessary components
from act4brd.data_types.brd_graph import (
    Bus,
    Route,
    ODFlow,
    FleetComposition,
    ServiceNetworkDesign,
    load_buses,
)
from act4brd.graph import create_street_grid
from act4brd.settings import BUS_DATA_PATH

# Step 1: Create the street network graph
street_network_graph, pos = create_street_grid(3, 3)

# Step 2: Define OD flows (toy example)
od_flows = [
    ODFlow(origin=1, destination=8, flow=190),
    ODFlow(origin=2, destination=6, flow=10),
]

# Step 3: Define routes
routes = [
    Route(name="Route_1", nodes=[1, 2, 5, 8, 7, 6]),
    Route(name="Route_2", nodes=[1, 2, 5, 4, 3, 6, 7, 8]),
]

# Step 4: Load buses from JSON files
buses = load_buses(
    emissions_file=BUS_DATA_PATH / "bus_per_mile_emissions.json",
    prices_file=BUS_DATA_PATH / "bus_prices.json",
    capacity_file=BUS_DATA_PATH / "bus_capacity.json",
    avg_mph_file=BUS_DATA_PATH / "bus_avg_mph.json",
    discomfort_level_file=BUS_DATA_PATH / "bus_discomfort_levels.json",
)

# Step 5: Create a fleet composition
fleet_composition = FleetComposition(
    fleet=[buses["Standard 40-Foot Diesel Bus"], buses["Articulated Diesel Bus"]]
)

# Step 6: Create the ServiceNetworkDesign object
toy_service_network = ServiceNetworkDesign(
    routes=routes,
    od_flows=od_flows,
    fleet_composition=fleet_composition,
    street_network_graph=street_network_graph,
)
