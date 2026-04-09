import os
from pathlib import Path
import yaml

import pandas as pd

from formulation.common import Bus, BusType, Depot

CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

FLEET_INPUT = CURRENT_FILE_DIR / ".." / "data" / "buses.csv"
FLEET_OUTPUT = CURRENT_FILE_DIR / ".." / "outputs" / "fleet_bus_count.dpc.yaml"

BUS_COSTS = {
    BusType.C: 50000,
    BusType.B: 70000,
    BusType.BWC: 80000,
    BusType.WC: 90000,
}

BUS_CAPACITIES = {
    BusType.C: 71,
    BusType.B: 48,
    BusType.BWC: 31,
    BusType.WC: 4,
}

# TODO: change formulation to use wheelchair capacity
BUS_WHEELCHAIRS = {
    BusType.C: False,
    BusType.B: False,
    BusType.BWC: True,
    BusType.WC: True,
}


def get_all_fleet_data():
    buses_df = pd.read_csv(
        FLEET_INPUT,
        dtype={
            "id": str,
            "depot_name": str,
            "capacity": int,
            "range": int,
            "has_wheelchair_access": bool,
            "type": str,
        },
    )

    depot = Depot(
        name="Depot",
        geographic_location=(0, 0),
        node_id=0,
    )

    fleet_data: list[Bus] = []
    for _, row in buses_df.iterrows():
        bus = Bus(
            name=row["id"],
            capacity=row["capacity"],
            range=row["range"],
            depot=depot,
            has_wheelchair_access=bool(row["has_wheelchair_access"]),
            type=BusType[row["type"]] if row["type"] in BusType.__members__ else None,
        )
        fleet_data.append(bus)
    return fleet_data


def get_amount_of_types(fleet: list[Bus]):
    type_counts = {bus_type: 0 for bus_type in BusType}
    for bus in fleet:
        if bus.type is not None:
            type_counts[bus.type] += 1
    return type_counts


def make_name_of_guideline(amounts: dict[BusType, int]):
    name = "guideline"
    for bus_type in BusType:
        count = amounts.get(bus_type, 0)
        name += f"_{count}"
    return name


def make_combos(max_amounts: dict[BusType, int]) -> list[dict[BusType, int]]:
    """Generate all combinations of bus types and amounts per type"""

    combos: list[dict[BusType, int]] = []
    bus_types = list(BusType)

    def backtrack(current_combo: dict[BusType, int], index: int):
        if index == len(bus_types):
            combos.append(current_combo.copy())
            return

        bus_type = bus_types[index]
        for amount in range(max_amounts[bus_type] + 1):
            current_combo[bus_type] = amount
            backtrack(current_combo, index + 1)

    backtrack({}, 0)
    return combos


def generate_fleet_yaml():
    fleet_data = get_all_fleet_data()
    type_counts = get_amount_of_types(fleet_data)
    combos = make_combos(type_counts)

    guidelines: dict[str, dict[str, list[str]]] = {}

    for combo in combos:
        guideline_name = make_name_of_guideline(combo)
        guidelines[guideline_name] = {
            "f_max": [
                # fleet size
                str(sum(combo.get(bus_type, 0) for bus_type in BusType)),
                # student capacity
                str(
                    sum(
                        BUS_CAPACITIES[bus_type] * combo.get(bus_type, 0)
                        for bus_type in BusType
                    )
                ),
                # wheelchair buses
                str(
                    sum(
                        (1 if BUS_WHEELCHAIRS[bus_type] else 0) * combo.get(bus_type, 0)
                        for bus_type in BusType
                    )
                ),
            ],
            "r_min": [
                str(
                    sum(
                        BUS_COSTS[bus_type] * combo.get(bus_type, 0)
                        for bus_type in BusType
                    )
                )
                + " USD",
            ],
        }

    return guidelines


def main() -> None:
    fleet_implementations = generate_fleet_yaml()

    data = {
        "F": ["Nat", "Nat", "Nat"],
        "R": ["USD"],
        "implementations": fleet_implementations,
    }

    with open(FLEET_OUTPUT, "w+", encoding="utf-8") as f:
        yaml.dump(data, f)


if __name__ == "__main__":
    main()
