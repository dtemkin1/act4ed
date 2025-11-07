import yaml
import json
from pathlib import Path


def write_service_network_design_yaml(
    fleet_composition: list[str],
    avg_travel_time: float,
    avg_discomfort_level: float,
    avg_transfers: float,
    avg_hops: float,
    total_emissions: float,
    output_file_path: Path,
    overwrite: bool = False
):
    # Check if file exists and load existing data if overwrite is False
    if output_file_path.exists():
        with output_file_path.open("r") as file:
            data = yaml.safe_load(file)
    else:
        data = {}

    # Initialize the YAML structure if file is new or overwrite is True
    if overwrite or "F" not in data or "R" not in data:
        data.update({
            "F": [
                "`satisfied_demand"
            ],
            "R": [
                "car",  # Bus type 1
                "car",  # Bus type 2
                "s",  # avg_travel_time
                "Reals",  # avg_discomfort_level
                "Reals",  # avg_transfers
                "Reals",  # avg_hops
                "kg/year",  # total_emissions
            ],
        })

    # Get the current implementation number and add the new entry
    implementations = data.get("implementations", {})
    model_number = len(implementations) + 1
    model_name = f"model{model_number}"

    # Add the new implementation
    implementations[model_name] = {
        "f_max": [
            f"`satisfied_demand: demand_1_8_190__2_6_10"
        ],
        "r_min": [
            f"{fleet_composition[0]} car",
            f"{fleet_composition[1]} car",
            f"{avg_travel_time} s",
            f"{avg_discomfort_level} Reals",
            f"{avg_transfers} Reals",
            f"{avg_hops} Reals",
            f"{total_emissions} kg/year"
        ]
    }

    data["implementations"] = implementations

    # Write the updated YAML file
    with output_file_path.open("w") as file:
        yaml.dump(data, file, default_flow_style=False)


def convert_json_to_yaml(json_path: Path, yaml_path: Path, overwrite: bool = False):
    with json_path.open("r") as file:
        data = json.load(file)

    for entry in data:
        write_service_network_design_yaml(
                fleet_composition=entry["Fleet_composition"],
                avg_travel_time=entry['Avg_tt'],
                avg_discomfort_level=entry['Discomfort'],
                avg_transfers=entry['Num_transfer'],
                avg_hops=entry['Num_hop'],
                total_emissions=entry['Emission'],
                output_file_path=yaml_path,
                overwrite = False
        )