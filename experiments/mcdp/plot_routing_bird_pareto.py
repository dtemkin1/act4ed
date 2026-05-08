from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIBRARY = PROJECT_ROOT / "routing.mcdplib"
DEFAULT_COSTS = PROJECT_ROOT / "experiments" / "mcdp" / "routing_costs.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "outputs" / "mcdp_solve" / "pareto"

BUS_TYPES = ("C", "B", "BWC", "WC")


def parse_number(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", str(value))
    if match is None:
        raise ValueError(f"Could not parse numeric value from {value!r}")
    return float(match.group(0))


def parse_poset_value(value: Any) -> str:
    text = str(value).strip().strip('"').strip("'").strip("`")
    if ":" in text:
        return text.split(":", 1)[1].strip()
    return text


def pareto_min(points: list[dict[str, Any]], x_key: str, y_key: str) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    best_y = math.inf
    for point in sorted(points, key=lambda p: (p[x_key], p[y_key])):
        y = float(point[y_key])
        if y < best_y:
            front.append(point)
            best_y = y
    return front


def route_records(library: Path, costs_path: Path) -> list[dict[str, Any]]:
    route_yaml = yaml.safe_load(
        (library / "yaml_catalogues" / "routing_service.dpc.yaml").read_text()
    )
    costs = yaml.safe_load(costs_path.read_text())

    school_days = float(costs["school_days"])
    capital = costs["capital"]
    distance_factor = costs["maintenance_distance_factor"]
    runtime_factor = costs["maintenance_runtime_factor"]

    records: list[dict[str, Any]] = []
    for label, impl in route_yaml["implementations"].items():
        f = [parse_number(v) for v in impl["f_max"]]
        r = impl["r_min"]

        used = {bus_type: int(parse_number(r[5 + i])) for i, bus_type in enumerate(BUS_TYPES)}
        distance = {
            bus_type: parse_number(r[9 + i]) for i, bus_type in enumerate(BUS_TYPES)
        }
        runtime = {
            bus_type: parse_number(r[13 + i]) for i, bus_type in enumerate(BUS_TYPES)
        }

        total_distance = sum(distance.values())
        capital_cost = sum(float(capital[bus_type]) * used[bus_type] for bus_type in BUS_TYPES)
        driver_cost = sum(used.values()) * float(costs["driver_yearly_pay"])
        monitor_cost = parse_number(r[4]) * float(costs["monitor_yearly_pay"])
        fuel_cost = total_distance * school_days * float(costs["fuel_cost_per_km"])
        maintenance_cost = school_days * sum(
            distance[bus_type] * float(distance_factor[bus_type])
            + runtime[bus_type] * float(runtime_factor[bus_type])
            for bus_type in BUS_TYPES
        )
        total_cost = capital_cost + driver_cost + monitor_cost + fuel_cost + maintenance_cost
        emissions = total_distance * school_days * float(costs["emissions_kg_per_km"])

        records.append(
            {
                "label": label,
                "students_served": f[0],
                "sped_students_served": f[1],
                "wheelchair_students_served": f[2],
                "students_unserved": parse_number(r[0]),
                "sped_students_unserved": parse_number(r[1]),
                "wheelchair_students_unserved": parse_number(r[2]),
                "stops_used": parse_number(r[3]),
                "monitor_buses": parse_number(r[4]),
                "used_C": used["C"],
                "used_B": used["B"],
                "used_BWC": used["BWC"],
                "used_WC": used["WC"],
                "total_distance_km": total_distance,
                "total_runtime_s": sum(runtime.values()),
                "capital_cost": capital_cost,
                "driver_cost": driver_cost,
                "monitor_cost": monitor_cost,
                "fuel_cost": fuel_cost,
                "maintenance_cost": maintenance_cost,
                "total_cost": total_cost,
                "emissions_kg": emissions,
                "bird_method": parse_poset_value(r[17]),
                "bird_lambda": parse_poset_value(r[18]),
                "bird_partial": parse_poset_value(r[19]),
                "bird_dwell": parse_poset_value(r[20]),
                "bird_arrival_window": parse_poset_value(r[21]),
            }
        )
    return records


def routing_simple_caps(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lambda_allowed = {"lambda_1e3", "lambda_1e4"}
    return [
        row
        for row in records
        if row["students_served"] >= 10
        and row["bird_method"] in {"lbh", "scenario"}
        and row["bird_lambda"] in lambda_allowed
        and row["bird_partial"] in {"partial_false", "partial_true"}
        and row["bird_dwell"] in {"dwell_0", "dwell_10"}
        and row["bird_arrival_window"] == "arrival_default"
    ]


def routing_simple_feasible(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in routing_simple_caps(records)
        if row["total_cost"] <= 4_500_000
        and row["emissions_kg"] <= 1_000_000
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_panel(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
) -> None:
    methods = {"lbh": "#3b82f6", "scenario": "#ef4444"}
    for method, color in methods.items():
        subset = [row for row in rows if row["bird_method"] == method]
        if subset:
            ax.scatter(
                [row[x_key] for row in subset],
                [row[y_key] for row in subset],
                s=42,
                alpha=0.7,
                color=color,
                label=method,
            )

    front = pareto_min(rows, x_key, y_key)
    if front:
        ax.plot(
            [row[x_key] for row in front],
            [row[y_key] for row in front],
            color="#111827",
            linewidth=2,
            marker="o",
            markersize=4,
            label="Pareto front",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)


def make_plot(rows: list[dict[str, Any]], output: Path, title: str) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    plot_panel(
        axes[0],
        rows,
        "students_unserved",
        "total_cost",
        "Students unserved",
        "Total annual cost (USD)",
    )
    plot_panel(
        axes[1],
        rows,
        "students_unserved",
        "emissions_kg",
        "Students unserved",
        "Annual emissions (kg)",
    )
    plot_panel(
        axes[2],
        rows,
        "stops_used",
        "total_cost",
        "Stops used",
        "Total annual cost (USD)",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3, frameon=False)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--costs", type=Path, default=DEFAULT_COSTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = route_records(args.library, args.costs)
    capped = routing_simple_caps(records)
    feasible = routing_simple_feasible(records)

    all_csv = args.output_dir / "routing_bird_mcdp_points.csv"
    capped_csv = args.output_dir / "routing_bird_mcdp_routing_simple_caps.csv"
    feasible_csv = args.output_dir / "routing_bird_mcdp_routing_simple_budget_feasible.csv"
    all_plot_png = args.output_dir / "routing_bird_mcdp_pareto_all.png"
    capped_plot_png = args.output_dir / "routing_bird_mcdp_pareto_routing_simple_caps.png"
    feasible_plot_png = args.output_dir / "routing_bird_mcdp_pareto_routing_simple_budget_feasible.png"

    write_csv(all_csv, records)
    write_csv(capped_csv, capped)
    write_csv(feasible_csv, feasible)
    make_plot(records, all_plot_png, "Routing BiRD MCDP Catalogue Pareto Fronts")
    if capped:
        make_plot(
            capped,
            capped_plot_png,
            "Routing BiRD MCDP Pareto Fronts: routing_simple BiRD Caps",
        )
    if feasible:
        make_plot(
            feasible,
            feasible_plot_png,
            "Routing BiRD MCDP Pareto Fronts: routing_simple Budget Feasible",
        )

    print(f"Loaded {len(records)} route catalogue points")
    print(f"Routing-simple BiRD-cap points: {len(capped)}")
    print(f"Routing-simple budget-feasible points: {len(feasible)}")
    print(f"Minimum catalogue cost: {min(row['total_cost'] for row in records):.2f} USD")
    if capped:
        print(f"Minimum routing-simple cap cost: {min(row['total_cost'] for row in capped):.2f} USD")
    print(f"Wrote {all_csv}")
    print(f"Wrote {capped_csv}")
    print(f"Wrote {feasible_csv}")
    print(f"Wrote {all_plot_png}")
    if capped:
        print(f"Wrote {capped_plot_png}")
    if feasible:
        print(f"Wrote {feasible_plot_png}")


if __name__ == "__main__":
    main()
