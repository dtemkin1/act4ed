from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.axes as axes
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


def fuel_cost_per_km(costs: dict[str, Any]) -> float:
    configured = costs.get("fuel_cost_per_km")
    if configured is not None:
        return float(configured)
    return (
        float(costs["diesel_cost_per_gallon"])
        * float(costs["emissions_kg_per_km"])
        / float(costs["diesel_co2_kg_per_gallon"])
    )


def pareto_min(
    points: list[dict[str, Any]], x_key: str, y_key: str
) -> list[dict[str, Any]]:
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
        (library / "yaml_catalogs" / "routing_service.dpc.yaml").read_text()
    )
    costs = yaml.safe_load(costs_path.read_text())

    school_days = float(costs["school_days"])
    capital = costs["capital"]
    distance_factor = costs["maintenance_distance_factor"]
    runtime_factor = costs["maintenance_runtime_factor"]
    fuel_cost_factor = fuel_cost_per_km(costs)

    records: list[dict[str, Any]] = []
    for label, impl in route_yaml["implementations"].items():
        f = [parse_number(v) for v in impl["f_max"]]
        r = impl["r_min"]

        unique_stops_used = parse_number(r[4])
        monitor_buses = parse_number(r[5])
        used = {
            bus_type: int(parse_number(r[6 + i]))
            for i, bus_type in enumerate(BUS_TYPES)
        }
        distance = {
            bus_type: parse_number(r[10 + i]) for i, bus_type in enumerate(BUS_TYPES)
        }
        runtime = {
            bus_type: parse_number(r[14 + i]) for i, bus_type in enumerate(BUS_TYPES)
        }
        config_offset = 6 + 3 * len(BUS_TYPES)

        total_distance = sum(distance.values())
        capital_cost = sum(
            float(capital[bus_type]) * used[bus_type] for bus_type in BUS_TYPES
        )
        driver_cost = sum(used.values()) * float(costs["driver_yearly_pay"])
        monitor_cost = monitor_buses * float(costs["monitor_yearly_pay"])
        fuel_cost = total_distance * school_days * fuel_cost_factor
        maintenance_cost = school_days * sum(
            distance[bus_type] * float(distance_factor[bus_type])
            + runtime[bus_type] * float(runtime_factor[bus_type])
            for bus_type in BUS_TYPES
        )
        total_cost = (
            capital_cost + driver_cost + monitor_cost + fuel_cost + maintenance_cost
        )
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
                "unique_stops_used": unique_stops_used,
                "monitor_buses": monitor_buses,
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
                "bird_method": parse_poset_value(r[config_offset]),
                "bird_lambda": parse_poset_value(r[config_offset + 1]),
                "bird_partial": parse_poset_value(r[config_offset + 2]),
                "bird_dwell": parse_poset_value(r[config_offset + 3]),
                "bird_arrival_window": parse_poset_value(r[config_offset + 4]),
                "bird_avg_speed": parse_poset_value(r[config_offset + 5]),
                "student_policy": (
                    parse_poset_value(r[config_offset + 6])
                    if len(r) > config_offset + 6
                    else "all_students"
                ),
            }
        )
    return records


def routing_simple_caps(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lambda_allowed = {"lambda_1e3", "lambda_1e4"}
    speed_allowed = {"speed_10mph", "speed_20mph", "speed_30mph"}
    return [
        row
        for row in records
        if row["students_served"] >= 10
        and row["bird_method"] in {"lbh", "scenario"}
        and row["bird_lambda"] in lambda_allowed
        and row["bird_partial"] in {"partial_false", "partial_true"}
        and row["bird_dwell"] in {"dwell_0", "dwell_10"}
        and row["bird_arrival_window"] == "arrival_default"
        and row["bird_avg_speed"] in speed_allowed
        and row["student_policy"] == "all_students"
    ]


def routing_simple_feasible(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in routing_simple_caps(records)
        if row["total_cost"] <= 4_500_000 and row["emissions_kg"] <= 1_000_000
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
    ax: axes.Axes,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    title: str = "",
) -> None:
    methods = {"lbh": "LBH", "scenario": "Scenario"}
    colors = {"lbh": "C0", "scenario": "C1"}

    front = pareto_min(rows, x_key, y_key)
    front_ids = {id(row) for row in front}

    for method, method_name in methods.items():
        subset = [
            row
            for row in rows
            if row["bird_method"] == method and id(row) not in front_ids
        ]
        if subset:
            ax.scatter(
                [row[x_key] for row in subset],
                [row[y_key] for row in subset],
                s=60,
                facecolors=colors[method],
                edgecolors="#111827",
                linewidth=0.45,
                alpha=0.5,
                label=method_name,
                zorder=2,
            )

    if front:
        front_sorted = sorted(front, key=lambda p: p[x_key])
        ax.step(
            [row[x_key] for row in front_sorted],
            [row[y_key] for row in front_sorted],
            where="post",
            color="#111827",
            linewidth=2,
            marker="o",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor="#111827",
            markeredgewidth=1.4,
            label="Pareto front",
            zorder=3,
            drawstyle="steps-post",
        )
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        first = front_sorted[0]
        last = front_sorted[-1]
        ax.plot(
            [first[x_key], first[x_key]],
            [first[y_key], y_max],
            color="#111827",
            linewidth=2,
            solid_capstyle="butt",
            zorder=3,
        )
        ax.plot(
            [last[x_key], x_max],
            [last[y_key], last[y_key]],
            color="#111827",
            linewidth=2,
            solid_capstyle="butt",
            zorder=3,
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    ax.set_xlabel(x_label, fontsize=11, fontweight="medium")
    ax.set_ylabel(y_label, fontsize=11, fontweight="medium")
    if title:
        ax.set_title(title, fontsize=12, fontweight="semibold")
    ax.grid(True, alpha=0.4, linestyle="--")

    # Optional nice formatting for axes
    if "cost" in y_key or "USD" in y_label:
        import matplotlib.ticker as ticker

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"${x:,.0f}"))
    elif "emission" in y_key or "kg" in y_label:
        import matplotlib.ticker as ticker

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:,.0f}"))


def make_plot(rows: list[dict[str, Any]], output: Path, main_title: str) -> None:
    plt.style.use("petroff10")

    # Smaller width so it scales better in the paper
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    plot_panel(
        axes[0],
        rows,
        "students_unserved",
        "total_cost",
        "Students unassigned",
        "Total annual cost (USD)",
        title="Cost vs. Unassigned Students",
    )
    plot_panel(
        axes[1],
        rows,
        "students_unserved",
        "emissions_kg",
        "Students unassigned",
        "Annual emissions (kg)",
        title="Emissions vs. Unassigned Students",
    )
    plot_panel(
        axes[2],
        rows,
        "stops_used",
        "students_unserved",
        "Stops used",
        "Students unassigned",
        title="Students Unassigned vs. Stops Used",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncols=3,
        frameon=True,
        fontsize=11,
    )
    fig.suptitle(main_title, y=1.12, fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--costs", type=Path, default=DEFAULT_COSTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    library: Path = args.library
    costs: Path = args.costs
    output_dir: Path = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    records = route_records(library, costs)
    capped = routing_simple_caps(records)
    feasible = routing_simple_feasible(records)

    all_csv = output_dir / "routing_bird_mcdp_points.csv"
    capped_csv = output_dir / "routing_bird_mcdp_routing_simple_caps.csv"
    feasible_csv = output_dir / "routing_bird_mcdp_routing_simple_budget_feasible.csv"
    all_plot_pdf = output_dir / "routing_bird_mcdp_pareto_all.pdf"
    capped_plot_pdf = output_dir / "routing_bird_mcdp_pareto_routing_simple_caps.pdf"
    feasible_plot_pdf = (
        output_dir / "routing_bird_mcdp_pareto_routing_simple_budget_feasible.pdf"
    )

    write_csv(all_csv, records)
    write_csv(capped_csv, capped)
    write_csv(feasible_csv, feasible)
    make_plot(records, all_plot_pdf, "Routing BiRD MCDP Catalog Pareto Fronts")
    if capped:
        make_plot(
            capped,
            capped_plot_pdf,
            "Routing BiRD MCDP Pareto Fronts: routing_simple BiRD Caps",
        )
    if feasible:
        make_plot(
            feasible,
            feasible_plot_pdf,
            "Routing BiRD MCDP Pareto Fronts: routing_simple Budget Feasible",
        )

    print(f"Loaded {len(records)} route catalog points")
    print(f"Routing-simple BiRD-cap points: {len(capped)}")
    print(f"Routing-simple budget-feasible points: {len(feasible)}")
    print(f"Minimum catalog cost: {min(row['total_cost'] for row in records):.2f} USD")
    if capped:
        print(
            f"Minimum routing-simple cap cost: {min(row['total_cost'] for row in capped):.2f} USD"
        )
    print(f"Wrote {all_csv}")
    print(f"Wrote {capped_csv}")
    print(f"Wrote {feasible_csv}")
    print(f"Wrote {all_plot_pdf}")
    if capped:
        print(f"Wrote {capped_plot_pdf}")
    if feasible:
        print(f"Wrote {feasible_plot_pdf}")


if __name__ == "__main__":
    main()
