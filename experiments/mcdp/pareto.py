import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_FILE = CURRENT_FILE_DIR / ".." / "outputs" / "pareto.png"


def get_antichain(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    # sort by x value
    points = sorted(points, key=lambda x: x[0])
    antichain = []
    current_y = float("inf")
    for p in points:
        if p[1] < current_y:
            antichain.append(p)
            current_y = p[1]
    return antichain


def main():
    df = pd.read_csv(CURRENT_FILE_DIR / "toy_results.csv")
    output = list(zip(10 - df["students_served"], df["total_cost"]))
    # plt.plot(10 - df["students_served"], df["total_cost"], marker="o")
    plt.xlabel("Students Left to Serve")
    plt.ylabel("Total Cost (USD)")
    plt.title("Pareto Front for Toy MCDP")

    points = [(float(t[0]), float(t[1])) for t in output]
    antichain = get_antichain(points)

    plot_x = []
    plot_y = []
    for i, (x, y) in enumerate(antichain):
        plot_x.append(x)
        plot_y.append(y)
        if i < len(antichain) - 1:
            x_next, y_next = antichain[i + 1]
            x_inter = min(x, x_next)
            y_inter = min(y, y_next)
            plot_x.append(x_inter)
            plot_y.append(y_inter)

    # make look nicer...

    plt.plot(plot_x, plot_y, linewidth=2)
    plt.scatter([p[0] for p in antichain], [p[1] for p in antichain], s=50)
    plt.fill_between(plot_x, y1=plot_y, y2=max(plot_y), alpha=0.1)

    plt.xlim(0, 10)
    plt.ylim(bottom=0)
    plt.tight_layout()

    plt.grid()
    plt.savefig(OUTPUT_FILE)
    print(f"Saved Pareto front plot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
