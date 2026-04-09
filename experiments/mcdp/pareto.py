import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_FILE = CURRENT_FILE_DIR / ".." / "outputs" / "pareto.png"


def main():
    df = pd.read_csv(CURRENT_FILE_DIR / "toy_results.csv")
    plt.plot(10 - df["students_served"], df["total_cost"], marker="o")
    plt.xlabel("Students Left to Serve")
    plt.ylabel("Total Cost (USD)")
    plt.title("Pareto Front for Toy MCDP")

    # make look nicer...
    plt.tight_layout()
    plt.xlim(0, 10)
    plt.ylim(bottom=0)
    plt.fill_between(10 - df["students_served"], df["total_cost"], alpha=0.2)

    plt.grid()
    plt.savefig(OUTPUT_FILE)
    print(f"Saved Pareto front plot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
