"""
---------------------------------------------------------
FIGURE — STEP 2 CLIFF PLOT
---------------------------------------------------------
Purpose:
    Visualize delay-induced phase collapse as a sharp
    survival cliff.

Physics Intent:
    - X-axis: Feedback Delay
    - Y-axis: Average Extinction Step (Lifetime)
    - Expectation: Non-smooth, discontinuous drop
---------------------------------------------------------
"""

import csv
import argparse
import matplotlib.pyplot as plt


def load_data(path):
    delays = []
    lifetimes = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            delays.append(int(row["delay"]))
            lifetimes.append(float(row["avg_extinction_step"]))

    return delays, lifetimes


def plot_cliff(delays, lifetimes, steps, out_path):
    plt.figure(figsize=(6, 4))

    plt.plot(
        delays,
        lifetimes,
        marker="o",
        linewidth=2,
        markersize=6,
    )

    plt.axhline(
        y=steps,
        linestyle="--",
        linewidth=1,
        label="Max Lifetime",
    )

    plt.xlabel("Feedback Delay")
    plt.ylabel("Average Extinction Step")
    plt.title("Delay-Induced Phase Collapse")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()

    delays, lifetimes = load_data(args.input)
    plot_cliff(delays, lifetimes, args.steps, args.out)


if __name__ == "__main__":
    main()
