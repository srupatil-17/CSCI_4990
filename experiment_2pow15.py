import networkx as nx
import random
import math
import os
import csv
import matplotlib.pyplot as plt

from two_twelve_dimensional_experiment.gowalla_loader import load_gowalla_graph, extract_lcc
from two_twelve_dimensional_experiment.dimension_experiment import generate_kleinberg, run_experiments


# before running dont forget about Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# and then .\venv\Scripts\Activate.ps1

# ==============================
# SETTINGS
# ==============================

TRIALS = 10000
TARGET_NODES = 2**15
DIMENSIONS = [3, 5, 15]

EDGE_FILE = "data/loc-gowalla_edges.txt.gz"

DIST_FOLDER = "distributions_2pow15"
PLOT_FOLDER = "plots_2pow15"


# ==============================
# Utility Functions
# ==============================

def compute_side_length(dimension, target):
    return round(target ** (1 / dimension))


def run_gowalla_shortest_paths(G, num_trials):

    nodes = list(G.nodes())
    path_lengths = []

    for _ in range(num_trials):
        start = random.choice(nodes)
        target = random.choice(nodes)

        while target == start:
            target = random.choice(nodes)

        try:
            d = nx.shortest_path_length(G, start, target)
            path_lengths.append(d)
        except nx.NetworkXNoPath:
            continue

    return path_lengths


def compute_distribution(path_lengths):

    from collections import Counter

    count = Counter(path_lengths)
    total = sum(count.values())

    return {k: count[k] / total for k in count}


def export_distribution_to_folder(folder, filename, dist):

    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path_length", "probability"])

        for k in sorted(dist.keys()):
            writer.writerow([k, dist[k]])

    print(f"Saved distribution: {filepath}")


def plot_kleinberg_vs_gowalla(folder, title, gowalla_dist, kleinberg_dist):

    os.makedirs(folder, exist_ok=True)

    plt.figure()

    # Gowalla
    x_g = sorted(gowalla_dist.keys())
    y_g = [gowalla_dist[k] for k in x_g]
    plt.plot(x_g, y_g, label="Gowalla")

    # Kleinberg
    x_k = sorted(kleinberg_dist.keys())
    y_k = [kleinberg_dist[k] for k in x_k]
    plt.plot(x_k, y_k, label="Kleinberg")

    plt.xlabel("Path Length")
    plt.ylabel("Probability")
    plt.title(title)
    plt.legend()

    filename = title.replace(" ", "_") + ".png"
    filepath = os.path.join(folder, filename)

    plt.savefig(filepath)
    plt.close()

    print(f"Saved plot: {filepath}")


def l2_distance(dist1, dist2):

    keys = set(dist1.keys()) | set(dist2.keys())

    total = 0
    for k in keys:
        total += (dist1.get(k, 0) - dist2.get(k, 0)) ** 2

    return math.sqrt(total)


# ==============================
# MAIN
# ==============================

def main():

    print("\n===== EXPERIMENT 2^15 =====\n")

    # Load Gowalla
    print("Loading Gowalla...")
    G_full = load_gowalla_graph(EDGE_FILE)
    G_lcc = extract_lcc(G_full)

    print("Running Gowalla trials...")
    gowalla_lengths = run_gowalla_shortest_paths(G_lcc, TRIALS)
    gowalla_dist = compute_distribution(gowalla_lengths)

    export_distribution_to_folder(
        DIST_FOLDER,
        "gowalla_full_distribution.csv",
        gowalla_dist
    )

    results = []

    # Kleinberg experiments
    for dim in DIMENSIONS:

        side_length = compute_side_length(dim, TARGET_NODES)

        print(f"\nGenerating Kleinberg {dim}D (side_length={side_length})")

        G_kleinberg, _ = generate_kleinberg(dim, side_length, r=dim)

        print("Running Kleinberg trials...")
        path_lengths = run_experiments(G_kleinberg, TRIALS)
        kleinberg_dist = compute_distribution(path_lengths)

        export_distribution_to_folder(
            DIST_FOLDER,
            f"kleinberg_dim{dim}.csv",
            kleinberg_dist
        )

        plot_kleinberg_vs_gowalla(
            PLOT_FOLDER,
            f"Kleinberg_{dim}D_vs_Gowalla_2pow15",
            gowalla_dist,
            kleinberg_dist
        )

        l2 = l2_distance(gowalla_dist, kleinberg_dist)
        results.append((dim, side_length, l2))

    print("\n===== RESULTS =====")
    for r in results:
        print(f"Dimension {r[0]} | side_length {r[1]} | L2 = {r[2]}")


if __name__ == "__main__":
    main()