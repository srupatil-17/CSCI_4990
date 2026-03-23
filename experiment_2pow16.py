import networkx as nx
import random
import math
import os
import csv
import matplotlib.pyplot as plt
from collections import Counter

from two_twelve_dimensional_experiment.gowalla_loader import (
    load_gowalla_graph,
    extract_lcc
)

# ------------------------------
# SETTINGS
# ------------------------------

TRIALS = 10000
TARGET_NODES = 2**16
DIMENSIONS = [2, 4, 8, 16]

EDGE_FILE = "data/loc-gowalla_edges.txt.gz"

DIST_FOLDER = "distributions_2pow16"
PLOT_FOLDER = "plots_2pow16"

SHORTCUT_SAMPLE_SIZE = 200
Q_SHORTCUTS = 1


# ------------------------------
# GRID CREATION
# ------------------------------

def create_nodes(dim, side_length):

    from itertools import product

    ranges = [range(side_length)] * dim
    return list(product(*ranges))


def manhattan(u, v):
    return sum(abs(a - b) for a, b in zip(u, v))


def add_local_edges(G, nodes):

    dim = len(nodes[0])
    side_length = max(coord for node in nodes for coord in node) + 1
    node_set = set(nodes)

    for u in nodes:
        for i in range(dim):

            neighbor = list(u)
            neighbor[i] += 1

            if neighbor[i] < side_length:
                neighbor = tuple(neighbor)

                if neighbor in node_set:
                    G.add_edge(u, neighbor)
                    G.add_edge(neighbor, u)


# ------------------------------
# OPTIMIZED KLEINBERG GENERATION
# ------------------------------

def generate_kleinberg(dim, side_length, r=None, q=1):

    if r is None:
        r = dim

    print(f"Generating {dim}D Kleinberg with side_length={side_length}")

    nodes = create_nodes(dim, side_length)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    add_local_edges(G, nodes)

    node_list = list(nodes)

    for u in node_list:

        for _ in range(q):

            candidates = random.sample(
                node_list,
                min(SHORTCUT_SAMPLE_SIZE, len(node_list))
            )

            weights = []

            for v in candidates:
                if v != u:
                    d = manhattan(u, v)
                    if d > 0:
                        weights.append((v, d ** (-r)))

            total = sum(w for _, w in weights)
            rand = random.random() * total

            cumulative = 0

            for v, w in weights:
                cumulative += w
                if cumulative >= rand:
                    G.add_edge(u, v)
                    break

    return G


# ------------------------------
# FAST PATH SAMPLING
# ------------------------------

def run_shortest_path_trials(G, trials):

    nodes = list(G.nodes())
    path_lengths = []

    for _ in range(trials):

        start = random.choice(nodes)

        distances = nx.single_source_shortest_path_length(G, start)

        target = random.choice(nodes)

        if target in distances:
            path_lengths.append(distances[target])

    return path_lengths


# ------------------------------
# DISTRIBUTION
# ------------------------------

def compute_distribution(lengths):

    count = Counter(lengths)
    total = sum(count.values())

    return {k: count[k] / total for k in count}


# ------------------------------
# EXPORT
# ------------------------------

def export_distribution(folder, filename, dist):

    os.makedirs(folder, exist_ok=True)

    filepath = os.path.join(folder, filename)

    with open(filepath, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["path_length", "probability"])

        for k in sorted(dist.keys()):
            writer.writerow([k, dist[k]])

    print("Saved:", filepath)


# ------------------------------
# PLOT
# ------------------------------

def plot_comparison(title, gowalla_dist, kleinberg_dist):

    os.makedirs(PLOT_FOLDER, exist_ok=True)

    plt.figure()

    x_g = sorted(gowalla_dist.keys())
    y_g = [gowalla_dist[k] for k in x_g]

    x_k = sorted(kleinberg_dist.keys())
    y_k = [kleinberg_dist[k] for k in x_k]

    plt.plot(x_g, y_g, label="Gowalla")
    plt.plot(x_k, y_k, label="Kleinberg")

    plt.xlabel("Path Length")
    plt.ylabel("Probability")

    plt.title(title)
    plt.legend()

    filename = title.replace(" ", "_") + ".png"

    filepath = os.path.join(PLOT_FOLDER, filename)

    plt.savefig(filepath)
    plt.close()

    print("Saved plot:", filepath)


# ------------------------------
# L2 DISTANCE
# ------------------------------

def l2_distance(d1, d2):

    keys = set(d1.keys()) | set(d2.keys())

    total = 0

    for k in keys:
        total += (d1.get(k, 0) - d2.get(k, 0)) ** 2

    return math.sqrt(total)


# ------------------------------
# SIDE LENGTH
# ------------------------------

def compute_side_length(dim, target):

    return round(target ** (1 / dim))


# ------------------------------
# MAIN
# ------------------------------

def main():

    print("\n===== 2^16 EXPERIMENT =====\n")

    # Load Gowalla

    print("Loading Gowalla...")

    G_full = load_gowalla_graph(EDGE_FILE)

    G_lcc = extract_lcc(G_full)

    print("Running Gowalla trials...")

    gowalla_lengths = run_shortest_path_trials(G_lcc, TRIALS)

    gowalla_dist = compute_distribution(gowalla_lengths)

    export_distribution(
        DIST_FOLDER,
        "gowalla_distribution.csv",
        gowalla_dist
    )

    results = []

    for dim in DIMENSIONS:

        side_length = compute_side_length(dim, TARGET_NODES)

        G_k = generate_kleinberg(
            dim,
            side_length,
            r=dim,
            q=Q_SHORTCUTS
        )

        print("Running Kleinberg trials...")

        k_lengths = run_shortest_path_trials(G_k, TRIALS)

        k_dist = compute_distribution(k_lengths)

        export_distribution(
            DIST_FOLDER,
            f"kleinberg_dim{dim}.csv",
            k_dist
        )

        plot_comparison(
            f"Kleinberg_{dim}D_vs_Gowalla_2pow16",
            gowalla_dist,
            k_dist
        )

        l2 = l2_distance(gowalla_dist, k_dist)

        results.append((dim, side_length, Q_SHORTCUTS, l2))

    print("\n===== RESULTS =====")

    for r in results:
        print(
            f"Dimension {r[0]} | side_length {r[1]} | q={r[2]} | L2={r[3]}"
        )


if __name__ == "__main__":
    main()