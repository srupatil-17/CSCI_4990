import networkx as nx
import random
import math
import os
import csv
import matplotlib.pyplot as plt
from collections import Counter
from itertools import product

from two_twelve_dimensional_experiment.gowalla_loader import (
    load_gowalla_graph,
    extract_lcc
)

# ------------------------------
# SETTINGS
# ------------------------------

TRIALS = 10000
DIMENSION = 12
Q_VALUES = [1, 2, 3]

TARGET_NODES = 2**12   # manageable experiment size

EDGE_FILE = "data/loc-gowalla_edges.txt.gz"

DIST_FOLDER = "q_experiment_distributions"
PLOT_FOLDER = "q_experiment_plots"


# ------------------------------
# GRID FUNCTIONS
# ------------------------------

def create_nodes(dim, side_length):

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
# KLEINBERG GENERATION
# ------------------------------

def generate_kleinberg(dim, side_length, q):

    print(f"Generating {dim}D Kleinberg with q={q}")

    nodes = create_nodes(dim, side_length)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    add_local_edges(G, nodes)

    node_list = list(nodes)

    sample_size = 200

    for u in node_list:

        for _ in range(q):

            candidates = random.sample(
                node_list,
                min(sample_size, len(node_list))
            )

            weights = []

            for v in candidates:
                if v != u:
                    d = manhattan(u, v)

                    if d > 0:
                        weights.append((v, d ** (-dim)))

            total = sum(w for _, w in weights)
            rand = random.random() * total

            cumulative = 0

            for v, w in weights:
                cumulative += w

                if cumulative >= rand:
                    G.add_edge(u, v)
                    break

    return G


# PATH SAMPLING


def run_shortest_path_trials(G, trials):

    nodes = list(G.nodes())
    lengths = []

    for _ in range(trials):

        start = random.choice(nodes)

        distances = nx.single_source_shortest_path_length(G, start)

        target = random.choice(nodes)

        if target in distances:
            lengths.append(distances[target])

    return lengths


# DISTRIBUTION


def compute_distribution(lengths):

    count = Counter(lengths)
    total = sum(count.values())

    return {k: count[k] / total for k in count}


# EXPORT


def export_distribution(filename, dist):

    os.makedirs(DIST_FOLDER, exist_ok=True)

    filepath = os.path.join(DIST_FOLDER, filename)

    with open(filepath, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["path_length", "probability"])

        for k in sorted(dist.keys()):
            writer.writerow([k, dist[k]])

    print("Saved distribution:", filepath)


# PLOTS


def plot_comparison(title, gowalla_dist, kleinberg_dist):

    os.makedirs(PLOT_FOLDER, exist_ok=True)

    plt.figure()

    xg = sorted(gowalla_dist.keys())
    yg = [gowalla_dist[k] for k in xg]

    xk = sorted(kleinberg_dist.keys())
    yk = [kleinberg_dist[k] for k in xk]

    plt.plot(xg, yg, label="Gowalla")
    plt.plot(xk, yk, label="Kleinberg")

    plt.xlabel("Path Length")
    plt.ylabel("Probability")
    plt.title(title)

    plt.legend()

    filename = title.replace(" ", "_") + ".png"

    path = os.path.join(PLOT_FOLDER, filename)

    plt.savefig(path)
    plt.close()

    print("Saved plot:", path)


# L2 DISTANCE


def l2_distance(d1, d2):

    keys = set(d1.keys()) | set(d2.keys())

    total = 0

    for k in keys:
        total += (d1.get(k, 0) - d2.get(k, 0)) ** 2

    return math.sqrt(total)


# DEGREE


def average_degree(G):

    degrees = [d for _, d in G.degree()]
    return sum(degrees) / len(degrees)


# SIDE LENGTH


def compute_side_length(dim, target):

    return round(target ** (1 / dim))

# RESULTS TABLE

def save_results_table(rows):

    filename = "q_experiment_L2_results.csv"

    with open(filename, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "nodes",
            "dimension",
            "q",
            "L2_norm"
        ])

        for r in rows:
            writer.writerow(r)

    print("Saved results table:", filename)


# MAIN


def main():

    print("\n===== Q EXPERIMENTS (12D) =====\n")

    # Load Gowalla

    print("Loading Gowalla...")

    G_full = load_gowalla_graph(EDGE_FILE)
    G_lcc = extract_lcc(G_full)

    print("Nodes:", G_lcc.number_of_nodes())
    print("Edges:", G_lcc.number_of_edges())

    avg_deg = average_degree(G_lcc)
    print("Average Gowalla degree:", avg_deg)

    # Gowalla path distribution

    print("Running Gowalla trials...")

    gowalla_lengths = run_shortest_path_trials(G_lcc, TRIALS)

    gowalla_dist = compute_distribution(gowalla_lengths)

    export_distribution(
        "gowalla_distribution.csv",
        gowalla_dist
    )

    results = []

    for q in Q_VALUES:

        side_length = compute_side_length(DIMENSION, TARGET_NODES)

        G_k = generate_kleinberg(
            DIMENSION,
            side_length,
            q
        )

        print("Running Kleinberg trials...")

        lengths = run_shortest_path_trials(G_k, TRIALS)

        kleinberg_dist = compute_distribution(lengths)

        export_distribution(
            f"kleinberg_12D_q{q}.csv",
            kleinberg_dist
        )

        plot_comparison(
            f"12D_q{q}_vs_Gowalla",
            gowalla_dist,
            kleinberg_dist
        )

        l2 = l2_distance(gowalla_dist, kleinberg_dist)

        results.append((TARGET_NODES, DIMENSION, q, l2))

    save_results_table(results)

    print("\n===== RESULTS =====")

    for r in results:
        print(
            f"nodes={r[0]} dim={r[1]} q={r[2]} L2={r[3]}"
        )


if __name__ == "__main__":
    main()