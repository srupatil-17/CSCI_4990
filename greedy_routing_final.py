import networkx as nx
import random
import math
import os
import csv
import matplotlib.pyplot as plt
from itertools import product
from collections import Counter

from true_greedy_routing_experiments.true_greedy_routing import greedy_route, lattice_distance, graph_distance

from shortest_path_experiments.two_twelve_dimensional_experiment.gowalla_loader import extract_lcc, load_gowalla_graph

# -----------------------------
# SETTINGS
# -----------------------------

TRIALS = 8000
TARGET_NODES = 2**12

DIMENSIONS = [4, 5, 8, 12]
Q_VALUES = [1, 2, 3]

EDGE_FILE = "data/loc-gowalla_edges.txt.gz"

RESULT_FOLDER = "final_greedy_experiment_results"

SHORTCUT_SAMPLE = 200



# -----------------------------
# GRID UTILITIES
# -----------------------------

def create_nodes(dim, side):

    ranges = [range(side)] * dim
    return list(product(*ranges))


def manhattan(a, b):

    return sum(abs(x - y) for x, y in zip(a, b))


def add_local_edges(G, nodes):

    dim = len(nodes[0])
    side = max(c for node in nodes for c in node) + 1

    node_set = set(nodes)

    for u in nodes:

        for i in range(dim):

            v = list(u)
            v[i] += 1

            if v[i] < side:

                v = tuple(v)

                if v in node_set:
                    G.add_edge(u, v)
                    G.add_edge(v, u)


# -----------------------------
# KLEINBERG GENERATOR
# -----------------------------

def generate_kleinberg(dim, side, q):

    print(f"Generating {dim}D Kleinberg (q={q})")

    nodes = create_nodes(dim, side)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    add_local_edges(G, nodes)

    node_list = list(nodes)

    for u in node_list:

        for _ in range(q):

            candidates = random.sample(
                node_list,
                min(SHORTCUT_SAMPLE, len(node_list))
            )

            weights = []

            for v in candidates:

                if v != u and manhattan(u, v) > 1:

                    d = manhattan(u, v)

                    if d > 0:
                        weights.append((v, d ** (-dim)))

            total = sum(w for _, w in weights)
            r = random.random() * total

            cumulative = 0

            for v, w in weights:

                cumulative += w

                if cumulative >= r:
                    G.add_edge(u, v)
                    break

    return G


# -----------------------------
# PATH SAMPLING
# -----------------------------

def run_trials_greedy(G, trials):

    nodes = list(G.nodes())
    lengths = []

    for _ in range(trials):

        start = random.choice(nodes)
        target = random.choice(nodes)

        if start == target:
            continue

        path, success = greedy_route(
            G,
            start,
            target,
            lattice_distance
        )

        if success:
            lengths.append(len(path) - 1)

    return lengths


# -----------------------------
# DISTRIBUTION
# -----------------------------

def compute_distribution(lengths):

    count = Counter(lengths)

    total = sum(count.values())

    return {k: count[k] / total for k in count}


# -----------------------------
# L2 DISTANCE
# -----------------------------

def l2_distance(a, b):

    keys = set(a.keys()) | set(b.keys())

    s = 0

    for k in keys:

        s += (a.get(k, 0) - b.get(k, 0))**2

    return math.sqrt(s)


# -----------------------------
# SIDE LENGTH
# -----------------------------

def side_length(dim):

    return round(TARGET_NODES ** (1 / dim))


# -----------------------------
# SAVE RESULTS TABLE
# -----------------------------

def save_table(rows):

    os.makedirs(RESULT_FOLDER, exist_ok=True)

    file = os.path.join(
        RESULT_FOLDER,
        "master_L2_results.csv"
    )

    with open(file, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "nodes",
            "dimension",
            "q",
            "L2_norm"
        ])

        for r in rows:
            writer.writerow(r)

    print("Saved table:", file)


# -----------------------------
# MASTER PLOT
# -----------------------------

def plot_all(gowalla_dist, distributions):

    os.makedirs(RESULT_FOLDER, exist_ok=True)

    plt.figure(figsize=(10,6))

    xg = sorted(gowalla_dist.keys())
    yg = [gowalla_dist[k] for k in xg]

    plt.plot(xg, yg, linewidth=3, label="Gowalla", color="black")

    colors = [
        "red","orange","gold",
        "green","blue","purple",
        "cyan","magenta","brown",
        "gray","pink","olive"
    ]

    i = 0

    for label, dist in distributions.items():

        x = sorted(dist.keys())
        y = [dist[k] for k in x]

        plt.plot(x, y, color=colors[i % len(colors)], label=label)

        i += 1

    plt.xlabel("Path Length")
    plt.ylabel("Probability")

    plt.title("Kleinberg vs Gowalla Path Distributions - True Greedy V.")

    plt.legend(fontsize=8)

    file = os.path.join(
        RESULT_FOLDER,
        "all_dimension_q_comparison.png"
    )

    plt.savefig(file)
    plt.close()

    print("Saved plot:", file)


# -----------------------------
# RUN TRIALS
# -----------------------------

def run_trials(G, trials, mode="greedy"):

    nodes = list(G.nodes())
    lengths = []

    for _ in range(trials):

        start = random.choice(nodes)
        target = random.choice(nodes)

        if start == target:
            continue

        # choose routing type
        if mode == "greedy":
            path, success = greedy_route(
                G,
                start,
                target,
                lattice_distance
            )
        else:  # graph / optimal
            path, success = greedy_route(
                G,
                start,
                target,
                graph_distance,
                cache={"G": G}
            )

        if success:
            lengths.append(len(path) - 1)

    return lengths



def run_trials_graph(G, trials):

    nodes = list(G.nodes())
    lengths = []

    for _ in range(trials):

        start = random.choice(nodes)
        target = random.choice(nodes)

        if start == target:
            continue

        path, success = greedy_route(
            G,
            start,
            target,
            graph_distance,
            cache={"G": G}
        )

        if success:
            lengths.append(len(path) - 1)

    return lengths

# -----------------------------
# MAIN
# -----------------------------

def main():

    print("\n===== FINAL EXPERIMENT =====\n")

    print("Loading Gowalla...")

    G_full = load_gowalla_graph(EDGE_FILE)

    G = extract_lcc(G_full)

    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    print("Running Gowalla trials...")

    gowalla_lengths = run_trials(G, TRIALS, mode="graph")

    gowalla_dist = compute_distribution(gowalla_lengths)

    results = []

    distributions = {}

    for dim in DIMENSIONS:

        side = side_length(dim)

        for q in Q_VALUES:

            print(f"\nRunning dimension {dim}, q={q}")

            Gk = generate_kleinberg(dim, side, q)

            lengths = run_trials(Gk, TRIALS, mode="greedy")

            dist = compute_distribution(lengths)

            l2 = l2_distance(gowalla_dist, dist)

            label = f"{dim}D_q{q}"

            distributions[label] = dist

            results.append((TARGET_NODES, dim, q, l2))

            print("L2:", l2)

    save_table(results)

    plot_all(gowalla_dist, distributions)

    print("\nExperiment complete.")



if __name__ == "__main__":
    main()