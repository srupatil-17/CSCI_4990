#EDITING IN PROGRESS

import networkx as nx
import random
import math
import os
import csv
import matplotlib.pyplot as plt
from itertools import product
from collections import Counter
from shortest_path_experiments.two_twelve_dimensional_experiment.gowalla_loader import (
    load_gowalla_graph,
    extract_lcc
)

#can you help me recreate a couple experiments, but put them into one file with this true greedy routing 
#function that would export into a plot, and l2 distributions? This one will be dimensions {4, 8, 12}


# before running dont forget about Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# and then .\venv\Scripts\Activate.ps1

# -----------------------------
# SETTINGS
# -----------------------------

TRIALS = 8000

EDGE_FILE = "data/loc-gowalla_edges.txt.gz"

RESULT_FOLDER = "final_experiment_resultspt2"

SHORTCUT_SAMPLE = 200

# Experiments to run
EXPERIMENTS = [
    (2**16, 4),
    (2**16, 8),
    (2**15, 5)
]

Q_VALUES = [1, 2, 3]


# -----------------------------
# GRID
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
# KLEINBERG
# -----------------------------

def generate_kleinberg(dim, side, q):

    print(f"Generating {dim}D (side={side}) q={q}")

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

                if v != u:

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

def run_trials(G, trials):

    nodes = list(G.nodes())
    lengths = []

    for _ in range(trials):

        start = random.choice(nodes)

        distances = nx.single_source_shortest_path_length(G, start)

        target = random.choice(nodes)

        if target in distances:
            lengths.append(distances[target])

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

    total = 0

    for k in keys:
        total += (a.get(k, 0) - b.get(k, 0))**2

    return math.sqrt(total)


# -----------------------------
# SIDE LENGTH
# -----------------------------

def side_length(nodes, dim):
    return round(nodes ** (1 / dim))


# -----------------------------
# SAVE TABLE
# -----------------------------

def save_table(rows):

    os.makedirs(RESULT_FOLDER, exist_ok=True)

    file = os.path.join(
        RESULT_FOLDER,
        "refined_L2_results.csv"
    )

    with open(file, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "nodes",
            "dimension",
            "q",
            "L2",
            "significant(<0.05)"
        ])

        for r in rows:
            writer.writerow(r)

    print("Saved table:", file)


# -----------------------------
# PLOT
# -----------------------------

def plot_all(gowalla_dist, distributions):

    os.makedirs(RESULT_FOLDER, exist_ok=True)

    plt.figure(figsize=(10,6))

    xg = sorted(gowalla_dist.keys())
    yg = [gowalla_dist[k] for k in xg]

    plt.plot(xg, yg, color="black", linewidth=3, label="Gowalla")

    colors = [
        "red","blue","green",
        "orange","purple","cyan",
        "magenta","brown","gray"
    ]

    i = 0

    for label, dist in distributions.items():

        x = sorted(dist.keys())
        y = [dist[k] for k in x]

        plt.plot(x, y, color=colors[i % len(colors)], label=label)

        i += 1

    plt.xlabel("Path Length")
    plt.ylabel("Probability")
    plt.title("Refined Kleinberg vs Gowalla")

    plt.legend(fontsize=8)

    file = os.path.join(
        RESULT_FOLDER,
        "refined_comparison_plot.png"
    )

    plt.savefig(file)
    plt.close()

    print("Saved plot:", file)


# -----------------------------
# MAIN
# -----------------------------

def main():

    print("\n===== REFINED EXPERIMENT =====\n")

    print("Loading Gowalla...")

    G_full = load_gowalla_graph(EDGE_FILE)
    G = extract_lcc(G_full)

    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    print("Running Gowalla trials...")

    gowalla_lengths = run_trials(G, TRIALS)
    gowalla_dist = compute_distribution(gowalla_lengths)

    results = []
    distributions = {}

    for nodes, dim in EXPERIMENTS:

        side = side_length(nodes, dim)

        for q in Q_VALUES:

            print(f"\nRunning {nodes} nodes, dim={dim}, q={q}")

            Gk = generate_kleinberg(dim, side, q)

            lengths = run_trials(Gk, TRIALS)
            dist = compute_distribution(lengths)

            l2 = l2_distance(gowalla_dist, dist)

            significant = l2 < 0.05

            label = f"{nodes}_dim{dim}_q{q}"
            distributions[label] = dist

            results.append((nodes, dim, q, l2, significant))

            print("L2:", l2, "| Significant:", significant)

    save_table(results)
    plot_all(gowalla_dist, distributions)

    print("\nDone.")


if __name__ == "__main__":
    main()