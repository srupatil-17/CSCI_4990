import networkx as nx
import random
from collections import Counter
import csv
import math
from gowalla_loader import load_gowalla_graph, extract_lcc, sample_connected_subgraph
from dimension_experiment import generate_kleinberg, run_experiments, compute_distribution
import os


def greedy_routing_gowalla(G, start, target):

    current = start
    path = [current]
    visited = set()

    while current != target:

        visited.add(current)

        neighbors = list(G.neighbors(current))

        if not neighbors:
            return path, False

        # Compute shortest-path distance to target for each neighbor
        distances = []

        for n in neighbors:
            try:
                d = nx.shortest_path_length(G, n, target)
                distances.append((n, d))
            except nx.NetworkXNoPath:
                continue

        if not distances:
            return path, False

        # choose neighbor closest to target
        best = min(distances, key=lambda x: x[1])[0]

        if best in visited:
            return path, False

        current = best
        path.append(current)

    return path, True

def run_gowalla_shortest_paths(G, num_trials=10000):

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

    count = Counter(path_lengths)
    total = sum(count.values())

    dist = {}

    for k in count:
        dist[k] = count[k] / total

    return dist

def export_distribution(filename, dist):

    # ensure distributions folder exists
    os.makedirs("distributions", exist_ok=True)

    filepath = os.path.join("distributions", filename)

    with open(filepath, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(["path_length", "probability"])

        for k in sorted(dist.keys()):
            writer.writerow([k, dist[k]])

    print(f"Saved: {filepath}")


def l2_distance(dist1, dist2):

    keys = set(dist1.keys()) | set(dist2.keys())

    total = 0

    for k in keys:

        p1 = dist1.get(k, 0)
        p2 = dist2.get(k, 0)

        total += (p1 - p2) ** 2

    return math.sqrt(total)

# Check if gowalla is actually random
def inspect_gowalla_sample(G, num_edges_to_print=20, save_to_csv=False):

    print("\n===== GOWALLA SAMPLE INSPECTION =====")

    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())

    edges = list(G.edges())

    print(f"\nShowing {min(num_edges_to_print, len(edges))} random edges:\n")

    sampled_edges = random.sample(edges, min(num_edges_to_print, len(edges)))

    for u, v in sampled_edges:
        print(f"{u} -- {v}")

    if save_to_csv:
        os.makedirs("debug", exist_ok=True)

        filepath = "debug/gowalla_sample_edges.csv"

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["node_u", "node_v"])

            for u, v in edges:
                writer.writerow([u, v])

        print(f"\nSaved all sampled edges to: {filepath}")

    print("===== END INSPECTION =====\n")


# double verifying
def verify_sample_edges(G_original, G_sample, num_checks=10):

    print("\n===== VERIFYING SAMPLE EDGES =====")

    edges = list(G_sample.edges())
    to_check = random.sample(edges, min(num_checks, len(edges)))

    for u, v in to_check:

        if G_original.has_edge(u, v):
            print(f"YES Edge ({u}, {v}) exists in original graph")
        else:
            print(f"NO Edge ({u}, {v}) NOT found in original graph")

    print("===== VERIFICATION COMPLETE =====\n")

# TRIPLE VERIFYING 
def inspect_node_degrees(G, num_nodes=10):

    print("\n===== NODE DEGREE CHECK =====")

    nodes = random.sample(list(G.nodes()), num_nodes)

    for node in nodes:
        print(f"Node {node} has degree {G.degree(node)}")

    print("===== DEGREE CHECK COMPLETE =====\n")





if __name__ == "__main__":

    random.seed(42)

    print("\n===== STARTING EXPERIMENT =====\n")

 
    # PARAMETERS


    trials = 10000
    r = 2  # Kleinberg clustering parameter

    # Keep total nodes = 4096
    dimension_settings = {
        2: 64,   # 64^2 = 4096
        3: 16,   # 16^3 = 4096
        4: 8,    # 8^4  = 4096
        6: 4,    # 4^6  = 4096
        12: 2    # 2^12 = 4096
    }

    kleinberg_distributions = {}

    # RUN KLEINBERG EXPERIMENTS


    for dim, side_length in dimension_settings.items():

        print(f"\nGenerating Kleinberg {dim}D...")

        G, _ = generate_kleinberg(dim, side_length, r=r)

        path_lengths = run_experiments(G, trials)

        dist = compute_distribution(path_lengths)

        filename = f"kleinberg_dim{dim}_distribution.csv"
        export_distribution(filename, dist)

        kleinberg_distributions[dim] = dist



    # LOAD AND PROCESS GOWALLA


    print("\nLoading Gowalla...")

    EDGE_FILE = "data/loc-gowalla_edges.txt.gz"

    G_full = load_gowalla_graph(EDGE_FILE)

    print("Extracting largest connected component...")
    G_lcc = extract_lcc(G_full)

    # checking gowalla for sure for sure
    inspect_gowalla_sample(G_lcc, num_edges_to_print=25, save_to_csv=True)

    # checking again AGAIN
    verify_sample_edges(G_full, G_lcc, 15)

    # checking again again AGAIN
    inspect_node_degrees(G_lcc, 10)

    # last check for real world preservation 
    print("Is graph connected? ", nx.is_connected(G_lcc))

    print("Running Gowalla shortest paths...")
    gowalla_paths = run_gowalla_shortest_paths(G_lcc, trials)

    gowalla_dist = compute_distribution(gowalla_paths)

    export_distribution("gowalla_distribution.csv", gowalla_dist)



    # L2 DISTANCE COMPARISON


    print("\n===== L2 DISTANCE RESULTS =====\n")

    for dim in sorted(kleinberg_distributions.keys()):

        l2 = l2_distance(kleinberg_distributions[dim], gowalla_dist)

        print(f"Dim {dim} vs Gowalla → L2 = {l2:.6f}")



    from plot_utils import plot_multiple_distributions

    files_to_plot = [
        ("distributions/gowalla_distribution.csv", "Gowalla"),
        ("distributions/kleinberg_dim2_distribution.csv", "Dim 2"),
        ("distributions/kleinberg_dim3_distribution.csv", "Dim 3"),
        ("distributions/kleinberg_dim4_distribution.csv", "Dim 4"),
        ("distributions/kleinberg_dim6_distribution.csv", "Dim 6"),
        ("distributions/kleinberg_dim12_distribution.csv", "Dim 12"),
    ]

    plot_multiple_distributions(files_to_plot)

    print("\n===== EXPERIMENT COMPLETE =====")



