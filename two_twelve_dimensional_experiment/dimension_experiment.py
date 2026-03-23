import random
import math
import csv
import networkx as nx
from collections import Counter
from itertools import product

# GENERALIZED D-DIMENSIONAL KLEINBERG

def manhattan(u, v):
    return sum(abs(a - b) for a, b in zip(u, v))

def create_nodes(dim, side_length):
    return list(product(range(side_length), repeat=dim))

def add_local_edges(G, nodes):
    dim = len(nodes[0])
    for u in nodes:
        for i in range(dim):
            neighbor_plus = list(u)
            neighbor_plus[i] += 1
            neighbor_plus = tuple(neighbor_plus)

            neighbor_minus = list(u)
            neighbor_minus[i] -= 1
            neighbor_minus = tuple(neighbor_minus)

            if neighbor_plus in G:
                G.add_edge(u, neighbor_plus)
                G.add_edge(neighbor_plus, u)
            
            if neighbor_minus in G:
                G.add_edge(u, neighbor_minus)
                G.add_edge(neighbor_minus, u)

def compute_probability_table(u, nodes, r):
    weights = []
    targets = []

    for v in nodes:
        if u == v:
            continue

        d = manhattan(u, v)
        if d <= 1:

            continue

        weight = 1 / (d ** r)

        targets.append(v)
        weights.append(weight)

    total = sum (weights)
    probs = [w / total for w in weights]

    return targets, probs

def pick_shortcut(u, nodes, r):
    targets, probs = compute_probability_table(u, nodes, r)
    rand = random.random()

    cumulative = 0

    for v, p in zip (targets, probs):
        cumulative += p
        if rand <= cumulative:
            return v
    return targets[-1]

def generate_kleinberg(dim, side_length, r, seed=42):
    random.seed(seed)
    G = nx.DiGraph()
    nodes = create_nodes(dim, side_length)
    G. add_nodes_from(nodes)
    add_local_edges(G, nodes)
    shortcuts = []
    for u in nodes:
        v = pick_shortcut(u, nodes, r)
        G.add_edge(u, v)
        shortcuts.append((u, v))
    return G, shortcuts

# GREEDY ROUTING

def greedy_routing(G, start, target):
    current = start
    path = [current]
    visited = set()
    while current != target:
        visited.add(current)
        neighbors = list(G.neighbors(current))
        if not neighbors:
            return path, False
        best = min(neighbors, key = lambda x: manhattan(x, target))
        if best in visited:
            return path, False
        current = best
        path.append(current)
    return path, True

# RUN EXPERIMENTS

def run_experiments(G, num_trials=10000):
    nodes = list(G.nodes())
    path_lengths = []
    for _ in range(num_trials):
        start = random.choice(nodes)
        target = random.choice(nodes)
        while target == start:
            target = random.choice(nodes)
        path, success = greedy_routing(G, start, target)
        if success:
            path_lengths.append(len(path) - 1)
    return path_lengths

# DISTRIBUTION

def compute_distribution(path_lengths):
    count = Counter(path_lengths)
    total = sum(count.values())
    dist = {}
    for k in count:
        dist[k] = count[k] / total
    return dist

# L2 NORM

def l2_norm(dist1, dist2):
    keys = set(dist1.keys()).union(dist2.keys())
    sum_sq = 0
    for k in keys:
        p = dist1.get(k, 0)
        q = dist2.get(k, 0)

        sum_sq += (p - q) ** 2
        return math.sqrt(sum_sq)
    

# EXPORT

def export_distribution(filename, dist):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path_length", "probability"])
        for k in sorted(dist.keys()):
            writer.writerow([k, dist[k]])

# MAIN EXPERIMENT

if __name__ == "__main__":

    configs = {

        2: 64,
        3: 16,
        4: 8,
        6: 4,
        12: 2

    }

    results = {}

    for dim, side_length in configs.items():

        print("\n===================================")
        print(f"Testing dimension {dim}")
        print(f"Side length {side_length}")
        print(f"Total nodes {side_length ** dim}")
        print("===================================")

        r = dim

        G, shortcuts = generate_kleinberg(dim, side_length, r)

        print("Running greedy routing experiments...")

        path_lengths = run_experiments(G, 10000)

        print("Computing distribution...")

        dist = compute_distribution(path_lengths)

        results[dim] = dist

        filename = f"dist_dim{dim}_4096nodes.csv"

        export_distribution(filename, dist)

        print(f"Saved {filename}")


