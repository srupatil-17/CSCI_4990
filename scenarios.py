import networkx as nx
import gzip
import random
import math
from collections import Counter
import matplotlib.pyplot as plt
import os
import csv

# -----------------------------
# SETTINGS
# -----------------------------

EDGE_FILE = "data/loc-gowalla_edges.txt.gz"
CHECKIN_FILE = "data/loc-gowalla_totalCheckins.txt.gz"

SAMPLE_SIZE = 100000
TRIALS = 10000

# -----------------------------
# HAVERSINE DISTANCE
# -----------------------------

def haversine(a, b):
    lat1, lon1 = a
    lat2, lon2 = b

    R = 6371

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    x = math.sin(dphi/2)**2 + \
        math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2

    return 2 * R * math.atan2(math.sqrt(x), math.sqrt(1 - x))

# -----------------------------
# LOAD GRAPH
# -----------------------------

def load_social_graph(file, sample_size):

    G = nx.Graph()

    with gzip.open(file, 'rt') as f:
        for line in f:
            u, v = line.strip().split()
            G.add_edge(u, v)

    nodes = list(G.nodes())
    sampled = set(random.sample(nodes, min(sample_size, len(nodes))))

    return G.subgraph(sampled).copy()

# -----------------------------
# LOAD CHECKINS
# -----------------------------

def load_checkins(file, allowed):

    data = {}

    with gzip.open(file, 'rt') as f:
        for line in f:
            parts = line.strip().split()

            user = parts[0]
            if user not in allowed:
                continue

            lat = float(parts[2])
            lon = float(parts[3])

            data.setdefault(user, []).append((lat, lon))

    return data

# -----------------------------
# HOME + LAST LOCATION
# -----------------------------

def compute_home_and_last(user_checkins):

    home = {}
    last = {}

    for user, locs in user_checkins.items():
        home[user] = locs[0]          # first check-in
        last[user] = locs[-1]         # last check-in

    return home, last

# -----------------------------
# SCENARIO 1: SHORTEST PATH
# -----------------------------

def scenario_shortest_path(G, trials):

    nodes = list(G.nodes())
    lengths = []

    for _ in range(trials):

        s = random.choice(nodes)
        t = random.choice(nodes)

        if s == t:
            continue

        try:
            d = nx.shortest_path_length(G, s, t)
            lengths.append(d)
        except:
            continue

    return lengths

# -----------------------------
# GENERIC GREEDY
# -----------------------------

def greedy_route(G, start, target, distance_func, max_steps=1000):

    current = start
    path = [current]
    visited = set([current])

    steps = 0

    while current != target and steps < max_steps:

        neighbors = list(G.neighbors(current))
        if not neighbors:
            return path, False

        best = None
        best_dist = float("inf")

        for n in neighbors:
            d = distance_func(n, target)

            if d < best_dist:
                best_dist = d
                best = n

        if best is None or best in visited:
            return path, False

        current = best
        path.append(current)
        visited.add(current)

        steps += 1

    return path, current == target

# -----------------------------
# SCENARIO 2: HOME → HOME
# -----------------------------

def scenario_home_home(G, home, trials):

    nodes = [u for u in G.nodes() if u in home]
    lengths = []

    def dist(u, t):
        return haversine(home[u], home[t])

    for _ in range(trials):

        s = random.choice(nodes)
        t = random.choice(nodes)

        if s == t:
            continue

        path, ok = greedy_route(G, s, t, dist)

        if ok:
            lengths.append(len(path) - 1)

    return lengths

# -----------------------------
# SCENARIO 3: HOME → TARGET-LAST
# -----------------------------

def scenario_home_target(G, home, last, trials):

    nodes = [u for u in G.nodes() if u in home and u in last]
    lengths = []

    def make_dist(target):
        return lambda u, _: haversine(home[u], last[target])

    for _ in range(trials):

        s = random.choice(nodes)
        t = random.choice(nodes)

        if s == t:
            continue

        dist = make_dist(t)

        path, ok = greedy_route(G, s, t, dist)

        if ok:
            lengths.append(len(path) - 1)

    return lengths

# -----------------------------
# SCENARIO 4: HYBRID
# -----------------------------

def scenario_hybrid(G, home, last, trials):

    nodes = [u for u in G.nodes() if u in home and u in last]
    lengths = []

    for _ in range(trials):

        s = random.choice(nodes)
        t = random.choice(nodes)

        if s == t:
            continue

        current = s
        visited = {s}
        path = [s]

        steps = 0

        while current != t and steps < 1000:

            neighbors = list(G.neighbors(current))
            if not neighbors:
                break

            best = None
            best_dist = float("inf")

            for v in neighbors:
                d = haversine(home[v], home[t])  # decision uses home

                if d < best_dist:
                    best_dist = d
                    best = v

            if best is None or best in visited:
                break

            # conceptually move to target location of best
            current = best
            path.append(current)
            visited.add(current)

            steps += 1

        if current == t:
            lengths.append(len(path) - 1)

    return lengths

# -----------------------------
# DISTRIBUTION SUMMARY
# -----------------------------

def summarize(name, lengths):

    if not lengths:
        print(f"{name}: No data")
        return

    avg = sum(lengths) / len(lengths)

    print(f"\n{name}")
    print("Samples:", len(lengths))
    print("Expected hops:", round(avg, 3))


def compute_distribution(lengths):

    from collections import Counter

    count = Counter(lengths)
    total = sum(count.values())

    return {k: count[k] / total for k in count}


# -----------------------------
# DISTRIBUTION EXPORTING
# -----------------------------

# visual
def plot_distributions(distributions, folder="results", filename=None):

    # create folder
    os.makedirs(folder, exist_ok=True)

    filepath = os.path.join(folder, filename)

    plt.figure(figsize=(10, 6))

    for label, dist in distributions.items():

        if not dist:
            continue  # skips empty distributions

        x = sorted(dist.keys())
        y = [dist[k] for k in x]

        plt.plot(x, y, marker='o', label=label)

    plt.xlabel("Number of Hops")
    plt.ylabel("Probability")
    plt.title("Probability Distribution of Routing Lengths")

    plt.legend()
    plt.grid()

    plt.savefig(filepath)
    plt.close()

    print(f"Saved plot to {filepath}")


# separate for each scenario
def export_distributions_to_csv(distributions, folder="results"):

    os.makedirs(folder, exist_ok=True)

    for label, dist in distributions.items():

        if not dist:
            continue

        # clean filename
        safe_label = label.replace(" ", "_").replace("→", "to")
        filename = f"{safe_label}_distribution.csv"

        filepath = os.path.join(folder, filename)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(["hops", "probability"])

            for k in sorted(dist.keys()):
                writer.writerow([k, dist[k]])

        print(f"Saved CSV to {filepath}")


# combined
def export_combined_csv(distributions, folder="results"):

    os.makedirs(folder, exist_ok=True)

    filepath = os.path.join(folder, "combined_distributions.csv")

    # collect all hop values
    all_keys = set()
    for dist in distributions.values():
        all_keys.update(dist.keys())

    all_keys = sorted(all_keys)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)

        header = ["hops"] + list(distributions.keys())
        writer.writerow(header)

        for k in all_keys:
            row = [k]
            for dist in distributions.values():
                row.append(dist.get(k, 0))
            writer.writerow(row)

    print(f"Saved combined CSV to {filepath}")

# -----------------------------
# MAIN
# -----------------------------

def main():

    print("\n===== LOADING DATA =====")

    G = load_social_graph(EDGE_FILE, SAMPLE_SIZE)

    user_checkins = load_checkins(CHECKIN_FILE, set(G.nodes()))

    home, last = compute_home_and_last(user_checkins)

    valid = set(home.keys()) & set(last.keys())
    G = G.subgraph(valid).copy()

    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    # Run scenarios
    s1 = scenario_shortest_path(G, TRIALS)
    s2 = scenario_home_home(G, home, TRIALS)
    s3 = scenario_home_target(G, home, last, TRIALS)
    s4 = scenario_hybrid(G, home, last, TRIALS)

    # Print results
    summarize("Scenario 1: Shortest Path", s1)
    summarize("Scenario 2: Greedy (Home to Home)", s2)
    summarize("Scenario 3: Greedy (Home to Target)", s3)
    summarize("Scenario 4: Hybrid", s4)

    # Compute distributions
    distributions = {
        "Shortest Path": compute_distribution(s1),
        "Home to Home": compute_distribution(s2),
        "Home to Target": compute_distribution(s3),
        "Hybrid": compute_distribution(s4)
    }

    # Plot
    plot_distributions(distributions, filename="gowalla_plot.png")
    export_distributions_to_csv(distributions)
    export_combined_csv(distributions)


if __name__ == "__main__":
    main()