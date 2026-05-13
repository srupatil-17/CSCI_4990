import networkx as nx
import gzip
import math
import csv
import os
from collections import Counter
import matplotlib.pyplot as plt

# before running dont forget about Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# and then .\venv\Scripts\Activate.ps1

# -----------------------------
# SETTINGS
# -----------------------------

EDGE_FILE = "data/loc-gowalla_edges.txt.gz"
CHECKIN_FILE = "data/loc-gowalla_totalCheckins.txt.gz"

MAX_STEPS = 1000

RESULT_FOLDER = "scenario_results"

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

    x = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(dlambda / 2) ** 2

    return 2 * R * math.atan2(math.sqrt(x), math.sqrt(1 - x))

# -----------------------------
# LOAD SOCIAL GRAPH
# -----------------------------

def load_social_graph(file):

    G = nx.Graph()

    with gzip.open(file, 'rt') as f:

        for line in f:

            u, v = line.strip().split()

            G.add_edge(u, v)

    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    return G

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
# HOME + CURRENT LOCATIONS
# -----------------------------

def compute_home_and_current(user_checkins):

    home = {}
    current = {}

    for user, locs in user_checkins.items():

        if len(locs) == 0:
            continue

        # first checkin
        home[user] = locs[0]

        # last checkin
        current[user] = locs[-1]

    return home, current

# -----------------------------
# GREEDY ROUTING
# -----------------------------

def greedy_route(
    G,
    start,
    target,
    neighbor_locations,
    target_locations,
    max_steps=MAX_STEPS
):

    current = start

    visited = set([current])

    path = [current]

    steps = 0

    while current != target and steps < max_steps:

        neighbors = list(G.neighbors(current))

        if not neighbors:
            return path, False, "dead_end"

        best = None
        best_dist = float("inf")

        for n in neighbors:

            if n not in neighbor_locations:
                continue

            d = haversine(
                neighbor_locations[n],
                target_locations[target]
            )

            if d < best_dist:

                best_dist = d
                best = n

        if best is None:
            return path, False, "no_choice"

        # loop detection
        if best in visited:

            path.append(best)

            return path, False, "loop"

        current = best

        path.append(current)

        visited.add(current)

        steps += 1

    if current == target:
        return path, True, "success"

    return path, False, "max_steps"

# -----------------------------
# SCENARIO 1
# SHORTEST PATH
# -----------------------------

def shortest_path_experiment(G):

    distribution = Counter()

    results_file = os.path.join(
        RESULT_FOLDER,
        "scenario1_shortest_paths.csv"
    )

    disconnected_file = os.path.join(
        RESULT_FOLDER,
        "scenario1_disconnected.csv"
    )

    nodes = list(G.nodes())

    with open(results_file, "w", newline="") as rf, \
         open(disconnected_file, "w", newline="") as df:

        writer = csv.writer(rf)
        disconnected_writer = csv.writer(df)

        writer.writerow([
            "source",
            "target",
            "hops"
        ])

        disconnected_writer.writerow([
            "source",
            "target"
        ])

        total = len(nodes)

        for i, source in enumerate(nodes):

            print(f"Scenario 1 source {i+1}/{total}")

            distances = nx.single_source_shortest_path_length(
                G,
                source
            )

            for target in nodes:

                if source == target:
                    continue

                if target in distances:

                    hops = distances[target]

                    writer.writerow([
                        source,
                        target,
                        hops
                    ])

                    distribution[hops] += 1

                else:

                    disconnected_writer.writerow([
                        source,
                        target
                    ])

    return distribution

# -----------------------------
# GENERIC GREEDY EXPERIMENT
# -----------------------------

def greedy_experiment(
    G,
    neighbor_locations,
    target_locations,
    scenario_name,
    prefix
):

    distribution = Counter()

    results_file = os.path.join(
        RESULT_FOLDER,
        f"{prefix}_results.csv"
    )

    failure_file = os.path.join(
        RESULT_FOLDER,
        f"{prefix}_failures.csv"
    )

    nodes = list(G.nodes())

    with open(results_file, "w", newline="") as rf, \
         open(failure_file, "w", newline="") as ff:

        writer = csv.writer(rf)
        failure_writer = csv.writer(ff)

        writer.writerow([
            "source",
            "target",
            "hops"
        ])

        failure_writer.writerow([
            "source",
            "target",
            "failure_type",
            "path"
        ])

        total = len(nodes)

        for i, source in enumerate(nodes):

            print(f"{scenario_name} source {i+1}/{total}")

            for target in nodes:

                if source == target:
                    continue

                path, ok, reason = greedy_route(
                    G,
                    source,
                    target,
                    neighbor_locations,
                    target_locations
                )

                if ok:

                    hops = len(path) - 1

                    writer.writerow([
                        source,
                        target,
                        hops
                    ])

                    distribution[hops] += 1

                else:

                    failure_writer.writerow([
                        source,
                        target,
                        reason,
                        path
                    ])

    return distribution

# -----------------------------
# NORMALIZE DISTRIBUTION
# -----------------------------

def normalize_distribution(distribution):

    total = sum(distribution.values())

    if total == 0:
        return {}

    return {
        k: v / total
        for k, v in distribution.items()
    }

# -----------------------------
# EXPECTED VALUE
# -----------------------------

def expected_value(distribution):

    total = sum(distribution.values())

    if total == 0:
        return 0

    s = 0

    for hops, count in distribution.items():

        s += hops * count

    return s / total

# -----------------------------
# EXPORT DISTRIBUTIONS
# -----------------------------

def export_distribution_csvs(distributions):

    for label, dist in distributions.items():

        safe = label.replace(" ", "_")

        file = os.path.join(
            RESULT_FOLDER,
            f"{safe}_distribution.csv"
        )

        with open(file, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                "hops",
                "probability"
            ])

            for hops in sorted(dist.keys()):

                writer.writerow([
                    hops,
                    dist[hops]
                ])

        print("Saved:", file)

# -----------------------------
# PLOT DISTRIBUTIONS
# -----------------------------

def plot_distributions(distributions):

    plt.figure(figsize=(10, 6))

    for label, dist in distributions.items():

        if not dist:
            continue

        x = sorted(dist.keys())

        y = [dist[k] for k in x]

        plt.plot(
            x,
            y,
            marker='o',
            label=label
        )

    plt.xlabel("Number of Hops")

    plt.ylabel("Probability")

    plt.title("Routing Probability Distributions")

    plt.legend()

    plt.grid()

    file = os.path.join(
        RESULT_FOLDER,
        "scenario_distributions.png"
    )

    plt.savefig(file)

    plt.close()

    print("Saved plot:", file)

# -----------------------------
# MAIN
# -----------------------------

def main():

    os.makedirs(RESULT_FOLDER, exist_ok=True)

    print("\n===== LOADING GRAPH =====")

    G = load_social_graph(EDGE_FILE)

    print("\n===== LOADING CHECKINS =====")

    user_checkins = load_checkins(
        CHECKIN_FILE,
        set(G.nodes())
    )

    home, current = compute_home_and_current(
        user_checkins
    )

    valid = set(home.keys()) & set(current.keys())

    G = G.subgraph(valid).copy()

    print("\nFiltered nodes:", G.number_of_nodes())
    print("Filtered edges:", G.number_of_edges())

    # ---------------------------------
    # Scenario 1
    # shortest path
    # ---------------------------------

    print("Computing Scenario 1...")

    print("\n===== SCENARIO 1 =====")

    s1 = shortest_path_experiment(G)

    # ---------------------------------
    # Scenario 2
    # current -> current
    # ---------------------------------

    print("Computing Scenario 2...")

    print("\n===== SCENARIO 2 =====")

    s2 = greedy_experiment(
        G,
        current,
        current,
        "Scenario 2",
        "scenario2_current_current"
    )

    # ---------------------------------
    # Scenario 3
    # home -> home
    # ---------------------------------

    print("Computing Scenario 3...")

    print("\n===== SCENARIO 3 =====")

    s3 = greedy_experiment(
        G,
        home,
        home,
        "Scenario 3",
        "scenario3_home_home"
    )

    # ---------------------------------
    # Scenario 4
    # home -> current
    # ---------------------------------

    print("Computing Scenario 4...")

    print("\n===== SCENARIO 4 =====")

    s4 = greedy_experiment(
        G,
        home,
        current,
        "Scenario 4",
        "scenario4_home_current"
    )

    # ---------------------------------
    # Scenario 5
    # current -> home
    # ---------------------------------

    print("Computing Scenario 5...")

    print("\n===== SCENARIO 5 =====")

    s5 = greedy_experiment(
        G,
        current,
        home,
        "Scenario 5",
        "scenario5_current_home"
    )

    # ---------------------------------
    # NORMALIZE
    # ---------------------------------

    distributions = {

        "Shortest Path":
            normalize_distribution(s1),

        "Current→Current":
            normalize_distribution(s2),

        "Home→Home":
            normalize_distribution(s3),

        "Home→Current":
            normalize_distribution(s4),

        "Current→Home":
            normalize_distribution(s5)
    }

    # ---------------------------------
    # EXPECTED VALUES
    # ---------------------------------

    print("\n===== EXPECTED VALUES =====")

    scenarios = [
        ("Scenario 1", s1),
        ("Scenario 2", s2),
        ("Scenario 3", s3),
        ("Scenario 4", s4),
        ("Scenario 5", s5)
    ]

    for name, dist in scenarios:

        total = sum(dist.values())

        exp = expected_value(dist)

        print(name)
        print("Samples:", total)
        print("Expected hops:", round(exp, 3))
        print()

    # ---------------------------------
    # EXPORT DISTRIBUTIONS
    # ---------------------------------

    export_distribution_csvs(distributions)

    # ---------------------------------
    # PLOT
    # ---------------------------------

    plot_distributions(distributions)

    print("\n===== DONE =====")


if __name__ == "__main__":
    main()