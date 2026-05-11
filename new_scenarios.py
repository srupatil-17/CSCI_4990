import networkx as nx
import gzip
import math
import os
import csv
from collections import Counter
import matplotlib.pyplot as plt


# before running dont forget about Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# and then .\venv\Scripts\Activate.ps1

# -----------------------------
# SETTINGS
# -----------------------------

EDGE_FILE = "data/loc-gowalla_edges.txt.gz"
CHECKIN_FILE = "data/loc-gowalla_totalCheckins.txt.gz"

RESULT_FOLDER = "new_scenario_results"

MAX_STEPS = 1000 # to make sure greedy doesn't run forever

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

    x = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2) ** 2
    )

    return 2 * R * math.atan2(math.sqrt(x), math.sqrt(1 - x))


# -----------------------------
# LOAD SOCIAL GRAPH
# -----------------------------

def load_social_graph(file):

    G = nx.Graph()

    with gzip.open(file, "rt") as f:

        for line in f:

            u, v = line.strip().split()

            G.add_edge(u, v)

    return G


# -----------------------------
# LOAD CHECKINS
# -----------------------------

def load_checkins(file):

    user_checkins = {}

    with gzip.open(file, "rt") as f:

        for line in f:

            parts = line.strip().split()

            user = parts[0]

            lat = float(parts[2])
            lon = float(parts[3])

            user_checkins.setdefault(user, []).append((lat, lon))

    return user_checkins


# -----------------------------
# HOME + CURRENT LOCATIONS
# -----------------------------

def compute_locations(user_checkins):

    home = {}
    current = {}

    for user, locs in user_checkins.items():

        if len(locs) == 0:
            continue

        home[user] = locs[0]
        current[user] = locs[-1]

    return home, current


# -----------------------------
# FILTER GRAPH
# -----------------------------

def filter_graph(G, home, current):

    valid = set(home.keys()) & set(current.keys())

    return G.subgraph(valid).copy()


# -----------------------------
# SHORTEST PATH
# -----------------------------

def shortest_path_experiment(G):

    lengths = []
    rows = []

    nodes = list(G.nodes())

    for source in nodes:

        distances = nx.single_source_shortest_path_length(G, source)

        for target in nodes:

            if source == target:
                continue

            if target in distances:

                hops = distances[target]

                lengths.append(hops)

                rows.append([
                    source,
                    target,
                    hops
                ])

    return lengths, rows


# -----------------------------
# GREEDY ROUTING
# -----------------------------

def greedy_route(
    G,
    source,
    target,
    neighbor_location_func,
    target_location_func
):

    current = source

    path = [current]
    visited = set([current])

    steps = 0

    while current != target and steps < MAX_STEPS:

        neighbors = list(G.neighbors(current))

        if not neighbors: 
            return path, False, "dead_end"
        # when the route reaches a node that is farther away from the target than the node before it
        # largely due to loops? will be recorded

        best = None
        best_dist = float("inf")

        target_loc = target_location_func(target)

        for n in neighbors:

            neighbor_loc = neighbor_location_func(n)

            d = haversine(neighbor_loc, target_loc)

            if d < best_dist:

                best_dist = d
                best = n

        if best is None:
            return path, False, "no_choice"

        # cycle detection
        if best in visited:
            path.append(best)
            return path, False, "cycle"

        current = best

        path.append(current)
        visited.add(current)

        steps += 1

    if current == target:
        return path, True, "success"

    return path, False, "max_steps"


# -----------------------------
# GENERIC GREEDY EXPERIMENT
# -----------------------------

def run_greedy_experiment(
    G,
    scenario_name,
    neighbor_location_func,
    target_location_func
):

    lengths = []

    success_rows = []
    loop_rows = []

    nodes = list(G.nodes())

    total_pairs = 0

    for source in nodes:

        for target in nodes:

            if source == target:
                continue

            total_pairs += 1

            path, success, reason = greedy_route(
                G,
                source,
                target,
                neighbor_location_func,
                target_location_func
            )

            if success:

                hops = len(path) - 1

                lengths.append(hops)

                success_rows.append([
                    source,
                    target,
                    hops
                ])

            else:

                loop_rows.append([
                    source,
                    target,
                    reason,
                    len(path) - 1,
                    path
                ])

    print(f"\n{scenario_name}")
    print("Total pairs:", total_pairs)
    print("Successful routes:", len(success_rows))
    print("Failures:", len(loop_rows))

    return lengths, success_rows, loop_rows


# -----------------------------
# DISTRIBUTION
# -----------------------------

def compute_distribution(lengths):

    count = Counter(lengths)

    total = sum(count.values())

    return {
        k: count[k] / total
        for k in count
    }


# -----------------------------
# SAVE CSV
# -----------------------------

def save_csv(rows, filename, header):

    os.makedirs(RESULT_FOLDER, exist_ok=True)

    filepath = os.path.join(RESULT_FOLDER, filename)

    with open(filepath, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(header)

        writer.writerows(rows)

    print("Saved:", filepath)


# -----------------------------
# PLOT DISTRIBUTIONS
# -----------------------------

def plot_distributions(distributions):

    os.makedirs(RESULT_FOLDER, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for label, dist in distributions.items():

        if not dist:
            continue

        x = sorted(dist.keys())
        y = [dist[k] for k in x]

        plt.plot(x, y, marker="o", label=label)

    plt.xlabel("Number of Hops")
    plt.ylabel("Probability")

    plt.title("Routing Distributions")

    plt.legend()
    plt.grid()

    filepath = os.path.join(
        RESULT_FOLDER,
        "scenario_distributions.png"
    )

    plt.savefig(filepath)
    plt.close()

    print("Saved:", filepath)


# -----------------------------
# MAIN
# -----------------------------

def main():

    print("\n===== LOADING DATA =====")

    G = load_social_graph(EDGE_FILE)

    user_checkins = load_checkins(CHECKIN_FILE)

    home, current = compute_locations(user_checkins)

    G = filter_graph(G, home, current)

    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    distributions = {}

    # -----------------------------
    # SCENARIO 1
    # -----------------------------

    print("Running scenario 1....")

    s1_lengths, s1_rows = shortest_path_experiment(G)

    save_csv(
        s1_rows,
        "scenario1_shortest_paths.csv",
        ["source", "target", "hops"]
    )

    distributions["Shortest Path"] = compute_distribution(s1_lengths)

    # -----------------------------
    # SCENARIO 2
    # CURRENT -> CURRENT
    # -----------------------------

    print("Running scenario 2....")

    s2_lengths, s2_success, s2_loops = run_greedy_experiment(
        G,
        "Scenario 2: Current to Current",
        lambda u: current[u],
        lambda t: current[t]
    )


    save_csv(
        s2_success,
        "scenario2_success.csv",
        ["source", "target", "hops"]
    )

    save_csv(
        s2_loops,
        "scenario2_loops.csv",
        ["source", "target", "reason", "steps", "path"]
    )

    distributions["Current->Current"] = compute_distribution(s2_lengths)

    # -----------------------------
    # SCENARIO 3
    # HOME -> HOME
    # -----------------------------

    print("Running scenario 3....")

    s3_lengths, s3_success, s3_loops = run_greedy_experiment(
        G,
        "Scenario 3: Home to Home",
        lambda u: home[u],
        lambda t: home[t]
    )

    save_csv(
        s3_success,
        "scenario3_success.csv",
        ["source", "target", "hops"]
    )

    save_csv(
        s3_loops,
        "scenario3_loops.csv",
        ["source", "target", "reason", "steps", "path"]
    )

    distributions["Home->Home"] = compute_distribution(s3_lengths)

    # -----------------------------
    # SCENARIO 4
    # HOME -> CURRENT
    # -----------------------------

    print("Running scenario 4....")

    s4_lengths, s4_success, s4_loops = run_greedy_experiment(
        G,
        "Scenario 4: Home to Current",
        lambda u: home[u],
        lambda t: current[t]
    )

    save_csv(
        s4_success,
        "scenario4_success.csv",
        ["source", "target", "hops"]
    )

    save_csv(
        s4_loops,
        "scenario4_loops.csv",
        ["source", "target", "reason", "steps", "path"]
    )

    distributions["Home->Current"] = compute_distribution(s4_lengths)

    # -----------------------------
    # SCENARIO 5
    # CURRENT -> HOME
    # -----------------------------

    print("Running scenario 5....")

    s5_lengths, s5_success, s5_loops = run_greedy_experiment(
        G,
        "Scenario 5: Current to Home",
        lambda u: current[u],
        lambda t: home[t]
    )

    save_csv(
        s5_success,
        "scenario5_success.csv",
        ["source", "target", "hops"]
    )

    save_csv(
        s5_loops,
        "scenario5_loops.csv",
        ["source", "target", "reason", "steps", "path"]
    )

    distributions["Current->Home"] = compute_distribution(s5_lengths)

    # -----------------------------
    # PLOT
    # -----------------------------

    plot_distributions(distributions)

    print("\n===== COMPLETE =====")


if __name__ == "__main__":
    main()