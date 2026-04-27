import networkx as nx
import gzip
import random
import math
from collections import Counter

# -----------------------------
# SETTINGS
# -----------------------------

EDGE_FILE = "data/loc-gowalla_edges.txt.gz"
CHECKIN_FILE = "data/loc-gowalla_totalCheckins.txt.gz"

SAMPLE_SIZE = 50000
TRIALS = 2000
DELTA = 0.2   # tolerance for ratio

# -----------------------------
# HAVERSINE
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

        # last check-in
        last[user] = locs[-1]

        # home = most frequent
        counts = Counter(locs)
        max_count = max(counts.values())

        candidates = [l for l, c in counts.items() if c == max_count]

        if len(candidates) == 1:
            home[user] = candidates[0]
        else:
            for loc in reversed(locs):
                if loc in candidates:
                    home[user] = loc
                    break

    return home, last

# -----------------------------
# GREEDY (GRAPH DISTANCE)
# -----------------------------

def greedy_route(G, start, target, coords, max_steps=1000):

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

            # ONLY local information
            if n not in coords or target not in coords:
                continue

            d = haversine(coords[n], coords[target])

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
# FINAL EXPERIMENT
# -----------------------------

def prediction_ratio_experiment(G, home, last, trials, delta):

    users = [u for u in G.nodes() if u in home and u in last]

    success = 0
    total = 0

    success = 0
    total = 0

    for _ in range(trials):

        start = random.choice(users)
        target = random.choice(users)

        if start == target:
            continue

        path, ok = greedy_route(G, start, target, home)

        # only requirement: at least one move
        if len(path) < 2:
            continue

        u = path[-2]

        predicted = haversine(home[u], last[target])
        actual = haversine(home[start], last[target])

        if actual == 0:
            continue

        ratio = predicted / actual

        if (1 - delta) <= ratio <= (1 + delta):
            success += 1

        total += 1

    print("\n===== RATIO EXPERIMENT =====")
    print("Trials:", total)
    print("Within delta:", success)

    if total > 0:
        print("Accuracy:", success / total)

# -----------------------------
# MAIN
# -----------------------------

def main():

    print("\n===== GOWALLA PREDICTION (RATIO + SECOND-LAST) =====\n")

    print("Loading graph...")
    G = load_social_graph(EDGE_FILE, SAMPLE_SIZE)

    print("Loading checkins...")
    user_checkins = load_checkins(CHECKIN_FILE, set(G.nodes()))

    home, last = compute_home_and_last(user_checkins)

    # filter graph
    valid = set(home.keys()) & set(last.keys())
    G = G.subgraph(valid).copy()

    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    prediction_ratio_experiment(G, home, last, TRIALS, DELTA)


if __name__ == "__main__":
    main()