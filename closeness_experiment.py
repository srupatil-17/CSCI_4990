import networkx as nx
import gzip
import random
import math
from collections import Counter

# before running dont forget about Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# and then .\venv\Scripts\Activate.ps1

# -----------------------------
# SETTINGS
# -----------------------------

CHECKIN_FILE = "data/loc-gowalla_totalCheckins.txt.gz"
TRIALS = 200
LOCAL_K = 4   # number of local neighbors

# -----------------------------
# HAVERSINE DISTANCE
# -----------------------------

def haversine(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    R = 6371  # Earth radius in km

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2

    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# -----------------------------
# LOAD CHECKINS
# -----------------------------

def load_checkins(file):

    user_checkins = {}

    with gzip.open(file, 'rt') as f:
        for line in f:
            parts = line.strip().split()

            user = parts[0]
            lat = float(parts[2])
            lon = float(parts[3])

            if user not in user_checkins:
                user_checkins[user] = []

            user_checkins[user].append((lat, lon))

    return user_checkins

# -----------------------------
# HOME LOCATION
# -----------------------------

def compute_home_locations(user_checkins):

    home = {}

    for user, locs in user_checkins.items():
        counts = Counter(locs)
        home[user] = counts.most_common(1)[0][0]

    return home

# -----------------------------
# LOCAL EDGES (NEAREST NEIGHBORS)
# -----------------------------

def add_local_edges(G, home, k=4):

    users = list(home.keys())

    for u in users:

        distances = []

        for v in users:
            if u == v:
                continue

            d = haversine(home[u], home[v])
            distances.append((v, d))

        distances.sort(key=lambda x: x[1])

        for v, _ in distances[:k]:
            G.add_edge(u, v)

# -----------------------------
# SHORTCUTS
# -----------------------------

def build_home_lookup(home):
    return {loc: user for user, loc in home.items()}

def add_shortcuts(G, user_checkins, home_lookup):

    shortcuts = []

    for u, checkins in user_checkins.items():

        for loc in checkins:

            if loc in home_lookup:

                v = home_lookup[loc]

                if u != v:
                    G.add_edge(u, v)
                    shortcuts.append((u, v))

    return shortcuts

# -----------------------------
# GREEDY ROUTING
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
# BUILD GRAPH
# -----------------------------

def build_graph(user_checkins):

    print("Computing home locations...")
    home = compute_home_locations(user_checkins)

    print("Building graph...")
    G = nx.Graph()

    for user in home:
        G.add_node(user)

    print("Adding local edges...")
    add_local_edges(G, home, LOCAL_K)

    print("Adding shortcuts...")
    home_lookup = build_home_lookup(home)
    shortcuts = add_shortcuts(G, user_checkins, home_lookup)

    return G, home, shortcuts

# -----------------------------
# RUN TRIALS
# -----------------------------

def run_trials(G, coords, trials):

    users = list(G.nodes())

    successes = 0
    lengths = []

    for _ in range(trials):

        start = random.choice(users)
        target = random.choice(users)

        if start == target:
            continue

        path, success = greedy_route(G, start, target, coords)

        if success:
            successes += 1
            lengths.append(len(path) - 1)

    print("\n===== RESULTS =====")
    print("Trials:", trials)
    print("Successes:", successes)

    if trials > 0:
        print("Success rate:", successes / trials)

    if lengths:
        avg_len = sum(lengths) / len(lengths)
        print("Average path length:", avg_len)

# -----------------------------
# MAIN
# -----------------------------

def main():

    print("\n===== GOWALLA GREEDY ROUTING EXPERIMENT =====\n")

    print("Loading checkins...")
    user_checkins = load_checkins(CHECKIN_FILE)

    print("Users loaded:", len(user_checkins))

    G, home, shortcuts = build_graph(user_checkins)

    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    print("Shortcuts:", len(shortcuts))

    run_trials(G, home, TRIALS)


if __name__ == "__main__":
    main()