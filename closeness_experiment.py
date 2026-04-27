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
SAMPLE_SIZE = 1500   # try 500–3000 depending on speed
TRIALS = 1000        
LOCAL_K = 4   # number of local neighbors
PRED_TRIALS = 1000


# -----------------------------
# HAVERSINE DISTANCE
# -----------------------------

def haversine(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    R = 6371

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

            user_checkins.setdefault(user, []).append((lat, lon))

    return user_checkins

# -----------------------------
# HOME LOCATION
# -----------------------------

def compute_home_locations(user_checkins):

    home = {}

    for user, locs in user_checkins.items():

        counts = Counter(locs)
        max_count = max(counts.values())

        candidates = [loc for loc, c in counts.items() if c == max_count]

        if len(candidates) == 1:
            home[user] = candidates[0]
        else:
            for loc in reversed(locs):
                if loc in candidates:
                    home[user] = loc
                    break

    return home

# -----------------------------
# LOCAL EDGES (FAST VERSION)
# -----------------------------

def add_local_edges_directional(G, home):

    users = list(home.keys())

    for u in users:

        lat_u, lon_u = home[u]

        best = {
            "north": (None, float("inf")),
            "south": (None, float("inf")),
            "east":  (None, float("inf")),
            "west":  (None, float("inf")),
        }

        for v in users:
            if u == v:
                continue

            lat_v, lon_v = home[v]
            d = haversine((lat_u, lon_u), (lat_v, lon_v))

            if lat_v > lat_u and d < best["north"][1]:
                best["north"] = (v, d)
            if lat_v < lat_u and d < best["south"][1]:
                best["south"] = (v, d)
            if lon_v > lon_u and d < best["east"][1]:
                best["east"] = (v, d)
            if lon_v < lon_u and d < best["west"][1]:
                best["west"] = (v, d)

        for direction in best.values():
            if direction[0] is not None:
                G.add_edge(u, direction[0])

# -----------------------------
# SHORTCUTS
# -----------------------------

def build_home_lookup(home):
    return {loc: user for user, loc in home.items()}

def add_shortcuts(G, user_checkins, home_lookup):

    shortcuts = []

    for u, checkins in user_checkins.items():

        if u not in G:
            continue

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

def greedy_route(G, start, target, coords, max_steps=100):

    current = start
    path = [current]
    visited = set([current])

    target_coord = coords[target]

    for _ in range(max_steps):

        if current == target:
            return path, True

        neighbors = list(G.neighbors(current))
        if not neighbors:
            return path, False

        best = None
        best_dist = float("inf")

        for n in neighbors:
            d = haversine(coords[n], target_coord)

            if d < best_dist:
                best_dist = d
                best = n

        if best is None or best in visited:
            return path, False

        current = best
        path.append(current)
        visited.add(current)

    return path, False

# -----------------------------
# BUILD GRAPH (WITH SAMPLING)
# -----------------------------

def build_graph(user_checkins):

    print("Computing home locations...")
    home_full = compute_home_locations(user_checkins)

    # SAMPLE USERS
    users = list(home_full.keys())
    sampled = random.sample(users, min(SAMPLE_SIZE, len(users)))

    home = {u: home_full[u] for u in sampled}

    print("Building graph...")
    G = nx.Graph()
    G.add_nodes_from(home.keys())

    print("Adding local edges...")
    add_local_edges_directional(G, home)

    print("Adding shortcuts...")
    home_lookup = build_home_lookup(home)

    user_checkins_sampled = {
        u: user_checkins[u] for u in sampled if u in user_checkins
    }

    shortcuts = add_shortcuts(G, user_checkins_sampled, home_lookup)

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

    print("\n===== GREEDY RESULTS =====")
    print("Trials:", trials)
    print("Success rate:", successes / trials if trials else 0)

    if lengths:
        print("Average path length:", sum(lengths) / len(lengths))

# -----------------------------
# PREDICTION EXPERIMENT
# -----------------------------

def prediction_experiment(G, home, trials=1000, delta=0.2):

    users = list(G.nodes())

    success = 0
    total = 0

    for _ in range(trials):

        start = random.choice(users)
        target = random.choice(users)

        if start == target:
            continue

        path, ok = greedy_route(G, start, target, home)

        if not ok:
            continue

        final_node = path[-1]

        predicted = haversine(home[final_node], home[target])
        actual = haversine(home[start], home[target])

        if actual == 0:
            continue

        ratio = predicted / actual

        if (1 - delta) <= ratio <= (1 + delta):
            success += 1

        total += 1

    print("\n=== PREDICTION RESULTS ===")
    print("Trials:", total)
    print("Accuracy:", success / total if total else 0)

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

    prediction_experiment(G, home, PRED_TRIALS)

#    for d in [0.1, 0.2, 0.5]:
#        prediction_experiment(G, home, 1000, d)
# Try after verifying it's right

    print("\nDone.")

if __name__ == "__main__":
    main()