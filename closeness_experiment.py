import networkx as nx
import random
import math
import csv
import os
import matplotlib.pyplot as plt

# -----------------------------
# HAVERSINE DISTANCE (KM)
# -----------------------------

def haversine(a, b):
    lat1, lon1 = a
    lat2, lon2 = b

    R = 6371

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    x = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2

    return 2 * R * math.atan2(math.sqrt(x), math.sqrt(1 - x))


# -----------------------------
# SAMPLING NODE PAIRS
# -----------------------------

def sample_pairs(nodes, k=20000):
    pairs = []

    for _ in range(k):
        u = random.choice(nodes)
        v = random.choice(nodes)

        if u != v:
            pairs.append((u, v))

    return pairs


# -----------------------------
# DISTANCE SAMPLING (FOR ANALYSIS)
# -----------------------------

def sample_distances(coords, num_samples=10000):
    nodes = list(coords.keys())
    dists = []

    for _ in range(num_samples):
        u = random.choice(nodes)
        v = random.choice(nodes)

        if u != v:
            dists.append(haversine(coords[u], coords[v]))

    return dists


# -----------------------------
# THRESHOLD METHODS
# -----------------------------

def threshold_percentiles(distances):
    distances = sorted(distances)

    percentiles = [5, 10, 25, 50]
    thresholds = []

    for p in percentiles:
        idx = int(len(distances) * (p / 100))
        thresholds.append(distances[idx])

    return thresholds


def threshold_fixed():
    return [1, 5, 10, 50, 100]


def threshold_sqrt_n(coords):
    n = len(coords)
    return [math.sqrt(n) * 0.001]  # scaled heuristic


# -----------------------------
# MAIN EXPERIMENT
# -----------------------------

def closeness_experiment(G, coords, thresholds, num_samples=20000):

    nodes = list(G.nodes())
    pairs = sample_pairs(nodes, num_samples)

    results = []

    for T in thresholds:

        total_close = 0
        connected_close = 0

        for u, v in pairs:

            if u not in coords or v not in coords:
                continue

            d = haversine(coords[u], coords[v])

            if d <= T:
                total_close += 1

                if G.has_edge(u, v) or G.has_edge(v, u):
                    connected_close += 1

        pct = connected_close / total_close if total_close > 0 else 0

        results.append((T, total_close, connected_close, pct))

        print(f"T={T:.3f} | close={total_close} | connected={connected_close} | pct={pct:.4f}")

    return results


# -----------------------------
# SAVE RESULTS
# -----------------------------

def save_results(results, filename="closeness_results.csv"):

    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", filename)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "threshold_km",
            "total_close_pairs",
            "connected_pairs",
            "fraction_connected"
        ])

        for row in results:
            writer.writerow(row)

    print("Saved:", path)


# -----------------------------
# PLOT
# -----------------------------

def plot_results(results):

    T = [r[0] for r in results]
    pct = [r[3] for r in results]

    plt.figure()
    plt.plot(T, pct, marker='o')

    plt.xlabel("Distance Threshold (km)")
    plt.ylabel("Fraction Connected")
    plt.title("Closeness vs Connectivity")

    plt.grid()
    plt.show()


# -----------------------------
# MAIN (PLACEHOLDER)
# -----------------------------

def main():
    print("Load your Gowalla graph + coordinates before running")


if __name__ == "__main__":
    main()
