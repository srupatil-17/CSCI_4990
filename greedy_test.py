import networkx as nx
import matplotlib.pyplot as plt
import random
import csv
import os

from true_greedy_routing_experiments.true_greedy_routing import greedy_route, lattice_distance, verify_greedy_property, lattice_distance_coords

# before running dont forget about Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# and then .\venv\Scripts\Activate.ps1


# -----------------------------
# SETTINGS
# -----------------------------

GRID_SIZE = 6
Q = 1   # change this anytime


# -----------------------------
# DISTANCE
# -----------------------------

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# -----------------------------
# GENERATE KLEINBERG GRID
# -----------------------------

def generate_kleinberg(n, q):

    G = nx.DiGraph()
    nodes = [(i, j) for i in range(n) for j in range(n)]
    G.add_nodes_from(nodes)

    shortcuts = []

    # local edges (grid)
    for i, j in nodes:
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            ni, nj = i + dx, j + dy
            if (ni, nj) in nodes:
                G.add_edge((i, j), (ni, nj))

    # shortcuts
    for u in nodes:

        added = 0

        while added < q:

            candidates = [
                v for v in nodes
                if v != u
                and manhattan(u, v) > 1
                and not G.has_edge(u, v)   # avoid duplicates
            ]

            if not candidates:
                break

            weights = []
            for v in candidates:
                d = manhattan(u, v)
                weights.append((v, d ** -2))  # Kleinberg distribution

            total = sum(w for _, w in weights)
            r = random.random() * total

            cumulative = 0

            for v, w in weights:
                cumulative += w
                if cumulative > r:
                    G.add_edge(u, v)
                    shortcuts.append((u, v))
                    added += 1
                    break

    return G, shortcuts


# -----------------------------
# VISUALIZATION
# -----------------------------

def visualize(G, shortcuts, path, start, target):

    pos = {node: node for node in G.nodes()}

    plt.figure(figsize=(7,7))

    # split edges
    grid_edges = []
    shortcut_edges = set(shortcuts)

    for u, v in G.edges():
        if (u, v) in shortcut_edges:
            continue
        grid_edges.append((u, v))

    # draw base grid
    nx.draw_networkx_edges(
        G, pos,
        edgelist=grid_edges,
        edge_color="lightgray"
    )

    # draw shortcuts
    nx.draw_networkx_edges(
        G, pos,
        edgelist=shortcuts,
        edge_color="orange",
        style="dashed",
        width=1.5
    )

    # draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=100,
        node_color="lightgray"
    )

    # draw path edges
    path_edges = list(zip(path, path[1:]))

    nx.draw_networkx_edges(
        G, pos,
        edgelist=path_edges,
        edge_color="blue",
        width=3
    )

    # highlight path nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=path,
        node_color="blue",
        node_size=150
    )

    # start & target
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[start],
        node_color="green",
        node_size=250,
        label="Start"
    )

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[target],
        node_color="red",
        node_size=250,
        label="Target"
    )

    plt.title("Greedy Routing with Kleinberg Shortcuts\n(orange dashed = shortcuts)")
    plt.legend()
    plt.show()


# -----------------------------
# STEP DEBUG 
# -----------------------------

def print_greedy_steps(G, path, target):

    print("\n--- GREEDY DECISIONS ---")

    for i in range(len(path) - 1):

        current = path[i]
        nxt = path[i + 1]

        neighbors = list(G.neighbors(current))

        print(f"\nAt {current} -> target {target}")

        best = None
        best_dist = float("inf")

        for n in neighbors:
            d = lattice_distance_coords(n, target)
            print(f"  Neighbor {n} | dist = {d}")

            if d < best_dist:
                best_dist = d
                best = n

        print(f"Chosen -> {best}")

        if nxt != best:
            print("WARNING: Not greedy!")
        else:
            print("Greedy choice")


# -----------------------------
# EXPORT TABLES TO CHECK
# -----------------------------

def export_full_routing_table(G, shortcuts):

    os.makedirs("routing_tables", exist_ok=True)
    filename = os.path.join("routing_tables", "all_pairs_lattice_distance.csv")

    shortcut_set = set(shortcuts)

    with open(filename, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "source",
            "target",
            "lattice_distance",
            "graph_distance",
            "reachable",
            "uses_shortcut_in_shortest_path"
        ])

        # precompute graph distances (THIS uses shortcuts)
        graph_distances = dict(nx.all_pairs_shortest_path_length(G))

        for source in G.nodes():

            for target in G.nodes():

                # TRUE lattice distance (independent of graph)
                lattice_dist = sum(abs(a - b) for a, b in zip(source, target))

                # graph distance (may use shortcuts)
                if target in graph_distances.get(source, {}):
                    graph_dist = graph_distances[source][target]
                    reachable = True

                    # get one shortest path to check shortcuts
                    try:
                        path = nx.shortest_path(G, source, target)

                        uses_shortcut = any(
                            (path[i], path[i+1]) in shortcut_set
                            for i in range(len(path) - 1)
                        )

                    except:
                        uses_shortcut = False

                else:
                    graph_dist = float("inf")
                    reachable = False
                    uses_shortcut = False

                writer.writerow([
                    source,
                    target,
                    lattice_dist,
                    graph_dist,
                    reachable,
                    uses_shortcut
                ])

    print(f"\nSaved full routing table -> {filename}")



# -----------------------------
# AUTOMATE CHECKING
# -----------------------------

def verify_greedy_property(G, path, target):
    for i in range(len(path) - 1):
        current = path[i]
        chosen = path[i+1]

        neighbors = list(G.neighbors(current))

        best = min(neighbors, key=lambda n: lattice_distance_coords(n, target))

        if chosen != best:
            return False

    return True

def does_path_match_shortest(G, path, target):
    """
    Returns True if every step in the path matches
    the optimal shortest-path choice (from routing table logic).
    """

    # shortest path distances TO target (respect direction)
    sp_dist = nx.single_source_shortest_path_length(G.reverse(), target)

    for i in range(len(path) - 1):

        current = path[i]
        chosen_next = path[i + 1]

        neighbors = list(G.neighbors(current))
        if not neighbors:
            return False

        # compute best possible next step
        best_dist = float("inf")
        best_neighbors = []

        for n in neighbors:
            d = sp_dist.get(n, float("inf"))

            if d < best_dist:
                best_dist = d
                best_neighbors = [n]
            elif d == best_dist:
                best_neighbors.append(n)

        # check if greedy choice matches optimal choice
        if chosen_next not in best_neighbors:
            return False

    return True


# -----------------------------
# RUN TRIALS
# -----------------------------

def run_trials(num_trials=50):

    matches = 0
    successes = 0

    for _ in range(num_trials):

        G, shortcuts = generate_kleinberg(GRID_SIZE, Q)
        nodes = list(G.nodes())

        start = random.choice(nodes)
        target = random.choice(nodes)

        while start == target:
            target = random.choice(nodes)

        path, success = greedy_route(
            G,
            start,
            target,
            lattice_distance
        )

        if success:
            successes += 1

            if does_path_match_shortest(G, path, target):
                matches += 1

    print("\n===== TRIAL RESULTS =====")
    print(f"Trials: {num_trials}")
    print(f"Successful routes: {successes}")
    print(f"Greedy matches optimal: {matches}")

    if successes > 0:
        print(f"Match rate: {matches / successes:.3f}")

# -----------------------------
# EXPORT GREEDY PATH
# -----------------------------

def export_greedy_path(G, path, target):

    os.makedirs("path_comparisons", exist_ok=True)
    filename = os.path.join("path_comparisons", "greedy_path.csv")

    with open(filename, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "step",
            "node",
            "lattice_distance_to_target"
        ])

        for i, node in enumerate(path):
            d = lattice_distance_coords(node, target)

            writer.writerow([
                i,
                node,
                d
            ])

    print(f"Saved greedy path to {filename}")

# -----------------------------
# EXPORT SHORTEST PATH
# -----------------------------

def export_shortest_path(G, start, target):

    os.makedirs("path_comparisons", exist_ok=True)
    filename = os.path.join("path_comparisons", "shortest_path.csv")

    try:
        path = nx.shortest_path(G, start, target)
    except:
        print("No shortest path found")
        return

    with open(filename, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "step",
            "node",
            "graph_distance_to_target"
        ])

        for i, node in enumerate(path):
            try:
                d = nx.shortest_path_length(G, node, target)
            except:
                d = float("inf")

            writer.writerow([
                i,
                node,
                d
            ])

    print(f"Saved shortest path to {filename}")


    # CHANGEEEEEEEEEEE
    # MAKE GREEDY VERSION - LATTICE DISTANCE 
    # DO NOT DELETE

# -----------------------------
# MAIN
# -----------------------------

def main():

    print("\n===== GREEDY + SHORTCUT VISUAL TEST =====\n")

    G, shortcuts = generate_kleinberg(GRID_SIZE, Q)

    nodes = list(G.nodes())

    start = random.choice(nodes)
    target = random.choice(nodes)

    while start == target:
        target = random.choice(nodes)

    export_full_routing_table(G, shortcuts)

    print("Start:", start)
    print("Target:", target)
    print("q (shortcuts per node):", Q)

    path, success = greedy_route(
        G,
        start,
        target,
        lattice_distance
    )

    print("Path:", path)
    print("Steps:", len(path) - 1)
    print("Success:", success)

    # verify greedy behavior
    print_greedy_steps(G, path, target)

    print("Strict greedy check:", verify_greedy_property(G, path, target))

    # export csv files
    export_greedy_path(G, path, target)
    export_shortest_path(G, start, target)



    visualize(G, shortcuts, path, start, target)
    #run_trials(100)


if __name__ == "__main__":
    main()