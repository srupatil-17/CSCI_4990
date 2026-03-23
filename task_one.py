import random
import networkx as nx
import matplotlib.pyplot as plt


# Parameters
n = 20          # grid size (n x n)
r = 2          # Kleinberg exponent
seed = 42
random.seed(seed)


# Distance function
def manhattan(u, v):
    return abs(u[0] - v[0]) + abs(u[1] - v[1])

def create_nodes(n):
    return [(i, j) for i in range(n) for j in range(n)]

def add_local_edges(G, nodes):
    for (x, y) in nodes:
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx_, ny_ = x + dx, y + dy
            if (nx_, ny_) in nodes:
                G.add_edge((x, y), (nx_, ny_))
                G.add_edge((nx_, ny_), (x, y))


def compute_probability_table(u, nodes, r):
    table = []
    total_weight = 0.0

    for v in nodes:
        if u == v:
            continue

        d = manhattan(u, v)

        # distance cannot be one for shortcuts
        if d <= 1:
            continue

        w = 1 / (d ** r)
        total_weight += w
        table.append((v, d, w))

    normalized = []
    cdf = 0.0
    for v, d, w in table:
        p = w / total_weight
        cdf += p
        normalized.append((v, d, w, p, cdf))

    return normalized, total_weight


# Pick ONE shortcut for node u
def pick_one(u, nodes, r):
    table, total_weight = compute_probability_table(u, nodes, r)
    rand = random.random()

    for v, d, w, p, cdf in table:
        if cdf >= rand:
            return v

    return table[-1][0]  # fallback (numerical safety)


# Build Kleinberg graph
def build_kleinberg_graph(n, r):
    G = nx.DiGraph()
    nodes = create_nodes(n)
    G.add_nodes_from(nodes)

    add_local_edges(G, nodes)

    shortcuts = []

    for u in nodes:
        v = pick_one(u, nodes, r)
        G.add_edge(u, v)
        shortcuts.append((u, v))

    return G, shortcuts

# Prints table
def print_probability_tables(nodes, r):
    for u in nodes:
        table, total_weight = compute_probability_table(u, nodes, r)
        print(f"\nNode {u}")
        print("v\t d\t weight\t\t p\t\t CDF")
        print("-" * 55)
        for v, d, w, p, cdf in table:
            print(f"{v}\t {d}\t {w:.5f}\t {p:.5f}\t {cdf:.5f}")
        print(f"TOTAL WEIGHT = {total_weight:.5f}")

# Shortcut time
def print_shortcuts(shortcuts):
    print("\nDirected shortcuts (u → v):")
    print("----------------------------")
    for u, v in shortcuts:
        print(f"{u} → {v}   d={manhattan(u, v)}")

# Visualization
def visualize_kleinberg(G, shortcuts):
    n = int(len(G.nodes()) ** 0.5)

    pos = {node: node for node in G.nodes()}

    local_edges = []
    shortcut_edges = set(shortcuts)

    for u, v in G.edges():
        if (u, v) in shortcut_edges:
            continue
        if manhattan(u, v) == 1:
            local_edges.append((u, v))

    # Scale visual parameters by graph size
    node_size = max(10, 400 // n)
    edge_width = max(0.5, 3 // n)
    shortcut_width = max(1, 6 // n)

    show_labels = n <= 10

    plt.figure(figsize=(8, 8))

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_size,
        node_color="black"
    )

    nx.draw_networkx_edges(
        G, pos,
        edgelist=local_edges,
        edge_color="lightgray",
        width=edge_width,
        arrows=False
    )

    nx.draw_networkx_edges(
        G, pos,
        edgelist=shortcut_edges,
        edge_color="red",
        width=shortcut_width,
        arrows=True,
        arrowstyle="->",
        arrowsize=10
    )

    if show_labels:
        nx.draw_networkx_labels(
            G, pos,
            font_size=8,
            font_color="white"
        )

    plt.title(f"{n}×{n} Kleinberg Small-World Network\nGray = local edges, Red = shortcuts")
    plt.axis("off")
    plt.tight_layout()
    plt.show()



# Approximate diameter
def approximate_diameter(G, trials=10):
    nodes = list(G.nodes())
    max_dist = 0

    for _ in range(trials):
        start = random.choice(nodes)
        lengths = nx.single_source_shortest_path_length(G, start)
        max_dist = max(max_dist, max(lengths.values()))

    return max_dist

def shortcut_sanity_check(shortcuts):
    from collections import Counter

    outgoing = Counter()
    incoming = Counter()
    distances = []

    for u, v in shortcuts:
        outgoing[u] += 1
        incoming[v] += 1
        distances.append(manhattan(u, v))

    print("\nShortcut Sanity Check")

    # Outgoing shortcuts per node
    outgoing_counts = list(outgoing.values())
    print(f"Outgoing shortcuts per node:")
    print(f"  min = {min(outgoing_counts)}")
    print(f"  max = {max(outgoing_counts)}")
    print(f"  avg = {sum(outgoing_counts) / len(outgoing_counts):.2f}")

    # Incoming shortcuts per node
    incoming_counts = list(incoming.values())
    print(f"\nIncoming shortcuts per node:")
    print(f"  min = {min(incoming_counts)}")
    print(f"  max = {max(incoming_counts)}")
    print(f"  avg = {sum(incoming_counts) / len(incoming_counts):.2f}")

    # Distances
    print(f"\nShortcut distances:")
    print(f"  min distance = {min(distances)}")
    print(f"  max distance = {max(distances)}")
    print(f"  avg distance = {sum(distances) / len(distances):.2f}")

    # Distance Probabilities
    dist_counts = Counter(distances)
    total_shortcuts = len(distances)
    print("\nShortcut distance probabilities:")
    for dist, count in sorted(dist_counts.items()):
        prob = count / total_shortcuts
        print(f"  distance {dist}: {count}/{total_shortcuts} = probability {prob:.3f}")


def run_experiment(grid_size, r):
    print(f"\n=== Running experiment for {grid_size}×{grid_size} grid ===")

    G, shortcuts = build_kleinberg_graph(grid_size, r)

    shortcut_sanity_check(shortcuts)
    plot_shortcut_distance_histogram(shortcuts, normalize=True)


def plot_shortcut_distance_histogram(shortcuts, normalize=False):
    distances = [manhattan(u, v) for u, v in shortcuts]

    avg_distance = sum(distances) / len(distances)
    print(f"Average shortcut distance: {avg_distance:.2f}")

    min_d = min(distances)
    max_d = max(distances)
    bins = range(min_d, max_d + 2)

    plt.figure(figsize=(8, 5))

    plt.hist(
        distances,
        bins=bins,
        align="left",
        rwidth=0.8,
        density=normalize
    )

    plt.xlabel("Shortcut distance (Manhattan)")
    plt.ylabel("Probability" if normalize else "Count")
    plt.title("Kleinberg Shortcut Distance Distribution")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# Main
if __name__ == "__main__":
    G, shortcuts = build_kleinberg_graph(n, r)


    # convert to undirected and extract LCC
    G_undir = G.to_undirected()
    lcc_nodes = max(nx.connected_components(G_undir), key=len)
    G_lcc = G_undir.subgraph(lcc_nodes).copy()

    # statistics
    num_nodes = G_lcc.number_of_nodes()
    num_edges = G_lcc.number_of_edges()
    avg_degree = sum(dict(G_lcc.degree()).values()) / num_nodes
    diam = approximate_diameter(G_lcc)



    print("\nTask 1.3.1 — Kleinberg Graph Statistics")
    print("--------------------------------------")
    print(f"|V| (LCC): {num_nodes}")
    print(f"|E| (LCC): {num_edges}")
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Approximate diameter: {diam}")
    print("Total shortcuts:", len(shortcuts))

    print("\nList of shortcuts (u → v):")
    print("---------------------------")
    for u, v in shortcuts:
        print(f"{u} → {v}   distance = {manhattan(u, v)}")


    shortcut_sanity_check(shortcuts)

    visualize_kleinberg(G, shortcuts)

    # Histogram (normalized = probability)
    plot_shortcut_distance_histogram(shortcuts, normalize=True)

    # Visualization (only for small grid)
    if n <= 10:
        visualize_kleinberg(G, shortcuts)


# This allows other programs to import and build the graph
def generate_graph(n, r, seed=42):
    random.seed(seed)
    G, shortcuts = build_kleinberg_graph(n, r)
    return G, shortcuts





