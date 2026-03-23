import random
import networkx as nx
import matplotlib.pyplot as plt

n = 5
random.seed(42)

def manhattan(u, v):
    return abs(u[0] - v[0]) + abs(u[1] - v[1])

nodes = [(i, j) for i in range(n) for j in range(n)]
G = nx.DiGraph()
G.add_nodes_from(nodes)

# ---- build local lattice (4 neighbors) ----
for x, y in nodes:
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        nx_, ny_ = x + dx, y + dy
        if (nx_, ny_) in nodes:
            G.add_edge((x, y), (nx_, ny_))

# ---- add ONE Kleinberg shortcut per node ----
for u in nodes:
    rows = []
    total_weight = 0.0

    # compute weights
    for v in nodes:
        if u == v:
            continue
        d = manhattan(u, v)
        w = 1 / (d ** 2)
        rows.append((v, w))
        total_weight += w

    # compute CDF
    cumulative = 0.0
    cdf_table = []
    for v, w in rows:
        p = w / total_weight
        cumulative += p
        cdf_table.append((v, cumulative))

    # pick random r
    r = random.uniform(0, 1)

    # select last v where CDF < r
    chosen_v = cdf_table[0][0]
    for v, cdf in cdf_table:
        if cdf < r:
            chosen_v = v


    # add directed shortcut
    G.add_edge(u, chosen_v)

def visualize_kleinberg(G, n):
    pos = {node: node for node in G.nodes()}  # grid layout

    local_edges = []
    shortcut_edges = []

    for u, v in G.edges():
        if manhattan(u, v) == 1:
            local_edges.append((u, v))
        else:
            shortcut_edges.append((u, v))

    plt.figure(figsize=(7, 7))

    # draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=300,
        node_color="black"
    )

    # draw local lattice edges
    nx.draw_networkx_edges(
        G, pos,
        edgelist=local_edges,
        edge_color="gray",
        width=2,
        arrows=False
    )

    # draw shortcut edges (directed)
    nx.draw_networkx_edges(
        G, pos,
        edgelist=shortcut_edges,
        edge_color="red",
        width=2,
        arrows=True,
        arrowstyle="->",
        arrowsize=15
    )


    nx.draw_networkx_labels(
        G, pos,
        font_size=8,
        font_color="white"
    )

    plt.title("5×5 Kleinberg Small-World Network\nGray = local edges, Red = directed shortcuts")
    plt.axis("off")
    plt.show()

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

print("\nKleinberg Shortcuts (one per node):")
print("----------------------------------")

for u in G.nodes():
    shortcuts = [
        v for v in G.successors(u)
        if manhattan(u, v) > 1
    ]

    if shortcuts:
        # should be exactly 1
        print(f"{u} -> {shortcuts[0]}")
    else:
        print(f"{u} -> NO SHORTCUT (ERROR)")


# ---- call visualization ----
visualize_kleinberg(G, n)

